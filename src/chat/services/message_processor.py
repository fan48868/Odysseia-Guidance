# -*- coding: utf-8 -*-

import discord
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import re
import asyncio
import aiohttp

from src import config
from src.chat.config import chat_config
from src.chat.utils.database import chat_db_manager

log = logging.getLogger(__name__)

# 定义一个正则表达式来匹配自定义表情
# <a:emoji_name:emoji_id> (动态) 或 <:emoji_name:emoji_id> (静态)
EMOJI_REGEX = re.compile(r"<a?:(\w+):(\d+)>")


def detect_bot_location(channel: Any) -> Dict[str, Any]:
    """
    通用的bot位置检测函数，用于检测bot当前所在的频道或帖子。

    Args:
        channel: Discord的channel对象（可能是TextChannel或Thread）

    Returns:
        Dict[str, Any]: 包含位置信息的字典：
            - location_type: "thread" | "channel" - 位置类型
            - location_id: int - 当前位置的ID（频道ID或帖子ID）
            - thread_id: int | None - 如果是帖子，返回帖子ID；否则为None
            - parent_channel_id: int | None - 如果是帖子，返回父频道ID；否则为None
            - is_thread: bool - 是否在帖子中
    """
    import discord

    if isinstance(channel, discord.Thread):
        return {
            "location_type": "thread",
            "location_id": channel.id,
            "thread_id": channel.id,
            "parent_channel_id": channel.parent_id,
            "is_thread": True,
        }
    elif isinstance(channel, discord.TextChannel):
        return {
            "location_type": "channel",
            "location_id": channel.id,
            "thread_id": None,
            "parent_channel_id": None,
            "is_thread": False,
        }
    else:
        # 处理未知类型的情况
        return {
            "location_type": "unknown",
            "location_id": getattr(channel, "id", None),
            "thread_id": None,
            "parent_channel_id": None,
            "is_thread": False,
        }


class MessageProcessor:
    """
    负责处理和解析 discord.Message 对象，提取用于 AI 对话所需的信息。
    """

    def _get_gif_size_limit_bytes(self, source: str = "generic") -> int:
        image_cfg = chat_config.IMAGE_PROCESSING_CONFIG
        if source == "emoji":
            max_mb = float(image_cfg.get("MAX_ANIMATED_EMOJI_SIZE_MB", 2))
        else:
            max_mb = float(image_cfg.get("MAX_GIF_SIZE_MB", 8))
        return int(max_mb * 1024 * 1024)

    async def _fetch_image_aio(
        self, session: aiohttp.ClientSession, url: str, proxy: Optional[str] = None
    ) -> Optional[bytes]:
        """下载图片"""
        try:
            headers = {
                "Accept": "image/gif,image/png,image/jpeg,image/webp,*/*",
                "User-Agent": "OdysseiaDiscordBot/1.0",
            }
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5),
                proxy=proxy,
                headers=headers,
            ) as response:
                response.raise_for_status()
                return await response.read()
        except asyncio.TimeoutError:
            log.warning(f"下载表情图片超时: {url}")
            return None
        except aiohttp.ClientError as e:
            log.warning(f"下载表情图片失败: {url}, 错误: {e}")
            return None

    def _guess_mime_type_from_url(self, url: str) -> str:
        """根据 URL 后缀猜测 MIME 类型。"""
        lowered = (url or "").lower().split("?", 1)[0]
        if lowered.endswith(".png"):
            return "image/png"
        if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
            return "image/jpeg"
        if lowered.endswith(".webp"):
            return "image/webp"
        if lowered.endswith(".gif"):
            return "image/gif"
        return "image/png"

    async def _extract_images_from_embed_urls(
        self, embeds: List[Any], source: str = "reply_embed"
    ) -> List[Dict[str, Any]]:
        """从 embed.image/embed.thumbnail 中提取图片并下载。"""
        if not embeds:
            return []

        urls: List[str] = []
        for embed in embeds:
            try:
                if (
                    getattr(embed, "image", None)
                    and getattr(embed.image, "url", None)
                ):
                    urls.append(embed.image.url)
                if (
                    getattr(embed, "thumbnail", None)
                    and getattr(embed.thumbnail, "url", None)
                ):
                    urls.append(embed.thumbnail.url)
            except Exception:
                continue

        # 去重并保持顺序
        unique_urls: List[str] = []
        seen = set()
        for u in urls:
            if u and u not in seen:
                unique_urls.append(u)
                seen.add(u)

        if not unique_urls:
            return []

        proxy_url = config.PROXY_URL
        results_list: List[Dict[str, Any]] = []
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(self._fetch_image_aio(session, u, proxy=proxy_url))
                for u in unique_urls
            ]
            fetched = await asyncio.gather(*tasks)

        for url, image_bytes in zip(unique_urls, fetched):
            if image_bytes:
                mime_type = self._guess_mime_type_from_url(url)
                if mime_type == "image/gif":
                    max_gif_size_bytes = self._get_gif_size_limit_bytes(source="generic")
                    if len(image_bytes) > max_gif_size_bytes:
                        log.warning(
                            "Embed GIF 超出大小限制，已跳过: %s (%s bytes > %s bytes)",
                            url,
                            len(image_bytes),
                            max_gif_size_bytes,
                        )
                        continue

                results_list.append(
                    {
                        "mime_type": mime_type,
                        "data": image_bytes,
                        "source": source,
                    }
                )

        return results_list

    async def _extract_emojis_as_images(
        self, content: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """从文本中提取自定义表情，下载图片，并用占位符替换文本"""
        emoji_images = []
        tasks = []
        matches = list(EMOJI_REGEX.finditer(content))

        if not matches:
            return content, []

        proxy_url = config.PROXY_URL
        async with aiohttp.ClientSession() as session:
            for match in matches:
                emoji_name, emoji_id = match.groups()
                extension = "gif" if match.group(0).startswith("<a:") else "png"
                url = f"https://cdn.discordapp.com/emojis/{emoji_id}.{extension}"
                tasks.append(
                    asyncio.create_task(
                        self._fetch_image_aio(session, url, proxy=proxy_url)
                    )
                )

            results = await asyncio.gather(*tasks)

        modified_content = content
        for match, image_bytes in zip(matches, results):
            if image_bytes:
                emoji_name = match.group(1)
                is_animated = match.group(0).startswith("<a:")
                mime_type = "image/gif" if is_animated else "image/png"

                if is_animated:
                    max_animated_emoji_size = self._get_gif_size_limit_bytes(source="emoji")
                    if len(image_bytes) > max_animated_emoji_size:
                        log.warning(
                            "动态表情 GIF 超出大小限制，已跳过: %s (%s bytes > %s bytes)",
                            emoji_name,
                            len(image_bytes),
                            max_animated_emoji_size,
                        )
                        continue

                emoji_images.append(
                    {
                        "mime_type": mime_type,
                        "data": image_bytes,
                        "source": "emoji",
                        "name": emoji_name,
                    }
                )
                modified_content = modified_content.replace(
                    match.group(0), f"__EMOJI_{emoji_name}__", 1
                )

        return modified_content, emoji_images

    async def process_message(
        self, message: discord.Message, bot: discord.Client
    ) -> Optional[Dict[str, Any]]:
        """
        处理传入的 discord 消息对象。
        如果消息来自一个不应被触发的频道（如永久面板或置顶帖子），则返回 None。
        """
        # 检查消息是否来自置顶帖子
        # 检查频道是否被禁言
        if await chat_db_manager.is_channel_muted(message.channel.id):
            channel_name = getattr(message.channel, "name", str(message.channel.id))
            log.debug(f"消息来自被禁言的频道 {channel_name}，已忽略。")
            return None

        # 检查消息是否来自置顶帖子
        if isinstance(message.channel, discord.Thread) and message.channel.flags.pinned:
            channel_name = getattr(message.channel, "name", str(message.channel.id))
            log.debug(f"消息来自置顶帖子 {channel_name}，已忽略。")
            return None

        # 检查消息是否来自配置中禁用的频道
        if message.channel.id in chat_config.DISABLED_INTERACTION_CHANNEL_IDS:
            channel_name = getattr(message.channel, "name", str(message.channel.id))
            log.debug(f"消息来自禁用的频道 {channel_name}，已忽略。")
            return None

        image_data_list = []
        video_data_list = []
        max_videos_per_message = int(
            chat_config.VIDEO_PROCESSING_CONFIG.get("MAX_VIDEOS_PER_MESSAGE", 1)
        )

        # 获取bot用户，优先使用 message.guild.me，如果是DM则使用 bot.user
        bot_user = message.guild.me if message.guild else bot.user

        if message.attachments:
            image_data_list.extend(
                await self._extract_images_from_attachments(message.attachments)
            )
            if len(video_data_list) < max_videos_per_message:
                remaining_video_slots = max_videos_per_message - len(video_data_list)
                video_data_list.extend(
                    await self._extract_videos_from_attachments(
                        message.attachments,
                        limit=remaining_video_slots,
                    )
                )

        replied_message_content = ""
        if message.reference and message.reference.message_id:
            try:
                ref_msg = await message.channel.fetch_message(
                    message.reference.message_id
                )
                if ref_msg:
                    # 核心修复：使用 'in' 和 '[]' 来访问 MessageSnapshot 的数据
                    if (
                        hasattr(ref_msg, "message_snapshots")
                        and ref_msg.message_snapshots
                    ):
                        log.debug(f"检测到消息快照，处理转发消息: {ref_msg.id}")
                        snapshot_content_parts = []

                        forwarder_name = ref_msg.author.display_name
                        original_author_name = "未知作者"

                        for snapshot in ref_msg.message_snapshots:
                            # 根据 discord.py 文档，MessageSnapshot 是一个对象，必须使用属性访问。
                            # 我们使用 hasattr() 来安全地检查属性是否存在。
                            if hasattr(snapshot, "author") and snapshot.author:  # type: ignore
                                # snapshot.author 是一个 User/Member 对象，它有 display_name 属性
                                original_author_name = snapshot.author.display_name  # type: ignore

                            if hasattr(snapshot, "content") and snapshot.content:
                                snapshot_content_parts.append(snapshot.content)

                            if hasattr(snapshot, "embeds") and snapshot.embeds:
                                for embed in snapshot.embeds:
                                    # embed 是一个 Embed 对象
                                    if embed.title:
                                        snapshot_content_parts.append(
                                            f"标题: {embed.title}"
                                        )
                                    if embed.description:
                                        snapshot_content_parts.append(
                                            f"描述: {embed.description}"
                                        )
                                    for field in embed.fields:
                                        snapshot_content_parts.append(
                                            f"{field.name}: {field.value}"
                                        )

                                image_data_list.extend(
                                    await self._extract_images_from_embed_urls(
                                        snapshot.embeds, source="snapshot_embed"
                                    )
                                )

                            if (
                                hasattr(snapshot, "attachments")
                                and snapshot.attachments
                            ):
                                # snapshot.attachments 是 Attachment 对象的列表
                                image_data_list.extend(
                                    await self._extract_images_from_attachments(
                                        snapshot.attachments
                                    )
                                )
                                if len(video_data_list) < max_videos_per_message:
                                    remaining_video_slots = (
                                        max_videos_per_message - len(video_data_list)
                                    )
                                    video_data_list.extend(
                                        await self._extract_videos_from_attachments(
                                            snapshot.attachments,
                                            limit=remaining_video_slots,
                                        )
                                    )

                        snapshot_full_text = "\n".join(
                            filter(None, snapshot_content_parts)
                        ).strip()
                        if snapshot_full_text:
                            lines = snapshot_full_text.split("\n")
                            formatted_quote = "\n> ".join(lines)
                            reply_header = f"> [{forwarder_name} 转发的来自 {original_author_name} 的消息]:"
                            replied_message_content = (
                                f"{reply_header}\n> {formatted_quote}\n\n"
                            )

                    else:
                        # 对非转发消息（包括embed命令）的常规处理
                        command_name = None
                        if ref_msg.embeds:
                            for embed in ref_msg.embeds:
                                if embed.footer and embed.footer.text:
                                    footer_text = embed.footer.text
                                    if "投喂" in footer_text:
                                        command_name = "/投喂"
                                    elif "忏悔" in footer_text:
                                        command_name = "/忏悔"
                                    break  # 找到一个就够了

                        embed_texts = []
                        if ref_msg.embeds:
                            for embed in ref_msg.embeds:
                                if embed.author and embed.author.name:
                                    author_label = (
                                        "投喂者"
                                        if command_name == "/投喂"
                                        else "忏悔者"
                                        if command_name == "/忏悔"
                                        else "作者"
                                    )
                                    embed_texts.append(
                                        f"{author_label}: {embed.author.name}"
                                    )
                                if embed.title:
                                    embed_texts.append(f"标题: {embed.title}")
                                if embed.description:
                                    embed_texts.append(f"描述: {embed.description}")
                                # 根据要求，不再将 embed 中的图片链接作为文本添加到上下文中
                                # if embed.image and embed.image.url: embed_texts.append(f"[图片]: {embed.image.url}")
                                for field in embed.fields:
                                    embed_texts.append(f"{field.name}: {field.value}")
                                if embed.footer and embed.footer.text:
                                    embed_texts.append(f"页脚: {embed.footer.text}")

                        embed_content = "\n".join(embed_texts)
                        ref_content_cleaned = self._clean_message_content(
                            ref_msg.content, ref_msg.mentions, bot_user
                        )

                        full_ref_content = [
                            ref for ref in [ref_content_cleaned, embed_content] if ref
                        ]
                        combined_content = "\n".join(full_ref_content).strip()

                        if combined_content:
                            lines = combined_content.split("\n")
                            formatted_quote = "\n> ".join(lines)

                            reply_header = ""
                            # 修复: ref_msg.embeds 是一个列表，我们应该从列表的第一个元素获取 author
                            embed_author_name = (
                                ref_msg.embeds[0].author.name
                                if ref_msg.embeds and ref_msg.embeds[0].author
                                else None
                            )

                            if (
                                bot_user
                                and ref_msg.author.id == bot_user.id
                                and embed_author_name
                            ):
                                command_context = (
                                    f"的 {command_name} 回应"
                                    if command_name
                                    else "的回应"
                                )
                                reply_header = f"> [类脑娘对 {embed_author_name} {command_context}]:"
                            else:
                                reply_header = f"> [{ref_msg.author.display_name}]:"

                            replied_message_content = (
                                f"{reply_header}\n> {formatted_quote}\n\n"
                            )

                        image_data_list.extend(
                            await self._extract_images_from_embed_urls(
                                ref_msg.embeds, source="reply_embed"
                            )
                        )

                        if ref_msg.attachments:
                            image_data_list.extend(
                                await self._extract_images_from_attachments(
                                    ref_msg.attachments
                                )
                            )
                            if len(video_data_list) < max_videos_per_message:
                                remaining_video_slots = (
                                    max_videos_per_message - len(video_data_list)
                                )
                                video_data_list.extend(
                                    await self._extract_videos_from_attachments(
                                        ref_msg.attachments,
                                        limit=remaining_video_slots,
                                    )
                                )

            except (discord.NotFound, discord.Forbidden):
                log.warning(
                    f"无法找到或无权访问被回复的消息 ID: {message.reference.message_id}"
                )
            except Exception as e:
                log.error(f"处理被回复消息时出错: {e}", exc_info=True)

        content_with_placeholders, emoji_images = await self._extract_emojis_as_images(
            message.content
        )
        image_data_list.extend(emoji_images)

        clean_content = self._clean_message_content(
            content_with_placeholders, message.mentions, bot_user
        )

        return {
            "user_content": clean_content,
            "replied_content": replied_message_content,
            "image_data_list": image_data_list,
            "video_data_list": video_data_list,
        }

    async def _extract_images_from_attachments(
        self, attachments: List[discord.Attachment]
    ) -> List[Dict[str, Any]]:
        """从附件列表中提取图片数据。"""
        image_data_list = []
        for attachment in attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    content_type = attachment.content_type.lower()
                    extension = os.path.splitext(attachment.filename or "")[1].lower()
                    is_gif = content_type == "image/gif" or extension == ".gif"

                    if is_gif and attachment.size:
                        max_gif_size_bytes = self._get_gif_size_limit_bytes(source="generic")
                        if attachment.size > max_gif_size_bytes:
                            log.warning(
                                "GIF 附件超出大小限制，已跳过: %s (%s bytes > %s bytes)",
                                attachment.filename,
                                attachment.size,
                                max_gif_size_bytes,
                            )
                            continue

                    image_bytes = await attachment.read()
                    if image_bytes:
                        if is_gif:
                            max_gif_size_bytes = self._get_gif_size_limit_bytes(source="generic")
                            if len(image_bytes) > max_gif_size_bytes:
                                log.warning(
                                    "GIF 附件读取后仍超出大小限制，已跳过: %s (%s bytes > %s bytes)",
                                    attachment.filename,
                                    len(image_bytes),
                                    max_gif_size_bytes,
                                )
                                continue

                        image_data_list.append(
                            {
                                "mime_type": attachment.content_type,
                                "data": image_bytes,
                                "source": "attachment",
                            }
                        )
                        log.debug(
                            f"成功读取图片附件: {attachment.filename}, 大小: {len(image_bytes)} 字节"
                        )
                except Exception as e:
                    log.error(f"读取图片附件 {attachment.filename} 时出错: {e}")
        return image_data_list

    def _guess_video_mime_type_from_extension(self, extension: str) -> str:
        ext = (extension or "").lower()
        mapping = {
            ".mp4": "video/mp4",
            ".mpeg": "video/mpeg",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".flv": "video/x-flv",
            ".mpg": "video/mpg",
            ".webm": "video/webm",
            ".wmv": "video/x-ms-wmv",
            ".3gp": "video/3gpp",
            ".3gpp": "video/3gpp",
        }
        return mapping.get(ext, "video/mp4")

    async def _extract_videos_from_attachments(
        self,
        attachments: List[discord.Attachment],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """从附件列表中提取视频数据，并执行格式/大小限制。"""
        video_data_list: List[Dict[str, Any]] = []
        if not attachments:
            return video_data_list

        video_config = chat_config.VIDEO_PROCESSING_CONFIG
        max_videos_per_message = int(video_config.get("MAX_VIDEOS_PER_MESSAGE", 1))
        if limit is not None:
            max_videos_per_message = min(max_videos_per_message, max(0, int(limit)))

        if max_videos_per_message <= 0:
            return video_data_list

        max_size_mb = float(video_config.get("MAX_VIDEO_SIZE_MB", 20))
        max_size_bytes = int(max_size_mb * 1024 * 1024)

        allowed_mime_types = {
            str(m).lower()
            for m in video_config.get("ALLOWED_VIDEO_MIME_TYPES", set())
        }
        allowed_extensions = {
            str(ext).lower()
            for ext in video_config.get("ALLOWED_VIDEO_EXTENSIONS", set())
        }

        for attachment in attachments:
            if len(video_data_list) >= max_videos_per_message:
                break

            content_type = (attachment.content_type or "").lower()
            extension = os.path.splitext(attachment.filename or "")[1].lower()

            mime_allowed = bool(content_type) and content_type in allowed_mime_types
            extension_allowed = bool(extension) and extension in allowed_extensions

            if not (mime_allowed or extension_allowed):
                continue

            if attachment.size and attachment.size > max_size_bytes:
                log.warning(
                    "视频附件超出限制，已跳过: %s (%s bytes > %s bytes)",
                    attachment.filename,
                    attachment.size,
                    max_size_bytes,
                )
                continue

            try:
                video_bytes = await attachment.read()
                if not video_bytes:
                    continue

                if len(video_bytes) > max_size_bytes:
                    log.warning(
                        "视频附件读取后仍超出限制，已跳过: %s (%s bytes > %s bytes)",
                        attachment.filename,
                        len(video_bytes),
                        max_size_bytes,
                    )
                    continue

                final_mime_type = (
                    content_type
                    if content_type in allowed_mime_types
                    else self._guess_video_mime_type_from_extension(extension)
                )

                video_data_list.append(
                    {
                        "mime_type": final_mime_type,
                        "data": video_bytes,
                        "source": "attachment",
                        "filename": attachment.filename,
                    }
                )
                log.debug(
                    "成功读取视频附件: %s, mime=%s, 大小=%s 字节",
                    attachment.filename,
                    final_mime_type,
                    len(video_bytes),
                )
            except Exception as e:
                log.error(f"读取视频附件 {attachment.filename} 时出错: {e}")

        return video_data_list

    def _clean_message_content(
        self,
        content: str,
        mentions: list,
        bot_user: Optional[discord.Member | discord.ClientUser],
    ) -> str:
        """
        清理消息内容，将对自身的@mention替换为名字，并移除其他@mention。
        """
        content = content.replace("\\_", "_")

        for user in mentions:
            mention_str_1 = f"<@{user.id}>"
            mention_str_2 = f"<@!{user.id}>"
            if bot_user and user.id == bot_user.id:
                replacement = f"@{bot_user.display_name}"  # type: ignore
                content = content.replace(mention_str_1, replacement).replace(
                    mention_str_2, replacement
                )
            # else:
            #     # 根据新需求，不再移除对其他用户的 @mention
            #     # 这样 AI 模型就可以接收到 <@user_id> 格式的字符串并提取 ID
            #     pass

        # content = regex_service.clean_user_input(content)
        content = content.strip()

        return content


# 创建一个单例
message_processor = MessageProcessor()
