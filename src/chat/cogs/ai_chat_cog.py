# -*- coding: utf-8 -*-

import discord
from discord.ext import commands
import logging
from typing import Optional
import re
import io
import asyncio

# 导入新的 Service
from src.chat.services.chat_service import chat_service
from src.chat.services.message_processor import message_processor
from src.chat.services.gemini_service import gemini_service
from src.chat.features.tools.functions.summarize_channel import text_to_summary_image


# 导入上下文服务

# 导入数据库管理器以进行黑名单检查和斜杠命令
from src import config
from src.chat.utils.database import chat_db_manager
from src.chat.config.chat_config import CHAT_ENABLED, MESSAGE_SETTINGS
from src.chat.config import chat_config
from src.chat.features.chat_settings.services.chat_settings_service import (
    chat_settings_service,
)
from src.chat.features.odysseia_coin.service.coin_service import coin_service

log = logging.getLogger(__name__)


class AIChatCog(commands.Cog):
    """处理AI聊天功能的Cog，包括@mention回复和斜杠命令"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # 服务实例的注入已由 main.py 统一处理，此处不再需要

    def _get_text_length_without_emojis(self, text: str) -> int:
        """计算移除Discord自定义表情后的文本长度。"""
        # 匹配 <a:name:id> 或 <:name:id> 格式的表情
        emoji_pattern = r"<a?:.+?:\d+>"
        text_without_emojis = re.sub(emoji_pattern, "", text)
        return len(text_without_emojis)

    async def _send_long_response_in_channel(
        self, message: discord.Message, response_text: str
    ) -> None:
        """将超长回复拆分后在当前频道连续发送，避免截断或私信。"""
        max_len = 1900
        chunks = []
        remaining = response_text

        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break

            split_pos = remaining.rfind("\n", 0, max_len)
            if split_pos == -1 or split_pos < max_len // 2:
                split_pos = max_len

            chunks.append(remaining[:split_pos].rstrip())
            remaining = remaining[split_pos:].lstrip()

        if not chunks:
            return

        await message.reply(chunks[0], mention_author=True)
        for chunk in chunks[1:]:
            await message.channel.send(chunk)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """
        监听所有消息，当bot被@mention时进行回复
        """
        if not CHAT_ENABLED:
            return

        # 忽略机器人自己的消息
        if message.author.bot:
            return

        # --- 核心前置检查 ---
        # 在处理任何逻辑之前，首先检查消息是否应该被 message_processor 忽略
        # 这会处理置顶帖和禁用频道的情况
        processed_data = await message_processor.process_message(message, self.bot)
        if processed_data is None:
            # 如果返回 None，说明消息来自一个应被忽略的源（如置顶帖），直接退出
            return

        # 检查消息是否符合处理条件：私聊 或 在服务器中被@
        is_dm = message.guild is None
        is_mentioned = self.bot.user in message.mentions

        if not is_dm and not is_mentioned:
            return

        # 新增：私信总开关（开发者豁免）
        if is_dm:
            dm_enabled = await chat_settings_service.is_global_dm_enabled()
            if not dm_enabled and message.author.id not in config.DEVELOPER_USER_IDS:
                log.info(
                    f"私信总开关已关闭，用户 {message.author.id} 非开发者，跳过私信处理。"
                )
                return

        # 新增：检查是否在帖子中，以及帖子创建者是否禁用了回复
        if isinstance(message.channel, discord.Thread):
            # 检查帖子的创建者
            thread_owner = message.channel.owner
            if thread_owner and await coin_service.blocks_thread_replies(
                thread_owner.id
            ):
                log.info(
                    f"帖子 '{message.channel.name}' 的创建者 {thread_owner.id} 已禁用回复，跳过消息处理。"
                )
                return

        # 黑名单检查
        if await chat_db_manager.is_user_globally_blacklisted(message.author.id):
            log.info(f"用户 {message.author.id} 在全局黑名单中，已跳过。")
            return

        # 在显示“输入中”之前执行所有前置检查
        if not await chat_service.should_process_message(message):
            return

        # 显示"正在输入"状态，直到AI响应生成完毕
        response_text = None
        async with message.channel.typing():
            # 注意：这里我们将已经处理过的数据传递下去
            response_text = await self.handle_chat_message(message, processed_data)

        # 在退出 typing 状态后发送回复
        if response_text:
            target_retry_error = "Cannot connect to host discord.com:443 ssl:default [None]"
            retry_delays = [2, 4, 6]  # 仅用于目标错误
            retry_count = 0

            while True:
                try:
                    # --- 响应发送逻辑 ---
                    # 动态获取上次调用的工具列表，如果不存在则为空列表
                    last_tools = getattr(gemini_service, "last_called_tools", [])

                    # 1. 如果调用了总结工具，总是转换为图片发送
                    if "summarize_channel" in last_tools:
                        log.info("调用了总结工具, 尝试转为图片发送。")
                        image_bytes = text_to_summary_image(response_text)
                        if image_bytes:
                            with io.BytesIO(image_bytes) as image_file:
                                await message.reply(
                                    file=discord.File(image_file, "summary.png"),
                                    mention_author=True,
                                )
                            # 发送成功后直接返回，不再执行后续逻辑
                            return
                        else:
                            log.error("总结图片生成失败，将作为文本尝试发送。")

                    # 2. 如果不是长篇总结，则检查是否在豁免频道或帖子 (常规长消息可直接发送)
                    is_unrestricted = (
                        message.channel.id in chat_config.UNRESTRICTED_CHANNEL_IDS
                        or isinstance(message.channel, discord.Thread)
                    )
                    if is_unrestricted:
                        await message.reply(response_text, mention_author=True)
                        return

                    # 3. 如果以上都不是，则检查是否为需要发送私信的普通长消息
                    if (
                        self._get_text_length_without_emojis(response_text)
                        > MESSAGE_SETTINGS["DM_THRESHOLD"]
                    ):
                        try:
                            channel_mention = (
                                message.channel.mention
                                if isinstance(
                                    message.channel, (discord.TextChannel, discord.Thread)
                                )
                                else "你们的私信"
                            )

                            await message.author.send(
                                f"刚刚在 {channel_mention} 频道里，你想听我说的话有点多，在这里悄悄告诉你哦：\n\n{response_text}"
                            )
                            log.info(
                                f"回复因过长已通过私信发送给 {message.author.display_name}"
                            )
                        except discord.Forbidden:
                            log.warning(
                                f"无法通过私信发送给 {message.author.display_name}，将在原频道回复提示信息。"
                            )
                            await message.reply(
                                "字太多啦，我不要刷屏。你的私信又关了，我就不给你讲啦！",
                                mention_author=True,
                            )
                        return

                    # 4. 默认情况：直接在频道回复短消息
                    await message.reply(response_text, mention_author=True)
                    return

                except discord.errors.HTTPException as e:
                    log.warning(f"发送回复时发生HTTP错误: {e}")
                    # 处理 400 Bad Request (error code: 50035): Invalid Form Body (Must be 2000 or fewer in length)
                    if (
                        e.status == 400
                        and e.code == 50035
                        and "Must be 2000 or fewer in length" in str(e)
                    ):
                        try:
                            await self._send_long_response_in_channel(
                                message, response_text
                            )
                            log.info(
                                f"因消息超长，已在原频道分段发送给 {message.author.display_name}"
                            )
                        except Exception as split_error:
                            log.error(
                                f"消息分段发送失败: {split_error}", exc_info=True
                            )
                            await message.reply(
                                "由于回复内容过长，分段发送也失败啦！",
                                mention_author=True,
                            )
                    return

                except Exception as e:
                    log.error(f"发送回复时发生未知错误: {e}", exc_info=True)

                    # 仅当命中特定网络错误时才重试
                    if target_retry_error not in str(e):
                        return

                    if retry_count >= len(retry_delays):
                        log.error("目标网络错误重试 3 次后仍失败，放弃本次回复。")
                        return

                    delay = retry_delays[retry_count]
                    retry_count += 1
                    log.warning(
                        f"命中目标网络错误，将在 {delay} 秒后进行第 {retry_count}/3 次重试。"
                    )
                    await asyncio.sleep(delay)

    async def handle_chat_message(
        self, message: discord.Message, processed_data: dict
    ) -> Optional[str]:
        """
        处理聊天消息（包括私聊和@mention），协调各个服务生成AI回复并返回其内容
        """
        try:
            # 1. MessageProcessor 的处理已前移到 on_message 中

            # 2. 使用 ChatService 获取AI回复
            # --- 新增：获取并传递位置信息 ---
            guild_name = message.guild.name if message.guild else "私信"
            location_name = ""
            if isinstance(message.channel, discord.Thread):
                # 如果是帖子（子区），显示“父频道 -> 帖子名”
                parent_channel_name = (
                    message.channel.parent.name
                    if message.channel.parent
                    else "未知频道"
                )
                location_name = f"{parent_channel_name} -> {message.channel.name}"
            elif isinstance(message.channel, discord.abc.GuildChannel):
                # 确保是服务器频道再获取名字
                location_name = message.channel.name
            else:
                # 否则（如私信），提供一个默认值
                location_name = "私信中"

            final_response = await chat_service.handle_chat_message(
                message, processed_data, guild_name, location_name
            )

            # 3. 返回回复内容
            return final_response

        except Exception as e:
            log.error(f"[AIChatCog] 处理@mention消息时发生顶层错误: {e}", exc_info=True)
            # 确保即使发生意外错误也有反馈
            return "抱歉，处理你的请求时遇到了一个未知错误。"


async def setup(bot: commands.Bot):
    """将这个Cog添加到机器人中"""
    await bot.add_cog(AIChatCog(bot))
