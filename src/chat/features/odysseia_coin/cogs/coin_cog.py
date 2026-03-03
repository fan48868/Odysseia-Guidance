import logging
import random
import re
import time

import discord
import httpx
from discord import app_commands
from discord.ext import commands

from src.chat.features.odysseia_coin.service.coin_service import coin_service
from src.chat.features.odysseia_coin.ui.shop_ui import SimpleShopView
from src.chat.config import chat_config
from src.chat.features.odysseia_coin.service.shop_service import shop_service
from src.chat.features.chat_settings.services.chat_settings_service import (
    chat_settings_service,
)

log = logging.getLogger(__name__)

EMOJI_RE = re.compile(
    r"((?:[\U0001F1E6-\U0001F1FF]{2})|(?:[\U0001F300-\U0001FAFF\u2600-\u27BF])(?:\uFE0F|\uFE0E)?(?:[\U0001F3FB-\U0001F3FF])?(?:\u200D(?:[\U0001F300-\U0001FAFF\u2600-\u27BF])(?:\uFE0F|\uFE0E)?(?:[\U0001F3FB-\U0001F3FF])?)*)"
)


class CoinCog(commands.Cog):
    """处理与类脑币相关的事件和命令"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._react_cd_until = 0.0
        self._react_cd_seconds = 120

    async def _ask_react_ai(self, text: str):
        cfg = chat_config.REACTION_AI
        base_url = str(cfg.get("url", "")).rstrip("/")
        if not base_url:
            return None

        api_url = (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )

        payload = {
            "model": cfg.get("model", "qwen2.5:1.5b"),
            "messages": [
                {"role": "system", "content": cfg.get("prompt", "")},
                {"role": "user", "content": text[:500]},
            ],
            "stream": False,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.2,
            "max_tokens": 16,
            "stop": ["\n", "<think>", "</think>"],
            # Ollama 原生参数双保险，限制输出长度并阻断思维链标记
            "options": {
                "num_predict": 16,
                "temperature": 0.2,
                "stop": ["\n", "<think>", "</think>"],
            },
        }

        try:
            timeout = int(cfg.get("timeout", 8))
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(api_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            log.warning(f"[react-ai] 调用失败，跳过本次主动反应: {e}")
            return None

        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception:
            log.warning(f"[react-ai] 返回格式异常，跳过本次主动反应: {data}")
            return None

    def _pick_emoji(self, raw):
        if not raw:
            return None

        match = EMOJI_RE.search(str(raw))
        if not match:
            return None

        return match.group(0)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """监听用户每日首次发言"""
        if message.author.bot:
            return

        # 全局开关：读取 /聊天设置 的“表情反应”独立开关
        reaction_enabled = True
        if message.guild:
            try:
                reaction_key = f"reaction_enabled:{message.guild.id}"
                reaction_raw = await chat_settings_service.db_manager.get_global_setting(
                    reaction_key
                )
                reaction_enabled = (
                    reaction_raw.lower() == "true"
                    if reaction_raw is not None
                    else True
                )
            except Exception as e:
                log.warning(f"[react-ai] 读取表情反应开关失败，默认允许反应: {e}")

        if reaction_enabled:
            # 自动反应：仅当消息 @ 了 bot 且包含指定关键词时，为该用户消息添加表情反应
            trigger_keywords = [
                "文爱",
                "亲",
                "抱",
                "摸",
                "脚",
                "腿",
                "主人",
                "色色",
                "mua",
            ]
            if (
                self.bot.user
                and self.bot.user in message.mentions
                and any(keyword in message.content for keyword in trigger_keywords)
            ):
                try:
                    await message.add_reaction("🥵")
                except (discord.Forbidden, discord.HTTPException) as e:
                    log.debug(f"自动添加反应失败（消息ID: {message.id}）: {e}")

            # 排除特定命令前缀的消息，避免与命令冲突
            command_prefix = self.bot.command_prefix
            # command_prefix can be a string, or a list/tuple of strings.
            # startswith requires a string or a tuple of strings.
            if isinstance(command_prefix, str):
                if message.content.startswith(command_prefix):
                    return
            elif isinstance(command_prefix, (list, tuple)):
                if message.content.startswith(tuple(command_prefix)):
                    return

            # 主动反应：按概率触发，调用本地模型返回 emoji
            try:
                rate = float(chat_config.REACTION_AI.get("rate", 0.2))
            except Exception:
                rate = 0.2

            now = time.monotonic()
            if now >= self._react_cd_until and rate > 0 and random.random() < rate:
                raw = await self._ask_react_ai(message.content or "")
                emoji = self._pick_emoji(raw)

                if emoji:
                    try:
                        await message.add_reaction(emoji)
                        # 仅在反应成功后进入冷却
                        self._react_cd_until = (
                            time.monotonic() + self._react_cd_seconds
                        )
                    except (discord.Forbidden, discord.HTTPException) as e:
                        log.error(
                            f"[react-ai] 点反应失败（消息ID: {message.id}）"
                            f" emoji={emoji!r} raw={raw!r} err={e}",
                            exc_info=True,
                        )

        try:
            reward_granted = await coin_service.grant_daily_message_reward(
                message.author.id
            )
            if reward_granted:
                log.info(
                    f"用户 {message.author.name} ({message.author.id}) 获得了每日首次发言奖励。"
                )
        except Exception as e:
            log.error(
                f"处理用户 {message.author.id} 的每日发言奖励时出错: {e}", exc_info=True
            )

    async def handle_new_thread_reward(
        self, thread: discord.Thread, first_message: discord.Message
    ):
        """
        由中央事件处理器调用的公共方法，用于处理新帖子的发币奖励。
        """
        try:
            author = first_message.author
            if author.bot:
                return

            # 检查服务器是否在奖励列表中已由中央处理器完成，这里直接执行逻辑
            log.info(f"[CoinCog] 接收到新帖子进行奖励处理: {thread.name} ({thread.id})")
            reward_amount = chat_config.COIN_CONFIG["FORUM_POST_REWARD"]
            channel_name = thread.parent.name if thread.parent else "未知频道"
            reason = f"在频道 {channel_name} 发布新帖"
            new_balance = await coin_service.add_coins(author.id, reward_amount, reason)
            log.info(
                f"[CoinCog] 用户 {author.name} ({author.id}) 因发帖获得 {reward_amount} 类脑币。新余额: {new_balance}"
            )

        except Exception as e:
            log.error(
                f"[CoinCog] 处理帖子 {thread.id} 的发帖奖励时出错: {e}", exc_info=True
            )

    @app_commands.command(name="类脑商店", description="打开商店，购买商品。")
    async def shop(self, interaction: discord.Interaction):
        """斜杠命令：打开商店"""
        try:
            # 1. 准备数据
            shop_data = await shop_service.prepare_shop_data(interaction.user.id)

            is_thread_author = False
            if isinstance(interaction.channel, discord.Thread):
                if interaction.user.id == interaction.channel.owner_id:
                    is_thread_author = True
                    shop_data.thread_id = interaction.channel.id
            shop_data.show_tutorial_button = is_thread_author

            # 2. 创建视图
            view = SimpleShopView(self.bot, interaction.user, shop_data)

            # 3. 启动视图（视图现在自己处理交互响应）
            await view.start(interaction)

        except Exception as e:
            log.error(f"打开商店时出错: {e}", exc_info=True)
            error_message = "打开商店时发生错误，请稍后再试。"
            # 根据交互是否已被响应来决定是使用 followup 还是 send_message
            if interaction.response.is_done():
                await interaction.followup.send(error_message, ephemeral=True)
            else:
                # 在初始 defer 之前就发生错误的情况
                await interaction.response.send_message(error_message, ephemeral=True)


async def setup(bot: commands.Bot):
    await bot.add_cog(CoinCog(bot))
