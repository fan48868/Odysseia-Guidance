# -*- coding: utf-8 -*-

from typing import Any, Awaitable, Callable, Dict

import discord
from discord import Interaction
from discord.ui import TextInput


class CustomModelConfigModal(discord.ui.Modal):
    """配置 custom(OpenAI 兼容) 模型所需的环境变量。"""

    API_KEY_INPUT_MAX_LENGTH = 4000

    def __init__(
        self,
        *,
        title: str,
        current_url: str,
        current_api_key: str,
        current_api_key_omitted: bool,
        current_model_name: str,
        current_enable_vision: str,
        current_enable_video_input: str,
        on_submit_callback: Callable[[Interaction, Dict[str, Any]], Awaitable[None]],
    ):
        super().__init__(title=title)
        self.on_submit_callback = on_submit_callback

        api_key_placeholder = (
            "当前 inline key 过长未展示；留空表示保持不变，也可填 /data/*.json（会自动映射）"
            if current_api_key_omitted
            else "例如：sk-xxxx / vck_xxxx / /data/CUSTOM_MODEL_API_KEY.json；留空保持当前值"
        )

        self.url_input = TextInput(
            label="CUSTOM_MODEL_URL",
            placeholder="例如：https://xxx/v1",
            default=current_url or "",
            required=True,
            max_length=300,
            custom_id="custom_model_url",
        )
        self.api_key_input = TextInput(
            label="CUSTOM_MODEL_API_KEY",
            placeholder=api_key_placeholder,
            default=current_api_key or "",
            required=False,
            style=discord.TextStyle.paragraph,
            max_length=self.API_KEY_INPUT_MAX_LENGTH,
            custom_id="custom_model_api_key",
        )
        self.model_name_input = TextInput(
            label="CUSTOM_MODEL_NAME",
            placeholder="例如：gpt-4o-mini / minimax/minimax-m2.5",
            default=current_model_name or "",
            required=True,
            max_length=200,
            custom_id="custom_model_name",
        )
        self.enable_vision_input = TextInput(
            label="开启识图工具",
            placeholder="true / false",
            default=current_enable_vision or "false",
            required=True,
            max_length=10,
            custom_id="custom_model_enable_vision",
        )
        self.enable_video_input = TextInput(
            label="开启视频输入",
            placeholder="true / false",
            default=current_enable_video_input or "false",
            required=True,
            max_length=10,
            custom_id="custom_model_enable_video_input",
        )

        self.add_item(self.url_input)
        self.add_item(self.api_key_input)
        self.add_item(self.model_name_input)
        self.add_item(self.enable_vision_input)
        self.add_item(self.enable_video_input)

    async def on_submit(self, interaction: Interaction):
        settings = {
            "custom_model_url": (self.url_input.value or "").strip(),
            "custom_model_api_key": (self.api_key_input.value or "").strip(),
            "custom_model_name": (self.model_name_input.value or "").strip(),
            "custom_model_enable_vision": (self.enable_vision_input.value or "").strip(),
            "custom_model_enable_video_input": (
                self.enable_video_input.value or ""
            ).strip(),
        }
        await self.on_submit_callback(interaction, settings)
