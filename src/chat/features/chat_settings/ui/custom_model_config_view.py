# -*- coding: utf-8 -*-

import logging
import os
from typing import Any, Dict, Optional, Tuple

from discord import ButtonStyle, Interaction
from discord.ui import Button, View
from dotenv import load_dotenv, set_key

from src import config
from src.chat.features.chat_settings.services.chat_settings_service import (
    ChatSettingsService,
)
from src.chat.features.chat_settings.ui.custom_model_config_modal import (
    CustomModelConfigModal,
)
from src.chat.services.gemini_service import gemini_service
from src.chat.services.openai_models import CustomModelClient
from src.chat.utils.custom_model_api_keys import (
    get_custom_model_api_key_raw_value,
    resolve_custom_model_api_keys,
)

log = logging.getLogger(__name__)


class CustomModelConfigView(View):
    """当用户选择 custom 模型时，引导其打开二次配置窗口。"""

    def __init__(
        self,
        *,
        opener_user_id: int,
        settings_service: ChatSettingsService,
        timeout: int = 180,
    ):
        super().__init__(timeout=timeout)
        self.opener_user_id = opener_user_id
        self.settings_service = settings_service

        open_btn = Button(
            label="(可选) 配置 custom 模型参数",
            style=ButtonStyle.green,
            custom_id="custom_model_config_open",
            row=0,
        )
        open_btn.callback = self.on_open
        self.add_item(open_btn)

    async def interaction_check(self, interaction: Interaction) -> bool:
        if interaction.user and interaction.user.id == self.opener_user_id:
            return True
        await interaction.response.send_message(
            "❌ 只有发起者可以操作该按钮。",
            ephemeral=True,
        )
        return False

    @staticmethod
    def _get_env_path() -> str:
        return os.path.join(config.BASE_DIR, ".env")

    @staticmethod
    def _normalize_bool_value(raw: str) -> Optional[str]:
        lowered = (raw or "").strip().lower()
        if lowered in {"true", "1", "yes"}:
            return "true"
        if lowered in {"false", "0", "no"}:
            return "false"
        return None

    @classmethod
    def _read_custom_model_env(cls) -> Tuple[str, str, str, str, str]:
        """读取 custom 模型配置：只读取当前进程里的实时环境变量。"""

        def _pick(key: str) -> str:
            return str(os.environ.get(key, "") or "").strip()

        url = _pick("CUSTOM_MODEL_URL")
        api_key = get_custom_model_api_key_raw_value()
        model_name = _pick("CUSTOM_MODEL_NAME")
        enable_vision = (
            cls._normalize_bool_value(_pick("CUSTOM_MODEL_ENABLE_VISION")) or "false"
        )
        enable_video_input = (
            cls._normalize_bool_value(_pick("CUSTOM_MODEL_ENABLE_VIDEO_INPUT"))
            or "false"
        )
        return url, api_key, model_name, enable_vision, enable_video_input

    @classmethod
    def _persist_custom_model_env(
        cls,
        *,
        url: str,
        api_key: str,
        model_name: str,
        enable_vision: str,
        enable_video_input: str,
    ) -> bool:
        """
        尝试写回 .env；失败则返回 False，但仍可通过 os.environ 在本进程生效。
        """
        env_path = cls._get_env_path()
        try:
            os.makedirs(os.path.dirname(env_path), exist_ok=True)

            set_key(env_path, "CUSTOM_MODEL_URL", url, quote_mode="always", encoding="utf-8")
            set_key(
                env_path,
                "CUSTOM_MODEL_API_KEY",
                api_key,
                quote_mode="always",
                encoding="utf-8",
            )
            set_key(
                env_path,
                "CUSTOM_MODEL_NAME",
                model_name,
                quote_mode="always",
                encoding="utf-8",
            )
            set_key(
                env_path,
                "CUSTOM_MODEL_ENABLE_VISION",
                enable_vision,
                quote_mode="always",
                encoding="utf-8",
            )
            set_key(
                env_path,
                "CUSTOM_MODEL_ENABLE_VIDEO_INPUT",
                enable_video_input,
                quote_mode="always",
                encoding="utf-8",
            )

            load_dotenv(env_path, override=True, encoding="utf-8")
            return True
        except Exception as e:
            log.warning("写回 .env 失败（将仅本次进程生效）: %s", e, exc_info=True)
            return False

    @staticmethod
    def _mask_key(key: str) -> str:
        key = (key or "").strip()
        if len(key) <= 8:
            return "*" * len(key) if key else ""
        return f"{key[:4]}****{key[-4:]}"

    async def on_open(self, interaction: Interaction) -> None:
        url, api_key, model_name, enable_vision, enable_video_input = (
            self._read_custom_model_env()
        )
        api_key_default_value = api_key
        api_key_omitted = False
        if len(api_key_default_value) > CustomModelConfigModal.API_KEY_INPUT_MAX_LENGTH:
            api_key_default_value = ""
            api_key_omitted = True

        async def _on_submit(modal_interaction: Interaction, settings: Dict[str, Any]):
            new_url = str(settings.get("custom_model_url", "")).strip()
            submitted_key = str(settings.get("custom_model_api_key", "")).strip()
            effective_key = submitted_key or api_key
            new_name = str(settings.get("custom_model_name", "")).strip()
            vision_raw = str(settings.get("custom_model_enable_vision", "")).strip()
            video_input_raw = str(
                settings.get("custom_model_enable_video_input", "")
            ).strip()

            new_enable_vision = self._normalize_bool_value(vision_raw)
            if new_enable_vision is None:
                await modal_interaction.response.send_message(
                    "❌ “开启识图工具” 只能填写 true 或 false。",
                    ephemeral=True,
                )
                return

            new_enable_video_input = self._normalize_bool_value(video_input_raw)
            if new_enable_video_input is None:
                await modal_interaction.response.send_message(
                    "❌ “开启视频输入” 只能填写 true 或 false。",
                    ephemeral=True,
                )
                return

            if (
                new_enable_vision == "true"
                and new_enable_video_input == "true"
            ):
                await modal_interaction.response.send_message(
                    "❌ 仅当“开启识图工具”为 false 时，才允许开启视频输入。",
                    ephemeral=True,
                )
                return

            if not new_url or not effective_key or not new_name:
                await modal_interaction.response.send_message(
                    "❌ CUSTOM_MODEL_URL / CUSTOM_MODEL_API_KEY / CUSTOM_MODEL_NAME 不能为空。",
                    ephemeral=True,
                )
                return

            try:
                resolved_key_config = resolve_custom_model_api_keys(effective_key)
            except ValueError as exc:
                await modal_interaction.response.send_message(
                    f"❌ {str(exc)}",
                    ephemeral=True,
                )
                return

            os.environ["CUSTOM_MODEL_URL"] = new_url
            os.environ["CUSTOM_MODEL_API_KEY"] = effective_key
            os.environ["CUSTOM_MODEL_NAME"] = new_name
            os.environ["CUSTOM_MODEL_ENABLE_VISION"] = new_enable_vision
            os.environ["CUSTOM_MODEL_ENABLE_VIDEO_INPUT"] = new_enable_video_input
            if "CUSTON_MODEL_API_KEY" in os.environ:
                os.environ["CUSTON_MODEL_API_KEY"] = effective_key

            persisted = self._persist_custom_model_env(
                url=new_url,
                api_key=effective_key,
                model_name=new_name,
                enable_vision=new_enable_vision,
                enable_video_input=new_enable_video_input,
            )

            try:
                gemini_service.openai_service.custom_model_client = CustomModelClient()
            except Exception as e:
                log.warning("刷新 CustomModelClient 失败: %s", e, exc_info=True)

            persist_note = (
                "✅ 已写入 .env 并立即生效"
                if persisted
                else "⚠️ 写入 .env 失败，已仅对当前进程生效（重启后可能失效）"
            )
            vision_note = "✅ 开启" if new_enable_vision == "true" else "❌ 关闭"
            video_input_note = (
                "✅ 开启" if new_enable_video_input == "true" else "❌ 关闭"
            )
            if resolved_key_config.source_type == "file":
                api_key_source_note = (
                    f"文件 `{resolved_key_config.file_path}`（从文件读取到 {resolved_key_config.key_count} 个 key）"
                )
                api_key_preview_note = None
            else:
                api_key_source_note = f"inline（解析到 {resolved_key_config.key_count} 个 key）"
                first_key = resolved_key_config.api_keys[0] if resolved_key_config.api_keys else ""
                api_key_preview_note = (
                    f"- API Key 预览: `{self._mask_key(first_key)}`\n"
                    if first_key
                    else ""
                )
            await modal_interaction.response.send_message(
                (
                    "✅ 已更新 custom 模型配置并生效。\n"
                    f"- URL: `{new_url}`\n"
                    f"- Model: `{new_name}`\n"
                    f"- API Key 来源: {api_key_source_note}\n"
                    f"{api_key_preview_note or ''}"
                    f"- 识图工具: {vision_note}\n"
                    f"- 视频输入: {video_input_note}\n"
                    f"- 持久化: {persist_note}"
                ),
                ephemeral=True,
            )

        modal = CustomModelConfigModal(
            title="配置 Custom 模型",
            current_url=url,
            current_api_key=api_key_default_value,
            current_api_key_omitted=api_key_omitted,
            current_model_name=model_name,
            current_enable_vision=enable_vision,
            current_enable_video_input=enable_video_input,
            on_submit_callback=_on_submit,
        )
        await interaction.response.send_modal(modal)
