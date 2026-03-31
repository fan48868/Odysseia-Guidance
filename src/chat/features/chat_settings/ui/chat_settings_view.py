import discord
from discord.ui import View, Button, Select
from discord import (
    ButtonStyle,
    SelectOption,
    Interaction,
    CategoryChannel,
)
from typing import List, Optional, Dict, Any

from src.chat.features.chat_settings.services.chat_settings_service import (
    chat_settings_service,
)
from src.chat.features.chat_settings.ui.channel_settings_modal import ChatSettingsModal
from src.database.services.token_usage_service import token_usage_service
from src.database.database import AsyncSessionLocal
from src.database.models import TokenUsage
from datetime import datetime
from src.chat.features.chat_settings.ui.warm_up_settings_view import WarmUpSettingsView
from src.chat.features.chat_settings.ui.components import PaginatedSelect
from src.chat.services.event_service import event_service
from src.chat.features.chat_settings.ui.ai_model_settings_modal import (
    AIModelSettingsModal,
)
from src.chat.features.personal_memory.services.personal_memory_service import (
    personal_memory_service,
)
from src.chat.features.chat_settings.ui.memory_settings_modal import (
    MemorySettingsModal,
)
from src.chat.services.gemini_service import gemini_service
from src.chat.features.chat_settings.ui.custom_model_config_view import (
    CustomModelConfigView,
)

_GLOBAL_TTS_MODE_KEY = "global_tts_mode"
_TTS_MODE_LEGACY = "legacy"  # tts_tool (edge_tts)
_TTS_MODE_NEW = "new"  # new_tts_tool (gradio / qwen3)


def _normalize_tts_mode(raw: Optional[str]) -> str:
    mode = (raw or _TTS_MODE_LEGACY).strip().lower()
    if mode not in {_TTS_MODE_LEGACY, _TTS_MODE_NEW}:
        return _TTS_MODE_LEGACY
    return mode


def _tts_mode_display(mode: str) -> str:
    if mode == _TTS_MODE_NEW:
        return "新版 Qwen3（new_tts_tool）"
    return "旧版 EdgeTTS（tts_tool）"


class TTSSwitchView(View):
    """仅自己可见的 TTS 切换面板（绿色按钮切换 + 持久化）。"""

    def __init__(self, current_mode: str):
        super().__init__(timeout=180)
        self.service = chat_settings_service
        self.mode = _normalize_tts_mode(current_mode)
        self._build_items()

    def render_content(self) -> str:
        return (
            f"当前使用的 TTS：**{_tts_mode_display(self.mode)}**\n\n"
            "点击下方绿色按钮即可切换。"
        )

    def _build_items(self) -> None:
        self.clear_items()

        target_mode = _TTS_MODE_NEW if self.mode == _TTS_MODE_LEGACY else _TTS_MODE_LEGACY
        switch_btn = Button(
            label=f"切换到：{_tts_mode_display(target_mode)}",
            style=ButtonStyle.green,
            custom_id="tts_switch_confirm",
            row=0,
        )
        switch_btn.callback = self.on_switch
        self.add_item(switch_btn)

    async def on_switch(self, interaction: Interaction):
        self.mode = _TTS_MODE_NEW if self.mode == _TTS_MODE_LEGACY else _TTS_MODE_LEGACY
        await self.service.db_manager.set_global_setting(_GLOBAL_TTS_MODE_KEY, self.mode)
        self._build_items()
        await interaction.response.edit_message(content=self.render_content(), view=self)


class ChatSettingsView(View):
    """聊天设置的主UI面板"""

    def __init__(self, interaction: Interaction):
        super().__init__(timeout=300)
        self.guild = interaction.guild
        self.service = chat_settings_service
        self.settings: Dict[str, Any] = {}
        self.model_usage_counts: Dict[str, int] = {}
        self.message: Optional[discord.Message] = None

        # 合并后的分页选择器：分类 + 频道
        self.entity_paginator: Optional[PaginatedSelect] = None

        self.factions: Optional[List[Dict[str, Any]]] = None
        self.selected_faction: Optional[str] = None
        self.token_usage: Optional[TokenUsage] = None
        self.reaction_enabled: bool = True
        self.dm_enabled: bool = True

    async def _initialize(self):
        """异步获取设置并构建UI。"""
        if not self.guild:
            return
        self.settings = await self.service.get_guild_settings(self.guild.id)
        self.model_usage_counts = await self.service.get_model_usage_counts()
        async with AsyncSessionLocal() as session:
            self.token_usage = await token_usage_service.get_token_usage(
                session, datetime.utcnow().date()
            )
        self.factions = event_service.get_event_factions()
        self.selected_faction = event_service.get_selected_faction()

        # 读取“表情反应”全局开关（按 guild 维度存储在 global_settings）
        reaction_key = f"reaction_enabled:{self.guild.id}"
        reaction_raw = await self.service.db_manager.get_global_setting(reaction_key)
        self.reaction_enabled = (
            reaction_raw.lower() == "true" if reaction_raw is not None else True
        )

        dm_raw = await self.service.db_manager.get_global_setting("global_dm_enabled")
        self.dm_enabled = dm_raw.lower() == "true" if dm_raw is not None else True

        self._create_paginators()
        self._create_view_items()

    @classmethod
    async def create(cls, interaction: Interaction):
        """工厂方法，用于异步创建和初始化View。"""
        view = cls(interaction)
        await view._initialize()
        return view

    def _create_paginators(self):
        """创建分页器实例（合并分类 + 频道）。"""
        if not self.guild:
            return

        options: List[SelectOption] = []

        # 分类
        for c in sorted(self.guild.categories, key=lambda c: c.position):
            label = f"【分类】{c.name}"
            options.append(SelectOption(label=label[:100], value=str(c.id)))

        # 频道（文本频道）
        for c in sorted(self.guild.text_channels, key=lambda c: c.position):
            label = f"【频道】{c.name}"
            options.append(SelectOption(label=label[:100], value=str(c.id)))

        self.entity_paginator = PaginatedSelect(
            placeholder="选择一个分类/频道进行设置...",
            custom_id_prefix="entity_select",
            options=options,
            on_select_callback=self.on_entity_select,
            label_prefix="列表",
        )

    def _create_view_items(self):
        """根据当前设置创建并添加所有UI组件。"""
        self.clear_items()

        # 全局开关 (第 0 行)
        global_chat_enabled = self.settings.get("global", {}).get("chat_enabled", True)
        self.add_item(
            Button(
                label=f"聊天总开关: {'开' if global_chat_enabled else '关'}",
                style=ButtonStyle.green if global_chat_enabled else ButtonStyle.red,
                custom_id="global_chat_toggle",
                row=0,
            )
        )

        self.add_item(
            Button(
                label=f"表情反应: {'开' if self.reaction_enabled else '关'}",
                style=ButtonStyle.green if self.reaction_enabled else ButtonStyle.red,
                custom_id="reaction_toggle",
                row=0,
            )
        )

        self.add_item(
            Button(
                label=f"私信开关: {'开' if self.dm_enabled else '关'}",
                style=ButtonStyle.green if self.dm_enabled else ButtonStyle.red,
                custom_id="global_dm_toggle",
                row=0,
            )
        )

        # 更换AI模型按钮（第 0 行）
        self.add_item(
            Button(
                label="更换AI模型",
                style=ButtonStyle.secondary,
                custom_id="ai_model_settings",
                row=0,
            )
        )

        # 活动派系选择器（第 1 行）
        if self.factions:
            faction_options = [
                SelectOption(
                    label="无 / 默认",
                    value="_default",
                    default=self.selected_faction is None,
                )
            ]
            for faction in self.factions:
                is_selected = self.selected_faction == faction["faction_id"]
                faction_options.append(
                    SelectOption(
                        label=f"{faction['faction_name']} ({faction['faction_id']})"[:100],
                        value=faction["faction_id"],
                        default=is_selected,
                    )
                )

            faction_select = Select(
                placeholder="设置当前活动派系人设...",
                options=faction_options,
                custom_id="faction_select",
                row=1,
            )
            faction_select.callback = self.on_faction_select
            self.add_item(faction_select)

        # 分类/频道合并选择器（第 2 行）
        if self.entity_paginator:
            self.add_item(self.entity_paginator.create_select(row=2))

        # 第 3 行：分页按钮（最多2个） + 三个设置按钮（总计最多5个）
        if self.entity_paginator:
            for btn in self.entity_paginator.get_buttons(row=3):
                self.add_item(btn)

        self.add_item(
            Button(
                label="记忆总结频率",
                style=ButtonStyle.secondary,
                custom_id="memory_settings",
                row=3,
            )
        )
        self.add_item(
            Button(
                label="今日 Token 统计",
                style=ButtonStyle.secondary,
                custom_id="show_token_usage",
                row=3,
            )
        )
        self.add_item(
            Button(
                label="临时调试",
                style=ButtonStyle.secondary,
                custom_id="temp_debug_once",
                row=3,
            )
        )

        # 最底下一行：切换 TTS（单独一行）
        self.add_item(
            Button(
                label="切换TTS",
                style=ButtonStyle.secondary,
                custom_id="tts_settings",
                row=4,
            )
        )

    async def _update_view(self, interaction: Interaction):
        """通过编辑附加的消息来刷新视图。"""
        await self._initialize()  # 重新获取所有数据，包括派系
        await interaction.response.edit_message(view=self)

    async def interaction_check(self, interaction: Interaction) -> bool:
        custom_id = interaction.data.get("custom_id") if interaction.data else None

        if custom_id == "global_chat_toggle":
            await self.on_global_toggle(interaction)
        elif custom_id == "reaction_toggle":
            await self.on_reaction_toggle(interaction)
        elif custom_id == "warm_up_toggle":
            await self.on_warm_up_toggle(interaction)
        elif custom_id == "global_dm_toggle":
            await self.on_global_dm_toggle(interaction)
        elif custom_id == "warm_up_settings":
            await self.on_warm_up_settings(interaction)
        elif (
            self.entity_paginator
            and custom_id
            and self.entity_paginator.handle_pagination(custom_id)
        ):
            # 分页时只更新视图组件，不重新初始化（避免 current_page 被重置）
            self._create_view_items()
            await interaction.response.edit_message(view=self)
        elif custom_id == "ai_model_settings":
            await self.on_ai_model_settings(interaction)
        elif custom_id == "show_token_usage":
            await self.on_show_token_usage(interaction)
        elif custom_id == "temp_debug_once":
            await self.on_temp_debug_once(interaction)
        elif custom_id == "memory_settings":
            await self.on_memory_settings(interaction)
        elif custom_id == "tts_settings":
            await self.on_tts_settings(interaction)

        return True

    async def on_tts_settings(self, interaction: Interaction):
        raw = await self.service.db_manager.get_global_setting(_GLOBAL_TTS_MODE_KEY)
        mode = _normalize_tts_mode(raw)
        view = TTSSwitchView(mode)
        await interaction.response.send_message(
            content=view.render_content(),
            view=view,
            ephemeral=True,
        )

    async def on_global_toggle(self, interaction: Interaction):
        current_state = self.settings.get("global", {}).get("chat_enabled", True)
        new_state = not current_state
        if not self.guild:
            return
        await self.service.db_manager.update_global_chat_config(
            self.guild.id, chat_enabled=new_state
        )
        await self._update_view(interaction)

    async def on_reaction_toggle(self, interaction: Interaction):
        if not self.guild:
            return
        new_state = not self.reaction_enabled
        reaction_key = f"reaction_enabled:{self.guild.id}"
        await self.service.db_manager.set_global_setting(
            reaction_key, "true" if new_state else "false"
        )
        await self._update_view(interaction)

    async def on_warm_up_toggle(self, interaction: Interaction):
        current_state = self.settings.get("global", {}).get("warm_up_enabled", True)
        new_state = not current_state
        if not self.guild:
            return
        await self.service.db_manager.update_global_chat_config(
            self.guild.id, warm_up_enabled=new_state
        )
        await self._update_view(interaction)

    async def on_global_dm_toggle(self, interaction: Interaction):
        new_state = not self.dm_enabled
        await self.service.db_manager.set_global_setting(
            "global_dm_enabled", "true" if new_state else "false"
        )
        await self._update_view(interaction)

    async def on_warm_up_settings(self, interaction: Interaction):
        """切换到暖贴频道设置视图。"""
        if not self.message:
            await interaction.response.send_message(
                "无法找到原始消息，请重新打开设置面板。", ephemeral=True
            )
            return

        await interaction.response.defer()
        warm_up_view = await WarmUpSettingsView.create(interaction, self.message)
        await interaction.edit_original_response(
            content="管理暖贴功能启用的论坛频道：", view=warm_up_view
        )
        self.stop()

    async def on_faction_select(self, interaction: Interaction):
        """处理派系选择事件。"""
        if not interaction.data or "values" not in interaction.data:
            await interaction.response.defer()
            return

        selected_faction_id = interaction.data["values"][0]

        if selected_faction_id == "_default":
            event_service.set_selected_faction(None)
        else:
            event_service.set_selected_faction(selected_faction_id)

        await self._update_view(interaction)

    async def on_entity_select(self, interaction: Interaction):
        """统一处理频道和分类的选择事件。"""
        if not interaction.data or "values" not in interaction.data:
            await interaction.response.defer()
            return

        values = interaction.data["values"]
        if not values or values[0] == "disabled":
            await interaction.response.defer()
            return

        entity_id = int(values[0])
        if not self.guild:
            await interaction.response.defer()
            return

        entity = self.guild.get_channel(entity_id)
        if not entity:
            await interaction.response.send_message("找不到该项目。", ephemeral=True)
            return

        entity_type = "category" if isinstance(entity, CategoryChannel) else "channel"
        current_config = self.settings.get("channels", {}).get(entity_id, {})

        async def modal_callback(
            modal_interaction: Interaction, settings: Dict[str, Any]
        ):
            await self._handle_modal_submit(
                modal_interaction, entity_id, entity_type, settings
            )
            # Modal 提交后刷新主视图
            if self.message:
                new_view = await ChatSettingsView.create(interaction)
                new_view.message = self.message
                await self.message.edit(content="设置已更新。", view=new_view)

        entity_type_label = "分类" if entity_type == "category" else "频道"

        modal = ChatSettingsModal(
            title=f"编辑{entity_type_label} {entity.name} 的设置",
            current_config=current_config,
            on_submit_callback=modal_callback,
            entity_name=f"{entity_type_label}:{entity.name}",
            include_active_chat_cache_option=entity_type == "channel",
        )
        await interaction.response.send_modal(modal)

    async def _handle_modal_submit(
        self,
        interaction: Interaction,
        entity_id: int,
        entity_type: str,
        settings: Dict[str, Any],
    ):
        """处理模态窗口提交的数据并保存。"""
        try:
            if not self.guild:
                await interaction.followup.send("❌ 服务器信息丢失。", ephemeral=True)
                return
            await self.service.set_entity_settings(
                guild_id=self.guild.id,
                entity_id=entity_id,
                entity_type=entity_type,
                is_chat_enabled=settings.get("is_chat_enabled"),
                cooldown_seconds=settings.get("cooldown_seconds"),
                cooldown_duration=settings.get("cooldown_duration"),
                cooldown_limit=settings.get("cooldown_limit"),
                active_chat_cache_enabled=settings.get("active_chat_cache_enabled"),
            )

            if not self.guild:
                entity_name = f"ID: {entity_id}"
            else:
                entity = self.guild.get_channel(entity_id)
                entity_name = entity.name if entity else f"ID: {entity_id}"

            is_chat_enabled = settings.get("is_chat_enabled")
            enabled_str = "继承"
            if is_chat_enabled is True:
                enabled_str = "✅ 开启"
            if is_chat_enabled is False:
                enabled_str = "❌ 关闭"

            cooldown_seconds = settings.get("cooldown_seconds")
            cd_sec_str = (
                f"{cooldown_seconds} 秒" if cooldown_seconds is not None else "继承"
            )

            cooldown_duration = settings.get("cooldown_duration")
            cooldown_limit = settings.get("cooldown_limit")
            freq_str = "继承"
            if cooldown_duration is not None and cooldown_limit is not None:
                freq_str = f"{cooldown_duration} 秒内最多 {cooldown_limit} 次"

            active_chat_cache_str = "继承"
            active_chat_cache_enabled = settings.get("active_chat_cache_enabled")
            if active_chat_cache_enabled is True:
                active_chat_cache_str = "✅ 开启"
            if active_chat_cache_enabled is False:
                active_chat_cache_str = "❌ 关闭"

            feedback = (
                f"✅ 已成功为 **{entity_name}** ({entity_type}) 更新设置。\n"
                f"🔹 **聊天总开关**: {enabled_str}\n"
                f"🔹 **固定冷却(秒)**: {cd_sec_str}\n"
                f"🔹 **频率限制**: {freq_str}"
            )
            if entity_type == "channel":
                feedback += f"\n🔹 **主动聊天缓存**: {active_chat_cache_str}"

            # 确保交互未被响应
            if not interaction.response.is_done():
                await interaction.response.defer()
            await interaction.followup.send(feedback, ephemeral=True)

        except Exception as e:
            if not interaction.response.is_done():
                await interaction.response.defer()
            await interaction.followup.send(f"❌ 保存设置时出错: {e}", ephemeral=True)

    async def on_ai_model_settings(self, interaction: Interaction):
        """打开AI模型设置模态框。"""
        current_model = await self.service.get_current_ai_model()
        current_summary_model = await self.service.get_current_summary_model()
        available_models = self.service.get_available_ai_models()

        async def modal_callback(
            modal_interaction: Interaction, settings: Dict[str, Any]
        ):
            new_model = (settings.get("ai_model") or "").strip()
            new_summary_model = (settings.get("summary_model") or "").strip()
            if not new_model:
                await modal_interaction.response.send_message(
                    "❌ 没有选择任何模型。", ephemeral=True
                )
                return
            if not new_summary_model:
                await modal_interaction.response.send_message(
                    "❌ 没有填写记忆摘要模型。", ephemeral=True
                )
                return

            normalized_model = "custom" if new_model.lower() == "custom" else new_model
            normalized_summary_model = (
                "custom"
                if new_summary_model.lower() == "custom"
                else new_summary_model
            )
            summary_persisted = await self.service.set_summary_model(
                normalized_summary_model
            )
            summary_persist_note = (
                "✅ 已写入 `.env` 并立即生效"
                if summary_persisted
                else "⚠️ 写入 `.env` 失败，已仅对当前进程生效（重启后可能失效）"
            )

            # custom 模型：立即切换，然后弹出可选配置提示
            if normalized_model == "custom":
                await self.service.set_ai_model("custom")
                view = CustomModelConfigView(
                    opener_user_id=modal_interaction.user.id,
                    settings_service=self.service,
                )
                await modal_interaction.response.send_message(
                    (
                        "✅ 已切换到 **custom** 聊天模型，并同步更新记忆摘要模型。\n"
                        f"- 聊天模型: `custom`\n"
                        f"- 记忆摘要模型: `{normalized_summary_model}`\n"
                        f"- 摘要模型持久化: {summary_persist_note}\n"
                        "可以配置 `CUSTOM_MODEL_URL` / `CUSTOM_MODEL_API_KEY` / `CUSTOM_MODEL_NAME`。\n"
                        "`CUSTOM_MODEL_API_KEY` 支持直接填写 key，或填写 `/data/*.json` 文件路径（会自动映射到 `/app/data`）。\n"
                        "点击下方按钮打开配置窗口（默认值将从项目根目录 `.env` 读取）。"
                    ),
                    view=view,
                    ephemeral=True,
                )
                return

            await self.service.set_ai_model(normalized_model)
            await modal_interaction.response.send_message(
                (
                    "✅ 已成功切换模型。\n"
                    f"- 聊天模型: `{normalized_model}`\n"
                    f"- 记忆摘要模型: `{normalized_summary_model}`\n"
                    f"- 摘要模型持久化: {summary_persist_note}"
                ),
                ephemeral=True,
            )

        modal = AIModelSettingsModal(
            title="更换全局AI模型",
            current_model=current_model,
            current_summary_model=current_summary_model,
            available_models=available_models,
            on_submit_callback=modal_callback,
        )
        await interaction.response.send_modal(modal)

    async def on_memory_settings(self, interaction: Interaction):
        """打开记忆总结频率设置模态框。"""
        current_threshold = personal_memory_service.get_summary_threshold()

        async def modal_callback(
            modal_interaction: Interaction, settings: Dict[str, Any]
        ):
            new_threshold = settings.get("threshold")
            if new_threshold is not None:
                personal_memory_service.set_summary_threshold(new_threshold)
                await modal_interaction.response.send_message(
                    f"✅ 已成功将记忆总结阈值更新为: **{new_threshold}** 条消息", ephemeral=True
                )

        modal = MemorySettingsModal(
            title="设置记忆总结频率",
            current_threshold=current_threshold,
            on_submit_callback=modal_callback,
        )
        await interaction.response.send_modal(modal)

    async def on_temp_debug_once(self, interaction: Interaction):
        """激活一次性临时调试 URL。"""
        debug_url = "http://host.docker.internal:1000"
        current_model = await self.service.get_current_ai_model()
        gemini_service.arm_one_time_debug_base_url(debug_url)

        await interaction.response.send_message(
            (
                f"✅ 已为当前模型 **{current_model}** 启用一次性临时调试。\n"
                f"本次将临时把 API URL 指向：`{debug_url}`\n"
                "下一次发送生效 1 次后会自动恢复原配置。"
            ),
            ephemeral=True,
        )

    async def on_show_token_usage(self, interaction: Interaction):
        """显示今天的 Token 使用情况。"""
        if not self.token_usage:
            await interaction.response.send_message(
                "今天还没有 Token 使用记录。", ephemeral=True
            )
            return

        input_tokens = self.token_usage.input_tokens or 0
        output_tokens = self.token_usage.output_tokens or 0
        total_tokens = self.token_usage.total_tokens or 0
        call_count = self.token_usage.call_count or 0
        average_per_call = total_tokens // call_count if call_count > 0 else 0
        usage_date = self.token_usage.date.strftime("%Y-%m-%d")

        embed = discord.Embed(
            title=f"📊 今日 Token 統計 ({usage_date})",
            color=discord.Color.blue(),
        )
        embed.add_field(name="📥 Input", value=f"{input_tokens:,}", inline=False)
        embed.add_field(name="📤 Output", value=f"{output_tokens:,}", inline=False)
        embed.add_field(name="📈 Total", value=f"{total_tokens:,}", inline=False)
        embed.add_field(name="🔢 呼叫次數", value=str(call_count), inline=False)
        embed.add_field(name="📊 平均每次", value=f"{average_per_call:,}", inline=False)

        await interaction.response.send_message(embed=embed, ephemeral=True)
