import discord
from discord import Interaction
from discord.ui import TextInput
from typing import Callable, Awaitable, Dict, Any


class MemorySettingsModal(discord.ui.Modal):
    """一个用于设置记忆总结频率的模态框。"""

    def __init__(
        self,
        *,
        title: str,
        current_threshold: int,
        on_submit_callback: Callable[[Interaction, Dict[str, Any]], Awaitable[None]],
    ):
        super().__init__(title=title)
        self.on_submit_callback = on_submit_callback

        self.threshold_input = TextInput(
            label="触发总结的消息数量阈值",
            placeholder="例如: 20",
            default=str(current_threshold),
            required=True,
            min_length=1,
            max_length=4,
        )
        self.add_item(self.threshold_input)

    async def on_submit(self, interaction: Interaction):
        try:
            val = int(self.threshold_input.value.strip())
            if val <= 0:
                raise ValueError
            await self.on_submit_callback(interaction, {"threshold": val})
        except ValueError:
            await interaction.response.send_message("请输入有效的正整数。", ephemeral=True)