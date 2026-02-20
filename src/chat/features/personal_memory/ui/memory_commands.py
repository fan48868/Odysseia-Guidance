# -*- coding: utf-8 -*-
import discord
from discord import app_commands
from discord.ext import commands
import logging

from src.chat.features.personal_memory.services.personal_memory_service import (
    personal_memory_service,
)

log = logging.getLogger(__name__)

class UserEditMemoryModal(discord.ui.Modal, title="编辑个人记忆"):
    def __init__(self, user_id: int, current_memory: str):
        super().__init__()
        self.user_id = user_id
        
        # 截断过长的记忆以防止 API 报错 (Discord 限制为 4000 字符)
        default_value = current_memory
        warning_msg = None
        if len(default_value) > 4000:
            warning_msg = f"记忆摘要过长 ({len(default_value)} 字符)，已截断至 4000 字符。"
            default_value = default_value[:4000]

        self.memory_input = discord.ui.TextInput(
            label="个人记忆摘要",
            style=discord.TextStyle.paragraph,
            placeholder="在这里输入你的个人记忆...",
            default=default_value,
            required=False,
            max_length=4000, 
        )
        self.add_item(self.memory_input)

        if warning_msg:
            self.add_item(discord.ui.TextInput(
                label="⚠️ 系统提示",
                default=warning_msg,
                style=discord.TextStyle.short,
                required=False
            ))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        new_memory = self.memory_input.value
        
        try:
            # 调用 service 更新记忆
            # 注意：这里假设 personal_memory_service 实现了 update_memory_summary 方法
            # 这与管理员面板中 EditMemoryModal 的行为保持一致
            await personal_memory_service.update_memory_summary(self.user_id, new_memory)
            await interaction.followup.send("✅ 你的个人记忆已成功更新。", ephemeral=True)
            
        except Exception as e:
            log.error(f"用户 {self.user_id} 更新记忆失败: {e}", exc_info=True)
            await interaction.followup.send(f"❌ 更新失败: {e}", ephemeral=True)

class MemoryView(discord.ui.View):
    def __init__(self, user_id: int, current_memory: str):
        super().__init__(timeout=60)
        self.user_id = user_id
        self.current_memory = current_memory

    @discord.ui.button(label="查看并修改记忆", style=discord.ButtonStyle.success, emoji="🧠")
    async def edit_memory(self, interaction: discord.Interaction, button: discord.ui.Button):
        # 点击按钮后弹出模态框
        modal = UserEditMemoryModal(self.user_id, self.current_memory)
        await interaction.response.send_modal(modal)

class MemoryCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="查看并修改记忆", description="查看并编辑你在这个服务器的个人记忆")
    async def view_and_edit_memory(self, interaction: discord.Interaction):
        user_id = interaction.user.id
        
        try:
            # 获取当前记忆
            # 逻辑仿照 CommunityMembersView.view_memory
            current_summary = await personal_memory_service.get_memory_summary(user_id)
            
            # 如果没有记忆，提供空字符串
            if current_summary is None:
                current_summary = ""
                
            view = MemoryView(user_id, current_summary)
            content = "点击下方绿色按钮即可查看或修改你的个人记忆摘要。"
            await interaction.response.send_message(content, view=view, ephemeral=True)
            
        except Exception as e:
            log.error(f"获取用户 {user_id} 记忆失败: {e}", exc_info=True)
            await interaction.response.send_message(f"无法获取记忆: {e}", ephemeral=True)

async def setup(bot):
    await bot.add_cog(MemoryCommands(bot))