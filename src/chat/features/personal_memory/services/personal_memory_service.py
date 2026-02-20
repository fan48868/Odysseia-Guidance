import logging
from typing import Optional
import re
import os
import datetime
from sqlalchemy.future import select
from sqlalchemy import update
from src.database.database import AsyncSessionLocal
from src.database.models import CommunityMemberProfile
from src.chat.config.chat_config import (
    PROMPT_CONFIG,
    SUMMARY_MODEL,
    GEMINI_SUMMARY_GEN_CONFIG,
    PERSONAL_MEMORY_CONFIG,
)
from src.chat.services.gemini_service import gemini_service

log = logging.getLogger(__name__)


class PersonalMemoryService:
    async def update_and_conditionally_summarize_memory(
        self, user_id: int, user_name: str, user_content: str, ai_response: str
    ):
        """
        核心入口：更新对话历史和计数，并在达到阈值时触发总结。
        所有数据库操作都在ParadeDB中完成。
        """
        history_to_summarize = None
        async with AsyncSessionLocal() as session:
            async with session.begin():
                stmt = (
                    select(CommunityMemberProfile)
                    .where(CommunityMemberProfile.discord_id == str(user_id))
                    .with_for_update()
                )
                result = await session.execute(stmt)
                profile = result.scalars().first()

                if not profile:
                    log.warning(f"用户 {user_id} 没有个人档案，无法记录记忆。")
                    return

                current_count = getattr(profile, "personal_message_count", 0)
                new_count = current_count + 1
                setattr(profile, "personal_message_count", new_count)

                new_turn = {"role": "user", "parts": [user_content]}
                new_model_turn = {"role": "model", "parts": [ai_response]}

                current_history = getattr(profile, "history", [])
                new_history = list(current_history or [])
                new_history.extend([new_turn, new_model_turn])
                setattr(profile, "history", new_history)

                log.debug(f"用户 {user_id} 的消息计数更新为: {new_count}")

                if new_count >= PERSONAL_MEMORY_CONFIG["summary_threshold"]:
                    log.info(f"用户 {user_id} 达到阈值，准备总结。")
                    history_to_summarize = list(new_history)
                    setattr(profile, "personal_message_count", 0)
                    setattr(profile, "history", [])

        if history_to_summarize:
            await self._summarize_memory(user_id, history_to_summarize)

    async def _summarize_memory(self, user_id: int, conversation_history: list):
        """私有方法：获取历史，生成摘要，并清空计数和历史。"""
        log.info(f"开始为用户 {user_id} 生成记忆摘要。")

        async with AsyncSessionLocal() as session:
            stmt = select(CommunityMemberProfile.personal_summary).where(
                CommunityMemberProfile.discord_id == str(user_id)
            )
            result = await session.execute(stmt)
            old_summary = result.scalars().first() or ""

        dialogue_text = "\n".join(
            f"{'用户' if turn.get('role') == 'user' else 'AI'}: {' '.join(map(str, turn.get('parts', [])))}"
            for turn in conversation_history
        ).strip()

        if not dialogue_text:
            log.warning(f"用户 {user_id} 的对话历史格式化后为空。")
            return

        # 3. 构建 Prompt 并调用 AI 生成新摘要
        prompt_template = PROMPT_CONFIG.get("personal_memory_summary")
        if not prompt_template:
            log.error("未找到 'personal_memory_summary' 的 prompt 模板。")
            return

        final_prompt = prompt_template.format(
            old_summary=old_summary if old_summary else "无",
            dialogue_history=dialogue_text,
        )

        log.info(f"---[MEMORY DEBUGGER]--- 用户 {user_id} 开始总结 ---")

        ai_response = await gemini_service.generate_simple_response(
            prompt=final_prompt,
            generation_config=GEMINI_SUMMARY_GEN_CONFIG,
            model_name=SUMMARY_MODEL,
        )

        if not ai_response:
            log.error(f"为用户 {user_id} 生成记忆摘要失败，AI 返回空。")
            return

        # 4. 解析 AI 响应并合并记忆
        # 提取新增的长期记忆
        new_long_match = re.search(
            r"<new_long_memory>(.*?)</new_long_memory>", ai_response, re.DOTALL
        )
        added_long_lines = []
        if new_long_match:
            content = new_long_match.group(1).strip()
            if content:
                added_long_lines = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip().startswith("-")
                ]

        # 提取新的近期动态
        new_recent_match = re.search(
            r"<recent_dynamics>(.*?)</recent_dynamics>", ai_response, re.DOTALL
        )
        new_recent_lines = []
        if new_recent_match:
            content = new_recent_match.group(1).strip()
            if content:
                new_recent_lines = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip().startswith("-")
                ]

        # --- 日志记录逻辑 ---
        log_dir = PERSONAL_MEMORY_CONFIG.get("log_dir")
        if log_dir and (added_long_lines or new_recent_lines):
            try:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                today_str = datetime.date.today().strftime("%Y-%m-%d")
                log_file_path = os.path.join(log_dir, f"memory_summary_{today_str}.log")
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"[{current_time}] User ID: {user_id}\n")
                    if added_long_lines:
                        f.write("【新增长期记忆】:\n" + "\n".join(added_long_lines) + "\n")
                    if new_recent_lines:
                        f.write("【近期动态】:\n" + "\n".join(new_recent_lines) + "\n")
                    f.write("-" * 50 + "\n")
            except Exception as e:
                log.error(f"写入记忆日志失败: {e}")

        # 解析旧摘要以获取现有的长期记忆
        old_long_lines = []
        if old_summary:
            lines = old_summary.split("\n")
            in_long_section = False
            for line in lines:
                line = line.strip()
                if line == "### 长期记忆":
                    in_long_section = True
                    continue
                elif line == "### 近期动态":
                    in_long_section = False
                    continue

                if in_long_section and line.startswith("-"):
                    old_long_lines.append(line)

        # 合并：旧长期 + 新增长期
        final_long_lines = old_long_lines + added_long_lines

        # 构建最终摘要字符串
        final_summary = (
            "### 长期记忆\n"
            + "\n".join(final_long_lines)
            + "\n\n### 近期动态\n"
            + "\n".join(new_recent_lines)
        )

        log.info(f"---[MEMORY DEBUGGER]--- 用户 {user_id} 总结完毕 ---")
        log.info(
            f"新增长期记忆: {len(added_long_lines)} 条, 总长期记忆: {len(final_long_lines)} 条"
        )
        log.debug(f"完整的新摘要:\n{final_summary}")

        await self.update_summary_manually(user_id, final_summary)
        log.info(f"用户 {user_id} 的总结流程完成。")

    async def get_memory_summary(self, user_id: int) -> str:
        """根据用户ID从 ParadeDB 获取其个人记忆摘要。"""
        async with AsyncSessionLocal() as session:
            stmt = select(CommunityMemberProfile.personal_summary).where(
                CommunityMemberProfile.discord_id == str(user_id)
            )
            result = await session.execute(stmt)
            summary = result.scalars().first()

            if summary:
                log.debug(f"从 ParadeDB 找到用户 {user_id} 的摘要。")
                return summary
            else:
                log.debug(f"在 ParadeDB 中未找到用户 {user_id} 的摘要。")
                return "该用户当前没有个人记忆摘要。"

    async def update_summary_manually(self, user_id: int, new_summary: str):
        """
        仅手动更新用户的个人记忆摘要，不影响计数或历史记录。
        主要用于管理员手动编辑。
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await self._update_summary(session, user_id, new_summary)
        log.info(f"为用户 {user_id} 手动更新了记忆摘要。")

    async def update_memory_summary(self, user_id: int, new_summary: str):
        """
        更新用户的个人记忆摘要 (用于命令调用)。
        """
        await self.update_summary_manually(user_id, new_summary)

    async def _update_summary(self, session, user_id: int, new_summary: Optional[str]):
        """私有方法：只更新摘要。"""
        stmt = (
            update(CommunityMemberProfile)
            .where(CommunityMemberProfile.discord_id == str(user_id))
            .values(personal_summary=new_summary)
        )
        await session.execute(stmt)

    async def _reset_history_and_count(self, session, user_id: int):
        """私有方法：只重置计数和历史。"""
        stmt = (
            update(CommunityMemberProfile)
            .where(CommunityMemberProfile.discord_id == str(user_id))
            .values(
                personal_message_count=0,
                history=[],
            )
        )
        await session.execute(stmt)

    async def update_summary_and_reset_history(
        self, user_id: int, new_summary: Optional[str]
    ):
        """
        在 ParadeDB 中更新摘要，同时重置个人消息计数和对话历史。
        (重构后，此函数调用两个独立的私有方法)
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await self._update_summary(session, user_id, new_summary)
                await self._reset_history_and_count(session, user_id)
        log.info(f"为用户 {user_id} 更新了记忆摘要，并重置了计数和历史。")

    async def clear_personal_memory(self, user_id: int):
        """
        清除指定用户的个人记忆摘要、对话历史和消息计数。
        """
        log.info(f"正在为用户 {user_id} 清除个人记忆...")
        await self.update_summary_and_reset_history(user_id, None)
        log.info(f"用户 {user_id} 的个人记忆已清除。")

    async def reset_memory_and_delete_history(self, user_id: int):
        """
        删除对话记录并重置记忆。
        这会清除用户的个人记忆摘要，并删除所有相关的对话历史记录。
        """
        log.info(f"正在为用户 {user_id} 重置记忆并删除对话历史...")
        await self.update_summary_and_reset_history(user_id, None)
        log.info(f"用户 {user_id} 的记忆和对话历史已清除。")

    async def delete_conversation_history(self, user_id: int):
        """
        单纯删除对话记录。
        这仅删除指定用户的对话历史记录和重置消息计数，不影响其个人记忆摘要。
        """
        log.info(f"正在为用户 {user_id} 删除对话历史...")
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await self._reset_history_and_count(session, user_id)
        log.info(f"用户 {user_id} 的对话历史已删除。")

    def set_summary_threshold(self, threshold: int):
        """运行时更新记忆总结阈值"""
        PERSONAL_MEMORY_CONFIG["summary_threshold"] = threshold
        log.info(f"个人记忆总结阈值已运行时更新为: {threshold}")

    def get_summary_threshold(self) -> int:
        """获取当前记忆总结阈值"""
        return PERSONAL_MEMORY_CONFIG.get("summary_threshold", 20)


# 单例实例
personal_memory_service = PersonalMemoryService()
