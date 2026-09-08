import asyncio
import logging
from typing import Optional, Any, List
import re
import datetime
from sqlalchemy.future import select
from sqlalchemy import update
from src.database.database import AsyncSessionLocal
from src.database.models import CommunityMemberProfile
from src.chat.config.chat_config import (
    PROMPT_CONFIG,
    get_summary_model,
    GEMINI_SUMMARY_GEN_CONFIG,
    PERSONAL_MEMORY_CONFIG,
)
from src.chat.services.gemini_service import gemini_service
from src.chat.features.personal_memory.services.personal_memory_vector_service import (
    personal_memory_vector_service,
)

log = logging.getLogger(__name__)


class PersonalMemoryService:
    def __init__(self):
        # 不再需要 force_summary 相关属性
        pass

    async def _sync_personal_memory_vectors_in_background(
        self, user_id: int, new_summary: Optional[str]
    ):
        """后台同步个人记忆向量表（community.personal_memory_chunks）。"""
        try:
            result = await personal_memory_vector_service.sync_vectors_for_user(
                discord_id=user_id,
                personal_summary=new_summary,
            )
            log.info(
                "个人记忆向量增量同步完成: user_id=%s, inserted=%s, deleted=%s, deduped=%s",
                user_id,
                result.get("inserted"),
                result.get("deleted"),
                result.get("deduped"),
            )
        except Exception as e:
            log.error(
                f"同步个人记忆向量失败: user_id={user_id}, err={e}",
                exc_info=True,
            )

    async def _filter_new_long_memory_lines(
        self,
        user_id: int,
        candidate_lines: List[str],
        existing_long_texts: set[str],
    ) -> List[str]:
        """对候选长期记忆行进行基础去重 + 语义去重，返回格式化的列表。"""
        normalized_texts: List[str] = []
        seen_texts: set[str] = set()
        for raw_line in candidate_lines or []:
            memory_text = (raw_line or "").lstrip("-").strip()
            if not memory_text:
                continue
            if memory_text in existing_long_texts or memory_text in seen_texts:
                continue
            seen_texts.add(memory_text)
            normalized_texts.append(memory_text)

        if not normalized_texts:
            return []

        dedupe_result = await personal_memory_vector_service.filter_semantically_unique_long_term_candidates(
            discord_id=user_id,
            candidate_memory_texts=normalized_texts,
            keep_unverified_on_failure=True,
        )

        filtered_lines: List[str] = []
        seen_filtered: set[str] = set()
        for entry in dedupe_result.get("accepted", []):
            memory_text = str(entry.get("memory_text") or "").strip()
            if not memory_text or memory_text in seen_filtered:
                continue
            seen_filtered.add(memory_text)
            filtered_lines.append(f"- {memory_text}")

        deduped_count = len(dedupe_result.get("deduped", []))
        if deduped_count:
            log.info(
                "个人记忆总结语义去重完成: user_id=%s, accepted=%s, deduped=%s",
                user_id,
                len(filtered_lines),
                deduped_count,
            )

        return filtered_lines

    async def update_and_conditionally_summarize_memory(
        self, user_id: int, user_name: str, user_content: str, ai_response: str
    ):
        """
        核心入口：更新对话历史和计数，并在达到阈值时触发总结。
        仅由计数阈值触发，不接受任何手动/强制请求。
        summary_enabled 为 False 时跳过所有历史累积和总结。
        """
        # 总开关：关闭时不累积历史、不总结（优先读数据库覆盖，回退到配置文件）
        _summary_enabled = True
        try:
            from src.chat.features.chat_settings.services.chat_settings_service import (
                chat_settings_service,
            )
            _se_raw = await chat_settings_service.db_manager.get_global_setting("summary_enabled")
            _summary_enabled = (_se_raw or "true").strip().lower() == "true"
        except Exception:
            _summary_enabled = PERSONAL_MEMORY_CONFIG.get("summary_enabled", True)
        if not _summary_enabled:
            log.debug("记忆总结已关闭，跳过历史累积和总结。user_id=%s", user_id)
            return

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

                # 仅检查计数阈值触发总结
                threshold = PERSONAL_MEMORY_CONFIG.get("summary_threshold", 20)
                if new_count >= threshold:
                    log.info(f"用户 {user_id} 达到阈值 {threshold}，准备总结。")
                    history_to_summarize = list(new_history)
                    setattr(profile, "personal_message_count", 0)
                    setattr(profile, "history", [])

        # 在事务外执行总结，避免长事务阻塞
        if history_to_summarize:
            await self._summarize_memory(user_id, history_to_summarize)

    async def _summarize_memory(
        self,
        user_id: int,
        conversation_history: list,
    ):
        """私有方法：获取历史，生成摘要，并更新数据库。"""
        log.info(f"开始为用户 {user_id} 生成记忆摘要。")

        async with AsyncSessionLocal() as session:
            stmt = select(CommunityMemberProfile.personal_summary).where(
                CommunityMemberProfile.discord_id == str(user_id)
            )
            result = await session.execute(stmt)
            old_summary = result.scalars().first() or ""

        # 解析旧摘要以获取现有的长期记忆（用于去重）
        old_long_lines = []
        if old_summary:
            old_lines = old_summary.split("\n")
            in_long_section = False
            for raw_line in old_lines:
                line = (raw_line or "").strip()
                if line == "### 长期记忆":
                    in_long_section = True
                    continue
                elif line == "### 近期动态":
                    in_long_section = False
                    continue

                if in_long_section and line.startswith("-"):
                    old_long_lines.append(line)

        def _normalize_long_memory_line(line: str) -> str:
            return (line or "").lstrip("-").strip()

        existing_long_texts = {
            _normalize_long_memory_line(line)
            for line in old_long_lines
            if _normalize_long_memory_line(line)
        }

        dialogue_text = "\n".join(
            f"{'用户' if turn.get('role') == 'user' else 'AI'}: {' '.join(map(str, turn.get('parts', [])))}"
            for turn in conversation_history
        ).strip()

        if not dialogue_text:
            log.warning(f"用户 {user_id} 的对话历史格式化后为空。")
            return

        # 构建 Prompt
        prompt_template = PROMPT_CONFIG.get("personal_memory_summary")
        if not prompt_template:
            log.error("未找到 'personal_memory_summary' 的 prompt 模板。")
            return

        final_prompt = prompt_template.format(
            old_summary=old_summary if old_summary else "无",
            dialogue_history=dialogue_text,
        )

        log.info("用户 %s 开始执行个人记忆总结。", user_id)

        max_retries = 3
        ai_response = None
        added_long_lines = []
        new_recent_lines = []

        for attempt in range(max_retries):
            ai_response = await gemini_service.generate_simple_response(
                prompt=final_prompt,
                generation_config=GEMINI_SUMMARY_GEN_CONFIG,
                model_name=get_summary_model(),
            )

            if ai_response:
                break

            log.error(f"为用户 {user_id} 生成记忆摘要失败，AI 返回空 (尝试 {attempt + 1}/{max_retries})。")

        if not ai_response:
            log.error(f"为用户 {user_id} 生成记忆摘要失败，重试多次后仍无有效响应。")
            return

        # 解析新增长期记忆
        new_long_match = re.search(
            r"<new_long_memory>(.*?)</new_long_memory>", ai_response, re.DOTALL
        )
        if new_long_match:
            content = new_long_match.group(1).strip()
            if content:
                added_long_lines = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip().startswith("-")
                ]

        # 条数限制：超过2条直接截断，不再调用AI筛选
        if len(added_long_lines) > 2:
            log.warning(
                f"生成的长期记忆条数 ({len(added_long_lines)}) 超过限制 (2条)，直接截断前2条。"
            )
            added_long_lines = added_long_lines[:2]

        # 提取近期动态
        new_recent_match = re.search(
            r"<recent_dynamics>(.*?)</recent_dynamics>", ai_response, re.DOTALL
        )
        if new_recent_match:
            content = new_recent_match.group(1).strip()
            if content:
                new_recent_lines = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip().startswith("-")
                ]

        # 合并长期记忆
        final_long_lines = old_long_lines + added_long_lines

        # 构建最终摘要
        final_summary = (
            "### 长期记忆\n"
            + "\n".join(final_long_lines)
            + "\n\n### 近期动态\n"
            + "\n".join(new_recent_lines)
        )

        log.info(
            "用户 %s 总结完毕：new_long=%s, total_long=%s, recent=%s",
            user_id,
            len(added_long_lines),
            len(final_long_lines),
            len(new_recent_lines),
        )

        await self.update_summary_manually(user_id, final_summary)
        log.info(f"用户 {user_id} 的总结流程完成。")

    async def get_memory_summary(self, user_id: int) -> str:
        """根据用户ID从数据库获取其个人记忆摘要。"""
        async with AsyncSessionLocal() as session:
            stmt = select(CommunityMemberProfile.personal_summary).where(
                CommunityMemberProfile.discord_id == str(user_id)
            )
            result = await session.execute(stmt)
            summary = result.scalars().first()

            if summary:
                log.debug(f"从数据库找到用户 {user_id} 的摘要。")
                return summary
            else:
                log.debug(f"在数据库中未找到用户 {user_id} 的摘要。")
                return "该用户当前没有个人记忆摘要。"

    async def update_summary_manually(self, user_id: int, new_summary: str):
        """
        仅手动更新用户的个人记忆摘要，不影响计数或历史记录。
        主要用于内部调用或管理员手动编辑。
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await self._update_summary(session, user_id, new_summary)
        log.info(f"为用户 {user_id} 手动更新了记忆摘要。")

        asyncio.create_task(
            self._sync_personal_memory_vectors_in_background(user_id, new_summary)
        )

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
        在数据库中更新摘要，同时重置个人消息计数和对话历史。
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await self._update_summary(session, user_id, new_summary)
                await self._reset_history_and_count(session, user_id)
        log.info(f"为用户 {user_id} 更新了记忆摘要，并重置了计数和历史。")

        asyncio.create_task(
            self._sync_personal_memory_vectors_in_background(user_id, new_summary)
        )

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