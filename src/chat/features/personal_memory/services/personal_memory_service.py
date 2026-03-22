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
    SUMMARY_MODEL,
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
        # In-process "force summary" requests (consumed on next message writeback).
        # Structure: {user_id: {"reason": Optional[str], "limit": int}}
        self._force_summary_requests: dict[int, dict[str, Any]] = {}
        self._force_summary_lock = asyncio.Lock()

    @staticmethod
    def _clamp_force_summary_limit(limit: Any) -> int:
        """
        Clamp the requested pair count for a forced summary.
        Default: 5
        Max: 20
        """
        try:
            limit_int = int(limit)
        except Exception:
            return 5

        if limit_int < 1:
            return 5

        return min(limit_int, 20)

    @staticmethod
    def _split_history_for_forced_summary(
        history: list, limit: int
    ) -> tuple[list, list, int]:
        """
        Split conversation history into:
        - history_to_summarize: last N pairs (user+model) to summarize
        - remaining_history: history with those pairs removed
        - pairs_to_summarize: actual pair count summarized

        Notes:
        - If history length is odd, keep the last dangling turn in remaining_history.
        - We always summarize complete (user+model) pairs only.
        """
        history_list = list(history or [])
        if not history_list:
            return [], [], 0

        dangling_turn = None
        if len(history_list) % 2 == 1:
            dangling_turn = history_list[-1]
            history_list = history_list[:-1]

        pairs_available = max(0, len(history_list) // 2)
        pairs_to_summarize = min(max(limit, 0), pairs_available)
        turns_to_summarize = pairs_to_summarize * 2

        if turns_to_summarize <= 0:
            remaining = list(history_list)
            if dangling_turn is not None:
                remaining.append(dangling_turn)
            return [], remaining, 0

        history_to_summarize = list(history_list[-turns_to_summarize:])
        remaining_history = list(history_list[:-turns_to_summarize])
        if dangling_turn is not None:
            remaining_history.append(dangling_turn)

        return history_to_summarize, remaining_history, pairs_to_summarize

    async def request_force_summary(
        self, user_id: int, reason: Optional[str] = None, limit: int = 5
    ) -> None:
        safe_limit = self._clamp_force_summary_limit(limit)
        async with self._force_summary_lock:
            self._force_summary_requests[int(user_id)] = {
                "reason": reason,
                "limit": safe_limit,
            }

    async def _consume_force_summary_request(
        self, user_id: int
    ) -> Optional[dict[str, Any]]:
        async with self._force_summary_lock:
            return self._force_summary_requests.pop(int(user_id), None)

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
        所有数据库操作都在ParadeDB中完成。
        """
        history_to_summarize = None
        forced_summary_context = None
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
                    # 避免工具被调用后，这个用户后续没有触发入库流程导致请求常驻内存
                    await self._consume_force_summary_request(user_id)
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

                force_request = await self._consume_force_summary_request(user_id)
                if force_request:
                    safe_limit = self._clamp_force_summary_limit(
                        force_request.get("limit")
                    )
                    reason = force_request.get("reason")

                    (
                        history_to_summarize,
                        remaining_history,
                        pairs_to_summarize,
                    ) = self._split_history_for_forced_summary(new_history, safe_limit)

                    if history_to_summarize:
                        setattr(profile, "history", remaining_history)
                        adjusted_count = max(0, new_count - pairs_to_summarize)
                        setattr(profile, "personal_message_count", adjusted_count)
                        forced_summary_context = {
                            "limit": safe_limit,
                            "pairs": pairs_to_summarize,
                            "reason": reason,
                        }
                        log.info(
                            "用户 %s 手动触发个人记忆总结：pairs=%s, limit=%s",
                            user_id,
                            pairs_to_summarize,
                            safe_limit,
                        )
                    else:
                        log.info(
                            "用户 %s 手动触发个人记忆总结，但没有可总结的历史。",
                            user_id,
                        )
                elif new_count >= PERSONAL_MEMORY_CONFIG["summary_threshold"]:
                    log.info(f"用户 {user_id} 达到阈值，准备总结。")
                    history_to_summarize = list(new_history)
                    setattr(profile, "personal_message_count", 0)
                    setattr(profile, "history", [])

        if history_to_summarize:
            if forced_summary_context:
                log.info(
                    "开始执行手动个人记忆总结：user_id=%s, pairs=%s, limit=%s",
                    user_id,
                    forced_summary_context.get("pairs"),
                    forced_summary_context.get("limit"),
                )
                await self._summarize_memory(
                    user_id,
                    history_to_summarize,
                    require_new_long_memory=True,
                    force_reason=forced_summary_context.get("reason"),
                )
            else:
                await self._summarize_memory(user_id, history_to_summarize)

    async def _summarize_memory(
        self,
        user_id: int,
        conversation_history: list,
        require_new_long_memory: bool = False,
        force_reason: Optional[str] = None,
    ):
        """私有方法：获取历史，生成摘要，并清空计数和历史。"""
        log.info(f"开始为用户 {user_id} 生成记忆摘要。")

        async with AsyncSessionLocal() as session:
            stmt = select(CommunityMemberProfile.personal_summary).where(
                CommunityMemberProfile.discord_id == str(user_id)
            )
            result = await session.execute(stmt)
            old_summary = result.scalars().first() or ""

        # 解析旧摘要以获取现有的长期记忆（用于手动触发时保证“新增”）
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

        # 3. 构建 Prompt 并调用 AI 生成新摘要
        prompt_template = PROMPT_CONFIG.get("personal_memory_summary")
        if not prompt_template:
            log.error("未找到 'personal_memory_summary' 的 prompt 模板。")
            return

        final_prompt = prompt_template.format(
            old_summary=old_summary if old_summary else "无",
            dialogue_history=dialogue_text,
        )

        if require_new_long_memory:
            reason_text = (force_reason or "").strip()
            extra_rules = (
                "\n\n[手动触发强制要求]\n"
                "- 这次是手动触发的个人记忆总结：必须在“<new_long_memory>”中输出至少 1 条**新增长期记忆**（仍然不超过 2 条）。\n"
                "- 新增长期记忆必须是客观陈述，不要复述或引用用户的具体对话句子，不要写“用户说...”。\n"
                "- 新增长期记忆不要与旧的长期记忆重复。\n"
            )
            if reason_text:
                extra_rules += f"- 触发原因/想记住的关键点：{reason_text}\n"
            final_prompt += extra_rules

        log.info("用户 %s 开始执行个人记忆总结。", user_id)

        max_retries = 3
        ai_response = None
        added_long_lines = []
        new_recent_lines = []

        # 仅对“AI 空响应”进行重试；超过条数限制时不重试，直接筛选
        for attempt in range(max_retries):
            ai_response = await gemini_service.generate_simple_response(
                prompt=final_prompt,
                generation_config=GEMINI_SUMMARY_GEN_CONFIG,
                model_name=SUMMARY_MODEL,
            )

            if ai_response:
                break

            log.error(f"为用户 {user_id} 生成记忆摘要失败，AI 返回空 (尝试 {attempt + 1}/{max_retries})。")

        if not ai_response:
            log.error(f"为用户 {user_id} 生成记忆摘要失败，重试多次后仍无有效响应。")
            return

        # 4. 解析 AI 响应并合并记忆
        # 提取新增的长期记忆
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

        # 检测条数限制：超过 2 条时，直接请求 AI 筛选，不再重试重新生成
        if len(added_long_lines) > 2:
            log.warning(
                f"生成的长期记忆条数 ({len(added_long_lines)}) 超过限制 (2条)，直接请求 AI 筛选最重要的 2 条。"
            )

            # 构建筛选 Prompt
            memory_items = [line.lstrip("- ").strip() for line in added_long_lines]
            memory_list_text = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(memory_items)])

            selection_prompt = (
                f"你为用户生成了以下 {len(memory_items)} 条长期记忆，但这超过了系统限制（最多保留 2 条）。\n"
                "请仔细评估，从中筛选出 **最重要、最具深层价值** 的 2 条记忆。\n\n"
                "**待筛选列表:**\n"
                f"{memory_list_text}\n\n"
                "**输出要求:**\n"
                "1. 只输出筛选后的 2 条内容。\n"
                "2. 保持原意，不要修改内容。\n"
                "3. 每条一行，严格以 `- ` 开头。\n"
                "4. 不要输出任何其他解释性文字。"
            )

            try:
                selection_response = await gemini_service.generate_simple_response(
                    prompt=selection_prompt,
                    generation_config=GEMINI_SUMMARY_GEN_CONFIG,
                    model_name=SUMMARY_MODEL,
                )

                if selection_response:
                    selected_lines = [
                        line.strip()
                        for line in selection_response.split("\n")
                        if line.strip().startswith("-")
                    ]
                    if selected_lines:
                        log.info(f"AI 筛选成功，保留了 {len(selected_lines)} 条记忆。")
                        added_long_lines = selected_lines[:2]
                    else:
                        log.warning("AI 筛选响应格式不符合预期，回退到强制截断。")
                        added_long_lines = added_long_lines[:2]
                else:
                    log.warning("AI 筛选响应为空，回退到强制截断。")
                    added_long_lines = added_long_lines[:2]
            except Exception as e:
                log.error(f"AI 筛选过程发生错误: {e}，回退到强制截断。")
                added_long_lines = added_long_lines[:2]

        added_long_lines = await self._filter_new_long_memory_lines(
            user_id=user_id,
            candidate_lines=added_long_lines,
            existing_long_texts=existing_long_texts,
        )

        # 提取新的近期动态
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

        # --- 手动触发：强制保证“新增长期记忆”至少 1 条 ---
        if require_new_long_memory:
            unique_added_long_lines = list(added_long_lines)

            if not unique_added_long_lines:
                max_force_attempts = 2
                reason_text = (force_reason or "").strip()
                for attempt in range(max_force_attempts):
                    force_prompt = (
                        "你是记忆管理专家。现在是一次【手动触发】的个人记忆总结。\n"
                        "请从对话中提炼出至少 1 条、最多 2 条【新的长期记忆】。\n\n"
                        "【本次对话】\n"
                        f"{dialogue_text}\n\n"
                        "【输出要求】\n"
                        "1. 只输出 1-2 条内容。\n"
                        "2. 每条一行，严格以 `- ` 开头。\n"
                        "3. 内容必须是客观陈述，不要复述具体对话句子，不要写“用户说...”。\n"
                        "4. 用户统一用“用户”代称，神社娘统一用“神社娘”代称。\n"
                        "5. 不要输出任何解释性文字。\n"
                    )
                    if reason_text:
                        force_prompt = (
                            force_prompt
                            + "\n【触发原因/想记住的关键点】\n"
                            + reason_text
                            + "\n"
                        )

                    force_response = await gemini_service.generate_simple_response(
                        prompt=force_prompt,
                        generation_config=GEMINI_SUMMARY_GEN_CONFIG,
                        model_name=SUMMARY_MODEL,
                    )

                    if not force_response:
                        log.warning(
                            "手动触发强制新增长期记忆失败：AI 返回空 (attempt=%s/%s)",
                            attempt + 1,
                            max_force_attempts,
                        )
                        continue

                    candidate_lines = [
                        line.strip()
                        for line in force_response.split("\n")
                        if line.strip().startswith("-")
                    ]
                    candidate_unique = await self._filter_new_long_memory_lines(
                        user_id=user_id,
                        candidate_lines=candidate_lines,
                        existing_long_texts=existing_long_texts,
                    )
                    if candidate_unique:
                        unique_added_long_lines = candidate_unique[:2]
                        log.info(
                            "手动触发强制新增长期记忆成功：user_id=%s, added=%s",
                            user_id,
                            len(unique_added_long_lines),
                        )
                        break

            if not unique_added_long_lines:
                # 最后兜底：尽量用 reason 生成一条可落库的长期记忆（避免新增 0 条）
                fallback_candidates = []
                reason_text = (force_reason or "").strip()
                if reason_text:
                    safe_reason = reason_text.replace("\n", " ").strip()
                    if len(safe_reason) > 160:
                        safe_reason = safe_reason[:160] + "..."
                    fallback_candidates.append(
                        f"- 用户希望被长期记住的关键点：{safe_reason}"
                    )
                fallback_candidates.extend(
                    [
                        "- 用户在关键节点会主动要求神社娘更新其长期记忆。",
                        "- 用户希望神社娘将关键要点写入长期记忆。",
                        "- 用户倾向于在重要信息出现时及时更新长期记忆。",
                    ]
                )

                for cand in fallback_candidates:
                    filtered_candidate = await self._filter_new_long_memory_lines(
                        user_id=user_id,
                        candidate_lines=[cand],
                        existing_long_texts=existing_long_texts,
                    )
                    if filtered_candidate:
                        unique_added_long_lines = filtered_candidate
                        break

            if not unique_added_long_lines:
                # 极端兜底：如果所有候选都重复，确保仍然能“新增”一条（避免手动触发新增 0 条）
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                unique_added_long_lines = [f"- 用户触发了一次手动记忆总结（{now_str}）。"]

            added_long_lines = unique_added_long_lines

        # 合并：旧长期 + 新增长期
        final_long_lines = old_long_lines + added_long_lines

        # 构建最终摘要字符串
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

        # 后台同步向量表，保证与 personal_summary 一致（不阻塞当前交互）
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
        在 ParadeDB 中更新摘要，同时重置个人消息计数和对话历史。
        (重构后，此函数调用两个独立的私有方法)
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await self._update_summary(session, user_id, new_summary)
                await self._reset_history_and_count(session, user_id)
        log.info(f"为用户 {user_id} 更新了记忆摘要，并重置了计数和历史。")

        # 后台同步向量表，保证与 personal_summary 一致（不阻塞当前交互）
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
