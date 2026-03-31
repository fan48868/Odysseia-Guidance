# -*- coding: utf-8 -*-

import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import delete, text, tuple_, update
from sqlalchemy.future import select

from src.chat.config.chat_config import PERSONAL_MEMORY_CONFIG
from src.chat.services.gemini_service import gemini_service
from src.database.database import AsyncSessionLocal
from src.database.models import CommunityMemberProfile, PersonalMemoryChunk

log = logging.getLogger(__name__)


class PersonalMemoryVectorService:
    """
    将 CommunityMemberProfile.personal_summary 中的“条目化记忆”解析出来，向量化并存储到
    community.personal_memory_chunks，供对话时向量召回使用。

    设计目标：
    - 支持“重建式写入”（先删后插），简单可靠
    - 当 embedding 不可用/失败时，尽量不破坏已有向量（避免把旧数据清空）
    """

    MAX_ITEMS_PER_USER: Optional[int] = None

    def parse_personal_summary(self, personal_summary: str) -> List[Dict[str, str]]:
        if not personal_summary or not personal_summary.strip():
            return []

        items: List[Dict[str, str]] = []
        current_type = "unknown"

        for raw_line in personal_summary.splitlines():
            line = (raw_line or "").strip()
            if not line:
                continue

            header_candidate = re.sub(r"^[#\s【\[\(（]+", "", line)
            header_candidate = re.sub(r"[】\]）\)\s:：]+$", "", header_candidate)
            header_candidate = header_candidate.strip()

            is_header_like = (
                line.startswith("#")
                or line.startswith("【")
                or line.startswith("[")
                or line.strip().endswith((":", "："))
                or header_candidate in {"长期记忆", "近期动态"}
            )

            if is_header_like:
                if "长期记忆" in header_candidate:
                    current_type = "long_term"
                    continue
                if "近期动态" in header_candidate:
                    current_type = "recent"
                    continue
                if line.startswith("#") or line.startswith("【") or line.startswith("["):
                    current_type = "unknown"
                    continue

            if current_type != "long_term":
                continue

            memory_text = ""
            bullet_match = re.match(r"^[-*•·－—]\s*(.+)$", line)
            if bullet_match:
                memory_text = bullet_match.group(1).strip()
            else:
                num_match = re.match(r"^\d+[\.\、]\s*(.+)$", line)
                if num_match:
                    memory_text = num_match.group(1).strip()

            if memory_text:
                items.append({"memory_type": "long_term", "memory_text": memory_text})

        deduped: List[Dict[str, str]] = []
        seen = set()
        for item in items:
            key = (item.get("memory_type"), item.get("memory_text"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        if self.MAX_ITEMS_PER_USER and len(deduped) > self.MAX_ITEMS_PER_USER:
            log.warning(
                "personal_summary 解析出 %s 条记忆，超过上限 %s，将截断。",
                len(deduped),
                self.MAX_ITEMS_PER_USER,
            )
            deduped = deduped[: self.MAX_ITEMS_PER_USER]

        return deduped

    @staticmethod
    def _vector_to_pg_str(vec: Sequence[float]) -> str:
        return "[" + ",".join(str(float(x)) for x in vec) + "]"

    @staticmethod
    def _normalize_dedupe_text(raw_text: Any) -> str:
        return str(raw_text or "").strip()

    @staticmethod
    def _cosine_distance(
        vec_a: Sequence[float], vec_b: Sequence[float]
    ) -> Optional[float]:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return None

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a_raw, b_raw in zip(vec_a, vec_b):
            a = float(a_raw)
            b = float(b_raw)
            dot += a * b
            norm_a += a * a
            norm_b += b * b

        if norm_a <= 0.0 or norm_b <= 0.0:
            return None

        cosine_similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        return 1.0 - cosine_similarity

    def _get_semantic_dedupe_max_distance(
        self, max_distance: Optional[float] = None
    ) -> float:
        if max_distance is not None:
            return float(max_distance)
        return float(PERSONAL_MEMORY_CONFIG.get("semantic_dedupe_max_distance", 0.18))

    async def _find_nearest_existing_long_term_match(
        self,
        session,
        discord_id: str,
        candidate_embedding: Sequence[float],
    ) -> Optional[Dict[str, Any]]:
        nearest_sql = text(
            """
            SELECT
                id,
                memory_text,
                embedding,
                (embedding <=> :candidate_vector) AS distance
            FROM community.personal_memory_chunks
            WHERE discord_id = :discord_id
              AND memory_type = 'long_term'
            ORDER BY distance ASC
            LIMIT 1;
            """
        )

        result = await session.execute(
            nearest_sql,
            {
                "discord_id": discord_id,
                "candidate_vector": self._vector_to_pg_str(candidate_embedding),
            },
        )
        row = result.first()
        if not row:
            return None

        mapping = row._mapping
        distance = mapping.get("distance")
        return {
            "id": mapping.get("id"),
            "memory_text": self._normalize_dedupe_text(mapping.get("memory_text")),
            "embedding": mapping.get("embedding"),
            "distance": float(distance) if distance is not None else None,
        }

    def _find_batch_match(
        self,
        representatives: List[Dict[str, Any]],
        candidate_embedding: Sequence[float],
        max_distance: float,
    ) -> Optional[Dict[str, Any]]:
        best_match: Optional[Dict[str, Any]] = None
        for rep in representatives:
            rep_embedding = rep.get("embedding")
            if not rep_embedding:
                continue
            distance = self._cosine_distance(candidate_embedding, rep_embedding)
            if distance is None or distance > max_distance:
                continue
            if best_match is None or distance < best_match["distance"]:
                best_match = {
                    "memory_text": self._normalize_dedupe_text(rep.get("memory_text")),
                    "distance": float(distance),
                }
        return best_match

    async def filter_semantically_unique_long_term_candidates(
        self,
        discord_id: int | str,
        candidate_memory_texts: List[str],
        max_distance: Optional[float] = None,
        fail_on_embedding_error: bool = False,
        keep_unverified_on_failure: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        discord_id_str = str(discord_id)
        max_distance_value = self._get_semantic_dedupe_max_distance(max_distance)
        accepted: List[Dict[str, Any]] = []
        deduped: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        representatives: List[Dict[str, Any]] = []

        cleaned_texts: List[str] = []
        seen_exact: set[str] = set()
        for raw_text in candidate_memory_texts or []:
            memory_text = self._normalize_dedupe_text(raw_text)
            if not memory_text:
                continue
            if memory_text in seen_exact:
                deduped.append(
                    {
                        "memory_text": memory_text,
                        "matched_text": memory_text,
                        "distance": 0.0,
                        "source": "batch",
                    }
                )
                continue

            seen_exact.add(memory_text)
            cleaned_texts.append(memory_text)

        if not cleaned_texts:
            return {"accepted": accepted, "deduped": deduped, "failed": failed}

        if not gemini_service.is_available():
            if fail_on_embedding_error:
                raise RuntimeError(
                    "RAG/embedding 未启用（未配置 GOOGLE_API_KEYS_LIST），无法执行个人记忆语义去重。"
                )

            if keep_unverified_on_failure:
                accepted.extend(
                    {"memory_text": memory_text, "embedding": None}
                    for memory_text in cleaned_texts
                )
            else:
                log.info(
                    "RAG/embedding 未启用，跳过个人记忆语义去重。"
                )

            return {"accepted": accepted, "deduped": deduped, "failed": failed}

        batch_duplicates_count = len(deduped)
        async with AsyncSessionLocal() as session:
            for idx, memory_text in enumerate(cleaned_texts, start=1):
                try:
                    embedding = await gemini_service.generate_embedding(
                        text=memory_text, task_type="retrieval_document"
                    )
                except Exception as exc:
                    if fail_on_embedding_error:
                        raise RuntimeError(
                            f"为候选记忆生成 embedding 失败（{idx}/{len(cleaned_texts)}）：{exc}"
                        ) from exc

                    if keep_unverified_on_failure:
                        accepted.append({"memory_text": memory_text, "embedding": None})
                        log.warning(
                            "个人记忆语义去重降级保留未校验候选: user_id=%s, err=%s",
                            discord_id_str,
                            exc,
                        )
                    else:
                        failed.append({"memory_text": memory_text, "reason": str(exc)})
                        log.error(
                            "为用户 %s 的候选记忆生成 embedding 失败（idx=%s/%s）: %s",
                            discord_id_str,
                            idx,
                            len(cleaned_texts),
                            exc,
                            exc_info=True,
                        )
                    continue

                if not embedding or not isinstance(embedding, list):
                    if fail_on_embedding_error:
                        raise RuntimeError(
                            f"为候选记忆生成 embedding 返回空（{idx}/{len(cleaned_texts)}）。"
                        )

                    if keep_unverified_on_failure:
                        accepted.append({"memory_text": memory_text, "embedding": None})
                        log.warning(
                            "个人记忆语义去重降级保留未校验候选: user_id=%s, err=empty_embedding",
                            discord_id_str,
                        )
                    else:
                        failed.append(
                            {"memory_text": memory_text, "reason": "empty_embedding"}
                        )
                        log.warning(
                            "为用户 %s 的候选记忆生成 embedding 返回空（idx=%s/%s），已跳过该条。",
                            discord_id_str,
                            idx,
                            len(cleaned_texts),
                        )
                    continue

                batch_match = self._find_batch_match(
                    representatives=representatives,
                    candidate_embedding=embedding,
                    max_distance=max_distance_value,
                )
                if batch_match:
                    deduped.append(
                        {
                            "memory_text": memory_text,
                            "matched_text": batch_match["memory_text"],
                            "distance": batch_match["distance"],
                            "source": "batch",
                        }
                    )
                    continue

                existing_match = await self._find_nearest_existing_long_term_match(
                    session=session,
                    discord_id=discord_id_str,
                    candidate_embedding=embedding,
                )
                if (
                    existing_match
                    and existing_match.get("distance") is not None
                    and float(existing_match["distance"]) <= max_distance_value
                ):
                    deduped.append(
                        {
                            "memory_text": memory_text,
                            "matched_text": existing_match["memory_text"],
                            "distance": float(existing_match["distance"]),
                            "source": "existing",
                            "matched_existing_id": existing_match.get("id"),
                            "matched_existing_text": existing_match["memory_text"],
                            "matched_existing_embedding": existing_match.get("embedding"),
                        }
                    )
                    continue

                accepted.append({"memory_text": memory_text, "embedding": embedding})
                representatives.append(
                    {
                        "memory_text": memory_text,
                        "embedding": embedding,
                        "source": "accepted",
                    }
                )

        semantic_deduped_count = len(deduped) - batch_duplicates_count
        if deduped or failed:
            log.info(
                "个人记忆语义去重完成: user_id=%s, received=%s, accepted=%s, deduped_exact=%s, deduped_semantic=%s, failed=%s",
                discord_id_str,
                len(cleaned_texts),
                len(accepted),
                batch_duplicates_count,
                semantic_deduped_count,
                len(failed),
            )

        return {"accepted": accepted, "deduped": deduped, "failed": failed}

    async def delete_vectors_for_user(self, discord_id: int | str) -> int:
        discord_id_str = str(discord_id)
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(
                    delete(PersonalMemoryChunk).where(
                        PersonalMemoryChunk.discord_id == discord_id_str
                    )
                )
                deleted = int(getattr(result, "rowcount", 0) or 0)
        return deleted

    async def delete_vectors_for_user_memory_texts(
        self,
        discord_id: int | str,
        memory_texts: List[str],
        memory_type: Optional[str] = "long_term",
    ) -> int:
        discord_id_str = str(discord_id)
        cleaned = [self._normalize_dedupe_text(t) for t in (memory_texts or [])]
        cleaned = [t for t in cleaned if t]
        if not cleaned:
            return 0

        cleaned = list(dict.fromkeys(cleaned))

        async with AsyncSessionLocal() as session:
            async with session.begin():
                stmt = delete(PersonalMemoryChunk).where(
                    PersonalMemoryChunk.discord_id == discord_id_str
                )
                if memory_type is not None:
                    stmt = stmt.where(PersonalMemoryChunk.memory_type == memory_type)
                stmt = stmt.where(PersonalMemoryChunk.memory_text.in_(cleaned))

                result = await session.execute(stmt)
                deleted = int(getattr(result, "rowcount", 0) or 0)

        log.info(
            "已删除用户的个人记忆向量条目"
        )
        return deleted

    async def cleanup_recent_vectors(
        self, discord_id: Optional[int | str] = None
    ) -> int:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                stmt = delete(PersonalMemoryChunk).where(
                    PersonalMemoryChunk.memory_type == "recent"
                )
                if discord_id is not None:
                    stmt = stmt.where(PersonalMemoryChunk.discord_id == str(discord_id))

                result = await session.execute(stmt)
                deleted = int(getattr(result, "rowcount", 0) or 0)

        return deleted

    async def _load_existing_vector_rows(
        self, discord_id: str
    ) -> List[tuple[int, str, str]]:
        async with AsyncSessionLocal() as session:
            res = await session.execute(
                select(
                    PersonalMemoryChunk.id,
                    PersonalMemoryChunk.memory_type,
                    PersonalMemoryChunk.memory_text,
                ).where(PersonalMemoryChunk.discord_id == discord_id)
            )
            return list(res.all())

    def _extract_target_keys(
        self, personal_summary: str
    ) -> List[tuple[str, str]]:
        items = self.parse_personal_summary(personal_summary)
        target_keys: List[tuple[str, str]] = []
        for item in items:
            memory_text = self._normalize_dedupe_text(item.get("memory_text"))
            if not memory_text:
                continue
            memory_type = str(item.get("memory_type") or "unknown")
            target_keys.append((memory_type, memory_text))
        return target_keys

    async def sync_vectors_for_user(
        self,
        discord_id: int | str,
        personal_summary: Optional[str],
    ) -> Dict[str, int]:
        discord_id_str = str(discord_id)

        if not personal_summary or not personal_summary.strip():
            deleted = await self.delete_vectors_for_user(discord_id_str)
            return {"inserted": 0, "deleted": deleted, "deduped": 0}

        new_keys = self._extract_target_keys(personal_summary)
        if not new_keys:
            if "### 长期记忆" in personal_summary:
                deleted = await self.delete_vectors_for_user(discord_id_str)
                return {"inserted": 0, "deleted": deleted, "deduped": 0}

            log.warning(
                "用户 %s 的 personal_summary 非空，但未解析出任何长期记忆条目，且未找到 '### 长期记忆' 标题，已跳过同步以避免误删旧向量。",
                discord_id_str,
            )
            return {"inserted": 0, "deleted": 0, "deduped": 0}

        new_set = set(new_keys)
        existing_rows = await self._load_existing_vector_rows(discord_id_str)
        existing_set: set[tuple[str, str]] = set()
        existing_id_by_key: dict[tuple[str, str], int] = {}
        for row in existing_rows:
            row_id = int(row[0])
            memory_type = str(row[1] or "unknown")
            memory_text = self._normalize_dedupe_text(row[2])
            if not memory_text:
                continue
            key = (memory_type, memory_text)
            existing_set.add(key)
            existing_id_by_key[key] = row_id

        to_add = [key for key in new_keys if key not in existing_set]
        if to_add and not gemini_service.is_available():
            return {"inserted": 0, "deleted": 0, "deduped": 0}

        dedupe_result = await self.filter_semantically_unique_long_term_candidates(
            discord_id=discord_id_str,
            candidate_memory_texts=[memory_text for _, memory_text in to_add],
            fail_on_embedding_error=False,
            keep_unverified_on_failure=False,
        )

        deduped_entries = dedupe_result["deduped"]
        failed_entries = dedupe_result["failed"]
        accepted_entries = dedupe_result["accepted"]

        deduped_existing_entries: List[Dict[str, Any]] = []
        seen_existing_ids: set[int] = set()
        for entry in deduped_entries:
            if entry.get("source") != "existing":
                continue
            existing_id = entry.get("matched_existing_id")
            if existing_id is None or existing_id in seen_existing_ids:
                continue
            seen_existing_ids.add(existing_id)
            deduped_existing_entries.append(entry)

        if to_add and not accepted_entries and failed_entries and not deduped_existing_entries:
            log.error(
                "用户 %s 的新增记忆 embedding 全部失败，已跳过本次同步，避免破坏旧向量。",
                discord_id_str,
            )
            return {"inserted": 0, "deleted": 0, "deduped": 0}

        retained_existing_keys = {
            ("long_term", self._normalize_dedupe_text(entry.get("matched_existing_text")))
            for entry in deduped_existing_entries
            if self._normalize_dedupe_text(entry.get("matched_existing_text"))
        }
        to_delete = [
            key for key in existing_set if key not in new_set and key not in retained_existing_keys
        ]

        rows_to_insert: List[PersonalMemoryChunk] = []
        for entry in accepted_entries:
            embedding = entry.get("embedding")
            memory_text = self._normalize_dedupe_text(entry.get("memory_text"))
            if not memory_text or not embedding:
                continue
            rows_to_insert.append(
                PersonalMemoryChunk(
                    discord_id=discord_id_str,
                    memory_type="long_term",
                    memory_text=memory_text,
                    embedding=embedding,
                )
            )

        updates_to_apply = [
            {
                "id": int(entry["matched_existing_id"]),
                "memory_text": self._normalize_dedupe_text(entry.get("memory_text")),
                "matched_existing_text": self._normalize_dedupe_text(
                    entry.get("matched_existing_text")
                ),
            }
            for entry in deduped_existing_entries
            if entry.get("matched_existing_id") is not None
            and self._normalize_dedupe_text(entry.get("memory_text"))
            and self._normalize_dedupe_text(entry.get("memory_text"))
            != self._normalize_dedupe_text(entry.get("matched_existing_text"))
        ]

        if not rows_to_insert and not to_delete and not updates_to_apply:
            return {
                "inserted": 0,
                "deleted": 0,
                "deduped": len(deduped_entries),
            }

        deleted_count = 0
        async with AsyncSessionLocal() as session:
            async with session.begin():
                if to_delete:
                    result = await session.execute(
                        delete(PersonalMemoryChunk).where(
                            PersonalMemoryChunk.discord_id == discord_id_str,
                            tuple_(
                                PersonalMemoryChunk.memory_type,
                                PersonalMemoryChunk.memory_text,
                            ).in_(to_delete),
                        )
                    )
                    deleted_count = int(getattr(result, "rowcount", 0) or 0)

                for update_item in updates_to_apply:
                    existing_key = (
                        "long_term",
                        update_item["matched_existing_text"],
                    )
                    fallback_id = existing_id_by_key.get(existing_key)
                    target_id = update_item["id"] or fallback_id
                    if target_id is None:
                        continue

                    await session.execute(
                        update(PersonalMemoryChunk)
                        .where(PersonalMemoryChunk.id == target_id)
                        .values(memory_text=update_item["memory_text"])
                    )

                if rows_to_insert:
                    session.add_all(rows_to_insert)

        log.info(
            "个人记忆向量已增量同步: discord_id=%s",
            discord_id_str,
        )
        return {
            "inserted": len(rows_to_insert),
            "deleted": deleted_count,
            "deduped": len(deduped_entries),
        }

    async def sync_vectors_for_user_strict(
        self,
        discord_id: int | str,
        personal_summary: Optional[str],
    ) -> Dict[str, int]:
        discord_id_str = str(discord_id)

        if not personal_summary or not personal_summary.strip():
            deleted = await self.delete_vectors_for_user(discord_id_str)
            return {"inserted": 0, "deleted": deleted, "deduped": 0}

        new_keys = self._extract_target_keys(personal_summary)
        if not new_keys:
            if "### 长期记忆" in personal_summary:
                deleted = await self.delete_vectors_for_user(discord_id_str)
                return {"inserted": 0, "deleted": deleted, "deduped": 0}
            return {"inserted": 0, "deleted": 0, "deduped": 0}

        new_set = set(new_keys)
        existing_rows = await self._load_existing_vector_rows(discord_id_str)
        existing_set: set[tuple[str, str]] = set()
        existing_id_by_key: dict[tuple[str, str], int] = {}
        for row in existing_rows:
            row_id = int(row[0])
            memory_type = str(row[1] or "unknown")
            memory_text = self._normalize_dedupe_text(row[2])
            if not memory_text:
                continue
            key = (memory_type, memory_text)
            existing_set.add(key)
            existing_id_by_key[key] = row_id

        to_add = [key for key in new_keys if key not in existing_set]
        if to_add and not gemini_service.is_available():
            raise RuntimeError(
                "RAG/embedding 未启用（未配置 GOOGLE_API_KEYS_LIST），无法为新增长期记忆生成向量。"
            )

        dedupe_result = await self.filter_semantically_unique_long_term_candidates(
            discord_id=discord_id_str,
            candidate_memory_texts=[memory_text for _, memory_text in to_add],
            fail_on_embedding_error=True,
            keep_unverified_on_failure=False,
        )

        deduped_entries = dedupe_result["deduped"]
        accepted_entries = dedupe_result["accepted"]

        deduped_existing_entries: List[Dict[str, Any]] = []
        seen_existing_ids: set[int] = set()
        for entry in deduped_entries:
            if entry.get("source") != "existing":
                continue
            existing_id = entry.get("matched_existing_id")
            if existing_id is None or existing_id in seen_existing_ids:
                continue
            seen_existing_ids.add(existing_id)
            deduped_existing_entries.append(entry)

        retained_existing_keys = {
            ("long_term", self._normalize_dedupe_text(entry.get("matched_existing_text")))
            for entry in deduped_existing_entries
            if self._normalize_dedupe_text(entry.get("matched_existing_text"))
        }
        to_delete = [
            key for key in existing_set if key not in new_set and key not in retained_existing_keys
        ]

        rows_to_insert: List[PersonalMemoryChunk] = []
        for entry in accepted_entries:
            embedding = entry.get("embedding")
            memory_text = self._normalize_dedupe_text(entry.get("memory_text"))
            if not memory_text or not embedding:
                continue
            rows_to_insert.append(
                PersonalMemoryChunk(
                    discord_id=discord_id_str,
                    memory_type="long_term",
                    memory_text=memory_text,
                    embedding=embedding,
                )
            )

        updates_to_apply = [
            {
                "id": int(entry["matched_existing_id"]),
                "memory_text": self._normalize_dedupe_text(entry.get("memory_text")),
                "matched_existing_text": self._normalize_dedupe_text(
                    entry.get("matched_existing_text")
                ),
            }
            for entry in deduped_existing_entries
            if entry.get("matched_existing_id") is not None
            and self._normalize_dedupe_text(entry.get("memory_text"))
            and self._normalize_dedupe_text(entry.get("memory_text"))
            != self._normalize_dedupe_text(entry.get("matched_existing_text"))
        ]

        if not rows_to_insert and not to_delete and not updates_to_apply:
            return {
                "inserted": 0,
                "deleted": 0,
                "deduped": len(deduped_entries),
            }

        deleted_count = 0
        async with AsyncSessionLocal() as session:
            async with session.begin():
                if to_delete:
                    result = await session.execute(
                        delete(PersonalMemoryChunk).where(
                            PersonalMemoryChunk.discord_id == discord_id_str,
                            tuple_(
                                PersonalMemoryChunk.memory_type,
                                PersonalMemoryChunk.memory_text,
                            ).in_(to_delete),
                        )
                    )
                    deleted_count = int(getattr(result, "rowcount", 0) or 0)

                for update_item in updates_to_apply:
                    existing_key = (
                        "long_term",
                        update_item["matched_existing_text"],
                    )
                    fallback_id = existing_id_by_key.get(existing_key)
                    target_id = update_item["id"] or fallback_id
                    if target_id is None:
                        continue

                    await session.execute(
                        update(PersonalMemoryChunk)
                        .where(PersonalMemoryChunk.id == target_id)
                        .values(memory_text=update_item["memory_text"])
                    )

                if rows_to_insert:
                    session.add_all(rows_to_insert)

        log.info(
            "个人记忆向量已严格增量同步: discord_id=%s, ",
            discord_id_str,
        )
        return {
            "inserted": len(rows_to_insert),
            "deleted": deleted_count,
            "deduped": len(deduped_entries),
        }

    async def rebuild_vectors_for_user(
        self,
        discord_id: int | str,
        personal_summary: Optional[str],
    ) -> int:
        discord_id_str = str(discord_id)

        if not personal_summary or not personal_summary.strip():
            await self.delete_vectors_for_user(discord_id_str)
            return 0

        if not gemini_service.is_available():
            log.info(
                "RAG/embedding 未启用（未配置 GOOGLE_API_KEYS_LIST），跳过个人记忆向量重建。discord_id=%s",
                discord_id_str,
            )
            return 0

        items = self.parse_personal_summary(personal_summary)
        if not items:
            if "### 长期记忆" in personal_summary:
                await self.delete_vectors_for_user(discord_id_str)
                return 0

            log.warning(
                "用户 %s 的 personal_summary 非空，但未解析出任何长期记忆条目，且未找到 '### 长期记忆' 标题，已跳过重建以避免清空旧向量。",
                discord_id_str,
            )
            return 0

        dedupe_result = await self.filter_semantically_unique_long_term_candidates(
            discord_id=discord_id_str,
            candidate_memory_texts=[
                self._normalize_dedupe_text(item.get("memory_text")) for item in items
            ],
            fail_on_embedding_error=False,
            keep_unverified_on_failure=False,
        )

        rows_to_insert: List[PersonalMemoryChunk] = []
        for entry in dedupe_result["accepted"]:
            embedding = entry.get("embedding")
            memory_text = self._normalize_dedupe_text(entry.get("memory_text"))
            if not memory_text or not embedding:
                continue

            rows_to_insert.append(
                PersonalMemoryChunk(
                    discord_id=discord_id_str,
                    memory_type="long_term",
                    memory_text=memory_text,
                    embedding=embedding,
                )
            )

        seen_reused_existing_ids: set[int] = set()
        for entry in dedupe_result["deduped"]:
            if entry.get("source") != "existing":
                continue

            existing_id = entry.get("matched_existing_id")
            if existing_id is None or existing_id in seen_reused_existing_ids:
                continue

            matched_embedding = entry.get("matched_existing_embedding")
            memory_text = self._normalize_dedupe_text(entry.get("memory_text"))
            if not memory_text or not matched_embedding:
                continue

            seen_reused_existing_ids.add(existing_id)
            rows_to_insert.append(
                PersonalMemoryChunk(
                    discord_id=discord_id_str,
                    memory_type="long_term",
                    memory_text=memory_text,
                    embedding=matched_embedding,
                )
            )

        if not rows_to_insert and dedupe_result["failed"]:
            log.error(
                "用户 %s 的个人记忆条目 embedding 全部失败，已跳过写入，避免清空旧向量。",
                discord_id_str,
            )
            return 0

        if not rows_to_insert:
            await self.delete_vectors_for_user(discord_id_str)
            return 0

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    delete(PersonalMemoryChunk).where(
                        PersonalMemoryChunk.discord_id == discord_id_str
                    )
                )
                session.add_all(rows_to_insert)

        return len(rows_to_insert)

    async def rebuild_vectors_for_user_from_db(self, discord_id: int | str) -> int:
        discord_id_str = str(discord_id)
        async with AsyncSessionLocal() as session:
            stmt = select(CommunityMemberProfile.personal_summary).where(
                CommunityMemberProfile.discord_id == discord_id_str
            )
            result = await session.execute(stmt)
            summary = result.scalars().first()

        return await self.rebuild_vectors_for_user(discord_id_str, summary)


personal_memory_vector_service = PersonalMemoryVectorService()
