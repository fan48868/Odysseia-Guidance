# -*- coding: utf-8 -*-
import logging
import re
from typing import Any, List, Dict

from sqlalchemy import text
from src.database.database import AsyncSessionLocal
from src.chat.services.gemini_service import gemini_service

log = logging.getLogger(__name__)


class KnowledgeSearchService:
    """
    提供对社区成员和通用知识的混合搜索功能。
    使用 ParadeDB 的 BM25 和向量搜索能力。
    """

    def __init__(self):
        log.info("KnowledgeSearchService 已初始化")
        # 在未来可以从配置中加载参数
        self.config = {
            "TOP_K_VECTOR": 30,  # 增加召回量，防止指令词(如"开始xx模式")导致核心词排名下降
            "TOP_K_FTS": 30,     # 增加召回量，提高关键字匹配的容错率
            "RRF_K": 60,
            "HYBRID_SEARCH_FINAL_K": 5, # 返回更多结果给上层，避免过早截断
            "VECTOR_DISTANCE_THRESHOLD": 0.5, # 向量搜索距离阈值，过滤不相关结果
            "KEYWORD_WEIGHT": 3.0, # 关键字搜索权重，提高精确匹配的重要性
        }

    def _clean_fts_query(self, query: str) -> str:
        """
        清理全文搜索查询，移除可能导致 paradedb 解析错误的特殊字符。
        只保留字母、数字、中日韩统一表意文字和空格。
        """
        # 正则表达式匹配所有非（字母、数字、CJK字符、空格）的字符
        cleaned_query = re.sub(r"[^\w\s\u4e00-\u9fff]", "", query)
        log.debug(f"原始 FTS 查询: '{query}' -> 清理后: '{cleaned_query}'")
        return cleaned_query

    async def _hybrid_search_chunks(
        self, session, query_text: str, query_vector: List[float]
    ) -> List[Dict[str, Any]]:
        """
        在 community.member_chunks 和 general_knowledge.knowledge_chunks
        两个表中执行混合搜索，并返回融合排序后的 chunk 结果。
        """
        # SQL 查询同时搜索两个 chunks 表
        sql_query = text(
            """
            WITH semantic_search AS (
                -- 社区成员向量搜索
                (SELECT
                    'community' as source_table,
                    id as chunk_id,
                    profile_id as parent_id,
                    chunk_text,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> :query_vector) as rank
                FROM community.member_chunks
                WHERE (embedding <=> :query_vector) < :max_distance
                ORDER BY embedding <=> :query_vector
                LIMIT :top_k_vector)
                UNION ALL
                -- 通用知识向量搜索
                (SELECT
                    'general_knowledge' as source_table,
                    id as chunk_id,
                    document_id as parent_id,
                    chunk_text,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> :query_vector) as rank
                FROM general_knowledge.knowledge_chunks
                WHERE (embedding <=> :query_vector) < :max_distance
                ORDER BY embedding <=> :query_vector
                LIMIT :top_k_vector)
            ),
            keyword_search AS (
                -- 社区成员 BM25 搜索 (使用 paradedb.score)
                (SELECT
                    'community' as source_table,
                    id as chunk_id,
                    profile_id as parent_id,
                    chunk_text,
                    ROW_NUMBER() OVER (ORDER BY paradedb.score(id) DESC) as rank
                FROM community.member_chunks
                WHERE chunk_text @@@ :query_text
                LIMIT :top_k_fts)
                UNION ALL
                -- 通用知识 BM25 搜索 (使用 paradedb.score)
                (SELECT
                    'general_knowledge' as source_table,
                    id as chunk_id,
                    document_id as parent_id,
                    chunk_text,
                    ROW_NUMBER() OVER (ORDER BY paradedb.score(id) DESC) as rank
                FROM general_knowledge.knowledge_chunks
                WHERE chunk_text @@@ :query_text
                LIMIT :top_k_fts)
            ),
            -- 使用 RRF (Reciprocal Rank Fusion) 融合排名
            fused_ranks AS (
                SELECT
                    COALESCE(s.chunk_id, k.chunk_id) as chunk_id,
                    COALESCE(s.parent_id, k.parent_id) as document_id,
                    COALESCE(s.source_table, k.source_table) as source_table,
                    COALESCE(s.chunk_text, k.chunk_text) as chunk_text,
                    (COALESCE(1.0 / (:rrf_k + s.rank), 0.0) + COALESCE(:keyword_weight / (:rrf_k + k.rank), 0.0)) as rrf_score
                FROM semantic_search s
                FULL OUTER JOIN keyword_search k ON s.chunk_id = k.chunk_id AND s.source_table = k.source_table
            )
            SELECT *
            FROM fused_ranks
            ORDER BY rrf_score DESC
            LIMIT :final_k;
            """
        )
        result = await session.execute(
            sql_query,
            {
                "query_text": query_text,
                "query_vector": str(query_vector),
                "top_k_vector": self.config["TOP_K_VECTOR"],
                "top_k_fts": self.config["TOP_K_FTS"],
                "rrf_k": self.config["RRF_K"],
                "final_k": self.config["HYBRID_SEARCH_FINAL_K"],
                "max_distance": self.config["VECTOR_DISTANCE_THRESHOLD"],
                "keyword_weight": self.config["KEYWORD_WEIGHT"],
            },
        )
        # SQLAlchemy 2.x 的 Row 对象需要通过 ._mapping 转换为字典
        return [dict(row._mapping) for row in result.fetchall()]

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        执行完整的 RAG 混合搜索流程。
        1. 生成查询嵌入。
        2. 在 chunks 表中进行混合搜索。
        3. (暂定)直接返回 chunks 内容，因为我们的场景下，chunk 可能就是全部。
        """
        log.info(f"收到知识库混合搜索请求: '{query}'")

        try:
            query_embedding = await gemini_service.generate_embedding(
                text=query, task_type="retrieval_query"
            )
            if not query_embedding or not isinstance(query_embedding, list):
                raise ValueError(f"Embedding 生成失败，返回为空或类型错误: {type(query_embedding)}")
        except Exception as e:
            log.error(f"为查询 '{query}' 生成 embedding 时出错: {e}", exc_info=True)
            return []

        search_results = []
        try:
            # 清理用于全文搜索的查询文本
            cleaned_fts_query = self._clean_fts_query(query)

            async with AsyncSessionLocal() as session:
                search_results = await self._hybrid_search_chunks(
                    session, cleaned_fts_query, query_embedding
                )
                log.info(f"混合搜索 RRF 结果: {search_results}")

        except Exception as e:
            log.error(f"在数据库中执行混合搜索时出错: {e}", exc_info=True)
            return []

        if not search_results:
            log.info(f"知识库混合搜索未找到 '{query}' 的相关文档。")
            return []

        # 转换为 world_book_service 期望的格式
        # 'content' 字段直接使用 chunk_text
        formatted_results = []
        for res in search_results:
            rrf_score = float(res["rrf_score"])  # 确保为浮点数
            formatted_results.append(
                {
                    "id": res.get("document_id"),
                    "content": res.get("chunk_text"),
                    "distance": 1.0 - rrf_score,  # 将rrf_score转换为类似距离的度量
                    "metadata": {"source_table": res.get("source_table")},
                }
            )

        return formatted_results


# 创建服务的单例
knowledge_search_service = KnowledgeSearchService()
