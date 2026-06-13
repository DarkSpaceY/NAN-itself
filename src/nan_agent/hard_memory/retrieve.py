"""记忆检索引擎（Retrieve）。

向量语义 + BM25 关键词 + 时间图 → RRF 三路融合 → 二次检索 → 内容截断。

二次检索：从首次检索结果中提取高 IDF 关键词，做补充 BM25 查询，合并去重。
"""

import math
from collections import defaultdict

from nan_agent.hard_memory.common import tokenize
from nan_agent.hard_memory.memory import Memory
from nan_agent.hard_memory.store import Store
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

MAX_CONTENT_CHARS = 600
RRF_K = 10
SECONDARY_QUERY_TOP_TERMS = 5  # 二次检索提取的关键词数


def _rrf_fuse(
    vector_results: list[str],
    bm25_results: list[str],
    temporal_results: list[str] | None = None,
    k: int = RRF_K,
) -> list[str]:
    """Reciprocal Rank Fusion 三路融合。"""
    scores: dict[str, float] = {}
    for rank, mem_id in enumerate(vector_results, 1):
        scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank)
    for rank, mem_id in enumerate(bm25_results, 1):
        scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank)
    if temporal_results:
        for rank, mem_id in enumerate(temporal_results, 1):
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [mem_id for mem_id, _ in ranked]


# 停用词表（常见英文停用词）
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very",
    "just", "because", "if", "when", "where", "how", "what", "which", "who",
    "whom", "this", "that", "these", "those", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "about", "up", "also", "there", "here", "much", "well", "really",
    "like", "know", "think", "get", "got", "go", "going", "come", "see",
    "make", "said", "say", "one", "two", "new", "now", "way", "even",
    "want", "tell", "thing", "things", "something", "anything", "nothing",
    "everything", "lot", "bit", "quite", "actually", "basically",
})


def _extract_high_idf_terms(
    texts: list[str],
    store_bm25: "Store",
    top_n: int = SECONDARY_QUERY_TOP_TERMS,
    exclude_terms: set[str] | None = None,
) -> str:
    """从检索结果中提取高 IDF 关键词，用于二次检索。

    IDF 越高的词越具区分性（专有名词、领域术语等）。
    排除原查询中已有的词，只提取原查询未覆盖的新线索。
    """
    exclude = exclude_terms or set()
    term_idf: dict[str, float] = {}
    total_docs = store_bm25._bm25._total_docs
    if total_docs == 0:
        return ""

    for text in texts:
        tokens = set(tokenize(text))
        for t in tokens:
            if t in _STOP_WORDS or len(t) < 3 or t in exclude:
                continue
            if t not in term_idf:
                n = len(store_bm25._bm25._inverted.get(t, {}))
                term_idf[t] = math.log((total_docs - n + 0.5) / (n + 0.5) + 1.0)

    if not term_idf:
        return ""

    sorted_terms = sorted(term_idf.items(), key=lambda x: x[1], reverse=True)
    return " ".join(t for t, _ in sorted_terms[:top_n])


class RetrieveEngine:
    """混合检索引擎。

    向量 + BM25 + 时间图 → RRF 三路融合 → 二次检索 → 截断。

    Attributes:
        _store: Store 实例。
    """

    def __init__(self, store: Store):
        self._store = store

    async def search(
        self,
        query: str,
        top_k: int = 10,
        query_timestamp: str = "",
        temporal_window_days: int = 3,
        secondary: bool = False,
        current_emotion: dict | None = None,
    ) -> list[Memory]:
        """三路混合检索 + 情绪引导 + 可选二次检索。

        流程：
        1. 向量语义检索
        2. BM25 关键词检索
        3. 时间图检索（如果提供了 query_timestamp）
        4. RRF 三路融合
        5. 情绪引导加分（MCM + SDM）
        6. 二次检索（从首次结果提取高 IDF 关键词，补充 BM25 查询）
        7. 截断 → top_k
        """
        candidate_n = max(top_k * 2, 10)

        # 1. 向量 + BM25
        vec_mems = await self._store.search_vector(query, top_k=candidate_n)
        vec_ids = [m.id for m in vec_mems]

        bm25_results = self._store.search_bm25(query, top_k=candidate_n)
        bm25_ids = [m.id for m, _ in bm25_results]

        # 2. 时间图检索
        temporal_ids: list[str] = []
        if query_timestamp:
            temporal_scored = self._store.search_temporal(
                query_timestamp, window_days=temporal_window_days, top_k=candidate_n
            )
            temporal_ids = [mid for mid, _ in temporal_scored]

        # 3. RRF 三路融合
        fused_ids = _rrf_fuse(vec_ids, bm25_ids, temporal_ids or None)

        # 4. 情绪引导加分（MCM 心境一致性 + SDM 状态依赖性）
        if current_emotion and fused_ids:
            fused_ids = self._apply_emotion_bonus(fused_ids, current_emotion)

        # 5. id → Memory
        id_to_mem: dict[str, Memory] = {}
        for m in vec_mems:
            id_to_mem[m.id] = m
        for m, _ in bm25_results:
            id_to_mem[m.id] = m

        # 6. 二次检索
        if secondary and fused_ids:
            top_texts = []
            for mid in fused_ids[:5]:
                mem = id_to_mem.get(mid) or self._store.get(mid)
                if mem:
                    top_texts.append(mem.content)
                    id_to_mem[mid] = mem

            if top_texts:
                query_terms = set(tokenize(query))
                secondary_query = _extract_high_idf_terms(
                    top_texts, self._store,
                    exclude_terms=query_terms,
                )
                if secondary_query:
                    logger.debug("secondary_query", original=query[:50], secondary=secondary_query)
                    sec_vec_mems = await self._store.search_vector(secondary_query, top_k=candidate_n)
                    sec_bm25_results = self._store.search_bm25(secondary_query, top_k=candidate_n)

                    sec_vec_ids = [m.id for m in sec_vec_mems]
                    sec_bm25_ids = [m.id for m, _ in sec_bm25_results]

                    sec_fused = _rrf_fuse(sec_vec_ids, sec_bm25_ids)

                    existing = set(fused_ids)
                    for mid in sec_fused:
                        if mid not in existing:
                            fused_ids.append(mid)
                            existing.add(mid)

                    for m in sec_vec_mems:
                        id_to_mem[m.id] = m
                    for m, _ in sec_bm25_results:
                        id_to_mem[m.id] = m

        # 补充时间图新结果
        for mid in fused_ids:
            if mid not in id_to_mem:
                mem = self._store.get(mid)
                if mem:
                    id_to_mem[mid] = mem

        # 7. 组装 + 截断
        result = []
        for mid in fused_ids:
            if mid in id_to_mem:
                mem = id_to_mem[mid]
                if len(mem.content) > MAX_CONTENT_CHARS:
                    mem.content = mem.content[:MAX_CONTENT_CHARS - 3] + "..."
                result.append(mem)
                if len(result) >= top_k:
                    break

        return result

    def _apply_emotion_bonus(
        self, fused_ids: list[str], current_emotion: dict
    ) -> list[str]:
        """情绪引导加分：心境一致性 (MCM) + 状态依赖性 (SDM)。

        原理：
        - MCM: 当前情绪效价与记忆效价一致 → 加分，对立 → 减分
        - SDM: 当前情绪与编码时情绪相似 → 加分
        - 高唤醒记忆 → 小幅加分（杏仁核增强效应）
        - 无情感标签记忆 → 不加分也不减分（中性处理）

        基于 Emotional RAG (Huang et al., 2024) 和 CMR3 (Cohen & Kahana, 2022)。
        """
        cur_val = current_emotion.get("valence", 0.0)
        cur_ars = current_emotion.get("arousal", 0.0)

        scored: list[tuple[float, str]] = []
        for i, mid in enumerate(fused_ids):
            base_score = 1.0 / (RRF_K + i + 1)

            mem = self._store.get(mid)
            mem_emotion = mem.metadata.get("emotional_state") if mem else None

            bonus = 0.0
            if mem_emotion:
                mem_val = mem_emotion.get("valence", 0.0)
                mem_ars = mem_emotion.get("arousal", 0.0)

                # MCM: 效价一致性（同号加分，异号减分）
                if cur_val * mem_val > 0:
                    bonus += 0.15
                elif cur_val * mem_val < 0:
                    bonus -= 0.05

                # SDM: 状态相似度
                val_diff = abs(cur_val - mem_val)
                ars_diff = abs(cur_ars - mem_ars)
                state_sim = max(0.0, 1.0 - (val_diff + ars_diff) / 4.0)
                bonus += 0.10 * state_sim

                # 高唤醒记忆优先（取绝对值，负 arousal 不应变成减分）
                bonus += 0.05 * max(0.0, mem_ars)

            # 无情感标签记忆：bonus = 0.0，不受惩罚

            scored.append((base_score + bonus, mid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mid for _, mid in scored]
