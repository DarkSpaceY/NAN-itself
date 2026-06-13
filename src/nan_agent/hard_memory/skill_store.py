"""技能存储模块（SkillStore）。

独立于 Episodic Store，专门存储 Exp（文字型 Skills）。
Exp 的格式参考 AutoGuide："When [situation], should [behavior]"。

检索方式：BM25 条件匹配 + 向量语义 → RRF 两路融合。
生命周期：效用追踪（use_count / success_count），长期无效的自动淘汰。
启动时从 ChromaDB 恢复内存索引。
"""

import json
import re
from collections import defaultdict
from typing import Optional

from nan_agent.hard_memory.common import BM25Index, new_id, tokenize
from nan_agent.hard_memory.memory import Memory
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

SKILL_COLLECTION = "skills"


def _extract_condition(exp_content: str) -> str:
    """从 Exp 内容中提取条件部分。

    "When discussing career with Caroline, support her transition..."
    → "discussing career with Caroline"
    """
    m = re.match(r'[Ww]hen\s+(.+?)[,.]\s*', exp_content)
    if m:
        return m.group(1).strip()
    return exp_content[:200]


class SkillStore:
    """技能存储层。

    独立于 Episodic Store，存储 Exp（文字型 Skills）。
    支持启动时从 ChromaDB 恢复内存索引。

    Attributes:
        _ollama: Ollama 模型提供者。
        _vs: ChromaDB VectorStore 实例。
        _bm25: BM25 条件索引。
        _skills: 内存中的 Exp 索引（id → Memory）。
        _utility: 效用追踪（id → {"use": n, "success": n}）。
    """

    def __init__(self, ollama_provider, vector_store):
        self._ollama = ollama_provider
        self._vs = vector_store
        self._bm25 = BM25Index()
        self._skills: dict[str, Memory] = {}
        self._utility: dict[str, dict] = {}

    async def load(self) -> None:
        """从 ChromaDB 恢复所有技能索引。"""
        try:
            client = self._vs._get_client()
            col = client.get_or_create_collection(name=SKILL_COLLECTION)
            count = col.count()
            if count == 0:
                return

            results = col.get(include=["metadatas", "documents"])
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])

            for i, mid in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else None
                if content is None:
                    content = meta.get("content", "")
                if not content:
                    continue  # 跳过空内容记录

                timestamp = meta.get("timestamp") or ""
                mem = Memory(
                    id=mid,
                    type="exp",
                    content=content,
                    episode_text="",
                    source="dream",
                    timestamp=timestamp,
                    parent_id=meta.get("parent_id", ""),
                )

                condition = _extract_condition(content)
                self._skills[mid] = mem
                self._bm25.add(mid, condition)
                self._utility[mid] = {"use": 0, "success": 0}

            logger.info("skill_store_loaded", count=len(self._skills))
        except Exception as e:
            logger.warning("skill_store_load_failed", error=str(e))

    async def add(self, content: str, source_facts: list[str] = None, timestamp: str = "") -> Memory:
        """添加一条 Exp。

        Args:
            content: Exp 全文，格式 "When [situation], should [behavior]"。
            source_facts: 来源 fact 的 ID 列表。
            timestamp: 时间戳。
        """
        condition = _extract_condition(content)
        embedding = await self._ollama.embed(condition)

        mem = Memory(
            id=new_id(),
            type="exp",
            content=content,
            episode_text="",
            source="dream",
            timestamp=timestamp,
            parent_id=",".join(source_facts) if source_facts else "",
        )

        chroma_meta: dict = {"content": content, "source": "dream", "type": "exp"}
        if timestamp:
            chroma_meta["timestamp"] = timestamp
        if source_facts:
            chroma_meta["parent_id"] = ",".join(source_facts)

        await self._vs.add(SKILL_COLLECTION, mem.id, embedding, metadata=chroma_meta)
        self._bm25.add(mem.id, condition)
        self._skills[mem.id] = mem
        self._utility[mem.id] = {"use": 0, "success": 0}

        logger.debug("exp_stored", id=mem.id)
        return mem

    async def search(self, situation: str, top_k: int = 5) -> list[Memory]:
        """按情境匹配检索 Exp。

        Args:
            situation: 当前情境描述。
            top_k: 返回的最大 Exp 数。
        """
        if not self._skills:
            return []

        # BM25 条件匹配
        bm25_results = self._bm25.search(situation, top_k=top_k * 2)

        # 向量语义匹配
        query_emb = await self._ollama.embed(situation)
        vec_results = await self._vs.search(SKILL_COLLECTION, query_emb, top_k=top_k * 2)

        # RRF 两路融合
        scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(bm25_results, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (10 + rank)
        for rank, r in enumerate(vec_results, 1):
            mid = r["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (10 + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for mid, _ in ranked[:top_k]:
            if mid in self._skills:
                result.append(self._skills[mid])
        return result

    def record_use(self, exp_id: str, success: bool = True) -> None:
        """记录 Exp 的使用效果。"""
        if exp_id in self._utility:
            self._utility[exp_id]["use"] += 1
            if success:
                self._utility[exp_id]["success"] += 1

    async def prune(self, min_uses: int = 5, min_success_rate: float = 0.3) -> list[str]:
        """淘汰长期无效的 Exp，同时清理 ChromaDB 和 BM25 索引。

        Args:
            min_uses: 最少使用次数，低于此数不淘汰。
            min_success_rate: 最低成功率，低于此数淘汰。

        Returns:
            被淘汰的 Exp ID 列表。
        """
        pruned = []
        for exp_id, util in list(self._utility.items()):
            if util["use"] >= min_uses:
                rate = util["success"] / util["use"]
                if rate < min_success_rate:
                    pruned.append(exp_id)

        for exp_id in pruned:
            self._skills.pop(exp_id, None)
            self._utility.pop(exp_id, None)
            self._bm25.remove(exp_id)
            try:
                await self._vs.delete(SKILL_COLLECTION, exp_id)
            except Exception as e:
                logger.warning("skill_prune_vs_delete_failed", id=exp_id, error=str(e))

        if pruned:
            logger.info("skills_pruned", count=len(pruned))
        return pruned

    def get(self, exp_id: str) -> Memory | None:
        return self._skills.get(exp_id)

    def get_all(self) -> list[Memory]:
        return list(self._skills.values())

    @property
    def count(self) -> int:
        return len(self._skills)
