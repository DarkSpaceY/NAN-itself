"""异步记忆巩固引擎（Dream Engine）。

三阶段流水线：
1. Episode → Fact（原子事实提取）
2. Fact → Exp（条件化技能合成）
3. Exp lifecycle（效用追踪/淘汰）

设计参考：
- Anthropic Claude Dreaming: Read → Curate → Output 三步法
- LightMem: 测试时软写入 + 睡眠时离线整理
- EverMemOS: MemCell 结构（Episode + AtomicFacts + Foresight）
- AutoGuide: "When...should..." 条件化技能格式
- ReMe: 效用精炼（自动添加/修剪）
"""

import asyncio
from typing import Optional

from nan_agent.hard_memory.common import new_id
from nan_agent.hard_memory.memory import Memory
from nan_agent.hard_memory.skill_store import SkillStore
from nan_agent.hard_memory.store import Store, COLLECTION_NAME
from nan_agent.logging.logger import get_logger
from nan_agent.model.cognition import Cognition
from nan_agent.model.types import MultiModalInput

logger = get_logger(__name__)

DEDUP_SIMILARITY_THRESHOLD = 0.95

FACT_EXTRACT_PROMPT = """Extract key facts from this text. One fact per line. Do not add numbers, bullets, or extra commentary. If the text contains no factual information, output nothing.

<text>
{text}
</text>

Facts:"""

EXP_SYNTHESIS_PROMPT = """Based on these facts, write ONE behavioral rule for interacting with this person/situation.

Facts:
{facts}

The rule must tell the agent what to DO and what to AVOID. Use this exact format:
When [specific situation], DO [specific action] and AVOID [specific mistake].

Rule:"""

EXP_MERGE_PROMPT = """These two behavioral rules overlap. Merge them into ONE rule that covers both situations.

Rule 1: {rule1}
Rule 2: {rule2}

Merged rule (When [situation], DO [action] and AVOID [mistake]):"""

CONSOLIDATION_PROMPT = """You are consolidating memories. The new fact may contradict an old fact.

Old fact: {old_fact}
New fact: {new_fact}

If they contradict, output the corrected fact. If they are consistent, output the new fact only.
Output one line only:"""


class DreamEngine:
    """异步记忆巩固引擎。

    三阶段流水线：
    Phase 1: Episode → Fact（去重 + LLM 事实提取）
    Phase 2: Fact → Exp（聚类 + LLM 技能合成）
    Phase 3: Exp lifecycle（效用追踪 + 淘汰）

    Attributes:
        _store: Episodic Store 实例。
        _skill_store: SkillStore 实例。
        _cognition: LLM 推理接口。
    """

    def __init__(self, store: Store, skill_store: SkillStore, cognition: Optional[Cognition] = None):
        self._store = store
        self._skill_store = skill_store
        self._cognition = cognition

    async def dream(self, max_items: int = 50) -> dict:
        """执行一次完整巩固周期。

        Returns:
            统计信息 dict。
        """
        stats = {"deduped": 0, "facts_extracted": 0, "exp_synthesized": 0, "exp_pruned": 0}

        # Phase 1: Episode → Fact
        pending = self._get_pending(max_items)
        if not pending:
            logger.info("dream_skip", reason="no_pending")
            return stats

        deduped = await self._deduplicate(pending)
        stats["deduped"] = len(pending) - len(deduped)

        all_facts = []
        if self._cognition:
            all_facts = await self._extract_facts(deduped)
            stats["facts_extracted"] = len(all_facts)

        # 写回 fact（通过 Store.write_fact 封装方法）
        for fact_data in all_facts:
            fact_mem = Memory(
                id=new_id(),
                type="fact",
                content=fact_data["content"],
                episode_text="",
                source="dream",
                timestamp=fact_data.get("timestamp", ""),
                parent_id=fact_data.get("source_id", ""),
            )
            await self._store.write_fact(fact_mem)

        # 标记原始 episode 为已巩固
        for mem in pending:
            mem.metadata["consolidated"] = True

        # Phase 2: Fact → Exp
        if all_facts and self._cognition:
            exp_count = await self._synthesize_exp(all_facts)
            stats["exp_synthesized"] = exp_count

        # Phase 3: Exp lifecycle
        pruned = await self._skill_store.prune(min_uses=5, min_success_rate=0.3)
        stats["exp_pruned"] = len(pruned)

        logger.info("dream_done", **stats, pending=len(pending))
        return stats

    # ── Phase 1: Episode → Fact ──────────────────────────────

    def _get_pending(self, max_items: int) -> list[Memory]:
        """获取待巩固的 episode 记忆。"""
        pending = []
        for mem in self._store._memories.values():
            if mem.type == "episode" and not mem.metadata.get("consolidated"):
                pending.append(mem)
                if len(pending) >= max_items:
                    break
        return pending

    async def _deduplicate(self, memories: list[Memory]) -> list[Memory]:
        """语义去重：先用文本包含快速去重，再用 embedding 余弦相似度精去重。"""
        if not memories:
            return []

        # 第一层：文本包含去重（快速，无需 embed）
        text_deduped: list[Memory] = []
        for mem in memories:
            is_dup = False
            for existing in text_deduped:
                if (mem.content in existing.content or existing.content in mem.content):
                    if len(mem.content) <= len(existing.content):
                        is_dup = True
                        break
            if not is_dup:
                text_deduped.append(mem)

        # 第二层：语义相似度去重（使用 DEDUP_SIMILARITY_THRESHOLD）
        if len(text_deduped) <= 1 or not self._store._ollama:
            return text_deduped

        try:
            embeddings = []
            for mem in text_deduped:
                emb = await self._store._ollama.embed(mem.content)
                embeddings.append(emb)

            keep: list[Memory] = []
            skip: set[int] = set()
            for i in range(len(text_deduped)):
                if i in skip:
                    continue
                keep.append(text_deduped[i])
                for j in range(i + 1, len(text_deduped)):
                    if j in skip:
                        continue
                    sim = self._cosine_sim(embeddings[i], embeddings[j])
                    if sim >= DEDUP_SIMILARITY_THRESHOLD:
                        skip.add(j)
            return keep
        except Exception as e:
            logger.warning("semantic_dedup_failed", error=str(e))
            return text_deduped

    async def _extract_facts(self, memories: list[Memory]) -> list[dict]:
        """LLM 从 episode 提取原子事实。"""
        if not self._cognition:
            return []

        all_facts = []
        for mem in memories:
            text = mem.content[:1000]
            prompt = FACT_EXTRACT_PROMPT.format(text=text)
            inp = MultiModalInput()
            inp.add_text(prompt)

            try:
                output = await self._cognition.infer_small(inp, temperature=0.0, skip_enrich=True)
                raw = output.text.strip()
                for line in raw.split("\n"):
                    line = line.strip().lstrip("-•*0123456789. ")
                    if len(line) > 10:
                        all_facts.append({
                            "content": line[:500],
                            "source_id": mem.id,
                            "timestamp": mem.timestamp,
                        })
            except Exception as e:
                logger.warning("fact_extract_failed", mem_id=mem.id, error=str(e))

        return all_facts

    # ── Phase 2: Fact → Exp ──────────────────────────────────

    async def _synthesize_exp(self, facts: list[dict]) -> int:
        """从 fact 聚类合成 Exp，然后去重合并。

        步骤：
        1. 按 timestamp 分组 fact
        2. 每个簇合成一条 Exp（功能性 DO/AVOID 格式）
        3. 语义去重：相似的 Exp 用 LLM 合并
        4. 写入 SkillStore

        Returns:
            合成的 Exp 数量。
        """
        if not facts or not self._cognition:
            return 0

        # 按 timestamp 分组（同一天的 fact 属于同一簇）
        clusters: dict[str, list[dict]] = {}
        for f in facts:
            key = f.get("timestamp", "")[:10]
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(f)

        # 如果簇太多，合并小簇
        if len(clusters) > 10:
            sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
            main_clusters = dict(sorted_clusters[:8])
            for date, cluster_facts in sorted_clusters[8:]:
                biggest = max(main_clusters.keys(), key=lambda k: len(main_clusters[k]))
                main_clusters[biggest].extend(cluster_facts)
            clusters = main_clusters

        # Phase 2a: 逐簇合成 Exp
        raw_exps: list[tuple[str, str, list[str]]] = []  # (content, date, source_ids)
        for date, cluster_facts in clusters.items():
            if len(cluster_facts) < 2:
                continue

            sample = cluster_facts[:8]
            facts_text = "\n".join(f"- {f['content']}" for f in sample)

            prompt = EXP_SYNTHESIS_PROMPT.format(facts=facts_text)
            inp = MultiModalInput()
            inp.add_text(prompt)

            try:
                output = await self._cognition.infer_small(inp, temperature=0.3, skip_enrich=True)
                raw = output.text.strip()
                logger.info("exp_raw_output", date=date, raw=raw[:200])
                if raw and "when" in raw.lower():
                    source_ids = [f["source_id"] for f in sample if f.get("source_id")]
                    raw_exps.append((raw[:500], date, source_ids))
            except Exception as e:
                logger.warning("exp_synthesis_failed", date=date, error=str(e))

        if not raw_exps:
            return 0

        # Phase 2b: 语义去重合并
        merged_exps = await self._merge_similar_exps(raw_exps)

        # 写入 SkillStore
        for content, date, source_ids in merged_exps:
            await self._skill_store.add(
                content=content,
                source_facts=source_ids,
                timestamp=date,
            )

        return len(merged_exps)

    async def _merge_similar_exps(
        self, exps: list[tuple[str, str, list[str]]]
    ) -> list[tuple[str, str, list[str]]]:
        """语义去重：相似的 Exp 用 LLM 合并为一条。

        使用 embedding 余弦相似度检测重叠，LLM 合并。
        """
        if len(exps) <= 1:
            return exps

        embeddings = []
        for content, _, _ in exps:
            emb = await self._store._ollama.embed(content)
            embeddings.append(emb)

        merged_indices: set[int] = set()
        merge_groups: list[list[int]] = []

        for i in range(len(exps)):
            if i in merged_indices:
                continue
            group = [i]
            for j in range(i + 1, len(exps)):
                if j in merged_indices:
                    continue
                sim = self._cosine_sim(embeddings[i], embeddings[j])
                if sim > 0.75:
                    group.append(j)
                    merged_indices.add(j)
            if len(group) > 1:
                merge_groups.append(group)
                merged_indices.add(i)

        result = []
        for i, (content, date, source_ids) in enumerate(exps):
            if i not in merged_indices:
                result.append((content, date, source_ids))

        for group in merge_groups:
            rule1 = exps[group[0]][0]
            rule2 = exps[group[1]][0]
            date = exps[group[0]][1]
            source_ids = []
            for idx in group:
                source_ids.extend(exps[idx][2])

            if self._cognition:
                prompt = EXP_MERGE_PROMPT.format(rule1=rule1, rule2=rule2)
                inp = MultiModalInput()
                inp.add_text(prompt)
                try:
                    output = await self._cognition.infer_small(inp, temperature=0.1, skip_enrich=True)
                    merged = output.text.strip()
                    if merged and "when" in merged.lower():
                        result.append((merged[:500], date, source_ids))
                        continue
                except Exception as e:
                    logger.warning("exp_merge_failed", error=str(e))

            result.append((rule1, date, source_ids))

        return result

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Phase 3: Exp lifecycle ────────────────────────────────
    # 效用追踪和淘汰在 SkillStore.prune() 中实现
    # dream() 每次执行时自动调用 prune()
