"""
元认知 (MetaCognition)
-----------------------
自我监控和推理质量评估模块，模拟"思考关于思考"的元认知能力。

核心功能：
1. self_verify: 多维度评估推理质量（PEARL 框架）
   - factual_accuracy: 事实准确性
   - logical_coherence: 逻辑连贯性
   - completeness: 完整性
   - clarity: 清晰度
   - conciseness: 简洁性

2. edge_cut: 边缘裁剪 - 基于置信度、冗余度和矛盾度清除低质量节点
   - confidence_cut: 置信度低于阈值的节点
   - redundancy_cut: 内容高度重复的节点（Jaccard 相似度）
   - contradiction_cut: 互相矛盾的节点（词嵌入 + LLM 双重检测）

3. iterative_think: 迭代式思考 - 对不满意的推理结果进行重试
4. coherence_check: 批量节点一致性检查
5. estimate_confidence: 基于文本特征的启发式置信度估算
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from typing import Optional

from nan_agent.inference.graph import GoTNode
from nan_agent.logging.logger import get_logger
from nan_agent.model.types import MultiModalInput

logger = get_logger(__name__)

CONTRADICTION_EMBED_THRESHOLD = 0.7
CONTRADICTION_LLM_PROMPT = (
    "Statement A: {content_a}\n"
    "Statement B: {content_b}\n\n"
    "Do these two statements contradict each other?\n"
    "Reply ONLY 'YES' or 'NO'."
)


@dataclass
class VerificationResult:
    """单次推理验证的结果

    Attributes:
        quality_score: 综合质量评分 (0.0-1.0)
        passed: 是否通过验证
        issues: 发现的问题列表
        suggestions: 改进建议列表
        confidence_adjustment: 置信度调整量（正数为提升，负数为降低）
    """
    quality_score: float
    passed: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0


@dataclass
class EdgeCutResult:
    """边缘裁剪结果

    Attributes:
        cut_node_ids: 被裁剪的节点 ID 列表
        reasons: 每个节点的裁剪原因 {node_id: reason}
        stats: 裁剪统计（各类别数量、存活率等）
    """
    cut_node_ids: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


class MetaCognition:
    """元认知模块

    提供推理质量的自我监控和评估能力。核心方法：
    - self_verify: 多维度 PEARL 框架评估推理质量
    - edge_cut: 基于置信度/冗余度/矛盾度的边缘裁剪
    - iterative_think: 迭代重试直到质量达标
    - coherence_check: 检查多个节点之间的一致性
    - estimate_confidence: 基于文本特征快速估算置信度

    矛盾检测采用两级策略：
    1. 词嵌入余弦相似度过滤（快速）
    2. LLM 微调确认（精确）
    """

    def __init__(
        self,
        cognition,
        embedding_provider: Optional[Callable[[str], Awaitable[list[float]]]] = None,
    ):
        self._cognition = cognition
        self._embed = embedding_provider
        self._contradiction_cache: dict[tuple[int, int], bool] = {}
        self._embed_cache: dict[str, list[float]] = {}

    async def self_verify(self, node: GoTNode, context_nodes: list[GoTNode] | None = None) -> VerificationResult:
        try:
            context_text = ""
            if context_nodes:
                context_lines = []
                for i, ctx in enumerate(context_nodes):
                    context_lines.append(f"[Context {i + 1}] {ctx.content}")
                context_text = "\n".join(context_lines) + "\n\n"

            prompt = f"""Evaluate the quality of this reasoning step across multiple dimensions.
Output ONLY valid JSON with these keys:

- factual_accuracy: 0.0-1.0 — are the stated facts correct and verifiable?
- logical_coherence: 0.0-1.0 — does the reasoning flow logically without gaps?
- completeness: 0.0-1.0 — does the answer fully address the implied question?
- clarity: 0.0-1.0 — is the expression clear and unambiguous?
- conciseness: 0.0-1.0 — is it free of unnecessary repetition or filler?
- groundedness: 0.0-1.0 — is the reasoning anchored to concrete mechanisms/formulas/code/empirical results rather than pure metaphor?

A high groundedness score means the content references specific methods, algorithms, equations, APIs, or empirical observations. A low score means it relies on vague analogies or poetic abstractions without operational substance.

{context_text}[Content to evaluate]
{node.content}

[Confidence: {node.confidence}]

Output ONLY the JSON object, nothing else:"""

            inp = MultiModalInput()
            inp.add_text(prompt)
            result = await self._cognition.infer(inp, temperature=0.0)
            text = self._extract_json(result.text)
            data = json.loads(text)

            # Check if PEARL+G multi-dimension keys are present
            if any(k in data for k in ("factual_accuracy", "logical_coherence", "completeness", "clarity", "conciseness", "groundedness")):
                factual = float(data.get("factual_accuracy", 0.5))
                logic = float(data.get("logical_coherence", 0.5))
                completeness = float(data.get("completeness", 0.5))
                clarity = float(data.get("clarity", 0.5))
                conciseness = float(data.get("conciseness", 0.5))
                groundedness = float(data.get("groundedness", 0.5))

                # Weighted aggregation: logic and factual carry highest weight
                quality_score = (
                    factual * 0.20 +
                    logic * 0.25 +
                    completeness * 0.15 +
                    clarity * 0.10 +
                    conciseness * 0.10 +
                    groundedness * 0.20
                )
                quality_score = max(0.0, min(1.0, quality_score))

                # If any core dimension is critically low (< 0.3), mark as fail
                passed = all(s >= 0.3 for s in [factual, logic, groundedness])

                issues = []
                if factual < 0.4:
                    issues.append("Low factual accuracy")
                if logic < 0.4:
                    issues.append("Weak logical coherence")
                if completeness < 0.4:
                    issues.append("Incomplete reasoning")
                if groundedness < 0.4:
                    issues.append("Over-reliance on metaphor, lacks concrete operational substance")

                suggestions = []
                if factual < 0.5:
                    suggestions.append("Verify facts against known knowledge")
                if logic < 0.5:
                    suggestions.append("Add intermediary reasoning steps")
                if clarity < 0.5:
                    suggestions.append("Use simpler language")
            else:
                # Backward compatibility: old format without dimension keys
                factual = 0.5
                logic = 0.5
                completeness = 0.5
                clarity = 0.5
                conciseness = 0.5
                quality_score = float(data.get("quality_score", 0.5))
                quality_score = max(0.0, min(1.0, quality_score))
                passed = bool(data.get("pass", True))
                issues = data.get("issues", [])
                suggestions = data.get("suggestions", [])

            if not isinstance(issues, list):
                issues = []
            if not isinstance(suggestions, list):
                suggestions = []

            confidence_adjustment = (quality_score - 0.5) * 0.2

            if not passed and quality_score < 0.5:
                confidence_adjustment = max(-0.5, (quality_score - 0.5) * 0.4)
            elif passed and quality_score >= 0.7:
                confidence_adjustment = min(0.5, (quality_score - 0.5) * 0.3)

            node.confidence = max(0.0, min(1.0, node.confidence + confidence_adjustment))

            return VerificationResult(
                quality_score=quality_score,
                passed=passed,
                issues=issues,
                suggestions=suggestions,
                confidence_adjustment=confidence_adjustment,
            )
        except Exception as e:
            logger.warning("self_verify_failed", error=str(e), node_id=node.node_id)
            return VerificationResult(
                quality_score=0.5,
                passed=True,
                issues=[],
                suggestions=[],
                confidence_adjustment=0.0,
            )

    async def iterative_think(
        self,
        node: GoTNode,
        reasoning_fn,
        max_retries: int = 3,
        quality_threshold: float = 0.7,
    ):
        best_result = None
        best_quality = 0.0
        feedback = ""

        for attempt in range(max_retries + 1):
            if feedback and attempt > 0:
                enhanced_node = GoTNode(
                    type=node.type,
                    content=f"[Previous attempt feedback: {feedback}]\n\n{node.content}",
                    confidence=node.confidence,
                    origin=node.origin,
                )
                result = await reasoning_fn(enhanced_node)
            else:
                result = await reasoning_fn(node)

            verification = await self.self_verify(node, context_nodes=result if isinstance(result, list) else None)

            if verification.quality_score > best_quality:
                best_quality = verification.quality_score
                best_result = result

            if verification.passed and verification.quality_score >= quality_threshold:
                return result

            if attempt < max_retries:
                feedback = self._build_feedback(verification.issues, verification.suggestions)

            logger.info(
                "iterative_think_retry",
                attempt=attempt,
                quality=verification.quality_score,
                node_id=node.node_id,
            )

        return best_result if best_result is not None else result

    def _rule_confidence(self, nodes: list[GoTNode], threshold: float) -> list[str]:
        return [n.node_id for n in nodes if n.confidence < threshold]

    def _rule_redundancy(self, nodes: list[GoTNode]) -> list[str]:
        cut_ids: list[str] = []
        for i, a in enumerate(nodes):
            if a.node_id in cut_ids:
                continue
            for j in range(i + 1, len(nodes)):
                if nodes[j].node_id in cut_ids:
                    continue
                if self._jaccard_similarity(a.content, nodes[j].content) >= 0.75:
                    lo = a if a.confidence <= nodes[j].confidence else nodes[j]
                    cut_ids.append(lo.node_id)
        return list(set(cut_ids))

    async def _rule_contradiction(self, nodes: list[GoTNode]) -> list[str]:
        cut_ids: list[str] = []
        for i, a in enumerate(nodes):
            if a.node_id in cut_ids:
                continue
            for j in range(i + 1, len(nodes)):
                if nodes[j].node_id in cut_ids:
                    continue
                contradiction = await self._detect_contradiction_async(
                    a.content, nodes[j].content, idx_a=i, idx_b=j,
                )
                if contradiction:
                    lo = a if a.confidence <= nodes[j].confidence else nodes[j]
                    cut_ids.append(lo.node_id)
        return cut_ids

    async def edge_cut(self, nodes: list[GoTNode], confidence_threshold: float = 0.3) -> EdgeCutResult:
        self._contradiction_cache.clear()
        self._embed_cache.clear()

        reasons: dict[str, str] = {}
        stats: dict = {"confidence_cut": 0, "redundancy_cut": 0, "contradiction_cut": 0}

        conf_cut_ids = self._rule_confidence(nodes, confidence_threshold)
        stats["confidence_cut"] = len(conf_cut_ids)
        for nid in conf_cut_ids:
            node = next((n for n in nodes if n.node_id == nid), None)
            reasons[nid] = f"confidence {node.confidence:.2f} below threshold {confidence_threshold}" if node else "unknown"

        redundancy_cut_ids = self._rule_redundancy(nodes)
        stats["redundancy_cut"] = len(redundancy_cut_ids)
        for nid in redundancy_cut_ids:
            reasons[nid] = "redundant content"

        remaining = [n for n in nodes if n.node_id not in set(conf_cut_ids + redundancy_cut_ids)]

        if self._embed is not None and remaining:
            try:
                embed_tasks = [self._get_embed_cached(n.content) for n in remaining]
                await asyncio.gather(*embed_tasks)
            except Exception:
                pass

        contradiction_cut_ids = await self._rule_contradiction(remaining)
        stats["contradiction_cut"] = len(contradiction_cut_ids)
        for nid in contradiction_cut_ids:
            reasons[nid] = "contradiction detected"

        cut_ids = conf_cut_ids + redundancy_cut_ids + contradiction_cut_ids
        cut_ids = list(dict.fromkeys(cut_ids))

        stats["total_cut"] = len(cut_ids)
        stats["total_nodes"] = len(nodes)
        stats["survival_rate"] = (stats["total_nodes"] - stats["total_cut"]) / max(1, stats["total_nodes"])

        for node_id in cut_ids:
            node = next((n for n in nodes if n.node_id == node_id), None)
            if node:
                node.mark_pruned(reasons.get(node_id, "edge_cut"))

        return EdgeCutResult(cut_node_ids=cut_ids, reasons=reasons, stats=stats)

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                content_lines = lines[1:]
                if content_lines and content_lines[-1].startswith("```"):
                    content_lines = content_lines[:-1]
                text = "\n".join(content_lines)
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text = text[brace_start:brace_end + 1]
        return text

    @staticmethod
    def _content_signature(content: str) -> tuple[str, set[str]]:
        text = content.strip().lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        words = text.split()
        word_set = set(words)
        if not word_set:
            return hashlib.md5(content.encode()).hexdigest()[:8], set()
        from collections import Counter
        word_counts = Counter(words)
        top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:5]
        sig = "|".join(w for w, _ in top_words)
        return sig, word_set

    @staticmethod
    def _detect_contradiction(content_a: str, content_b: str) -> bool:
        """DEPRECATED: Sync heuristic fallback, language-agnostic negation check."""
        a_set = set(content_a.lower().split())
        b_set = set(content_b.lower().split())
        overlap = len(a_set & b_set)
        if overlap == 0 or overlap < min(len(a_set), len(b_set)) * 0.3:
            return False

        negation_words = {
            "not", "no", "never", "neither", "nor", "none", "nothing",
            "不", "没", "无", "非", "否", "别", "未", "勿",
        }
        has_neg_a = any(w in a_set for w in negation_words)
        has_neg_b = any(w in b_set for w in negation_words)
        return has_neg_a != has_neg_b

    async def _detect_contradiction_async(
        self, content_a: str, content_b: str, idx_a: int = 0, idx_b: int = 0
    ) -> bool:
        """Language-agnostic contradiction detection via embedding filter + micro LLM."""
        pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
        if pair_key in self._contradiction_cache:
            return self._contradiction_cache[pair_key]

        if self._embed is not None:
            try:
                emb_a, emb_b = await asyncio.gather(
                    self._get_embed_cached(content_a),
                    self._get_embed_cached(content_b),
                )
                sim = self._cosine_sim(emb_a, emb_b)
            except Exception:
                sim = 0.0

            if sim < CONTRADICTION_EMBED_THRESHOLD:
                self._contradiction_cache[pair_key] = False
                return False

            if sim > 0.95:
                words_a = set(content_a.lower().split())
                words_b = set(content_b.lower().split())
                if len(words_a ^ words_b) < 3:
                    self._contradiction_cache[pair_key] = False
                    return False

        try:
            prompt = CONTRADICTION_LLM_PROMPT.format(
                content_a=content_a, content_b=content_b,
            )
            inp = MultiModalInput()
            inp.add_text(prompt)
            output = await self._cognition.infer_small(inp, temperature=0.0)
            answer = output.text.strip().upper().replace(".", "")
            result = answer == "YES"
        except Exception:
            logger.debug("contradiction_llm_failed", fallback="heuristic")
            result = self._detect_contradiction(content_a, content_b)

        self._contradiction_cache[pair_key] = result
        return result

    async def _get_embed_cached(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self._embed_cache:
            self._embed_cache[key] = await self._embed(text)
        return self._embed_cache[key]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _jaccard_similarity(text_a, text_b) -> float:
        if isinstance(text_a, set) and isinstance(text_b, set):
            set_a, set_b = text_a, text_b
        elif isinstance(text_a, str) and isinstance(text_b, str):
            if not text_a.strip() or not text_b.strip():
                return 0.0
            set_a = set(text_a.lower().split())
            set_b = set(text_b.lower().split())
        else:
            return 0.0

        if not set_a or not set_b:
            return 0.0

        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    @staticmethod
    def _build_feedback(issues: list[str], suggestions: list[str]) -> str:
        parts = []
        if issues:
            parts.append("Issues to address:")
            for issue in issues:
                parts.append(f"  - {issue}")
        if suggestions:
            parts.append("Suggestions for improvement:")
            for suggestion in suggestions:
                parts.append(f"  - {suggestion}")
        if not parts:
            parts.append("Please reconsider and improve the reasoning.")
        return "\n".join(parts)