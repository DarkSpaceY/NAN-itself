"""
推理循环 (Reasoning Loop)
--------------------------
对单个推理节点执行完整的 Graph of Thought 推理流水线。

流程：
1. 分析 (analyze): 从记忆和人格中获取上下文
2. 思考 (think): 通过 LLM 执行推理，支持工具调用和画图思考
   模型自主选择扩散模式（hot / cold），决定推理的深度和广度
3. 元认知 (metacognate): 自检验证推理质量
4. 决策分支: 根据 LLM 返回的 action 决定下一步
   - branch: 产生多个子推理分支
   - prune: 裁剪当前路径
   - merge: 合并多条推理路径
   - action_output: 输出具体动作

两阶段输出格式（基于 "Let Me Speak Freely?" 2024 论文）：
- Phase 1: LLM 自由思考，用自然语言探索问题
- Phase 2: LLM 在末尾提供标签化摘要 [KEY: value]
- 解析时优先提取标签，回退到 JSON 解析

扩散模式（Diffusion Mode）：
- cold: 精确收敛，低温度 (0.3)，有限分支 (3)，适合逻辑推理和验证
- hot:  发散探索，高温度 (0.9)，多分支 (5)，适合创意和联想
模型在每一次推理中自主选择，默认使用 cold 模式。
"""

import base64
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from nan_agent.inference.graph import GoTEdge, GoTGraph, GoTNode, EdgeType, NodeType, NodeOrigin
from nan_agent.inference.tools import GoTToolkit, ToolCall
from nan_agent.logging.logger import get_logger
from nan_agent.model.types import MultiModalInput, MultiModalOutput

logger = get_logger(__name__)

# ── 扩散模式配置 ──
# 模型自主选择 hot/cold，对应两套固定参数
DIFFUSION_CONFIGS = {
    "cold": {
        "temperature": 0.3,
        "max_branches": 3,
        "description": "精确收敛，适合逻辑推理和验证",
    },
    "hot": {
        "temperature": 0.9,
        "max_branches": 5,
        "description": "发散探索，适合创意和联想",
    },
}
_DEFAULT_DIFFUSION = "cold"
_MAX_TOOL_ROUNDS = 3
_MAX_ITERATIONS = 5
_LOW_QUALITY_THRESHOLD = 0.3


@dataclass
class LoopResult:
    """单次推理循环的结果

    Attributes:
        new_nodes: 本轮推理产生的新节点列表
        action_outputs: 产生的动作输出列表
        pruned: 该节点是否被裁剪
        quality_score: 推理质量评分 (0.0-1.0)
        diffusion_mode: 扩散模式 ('cold' / 'hot')
        iterations: 实际迭代次数
        node_id: 处理的节点 ID
        status: 推理状态 ('ok' / 'pruned' / 'failed' / 'branched' / 'merged' / 'action_output')
        error: 错误信息（如有）
    """
    new_nodes: list = field(default_factory=list)
    action_outputs: list[dict] = field(default_factory=list)
    pruned: bool = False
    quality_score: float = 0.5
    diffusion_mode: str = "cold"
    iterations: int = 1
    node_id: str = ""
    status: str = "ok"
    error: Optional[str] = None


class ReasoningLoop:
    """GoT 推理循环

    对单个节点执行完整的推理流水线：分析 → 思考 → 元认知 → 决策分支。
    支持多种推理路径：分支扩展、路径裁剪、多路合并和动作输出。

    两阶段输出：模型先自由思考，再提供标签化摘要，兼顾推理深度和结构化输出。
    扩散模式由模型在推理过程中自主选择（hot/cold），无需外部预判。

    依赖：
    - cognition: LLM 推理接口
    - graph: GoT 推理图
    - hard_memory: 记忆检索（提供上下文）
    - tools: 工具集（提供代码执行、搜索等能力）
    - meta: 元认知（质量验证和边缘裁剪）
    - draw_of_thought: 画图思考（可视化推理）
    """

    def __init__(
        self,
        cognition,
        graph,
        hard_memory=None,
        tools=None,
        meta=None,
        action_queue=None,
        draw_of_thought=None,
        self_value=None,
        soft_memory=None,
        embed_fn=None,
    ):
        self.cognition = cognition
        self.graph = graph
        self.hard_memory = hard_memory
        self.tools = tools
        self.meta = meta
        self.action_queue = action_queue
        self.draw_of_thought = draw_of_thought
        self.self_value = self_value
        self.soft_memory = soft_memory
        self._embed_fn = embed_fn

    def _get_emotional_state(self) -> Optional[dict]:
        if self.self_value is not None:
            try:
                return self.self_value.get_emotional_state()
            except Exception as e:
                logger.warning("reasoning_emotional_state_failed", error=str(e))
        return None

    async def run(self, node: GoTNode) -> LoopResult:
        if node.node_id not in self.graph._nodes:
            self.graph.add_node(node)

        # Memory 通过工具显式获取，不做被动 recollect
        # await self._analyze(node)

        data, iterations = await self._think(node)
        # 防御：_think 返回值类型检查
        if not isinstance(data, dict):
            logger.warning(
                "run_bad_data_after_think",
                node_id=node.node_id[:8],
                actual_type=type(data).__name__,
                raw_value=str(data)[:200],
            )
            data = {}

        quality_score = data.get("confidence", 0.6)
        if self.meta is not None:
            async def _reason(n: GoTNode):
                d, _ = await self._think(n)
                return d
            data = await self.meta.iterative_think(
                node=node,
                reasoning_fn=_reason,
                max_retries=_MAX_ITERATIONS,
                quality_threshold=_LOW_QUALITY_THRESHOLD,
            )
            # 防御：iterative_think 返回值类型检查
            if not isinstance(data, dict):
                logger.warning(
                    "run_bad_data_after_iterative",
                    node_id=node.node_id[:8],
                    actual_type=type(data).__name__,
                    raw_value=str(data)[:200],
                )
                data = {}
            quality_score = await self._metacognate(node, data)

        # 从模型输出中读取扩散模式，未指定则默认 cold
        diffusion_mode = data.get("diffusion_mode", _DEFAULT_DIFFUSION)
        if diffusion_mode not in DIFFUSION_CONFIGS:
            diffusion_mode = _DEFAULT_DIFFUSION
        diff_config = DIFFUSION_CONFIGS[diffusion_mode]

        action = data.get("action", "branch")
        node_type = self._resolve_node_type(action, data.get("node_type", ""))
        edge_type = self._resolve_edge_type(action, data.get("edge_type", ""))
        edge_target_id = self._resolve_edge_target(data.get("edge_target", "")) if data.get("edge_target") else None

        if action == "prune":
            self.handle_prune(node.node_id, reason=data.get("content", "model decided to prune"))
            return LoopResult(
                node_id=node.node_id,
                new_nodes=[],
                action_outputs=[],
                pruned=True,
                quality_score=quality_score,
                diffusion_mode=diffusion_mode,
                iterations=iterations,
                status="pruned",
            )

        # ── 单节点动作：question, contradiction, analogy, insight ──
        if action in ("question", "contradiction", "analogy", "insight"):
            single_node = GoTNode(
                type=node_type,
                content=data.get("content", ""),
                confidence=quality_score,
                origin=node.origin,
            )
            parent_activation = node.metadata.get("_activation_before_consume", node.activation)
            single_node.inject_activation(parent_activation * 0.5)
            self.graph.add_node(single_node)
            self.graph.add_edge(GoTEdge(
                source_id=node.node_id,
                target_id=single_node.node_id,
                type=edge_type,
            ))
            # 跨节点边
            if edge_target_id and edge_target_id != node.node_id:
                self.graph.add_edge(GoTEdge(
                    source_id=single_node.node_id,
                    target_id=edge_target_id,
                    type=edge_type,
                ))
                logger.info("cross_node_edge_created",
                    source=single_node.node_id[:8],
                    target=edge_target_id[:8],
                    edge_type=edge_type.value,
                )

            await self._store_to_memory(single_node.content, single_node)
            await self._compute_surprises([single_node])
            return LoopResult(
                node_id=node.node_id,
                new_nodes=[single_node],
                action_outputs=[],
                pruned=False,
                quality_score=quality_score,
                diffusion_mode=diffusion_mode,
                iterations=iterations,
                status=action,
            )

        if action == "action_output":
            self.handle_action_output(data, self.action_queue)
            is_tool_return = bool(node.metadata.get("tool_call_history"))
            action_content = data.get("content", "")
            if not action_content:
                action_content = data.get("action_params", {}).get("task_desc", "")
            if not action_content:
                action_content = data.get("action_params", {}).get("description", "")
            if not action_content:
                action_content = node.content[:300] if node.content else f"[action_output] {data.get('action', 'unknown')}"
            action_node = GoTNode(
                type=NodeType.ACTION_OUTPUT,
                content=action_content,
                confidence=quality_score,
                origin=NodeOrigin.TOOL_RETURN if is_tool_return else node.origin,
                action_params=data.get("action_params", {}),
            )
            parent_activation = node.metadata.get("_activation_before_consume", node.activation)
            action_node.inject_activation(parent_activation * 0.3)
            self.graph.add_node(action_node)
            self.graph.add_edge(GoTEdge(
                source_id=node.node_id,
                target_id=action_node.node_id,
                type=EdgeType.BRANCH,
            ))
            # 跨节点边
            if edge_target_id and edge_target_id != node.node_id:
                self.graph.add_edge(GoTEdge(
                    source_id=action_node.node_id,
                    target_id=edge_target_id,
                    type=EdgeType.SUPPORT,
                ))

            if self.hard_memory is not None:
                await self._store_to_memory(action_node.content, action_node)

            await self._compute_surprises([action_node])
            return LoopResult(
                node_id=node.node_id,
                new_nodes=[action_node],
                action_outputs=[data.get("action_params", {})],
                pruned=False,
                quality_score=quality_score,
                diffusion_mode=diffusion_mode,
                iterations=iterations,
                status="action_output",
            )

        if action == "merge":
            merged = self.handle_merge(data.get("content", ""), [node])
            merged.type = node_type  # 使用模型指定的节点类型
            self.graph.add_node(merged)
            self.graph.add_edge(GoTEdge(
                source_id=node.node_id,
                target_id=merged.node_id,
                type=edge_type,
            ))

            if self.hard_memory is not None:
                await self._store_to_memory(merged.content, merged)

            await self._compute_surprises([merged])
            return LoopResult(
                node_id=node.node_id,
                new_nodes=[merged],
                action_outputs=[],
                pruned=False,
                quality_score=quality_score,
                diffusion_mode=diffusion_mode,
                iterations=iterations,
                status="merged",
            )

        # ── 默认：branch ──
        new_nodes = self.handle_branch(
            node,
            data.get("branches", []),
            diff_config,
            branch_node_type=node_type,
        )

        if self.meta is not None and len(new_nodes) > 1:
            try:
                cut_result = await self.meta.edge_cut(new_nodes)
                if cut_result.cut_node_ids:
                    new_nodes = [n for n in new_nodes if n.node_id not in cut_result.cut_node_ids]
                    for nid in cut_result.cut_node_ids:
                        self.graph._nodes[nid].mark_pruned("edge_cut") if nid in self.graph._nodes else None

                    # ── 元认知可视化节点 ──
                    reason_summary = "; ".join(
                        f"{nid[:6]}:{cut_result.reasons.get(nid, 'unknown')}"
                        for nid in cut_result.cut_node_ids[:5]
                    )
                    if len(cut_result.cut_node_ids) > 5:
                        reason_summary += f" (+{len(cut_result.cut_node_ids)-5} more)"
                    meta_node = GoTNode(
                        type=NodeType.INSIGHT,
                        content=f"[MetaCognition: Edge Cut] Pruned {cut_result.stats.get('total_cut', len(cut_result.cut_node_ids))} of {cut_result.stats.get('total_nodes', len(new_nodes))} nodes (survival {cut_result.stats.get('survival_rate', 0):.0%}). Reasons: {reason_summary}",
                        confidence=0.85,
                        origin=NodeOrigin.METACOGNITIVE,
                    )
                    meta_node.inject_activation(0.3)
                    self.graph.add_node(meta_node)
                    self.graph.add_edge(GoTEdge(
                        source_id=node.node_id,
                        target_id=meta_node.node_id,
                        type=EdgeType.ELABORATES,
                    ))
                    new_nodes.insert(0, meta_node)
                    logger.info("metacog_edge_cut_node_created",
                        cut_count=len(cut_result.cut_node_ids),
                        survival=cut_result.stats.get('survival_rate', 0),
                    )
            except Exception as e:
                logger.warning("edge_cut_failed", error=str(e))

        for n in new_nodes:
            await self._store_to_memory(n.content, n)
        await self._compute_surprises(new_nodes)
        return LoopResult(
            node_id=node.node_id,
            new_nodes=new_nodes,
            action_outputs=[],
            pruned=False,
            quality_score=quality_score,
            diffusion_mode=diffusion_mode,
            iterations=iterations,
            status="branched",
        )

    async def _compute_surprises(self, new_nodes: list) -> None:
        """使用 embedding 余弦距离计算新节点的惊奇度。
        惊奇度过低的节点自动标记为冗余并裁剪。
        高惊奇度节点获得额外激活能量注入（资源分配偏向高信息价值路径）。

        Args:
            new_nodes: 新创建的节点列表
        """
        if self._embed_fn is None:
            return
        for node in new_nodes:
            if node is None:
                continue
            try:
                node.surprise = await self.graph.compute_node_surprise(
                    node, self._embed_fn,
                )
                # 冗余检测：惊奇度过低说明与已有节点高度相似
                if node.surprise < 0.12:
                    node.mark_pruned(f"redundant (surprise={node.surprise:.2f})")
                    logger.info("node_pruned_redundant",
                        node_id=node.node_id[:8],
                        surprise=round(node.surprise, 3),
                    )
                    continue

                # ── 惊奇度驱动的激活能注入 ──
                # 高惊奇度节点 = 高信息价值 = 应获得更多计算资源
                if node.surprise > 0.5:
                    # 高度令人惊讶：大幅提升激活（2-3x），推动深入探索
                    boost = 1.5 + (node.surprise - 0.5) * 2.0  # 1.5 - 2.5x
                    node.inject_activation(node.activation * boost)
                    logger.debug("activation_boost_surprise_high",
                        node_id=node.node_id[:8],
                        surprise=round(node.surprise, 2),
                        boost=round(boost, 2),
                    )
                elif node.surprise > 0.25:
                    # 中等惊奇：适度提升（1.2x）
                    node.inject_activation(node.activation * 0.2)
                else:
                    # 低惊奇（0.12-0.25）：降低激活（0.6x），回收资源
                    node.activation *= 0.6
            except Exception:
                pass

    async def _store_to_memory(self, content: str, node: GoTNode | None = None) -> None:
        """存储推理内容到硬记忆，支持多模态附件。"""
        if self.hard_memory is None or not content:
            return
        try:
            # 提取 DoT 可视化图片作为多模态附件
            attachments = None
            if node is not None:
                dot_svg = node.metadata.get("dot_svg_base64")
                if dot_svg:
                    attachments = [{
                        "type": "image",
                        "mime_type": node.metadata.get("dot_svg_mime", "image/svg+xml"),
                        "description": "Draw-of-Thought visualization",
                        "data": dot_svg,
                    }]
            await self.hard_memory.add_memcell(
                content, source="reasoning_loop",
                emotional_state=self._get_emotional_state(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                multimodal_attachments=attachments,
            )
        except Exception as e:
            logger.warning("reasoning_hard_memory_store_failed", error=str(e))

    async def _analyze(self, node: GoTNode) -> None:
        memory_parts = []

        if self.hard_memory is not None:
            try:
                memories = await self.hard_memory.recollect(node.content)
                if memories:
                    memory_lines = "\n".join(
                        m.content if hasattr(m, "content") else str(m)
                        for m in memories
                    )
                    memory_parts.append(memory_lines)
            except Exception as e:
                logger.warning("analyze_memory_failed", error=str(e))

        if memory_parts:
            node.metadata["memory_prefix"] = "\n\n".join(memory_parts) + "\n\n"

    async def _think(
        self,
        node: GoTNode,
        retry: bool = False,
    ) -> tuple[dict, int]:
        iterations = 1
        tools_desc = self.tools.format_tools_for_llm() if self.tools else ""

        for _ in range(_MAX_TOOL_ROUNDS):
            prompt = self.build_prompt(node, tools_desc, retry=retry)

            inp = MultiModalInput()
            inp.add_text(prompt)
            # ── 多模态图片注入：当前节点有 DoT 可视化时作为 ImagePart 注入 ──
            dot_svg = node.metadata.get("dot_svg_base64")
            if dot_svg:
                inp.add_image_base64(dot_svg, mime_type=node.metadata.get("dot_svg_mime", "image/svg+xml"))

            output: MultiModalOutput = await self.cognition.infer(
                inp,
                temperature=0.5,  # 默认温度，将被扩散模式覆盖
                max_context_tokens=8192,  # GoT 推理需要足够上下文容纳 graph context + memory
            )

            response_text = output.text if output else ""
            # 使用两阶段解析：优先提取标签，回退到 JSON
            data = self.extract_thinking(response_text)

            # 解析扩散模式，用对应温度重新推理如果模式与默认不同
            raw_mode = data.get("diffusion_mode", _DEFAULT_DIFFUSION)
            diffusion_mode = raw_mode if raw_mode in DIFFUSION_CONFIGS else _DEFAULT_DIFFUSION
            diff_config = DIFFUSION_CONFIGS[diffusion_mode]

            # 如果模型选择了 hot 但当前温度是 0.5，用 hot 的 temperature 重新推理
            if diffusion_mode == "hot" and iterations == 1:
                prompt = self.build_prompt(node, tools_desc, retry=retry)
                inp = MultiModalInput()
                inp.add_text(prompt)
                dot_svg = node.metadata.get("dot_svg_base64")
                if dot_svg:
                    inp.add_image_base64(dot_svg, mime_type=node.metadata.get("dot_svg_mime", "image/svg+xml"))
                output = await self.cognition.infer(
                    inp,
                    temperature=diff_config["temperature"],
                    max_context_tokens=8192,
                )
                response_text = output.text if output else ""
                data = self.extract_thinking(response_text)

            tool_call = data.get("tool_call")
            if (
                tool_call
                and isinstance(tool_call, dict)
                and self.tools is not None
            ):
                tool_name = tool_call.get("name", "")
                tool_params = tool_call.get("parameters", {})
                tool_result = await self.tools.execute_tool(tool_name, tool_params)

                _tc = ToolCall(
                    tool_name=tool_name,
                    parameters=tool_params,
                    result=tool_result,
                )
                if "tool_call_history" not in node.metadata:
                    node.metadata["tool_call_history"] = []
                node.metadata["tool_call_history"].append(_tc)

                tool_result_str = f"\n[Tool Result: {tool_name}]\n{json.dumps(tool_result, default=str)}"
                node.content += tool_result_str
                iterations += 1
                retry = False
                continue

            if (
                self.draw_of_thought is not None
                and data.get("trigger_draw_of_thought")
            ):
                try:
                    svg_result = await self.draw_of_thought.generate(
                        node.content, ""
                    )
                    # 存储 SVG 图片供后续节点多模态感知
                    svg_b64 = base64.b64encode(svg_result.svg_code.encode()).decode()
                    node.metadata["dot_svg_base64"] = svg_b64
                    node.metadata["dot_svg_mime"] = "image/svg+xml"
                    # 仅保留简短文本标记，不存储完整 SVG 文本
                    node.content += f"\n[Draw-of-Thought: visualization generated ({len(svg_result.svg_code)} chars)]"
                except Exception as e:
                    logger.warning("dwt_integration_failed", error=str(e))

            break

        return data, iterations

    async def _metacognate(
        self, node: GoTNode, data: dict
    ) -> float:
        if self.meta is not None:
            try:
                # 跳过非推理内容节点（任务描述不应被质量评估）
                if node.type == NodeType.EXTERNAL_TASK:
                    return data.get("confidence", 0.6)

                result = await self.meta.self_verify(node)

                # ── 元认知可视化节点：严重问题时生成质疑 ──
                if (result.quality_score < 0.4 or len(result.issues) >= 2) and not node.pruned:
                    issue_text = "; ".join(result.issues[:3]) if result.issues else f"quality={result.quality_score:.2f}"
                    meta_node = GoTNode(
                        type=NodeType.QUESTION,
                        content=f"[MetaCognition: Self-Verify Failed] Score={result.quality_score:.2f}. Issues: {issue_text}. Suggestions: {'; '.join(result.suggestions[:2])}",
                        confidence=0.7,
                        origin=NodeOrigin.METACOGNITIVE,
                    )
                    meta_node.inject_activation(0.5)
                    self.graph.add_node(meta_node)
                    self.graph.add_edge(GoTEdge(
                        source_id=node.node_id,
                        target_id=meta_node.node_id,
                        type=EdgeType.QUESTIONS,
                    ))
                    logger.info("metacog_self_verify_question_created",
                        node_id=node.node_id[:8],
                        quality=round(result.quality_score, 2),
                    )

                return result.quality_score
            except Exception as e:
                logger.warning("metacognition_failed", error=str(e))
                return data.get("confidence", 0.6)

        confidence = data.get("confidence", 0.6)
        content = data.get("content", "")
        if len(content) < 10:
            confidence = min(confidence, 0.3)
        if data.get("action") == "prune" and confidence > 0.8:
            confidence = max(confidence - 0.2, 0.3)
        return confidence

    def handle_branch(
        self,
        node: GoTNode,
        branches: list[dict],
        diff_config: dict,
        branch_node_type: NodeType = NodeType.INFERENCE,
    ) -> list[GoTNode]:
        new_nodes: list[GoTNode] = []
        if not branches:
            return new_nodes

        limit = min(len(branches), diff_config["max_branches"])
        for i in range(limit):
            branch = branches[i]
            content = branch.get("content", "").strip()
            if not content:
                logger.warning("branch_content_empty", node_id=node.node_id[:8], branch_index=i)
                continue
            child = GoTNode(
                type=branch_node_type,
                content=content,
                confidence=branch.get("confidence", node.confidence * 0.9),
                origin=node.origin,
            )
            parent_activation = node.metadata.get("_activation_before_consume", node.activation)
            child.inject_activation(parent_activation * 0.5)
            self.graph.add_node(child)
            self.graph.add_edge(GoTEdge(
                source_id=node.node_id,
                target_id=child.node_id,
                type=EdgeType.BRANCH,
            ))
            new_nodes.append(child)

        return new_nodes

    # ═══════════════════════════════════════════════════════════
    # 节点/边类型解析
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_node_type(action: str, node_type_str: str) -> NodeType:
        """从模型输出解析节点认知角色。

        Args:
            action: 模型选择的动作
            node_type_str: NODE_TYPE 标签值，空字符串表示使用默认值

        Returns:
            对应的 NodeType
        """
        # 模型明确指定了节点类型
        if node_type_str:
            try:
                return NodeType(node_type_str)
            except ValueError:
                pass

        # 根据 action 推断默认节点类型
        defaults = {
            "question": NodeType.QUESTION,
            "contradiction": NodeType.CONTRADICTION,
            "analogy": NodeType.ANALOGY,
            "insight": NodeType.INSIGHT,
            "merge": NodeType.CONCLUSION,
            "action_output": NodeType.ACTION_OUTPUT,
        }
        return defaults.get(action, NodeType.INFERENCE)

    @staticmethod
    def _resolve_edge_type(action: str, edge_type_str: str) -> EdgeType:
        """从模型输出解析边关系类型。

        Args:
            action: 模型选择的动作
            edge_type_str: EDGE_TYPE 标签值

        Returns:
            对应的 EdgeType
        """
        if edge_type_str:
            try:
                return EdgeType(edge_type_str)
            except ValueError:
                pass

        # 根据 action 推断默认边类型
        defaults = {
            "question": EdgeType.QUESTIONS,
            "contradiction": EdgeType.CONTRADICT,
            "analogy": EdgeType.ANALOGIZES,
            "insight": EdgeType.ELABORATES,
            "merge": EdgeType.MERGE,
        }
        return defaults.get(action, EdgeType.BRANCH)

    def _resolve_edge_target(self, edge_target_str: str) -> Optional[str]:
        """在现有图中查找与 EDGE_TARGET 内容最匹配的节点。

        匹配策略：
        1. 先尝试精确匹配 ref:id 格式 (ref:XXXXXXXX)
        2. 然后尝试子串匹配
        3. 最后选字符串相似度最高的节点

        Args:
            edge_target_str: 模型输出的目标节点描述

        Returns:
            匹配到的 node_id，未找到返回 None
        """
        if not edge_target_str:
            return None

        # 策略1: ref:id 精确匹配
        ref_match = re.match(r"^ref:([a-f0-9]{1,12})$", edge_target_str.strip(), re.IGNORECASE)
        if ref_match:
            prefix = ref_match.group(1)
            for nid in self.graph._nodes:
                if nid.startswith(prefix):
                    return nid

        # 策略2: 子串匹配（取前 30 字符作为查询）
        query = edge_target_str[:80].strip()
        best_id = None
        best_score = 0.0
        for nid, node in self.graph._nodes.items():
            if node.pruned:
                continue
            content = node.content[:80].strip()
            # 子串匹配得分
            if query.lower() in content.lower() or content.lower() in query.lower():
                score = len(set(query.lower().split()) & set(content.lower().split())) / max(len(query.split()), 1)
                if score > best_score:
                    best_score = score
                    best_id = nid

        if best_id and best_score > 0.3:
            return best_id
        return None

    # ═══════════════════════════════════════════════════════════
    # 修剪/合并/动作输出
    # ═══════════════════════════════════════════════════════════

    def handle_prune(self, node_id: str = "", reason: str = ""):
        # 周期性批量裁剪：清理低置信度节点和死分支
        if not node_id:
            count = 0
            for nid, node in list(self.graph._nodes.items()):
                if node.pruned:
                    continue
                if node.confidence < 0.2 and node.activation < 0.05:
                    self.graph.prune_subtree(nid, "periodic_low_confidence")
                    count += 1
            if count > 0:
                logger.info("periodic_prune_bulk", nodes_pruned=count)
            return count

        count = self.graph.prune_subtree(node_id, reason)
        if count > 0:
            logger.info("pruned_subtree", node_id=node_id, count=count, reason=reason)
        return count

    def handle_merge(
        self,
        content: str,
        source_nodes: list[GoTNode],
    ) -> GoTNode:
        merged_confidence = (
            sum(n.confidence for n in source_nodes) / len(source_nodes)
            if source_nodes
            else 0.5
        )
        merged = GoTNode(
            type=NodeType.CONCLUSION,
            content=content,
            confidence=merged_confidence,
            origin=source_nodes[0].origin if source_nodes else NodeOrigin.INHERITED,
        )
        # 合并节点继承源节点的激活能量之和（使用 consume 前的值）
        total_activation = sum(
            n.metadata.get("_activation_before_consume", n.activation) for n in source_nodes
        )
        merged.inject_activation(total_activation * 0.5)
        self.graph.add_node(merged)
        for src in source_nodes:
            self.graph.add_edge(GoTEdge(
                source_id=src.node_id,
                target_id=merged.node_id,
                type=EdgeType.MERGE,
            ))
        return merged

    def handle_action_output(
        self,
        data: dict,
        action_queue: Optional[list] = None,
    ) -> None:
        params = data.get("action_params", {})
        if action_queue is not None:
            action_queue.append(params)

    def _build_graph_context(self, node: GoTNode) -> str:
        """Build list-based graph context so the node can 'see' its neighbors.
        
        Each node is labeled with a short ref:id for use in EDGE_TARGET."""
        parts: list[str] = []
        _PREVIEW_LEN = 120  # 截断长度，避免 prompt 膨胀
        _ID_LEN = 8  # ID 前缀长度

        parents = self.graph.get_parents(node.node_id)
        if parents:
            parent_lines = ["Parent thoughts:"]
            for p in parents[:3]:
                ref = p.node_id[:_ID_LEN]
                preview = p.content.replace("\n", " ")[:_PREVIEW_LEN]
                parent_lines.append(f"  - ref:{ref} [{p.type.value}] {preview}")
            # 如果父节点很多，加一个省略提示
            if len(parents) > 3:
                parent_lines.append(f"  ... and {len(parents) - 3} more")
            parts.append("\n".join(parent_lines))

        siblings = self.graph.get_siblings(node.node_id)
        if siblings:
            sib_lines = ["Sibling branches (competing/alternative):"]
            shown = 0

            # ── 智能排序：信息密度 × 多样性 ──
            # 类型权重：洞察/矛盾/类比 > 问题 > 推理 > 思考 > 结论
            _TYPE_WEIGHT = {
                "insight": 2.5, "contradiction": 2.5, "analogy": 2.0,
                "question": 1.5, "inference": 1.2,
                "conclusion": 0.8,
            }
            active_siblings = [s for s in siblings if not s.pruned]
            scored = []
            for s in active_siblings:
                tw = _TYPE_WEIGHT.get(s.type.value, 1.0)
                score = s.activation * (1.0 + s.surprise) * tw
                scored.append((score, s))
            scored.sort(key=lambda x: x[0], reverse=True)

            # ── 去重：跳过与已选节点高度相似的 ──
            selected_content: list[set[str]] = []
            for score, s in scored:
                if shown >= 12:  # 略超 8 以补偿去重损失
                    break
                s_words = set(s.content.lower().split())
                # 检查是否与已选节点重复
                duplicate = False
                for prev_words in selected_content:
                    if s_words and prev_words:
                        intersect = len(s_words & prev_words)
                        union = len(s_words | prev_words)
                        if union > 0 and intersect / union > 0.6:
                            duplicate = True
                            break
                if duplicate:
                    continue

                selected_content.append(s_words)
                ref = s.node_id[:_ID_LEN]
                preview = s.content.replace("\n", " ")[:_PREVIEW_LEN]
                rel = ""
                for p in parents:
                    edge = self.graph.get_edge(p.node_id, s.node_id)
                    if edge and edge.type.value != "branch":
                        rel = f" [{edge.type.value}]"
                        break
                sib_lines.append(f"  - ref:{ref} [{s.type.value}]{rel} {preview}")
                shown += 1

            if shown > 0:
                parts.append("\n".join(sib_lines))

        chain = self._get_ancestor_chain(node, max_depth=4)
        if chain:
            parts.append(f"Reasoning path (root→here): {' → '.join(chain)}")

        return "\n\n".join(parts) if parts else ""

    def _get_ancestor_chain(self, node: GoTNode, max_depth: int = 4) -> list[str]:
        """Walk up the parent tree to build a reasoning path."""
        chain: list[str] = []
        current_id = node.node_id
        for _ in range(max_depth):
            parents = self.graph.get_parents(current_id)
            if not parents:
                break
            p = parents[0]
            preview = p.content.replace("\n", " ").strip()[:80]
            chain.insert(0, f"[{p.type.value}] {preview}")
            current_id = p.node_id
        return chain

    def _build_entropy_status(self) -> str:
        """根据结构熵计算图状态描述，用于 prompt 中的 [Graph State] 段

        结构熵区间与状态映射（参考 Buehler 2025 自组织临界态）：
        - < 0.3: 图过于有序，建议发散
        - 0.3-0.7: 接近临界态，保持当前策略
        - > 0.7: 图过于混乱，建议收敛

        Returns:
            格式如 "entropy=0.45, near critical" 的状态字符串
        """
        entropy = self.graph.structural_entropy()
        if entropy < 0.1:
            status = "no exploration yet - you MUST branch, do NOT merge"
        elif entropy < 0.3:
            status = "too ordered - prefer branch to explore"
        elif entropy > 0.7:
            status = "too chaotic - STRONGLY prefer merge/prune to converge; avoid creating new branches"
        else:
            status = "near critical - maintain current strategy"
        return f"entropy={entropy:.2f}, {status}"

    def build_prompt(
        self,
        node: GoTNode,
        tools_desc: str = "",
        retry: bool = False,
    ) -> str:
        lines = []

        # ── Header ──
        lines.append("You are a graph-of-thought (GoT) reasoning agent.")
        if retry:
            lines.append("[Retry] Previous response quality was low. Think deeper.")
        lines.append("")

        # ── Graph state ──
        entropy_status = self._build_entropy_status()
        lines.append(f"[Graph State: {entropy_status}]")

        graph_context = self._build_graph_context(node)
        if graph_context:
            lines.append(graph_context)

        lines.append(f"[Node Content]\n{node.content}")

        if tools_desc:
            lines.append(tools_desc)

        lines.append("")

        # ── Single-hop novelty requirement (compact) ──
        lines.append("Add NEW information beyond the parent — no poetic restatements.")

        # ═══════════════════════════════════════════════════════
        # Phase 1: Action (compact)
        # ═══════════════════════════════════════════════════════
        lines.append("")
        lines.append("## Phase 1: Action")
        lines.append("[ACTION: choose ONE: branch|merge|question|contradiction|analogy|insight|prune|action_output]")
        lines.append("Guide: entropy=0→branch, >0.7→merge/prune. Only merge with sibling insights to converge.")

        # ═══════════════════════════════════════════════════════
        # Phase 2: Output Format (compact table)
        # ═══════════════════════════════════════════════════════
        lines.append("")
        lines.append("## Phase 2: Output Format (read only YOUR action's line below)")
        lines.append("")
        lines.append("  BRANCH      → [NODE_TYPE: inference] [DIFFUSION: cold|hot] [CONTENT: 1-2 sentences] [BRANCHES: direction1 (0.8) | direction2 (0.6)]")
        lines.append("                cold=extend established line, hot=need new angles. Branches must be FULL SENTENCES, not titles.")
        lines.append("")
        lines.append("  MERGE       → [NODE_TYPE: conclusion] [EDGE_TYPE: merge] [DIFFUSION: cold|hot] [CONTENT: 2-4 sentences]")
        lines.append("                cold=convergence, hot=synthesis across domains.")
        lines.append("")
        lines.append("  QUESTION    → [NODE_TYPE: question] [EDGE_TYPE: questions] [DIFFUSION: hot] [CONTENT: 1-2 sentences]")
        lines.append("                [EDGE_TARGET: ref:XXXXXXXX] (optional)")
        lines.append("")
        lines.append("  CONTRADICT  → [NODE_TYPE: contradiction] [EDGE_TYPE: contradict] [DIFFUSION: hot] [CONTENT: conflict + new direction in 2-3 sentences]")
        lines.append("                Ask: what assumption must be dropped? Consider going to a higher abstraction level.")
        lines.append("                [EDGE_TARGET: ref:XXXXXXXX]")
        lines.append("")
        lines.append("  ANALOGY     → [NODE_TYPE: analogy] [EDGE_TYPE: analogizes] [DIFFUSION: hot] [CONTENT: 2-3 sentences citing concrete mechanisms from BOTH domains]")
        lines.append("                [EDGE_TARGET: ref:XXXXXXXX]")
        lines.append("")
        lines.append("  INSIGHT     → [NODE_TYPE: insight] [EDGE_TYPE: elaborates] [DIFFUSION: hot] [CONTENT: 2-3 sentences grounded in a concrete mechanism, NOT metaphor]")
        lines.append("")
        lines.append("  PRUNE       → [CONTENT: reason in 1 sentence] (only if redundant, dead end, or too vague)")
        lines.append("")
        lines.append("  ACTION_OUT  → [NODE_TYPE: action_output] [CONTENT: describe the action] [TOOL_CALL: ...] (optional)")
        lines.append("")
        lines.append("[TRIGGER_DRAW_OF_THOUGHT: true|false]")

        return "\n".join(lines)

    @staticmethod
    def extract_thinking(text: str) -> dict:
        """两阶段输出解析：优先提取 [KEY: value] 标签，回退到 JSON 解析

        基于 "Let Me Speak Freely?" (2024) 论文的两阶段输出格式：
        - Phase 1: 自由思考文本（不参与结构化解析）
        - Phase 2: 标签化摘要 [KEY: value]，从中提取结构化数据

        解析策略：
        1. 尝试从文本中提取所有 [KEY: value] 标签
        2. 如果找到标签，构建与 JSON 格式兼容的 dict
        3. 如果未找到标签，回退到 extract_json（兼容旧格式）

        BRANCHES 标签的特殊解析：
        - 用 | 分隔多个分支
        - 每个分支可附带置信度，格式如 "idea (0.7)"
        - 解析为 [{"content": "idea", "confidence": 0.7}, ...]

        Args:
            text: LLM 的完整输出文本（包含自由思考和标签化摘要）

        Returns:
            与 extract_json 返回格式兼容的 dict
        """
        if not text:
            return {}

        # ── 第一阶段：提取 [KEY: value] 标签 ──
        # 使用前瞻断言确保每个标签的值在遇到下一个 [KEY: 时停止
        # 这避免了 CONTENT 值跨多行时吞入后续标签的问题
        tag_pattern = re.compile(
            r"\[([A-Z_]+):\s*(.*?)(?=\s*\[[A-Z_]+:\s|\s*$)",
            re.DOTALL,
        )
        tags: dict[str, str] = {}

        for match in tag_pattern.finditer(text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            # 去除尾部可能残留的 ]
            value = value.rstrip("]").strip()
            tags[key] = value

        # 如果没有找到任何标签，回退到 JSON 解析
        if not tags:
            return ReasoningLoop.extract_json(text)

        # ── 第二阶段：将标签转换为兼容 dict ──
        result: dict = {}

        # ACTION 标签
        action = tags.get("ACTION", "branch").strip().lower()
        result["action"] = action

        # CONFIDENCE 标签：解析为浮点数
        confidence_str = tags.get("CONFIDENCE", "0.5").strip()
        try:
            result["confidence"] = float(confidence_str)
        except ValueError:
            result["confidence"] = 0.5

        # DIFFUSION 标签：映射到 diffusion_mode
        diffusion = tags.get("DIFFUSION", "cold").strip().lower()
        result["diffusion_mode"] = diffusion if diffusion in ("cold", "hot") else "cold"

        # CONTENT 标签
        result["content"] = tags.get("CONTENT", "").strip()

        # BRANCHES 标签：用 | 分隔，支持 "idea (0.7)" 格式的置信度
        branches_raw = tags.get("BRANCHES", "").strip()
        branches: list[dict] = []
        if branches_raw:
            # 用 | 分隔各分支
            branch_parts = [b.strip() for b in branches_raw.split("|") if b.strip()]
            for part in branch_parts:
                # 尝试提取 "idea (0.7)" 格式中的置信度
                conf_match = re.search(r"\((\d+\.?\d*)\)\s*$", part)
                if conf_match:
                    branch_content = part[:conf_match.start()].strip()
                    try:
                        branch_conf = float(conf_match.group(1))
                    except ValueError:
                        branch_conf = 0.5
                else:
                    branch_content = part
                    branch_conf = 0.5
                branches.append({
                    "content": branch_content,
                    "confidence": branch_conf,
                })
        result["branches"] = branches

        # TOOL_CALL 标签：尝试解析为 JSON 对象
        tool_call_raw = tags.get("TOOL_CALL", "None").strip()
        if tool_call_raw and tool_call_raw.lower() != "none":
            try:
                result["tool_call"] = json.loads(tool_call_raw)
            except json.JSONDecodeError:
                # 如果不是有效 JSON，保留为字符串标记
                result["tool_call"] = {"name": tool_call_raw, "parameters": {}}
        else:
            result["tool_call"] = None

        # TRIGGER_DRAW_OF_THOUGHT 标签（可选）
        trigger_dot = tags.get("TRIGGER_DRAW_OF_THOUGHT", "false").strip().lower()
        result["trigger_draw_of_thought"] = trigger_dot in ("true", "1", "yes")

        # ACTION_PARAMS 标签（可选，用于 action_output）
        action_params_raw = tags.get("ACTION_PARAMS", "").strip()
        if action_params_raw:
            try:
                result["action_params"] = json.loads(action_params_raw)
            except json.JSONDecodeError:
                result["action_params"] = {"raw": action_params_raw}
        else:
            result["action_params"] = {}

        # NODE_TYPE 标签（可选，默认值在 run() 中根据 action 决定）
        node_type = tags.get("NODE_TYPE", "").strip().lower()
        result["node_type"] = node_type

        # EDGE_TYPE 标签（可选，默认值在 run() 中根据 action 决定）
        edge_type = tags.get("EDGE_TYPE", "").strip().lower()
        result["edge_type"] = edge_type

        # EDGE_TARGET 标签（可选，用于跨节点边）
        edge_target = tags.get("EDGE_TARGET", "").strip()
        result["edge_target"] = edge_target

        return result

    @staticmethod
    def extract_json(text: str) -> dict:
        """从文本中提取 JSON（作为标签解析的回退方案）

        保留原有 JSON 提取逻辑，当标签化格式不存在时使用。
        """
        if not text:
            return {}

        code_block_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL
        )
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # 尝试修复截断的 JSON
                fixed = ReasoningLoop._repair_truncated_json(candidate)
                if fixed:
                    return fixed

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                fixed = ReasoningLoop._repair_truncated_json(candidate)
                if fixed:
                    return fixed

        # 尝试匹配不完整的 JSON（截断的）
        brace_start = re.search(r"\{", text)
        if brace_start:
            candidate = text[brace_start.start():]
            fixed = ReasoningLoop._repair_truncated_json(candidate)
            if fixed:
                return fixed

        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            candidate = bracket_match.group(0).strip()
            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return {"action": "branch", "content": text, "confidence": 0.4, "branches": result}
            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def _repair_truncated_json(text: str) -> dict | None:
        """尝试修复被截断的 JSON 字符串。

        常见场景：LLM 输出过长被 max_tokens 截断，导致 JSON 不完整。
        策略：逐步关闭未闭合的字符串和花括号，然后尝试解析。
        """
        if not text:
            return None

        # 先尝试直接解析
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

        # 尝试逐步修复：关闭未闭合的字符串和括号
        repaired = text.rstrip()

        # 关闭未闭合的字符串值（在截断点补 "）
        if repaired.endswith("\\"):
            repaired = repaired[:-1] + '"'
        elif repaired.count('"') % 2 == 1:
            # 奇数个引号，说明最后一个字符串未闭合
            repaired += '"'

        # 逐层关闭未闭合的括号
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        repaired += "]" * max(open_brackets, 0) + "}" * max(open_braces, 0)

        try:
            result = json.loads(repaired)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

        # 最后手段：用正则提取关键字段
        action_m = re.search(r'"action"\s*:\s*"(\w+)"', text)
        confidence_m = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
        diffusion_m = re.search(r'"diffusion_mode"\s*:\s*"(cold|hot)"', text)
        content_m = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)

        if action_m or content_m:
            return {
                "action": action_m.group(1) if action_m else "branch",
                "content": content_m.group(1) if content_m else text,
                "confidence": float(confidence_m.group(1)) if confidence_m else 0.5,
                "diffusion_mode": diffusion_m.group(1) if diffusion_m else "cold",
                "branches": [],
            }

        return None
