"""
子代理调度器 - 接收自然语言任务，自主搜索/加载/执行技能

Main Agent 只需调用 delegate_task(task, context)，Sub-Agent 在独立
LLM 会话中自主完成：搜索技能 → 选择最匹配 → 激活 → 执行 → 返回摘要。

核心组件：
- SubAgentResult: 子代理执行结果
- SubAgentDispatcher: 子代理调度器
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from nan_agent.action_room.code_executor import CodeExecutor, ExecutionResult
from nan_agent.action_room.skill_loader import LoadedSkill, SkillLoader
from nan_agent.action_room.skill_trees import SkillTreeManager
from nan_agent.action_room.registry import Tool, ToolRegistry
from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger
from nan_agent.model.types import MultiModalInput, MultiModalOutput
from nan_agent.model.provider import InferenceRequest

logger = get_logger(__name__)


@dataclass
class SubAgentResult:
    """子代理执行结果。"""

    success: bool
    summary: str
    skill_name: str
    execution_time_ms: float
    tools_used: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "summary": self.summary,
            "skill_name": self.skill_name,
            "execution_time_ms": self.execution_time_ms,
            "tools_used": self.tools_used,
            "error": self.error,
        }


class SubAgentDispatcher:
    """子代理调度器 — 接收自然语言任务，自主搜索/加载/执行技能。

    Main Agent 只需调用 delegate_task(task)，Sub-Agent 在独立 LLM 会话中
    自主完成：搜索技能 → 选择最匹配 → 激活 → 执行 → 返回摘要。

    Sub-Agent 拥有的内部工具：
    - search_skills: 搜索技能树
    - activate_skill: 激活技能并获取指令
    - get_skill_stats: 获取技能统计
    - recommend_skills: 推荐技能
    - execute_script: 执行技能脚本

    Usage::

        dispatcher = SubAgentDispatcher(loader, tree_manager, registry, executor, llm_client)
        result = await dispatcher.delegate("Generate a Flask app scaffold")
    """

    def __init__(
        self,
        skill_loader: SkillLoader,
        skill_tree_manager: SkillTreeManager,
        tool_registry: ToolRegistry,
        code_executor: CodeExecutor,
        llm_client: Any = None,
        default_timeout: float = 120.0,
    ):
        self._loader = skill_loader
        self._tree_manager = skill_tree_manager
        self._registry = tool_registry
        self._executor = code_executor
        self._llm = llm_client
        self._default_timeout = default_timeout

    # ── 公开接口 ──────────────────────────────────────────

    async def delegate(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> SubAgentResult:
        """派遣子代理执行自然语言任务。

        子代理将自主搜索技能树、选择最匹配的技能、激活并执行。

        Args:
            task: 自然语言任务描述
            context: 附加上下文信息

        Returns:
            SubAgentResult 包含执行结果或错误信息

        Raises:
            ActionError: LLM 不可用时
        """
        start_time = time.perf_counter()

        if self._llm is None:
            raise ActionError(
                "LLM client not available for Sub-Agent delegation",
                error_code="E540",
            )

        # 1. 构建 Sub-Agent 内部工具集
        tools = self._build_internal_tools()

        # 2. 构建系统提示
        system_prompt = self._build_delegation_prompt()

        # 3. 构建用户消息
        user_message = task
        if context:
            user_message = f"{task}\n\n## Additional Context\n{context}"

        # 4. 执行子代理
        try:
            result_text = await asyncio.wait_for(
                self._run_sub_agent(system_prompt, user_message, tools),
                timeout=self._default_timeout,
            )
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # 尝试解析结构化结果
            skill_name, summary = self._parse_result(result_text)

            return SubAgentResult(
                success=True,
                summary=summary,
                skill_name=skill_name,
                execution_time_ms=execution_time_ms,
                tools_used=[t.name for t in tools],
            )

        except asyncio.TimeoutError:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                summary="",
                skill_name="",
                execution_time_ms=execution_time_ms,
                error=f"Sub-Agent timed out after {self._default_timeout}s (E543)",
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.exception("sub_agent_delegate_error", error=str(e))
            return SubAgentResult(
                success=False,
                summary="",
                skill_name="",
                execution_time_ms=execution_time_ms,
                error=str(e),
            )

    # ── Sub-Agent 内部工具构建 ─────────────────────────────

    def _build_internal_tools(self) -> List[Tool]:
        """构建 Sub-Agent 可用的内部工具集。

        包含：
        - search_skills: 搜索技能树
        - activate_skill: 激活技能
        - get_skill_stats: 获取技能统计
        - recommend_skills: 推荐技能
        - execute_script: 执行技能脚本
        - ActionRoom 默认工具（Sub-Agent 可复用的基础工具）
        """
        tools: List[Tool] = []

        # ── search_skills ──
        tree_mgr = self._tree_manager

        def _search_skills(query: str) -> str:
            results = tree_mgr.search_all(query)
            if not results:
                return json.dumps({"found": False, "message": f"No skills matching '{query}'"})
            return json.dumps({
                "found": True,
                "skills": [
                    {"name": n.name, "description": n.description, "is_leaf": getattr(n, "is_leaf", False)}
                    for n in results[:10]
                ],
            }, ensure_ascii=False)

        tools.append(Tool(
            name="search_skills",
            description="Search the skill tree for skills matching a query. Returns matching skill names and descriptions.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query describing the desired capability"},
                },
                "required": ["query"],
            },
            handler=_search_skills,
            category="skill_navigation",
        ))

        # ── activate_skill ──
        loader = self._loader

        def _activate_skill(name: str) -> str:
            skill = loader.activate(name)
            if skill is None:
                return json.dumps({"error": f"Skill '{name}' not found"})

            # 验证兼容性
            if not loader.validate_compatibility(name):
                return json.dumps({
                    "error": f"Skill '{name}' compatibility not satisfied: {skill.compatibility}",
                    "compatibility": skill.compatibility,
                })

            result = {
                "name": skill.name,
                "description": skill.description,
                "instructions": skill.instructions,
                "is_leaf": skill.is_leaf,
                "allowed_tools": skill.allowed_tools,
                "scripts": list(skill.scripts.keys()),
                "references": list(skill.references.keys()),
            }
            return json.dumps(result, ensure_ascii=False)

        tools.append(Tool(
            name="activate_skill",
            description="Activate a skill by name to load its full instructions, scripts, and references. Always call this after finding a skill via search_skills.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact skill name from search results"},
                },
                "required": ["name"],
            },
            handler=_activate_skill,
            category="skill_navigation",
        ))

        # ── get_skill_stats ──
        def _get_skill_stats() -> str:
            # 用空查询获取所有节点
            all_nodes = tree_mgr.search_all("")
            leaf_count = sum(1 for n in all_nodes if getattr(n, "is_leaf", False))
            categories = list(set(n.category for n in all_nodes if hasattr(n, "category") and n.category))
            return json.dumps({
                "total_skills": len(all_nodes),
                "leaf_skills": leaf_count,
                "categories": categories,
            }, ensure_ascii=False)

        tools.append(Tool(
            name="get_skill_stats",
            description="Get statistics about available skills in the skill tree.",
            parameters={"type": "object", "properties": {}},
            handler=_get_skill_stats,
            category="skill_navigation",
        ))

        # ── recommend_skills ──
        def _recommend_skills(task_description: str, top_k: int = 5) -> str:
            # 先用 search 搜索，再取 top_k
            results = tree_mgr.search_all(task_description)
            if not results:
                return json.dumps({"recommendations": []})
            return json.dumps({
                "recommendations": [
                    {"name": n.name, "description": n.description}
                    for n in results[:top_k]
                ],
            }, ensure_ascii=False)

        tools.append(Tool(
            name="recommend_skills",
            description="Get skill recommendations based on a task description.",
            parameters={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string", "description": "Description of the task to find skills for"},
                    "top_k": {"type": "integer", "description": "Maximum number of recommendations", "default": 5},
                },
                "required": ["task_description"],
            },
            handler=_recommend_skills,
            category="skill_navigation",
        ))

        # ── execute_script ──
        executor = self._executor

        def _execute_script(skill_name: str, script_name: str, args: dict = None) -> str:
            script_content = loader.load_script(skill_name, script_name)
            if script_content is None:
                return json.dumps({"error": f"Script '{script_name}' not found in skill '{skill_name}'"})

            result: ExecutionResult = executor.execute(
                code=script_content,
                language="python",
                input_vars=args or {},
            )
            if result.exit_code != 0:
                return json.dumps({"error": f"Script execution failed: {result.stderr}", "exit_code": result.exit_code})
            return json.dumps({"output": result.stdout, "exit_code": result.exit_code}, ensure_ascii=False)

        tools.append(Tool(
            name="execute_script",
            description="Execute a skill script by name. Use after activate_skill to identify available scripts.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Name of the skill"},
                    "script_name": {"type": "string", "description": "Name of the script to execute"},
                    "args": {"type": "object", "description": "Arguments to pass to the script", "default": {}},
                },
                "required": ["skill_name", "script_name"],
            },
            handler=_execute_script,
            category="skill_execution",
        ))

        # ── 从 ToolRegistry 获取基础工具 ──
        # Sub-Agent 可以复用 ActionRoom 的基础工具（文件系统、代码执行等）
        base_tool_names = [
            "read_file", "write_file", "list_dir", "search_files",
            "execute_python", "execute_bash",
            "search_web", "fetch_content",
        ]
        for tool_name in base_tool_names:
            tool = self._registry.get_tool(tool_name)
            if tool is not None and tool.enabled:
                tools.append(tool)

        return tools

    # ── 提示构建 ──────────────────────────────────────────

    def _build_delegation_prompt(self) -> str:
        """构建子代理的系统提示。

        指导子代理自主搜索、选择、激活并执行技能。
        """
        return (
            "You are a specialized Sub-Agent with access to a skill tree. "
            "Your job is to complete the given task by:\n\n"
            "1. **Search** — Use `search_skills` or `recommend_skills` to find relevant skills\n"
            "2. **Activate** — Use `activate_skill` to load the best matching skill's instructions\n"
            "3. **Execute** — Follow the skill's instructions. Use `execute_script` for skill scripts, "
            "or use other available tools as needed\n"
            "4. **Report** — Return a structured result\n\n"
            "## Output Format\n\n"
            "After completing the task, output your result in this exact format:\n\n"
            "```\n"
            "SKILL_USED: <skill_name>\n"
            "RESULT: <your summary of what was accomplished>\n"
            "```\n\n"
            "If no suitable skill is found, output:\n\n"
            "```\n"
            "SKILL_USED: none\n"
            "RESULT: <explanation of why no skill matched and what you attempted>\n"
            "```\n\n"
            "Always try to find and use the most relevant skill before falling back to general tools."
        )

    def _parse_result(self, result_text: str) -> tuple[str, str]:
        """解析子代理的结构化输出。

        Args:
            result_text: 子代理返回的文本

        Returns:
            (skill_name, summary) 元组
        """
        skill_name = ""
        summary = result_text

        # 尝试提取 SKILL_USED 和 RESULT
        for line in result_text.split("\n"):
            line = line.strip()
            if line.startswith("SKILL_USED:"):
                skill_name = line[len("SKILL_USED:"):].strip()
            elif line.startswith("RESULT:"):
                summary = line[len("RESULT:"):].strip()

        # 如果没有提取到任何结构化字段，整个文本作为 summary
        if not skill_name and summary == result_text:
            summary = result_text

        return skill_name, summary

    # ── LLM 调用 ──────────────────────────────────────────

    async def _run_sub_agent(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Tool],
    ) -> str:
        """在独立 LLM 会话中执行子代理。

        Args:
            system_prompt: 系统提示
            user_message: 用户消息
            tools: 可用工具列表

        Returns:
            LLM 响应文本
        """
        # 构建工具描述
        tool_descriptions = ""
        if tools:
            tool_lines = []
            for tool in tools:
                params_desc = ""
                props = tool.parameters.get("properties", {})
                if props:
                    param_parts = []
                    for pname, pschema in props.items():
                        ptype = pschema.get("type", "any")
                        pdesc = pschema.get("description", "")
                        param_parts.append(f"{pname}({ptype}): {pdesc}")
                    params_desc = ", ".join(param_parts)
                tool_lines.append(
                    f"- **{tool.name}**({params_desc}): {tool.description}"
                )
            tool_descriptions = "\n\n## Available Tools\n" + "\n".join(tool_lines) + "\n"

        # 构建完整提示
        full_prompt = system_prompt + tool_descriptions + "\n\n## Task\n" + user_message

        # 创建推理请求
        mm_input = MultiModalInput()
        mm_input.add_text(full_prompt)

        request = InferenceRequest(
            input=mm_input,
            temperature=0.3,
        )

        # 调用 LLM
        output: MultiModalOutput = await self._llm.infer(request)
        return output.text.strip() if output.text else "No response from Sub-Agent"
