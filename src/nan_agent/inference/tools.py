"""
GoT 工具集 (Toolkit)
--------------------
为推理循环提供可调用的外部工具能力，支持在推理过程中执行代码、搜索记忆和检索网络信息。

工具列表：
- code_execute: 在沙盒环境中执行 Python 代码，用于计算和验证
- memory_search: 在 HardMemory 中搜索相关的历史经验和知识
- web_search: 通过搜索引擎获取外部信息

工具调用通过 LLM 在推理过程中发起，结果会被注入到节点内容中。
"""

import asyncio
import io
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

_SAFE_EVAL_BUILTINS = {
    "abs": abs, "all": all, "any": any, "ascii": ascii,
    "bin": bin, "bool": bool, "bytes": bytes,
    "chr": chr, "complex": complex,
    "dict": dict, "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
    "hex": hex,
    "int": int, "isinstance": isinstance,
    "len": len, "list": list,
    "map": map, "max": max, "min": min,
    "oct": oct, "ord": ord,
    "pow": pow, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type,
    "zip": zip,
    "True": True, "False": False, "None": None,
    "math": __import__("math"),
    "json": __import__("json"),
    "re": __import__("re"),
    "datetime": __import__("datetime"),
    "collections": __import__("collections"),
    "itertools": __import__("itertools"),
    "functools": __import__("functools"),
    "random": __import__("random"),
    "statistics": __import__("statistics"),
    "hashlib": __import__("hashlib"),
}


@dataclass
class ToolCall:
    """单次工具调用记录

    记录推理过程中 LLM 发起的一次工具调用，包括工具名、参数和执行结果。
    结果会追加到节点的 metadata.tool_call_history 中，供后续推理参考。
    """
    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None


class GoTToolkit:
    """GoT 推理工具集

    为推理循环提供三种核心工具：代码执行、记忆搜索和网络搜索。
    工具列表和参数描述可以格式化为 LLM prompt 的一部分，
    使 LLM 在推理过程中能够主动调用这些工具来辅助思考。
    """

    def __init__(
        self,
        cognition: Any = None,
        hard_memory: Any = None,
        web_search: Any = None,
        code_executor: Any = None,
    ):
        self._cognition = cognition
        self._hard_memory = hard_memory
        self._web_search = web_search
        self._code_executor = code_executor

    def code_execute(self, code: str) -> dict[str, Any]:
        if self._code_executor is not None:
            try:
                result = self._code_executor.execute(code, language="python")
                return {
                    "success": result.exit_code == 0,
                    "data": result.stdout.strip() if result.stdout else None,
                    "error": result.stderr.strip() if result.stderr else None,
                    "exit_code": result.exit_code,
                    "execution_time": result.execution_time_ms,
                }
            except Exception as e:
                logger.warning("code_executor_error", error=str(e))
                return {
                    "success": False,
                    "data": None,
                    "error": str(e),
                    "exit_code": -1,
                    "execution_time": 0.0,
                }

        t0 = time.perf_counter()
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            safe_globals = {"__builtins__": _SAFE_EVAL_BUILTINS}
            safe_locals: dict[str, Any] = {}
            compiled = compile(code, "<got_sandbox>", "exec")
            exec(compiled, safe_globals, safe_locals)
            output = captured.getvalue().strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result_data = output if output else None
            if result_data is None and safe_locals:
                result_data = str(safe_locals.get("_result", safe_locals))
            return {
                "success": True,
                "data": result_data,
                "error": None,
                "exit_code": 0,
                "execution_time": elapsed_ms,
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "exit_code": -1,
                "execution_time": elapsed_ms,
            }
        finally:
            sys.stdout = old_stdout

    async def memory_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if self._hard_memory is None:
            return [{"status": "unavailable", "reason": "HardMemory not configured"}]
        try:
            results = await self._hard_memory.recollect(query, k=k)
            return results
        except Exception as e:
            logger.warning("memory_search_error", error=str(e))
            return [{"status": "error", "reason": str(e)}]

    async def web_search_fn(
        self, query: str, num_results: int = 5
    ) -> list[dict[str, Any]]:
        if self._web_search is None:
            return [{"status": "unavailable", "reason": "WebSearch not configured"}]
        try:
            results = await self._web_search.search(query, max_results=num_results)
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "relevance_score": r.relevance_score,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("web_search_error", error=str(e))
            return [{"status": "error", "reason": str(e)}]

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "code_execute",
                "description": "Execute Python code in a sandboxed environment for computation and verification",
                "parameters": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                        "required": True,
                    }
                },
            },
            {
                "name": "memory_search",
                "description": "Search HardMemory for relevant past experiences and knowledge",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query for memory retrieval",
                        "required": True,
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "required": False,
                    },
                },
            },
            {
                "name": "web_search",
                "description": "Search the web for external information using multiple search engines",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                        "required": True,
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "required": False,
                    },
                },
            },
        ]

    def format_tools_for_llm(self) -> str:
        tools = self.list_tools()
        lines = ["Available tools:"]
        for t in tools:
            params_desc = []
            for name, info in t["parameters"].items():
                req = "(required)" if info.get("required") else "(optional)"
                params_desc.append(
                    f"    - {name} ({info['type']}) {req}: {info['description']}"
                )
            lines.append(f"- {t['name']}: {t['description']}")
            if params_desc:
                lines.extend(params_desc)
        return "\n".join(lines)

    def has_tool(self, name: str) -> bool:
        """Check if a tool with the given name is available."""
        return name in {"code_execute", "memory_search", "web_search"}

    async def execute_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if name == "code_execute":
            return self.code_execute(code=params.get("code", ""))
        elif name == "memory_search":
            return {
                "results": await self.memory_search(
                    query=params.get("query", ""),
                    k=params.get("k", 5),
                )
            }
        elif name == "web_search":
            return {
                "results": await self.web_search_fn(
                    query=params.get("query", ""),
                    num_results=params.get("num_results", 5),
                )
            }
        else:
            return {"success": False, "error": f"Unknown tool: {name}"}