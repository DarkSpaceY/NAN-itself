import asyncio
import json
import math
import time
from collections.abc import AsyncIterator
from typing import Optional

from nan_agent.logging.logger import get_logger
from nan_agent.model.provider import InferenceRequest, ModelProvider
from nan_agent.model.types import (
    AudioPart,
    ImagePart,
    MultiModalInput,
    MultiModalOutput,
    VideoPart,
)

logger = get_logger(__name__)

_SUMMARY_SEMAPHORE = asyncio.Semaphore(5)
TARGET_CHUNK_MIN = 150
TARGET_CHUNK_MAX = 250
TREE_GROUP_SIZE = 5

# ── Token estimation & summary template ────────────────────────────

SUMMARY_TEMPLATE = """Extract information from the {module_name} context that is relevant to the query.

<query>
{query}
</query>

<context>
{raw_context}
</context>

<instructions>
Output a 2-4 sentence summary in natural language. Include only information
relevant to the query. Be concise and factual.
</instructions>

Relevant summary:"""


def _estimate_tokens(text: str) -> int:
    """Conservative character-based token count (no external tokenizer).
    Latin ~4 chars/token, CJK ~2 chars/token. ±20% accuracy."""
    if not text:
        return 0
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    cjk_chars = len(text) - latin_chars
    return max(1, (latin_chars // 4) + (cjk_chars // 2))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_tool_entries(tools_desc: str) -> list[dict]:
    """Parse tools description string into individual tool entries.
    Format: "- **tool_name** (category): description\n  Parameters: {...}"
    """
    entries = []
    current = None
    for line in tools_desc.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- **"):
            if current is not None:
                entries.append(current)
            # Parse: - **tool_name** (category): description
            try:
                rest = stripped[4:]  # Remove "- **"
                name_end = rest.index("**")
                name = rest[:name_end]
                rest = rest[name_end + 2:].strip()
                if rest.startswith("("):
                    cat_end = rest.index(")")
                    category = rest[1:cat_end]
                    rest = rest[cat_end + 1:].strip()
                    if rest.startswith(":"):
                        description = rest[1:].strip()
                    else:
                        description = rest
                else:
                    category = ""
                    description = rest
                current = {"name": name, "category": category, "description": description, "params": ""}
            except (ValueError, IndexError):
                current = {"name": stripped, "category": "", "description": "", "params": ""}
        elif stripped.startswith("Parameters:") and current is not None:
            current["params"] = stripped[len("Parameters:"):].strip()
    if current is not None:
        entries.append(current)
    return entries


class Cognition:
    def __init__(
        self,
        provider: ModelProvider,
        hard_memory=None,
        self_value=None,
        soft_memory=None,
        skill_trees=None,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9,
        default_max_tokens: int = 16384,
    ):
        self.provider = provider
        self.hard_memory = hard_memory
        self.self_value = self_value
        self.soft_memory = soft_memory
        self.skill_trees = skill_trees
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_max_tokens = default_max_tokens

    async def infer(
        self,
        input: MultiModalInput,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enrich_query: str = "",
        fixed_prefix_len: int = 0,
        **kwargs,
    ) -> MultiModalOutput:
        """推理。

        Args:
            input: 输入。
            temperature: 采样温度。
            top_p: 核采样阈值。
            max_tokens: 最大生成长度。
            enrich_query: 上下文丰富查询提示。
            fixed_prefix_len: 不可压缩的前缀长度。
        """
        t0 = time.monotonic()
        enriched = await self._enrich_input(input, max_context_tokens=kwargs.pop("max_context_tokens", 1024), query_hint=enrich_query, fixed_prefix_len=fixed_prefix_len)

        request = InferenceRequest(
            input=enriched,
            temperature=temperature if temperature is not None else self.default_temperature,
            top_p=top_p if top_p is not None else self.default_top_p,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            **kwargs,
        )

        result = await self.provider.infer(request)
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        output_len = len(result.text) if result and result.text else 0
        logger.info("inference_done", model="main", elapsed_ms=elapsed_ms, output_len=output_len)
        return result

    async def infer_small(
        self,
        input: MultiModalInput,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        skip_enrich: bool = False,
        **kwargs,
    ) -> MultiModalOutput:
        """Lightweight inference using the small chat model. No memory storage.

        Args:
            input: 输入。
            temperature: 采样温度。
            top_p: 核采样阈值。
            max_tokens: 最大生成长度。
            skip_enrich: 是否跳过 prompt enrichment / ToA 压缩。
                用于提取等场景（prompt 已自带完整指令，无需压缩）。
        """

        if skip_enrich:
            enriched = input
        else:
            enriched = await self._enrich_input(input)

        request = InferenceRequest(
            input=enriched,
            temperature=temperature if temperature is not None else self.default_temperature,
            top_p=top_p if top_p is not None else self.default_top_p,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            **kwargs,
        )

        t0 = time.monotonic()
        result = await self.provider.infer_small(request)
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        output_len = len(result.text) if result and result.text else 0
        logger.info("inference_done", model="small", elapsed_ms=elapsed_ms, output_len=output_len)
        return result

    async def infer_stream(
        self,
        input: MultiModalInput,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        enriched = await self._enrich_input(input)

        request = InferenceRequest(
            input=enriched,
            temperature=temperature if temperature is not None else self.default_temperature,
            top_p=top_p if top_p is not None else self.default_top_p,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            **kwargs,
        )

        async for chunk in self.provider.infer_stream(request):
            yield chunk

    async def _build_context_sections(
        self,
        query: str,
        modules: list,
        per_module_budget: int,
    ) -> list:
        """Build labeled context sections for a list of modules in parallel.

        Each module is (module_name, label, fetch_fn) where fetch_fn is an
        async callable `(query, budget) -> str`.

        Returns list of non-empty section strings. Failures return "" and are logged.
        """
        if not modules or per_module_budget < 32:
            return []

        async def _build_one(name, label, fetch_fn):
            try:
                result = await fetch_fn(query, per_module_budget)
                return result if result else ""
            except Exception as e:
                logger.warning(
                    "context_section_failed",
                    module=name,
                    error=str(e),
                )
                return ""

        results = await asyncio.gather(
            *[_build_one(name, label, fetch) for name, label, fetch in modules],
            return_exceptions=True,
        )

        sections = []
        for (name, label, _), result in zip(modules, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "context_section_failed",
                    module=name,
                    error=str(result),
                )
            elif result:
                sections.append(result)

        return sections

    async def _enrich_input(
        self, input: MultiModalInput, max_context_tokens: int = 1024, query_hint: str = "",
        fixed_prefix_len: int = 0,
    ) -> MultiModalInput:
        """Enrich prompt with emotional state only.
        Memory/Experience/Skills/Adaptors are injected by enrich_task_context.
        Tree-of-Agents applied on full assembled text if over budget.
        query_hint (user's original question) is used as the ToA relevance anchor.
        fixed_prefix_len: characters at the start of the text that must NEVER be compressed
        (identity, personality, instructions, critical, sensors)."""
        prompt_text = input.get_text()
        prompt_tokens = _estimate_tokens(prompt_text)

        # Split into fixed prefix and compressible suffix
        if fixed_prefix_len > 0 and fixed_prefix_len < len(prompt_text):
            fixed_prefix = prompt_text[:fixed_prefix_len]
            compressible_text = prompt_text[fixed_prefix_len:]
            fixed_tokens = _estimate_tokens(fixed_prefix)
            compressible_tokens = _estimate_tokens(compressible_text)
        else:
            fixed_prefix = ""
            compressible_text = prompt_text
            fixed_tokens = 0
            compressible_tokens = prompt_tokens

        budget = max_context_tokens - compressible_tokens

        # ── Tree-of-Agents: compress oversized prompt BEFORE enrichment ──
        if budget < 128:
            logger.warning(
                "enrich_budget_exhausted",
                prompt_tokens=prompt_tokens,
                prompt_chars=len(prompt_text),
                max_context_tokens=max_context_tokens,
                budget=budget,
                fixed_tokens=fixed_tokens,
                compressible_tokens=compressible_tokens,
                query_hint=query_hint[:100] if query_hint else "",
            )
            toa_query = query_hint or compressible_text
            toa_start = time.monotonic()
            compressed_text = await self._summarize_with_chunking(
                "context", compressible_text, toa_query, max_context_tokens,
            )
            toa_elapsed = time.monotonic() - toa_start
            compressed_tokens = _estimate_tokens(compressed_text)
            final_text = fixed_prefix + compressed_text
            final_tokens = _estimate_tokens(final_text)
            logger.info(
                "toa_compressed_prompt",
                before_tokens=prompt_tokens,
                before_chars=len(prompt_text),
                fixed_tokens=fixed_tokens,
                compressible_before_tokens=compressible_tokens,
                compressible_after_tokens=compressed_tokens,
                after_tokens=final_tokens,
                after_chars=len(final_text),
                compression_ratio=round(compressed_tokens / max(compressible_tokens, 1), 4),
                toa_elapsed_s=round(toa_elapsed, 2),
                compressed_preview=compressed_text[:300],
            )
            result = MultiModalInput()
            result.add_text(final_text)
            non_text_parts = [p for p in input.parts if isinstance(p, (ImagePart, AudioPart, VideoPart))]
            for part in non_text_parts:
                result.parts.append(part)
            return result

        modules = [
            ("emotional_state", "<internal_state>", self._fetch_emotional_state),
        ]
        sections = await self._build_context_sections(
            compressible_text, modules, budget,
        )

        parts = sections + [compressible_text]
        enriched_text = "\n\n".join(parts)

        full_tokens = _estimate_tokens(enriched_text)
        if full_tokens > max_context_tokens:
            toa_query = query_hint or compressible_text
            enriched_text = await self._summarize_with_chunking(
                "context", enriched_text, toa_query, max_context_tokens,
            )

        final_text = fixed_prefix + enriched_text

        logger.debug(
            "enrich_done",
            prompt_tokens=prompt_tokens,
            context_tokens=_estimate_tokens(final_text) - prompt_tokens,
            budget=max_context_tokens,
        )

        non_text_parts = [
            p
            for p in input.parts
            if isinstance(p, (ImagePart, AudioPart, VideoPart))
        ]

        result = MultiModalInput()
        result.add_text(final_text)
        for part in non_text_parts:
            result.parts.append(part)

        return result

    async def retrieve_relevant_tools(
        self, query: str, tools_desc: str, top_k: int = 5,
    ) -> str:
        """Retrieve top-k most relevant tools for the given query using semantic search.
        Uses the provider's embed method to compute embeddings, then cosine similarity.

        Falls back to full tools_desc if embedding fails or tools count <= top_k.
        """
        if not query or not tools_desc or not tools_desc.strip():
            return tools_desc

        # Parse individual tool entries
        tool_entries = _parse_tool_entries(tools_desc)
        if len(tool_entries) <= top_k:
            return tools_desc

        try:
            # Check if provider supports embedding
            if not hasattr(self.provider, "embed") or not callable(self.provider.embed):
                logger.debug("tool_rag_no_embed", provider_type=type(self.provider).__name__)
                return tools_desc

            # Embed query
            query_emb = await self.provider.embed(query)
            if not query_emb or all(v == 0.0 for v in query_emb):
                return tools_desc

            # Embed each tool description (name + category + description as the embedding text)
            tool_texts = [
                f"{entry['name']} ({entry['category']}): {entry['description']}"
                for entry in tool_entries
            ]
            tool_embs = []
            for text in tool_texts:
                emb = await self.provider.embed(text)
                tool_embs.append(emb if emb and any(v != 0.0 for v in emb) else [0.0] * len(query_emb))

            # Compute cosine similarities
            scores = []
            for i, emb in enumerate(tool_embs):
                sim = _cosine_similarity(query_emb, emb)
                scores.append((sim, i))

            # Sort by similarity descending, take top_k
            scores.sort(key=lambda x: x[0], reverse=True)
            selected_indices = sorted([idx for _, idx in scores[:top_k]])

            # Rebuild tools_desc with only selected tools
            selected_entries = [tool_entries[i] for i in selected_indices]
            lines = []
            for entry in selected_entries:
                lines.append(f"- **{entry['name']}** ({entry['category']}): {entry['description']}")
                if entry.get("params"):
                    lines.append(f"  Parameters: {entry['params']}")
            result = "\n".join(lines)

            logger.info(
                "tool_rag_selection",
                query_preview=query[:80],
                total_tools=len(tool_entries),
                selected=len(selected_entries),
                selected_names=[e["name"] for e in selected_entries],
                top_scores=[round(s, 4) for s, _ in scores[:top_k]],
            )
            return result

        except Exception as e:
            logger.warning("tool_rag_failed", error=str(e), query_preview=query[:80])
            return tools_desc

    async def enrich_task_context(
        self,
        text: str,
        task_intent: str = "",
        max_context_tokens: int = 4096,
    ) -> str:
        """Enrich task context with personality, memory, skills, adaptors, and task intent.
        Returns raw assembled context blocks. Tree-of-Agents is NOT applied here —
        it runs later in _enrich_input on the complete prompt (including sensors,
        filesystem view, tools, history, and system identity)."""
        prompt_tokens = _estimate_tokens(text) + _estimate_tokens(task_intent)
        budget_pool = max_context_tokens - prompt_tokens

        if budget_pool < 160:
            logger.warning(
                "task_context_budget_exhausted",
                prompt_tokens=prompt_tokens,
                budget_pool=budget_pool,
            )
            parts = []
            if task_intent:
                parts.append(f"<task_intent>\n{task_intent}\n</task_intent>")
            parts.append(text)
            return "\n\n".join(parts)

        modules = [
            ("personality", "[Personality]", self._fetch_personality_raw),
            ("memory", "[Memory]", self._fetch_memory_raw),
            ("experience", "[Experience]", self._fetch_exp_raw),
            ("skills", "[Skills]", self._fetch_skills_raw),
            ("adaptors", "[Adaptors]", self._fetch_adaptors_raw),
        ]
        sections = await self._build_context_sections(
            text, modules, max(budget_pool // len(modules), 32),
        )

        # Log token contribution per module
        section_tokens = {}
        for (name, _, _), section in zip(modules, sections):
            section_tokens[name] = _estimate_tokens(section) if section else 0

        parts = []
        if task_intent:
            parts.append(f"<task_intent>\n{task_intent}\n</task_intent>")

        parts.extend(sections)
        parts.append(text)

        full_text = "\n\n".join(parts)

        logger.info(
            "task_context_assembled",
            prompt_tokens=prompt_tokens,
            full_tokens=_estimate_tokens(full_text),
            budget=max_context_tokens,
            num_sections=len(sections),
            section_tokens=section_tokens,
        )

        return full_text

    async def _llm_summarize(
        self,
        module_name: str,
        raw_context: str,
        query: str,
        max_tokens: int = 16384,
    ) -> str:
        """Use LLM to extract query-relevant summary from raw context."""
        prompt = SUMMARY_TEMPLATE.format(
            query=query,
            module_name=module_name,
            raw_context=raw_context,
        )
        prompt_tokens = _estimate_tokens(prompt)
        raw_tokens = _estimate_tokens(raw_context)
        logger.info(
            "toa_llm_summary_input",
            module=module_name,
            prompt_tokens=prompt_tokens,
            prompt_chars=len(prompt),
            raw_tokens=raw_tokens,
            raw_chars=len(raw_context),
            query_preview=query[:200],
            full_prompt=prompt,
        )
        summary_input = MultiModalInput()
        summary_input.add_text(prompt)
        request = InferenceRequest(
            input=summary_input,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        result = await self.provider.infer_small(request)
        summary = result.text.strip()
        logger.info(
            "toa_llm_summary_output",
            module=module_name,
            raw_tokens=raw_tokens,
            raw_chars=len(raw_context),
            summary_tokens=_estimate_tokens(summary),
            summary_chars=len(summary),
            summary_full=summary,
            max_tokens=max_tokens,
        )
        return summary

    def _split_text_semantic(self, text: str) -> list[str]:
        """Split text into 300-500 token chunks preserving semantic boundaries.

        Strategy: paragraph grouping → over-large chunks split by sentences → final.
        """
        tokens = _estimate_tokens(text)
        if tokens <= TARGET_CHUNK_MAX:
            return [text]

        paragraphs = text.split("\n")
        target = (TARGET_CHUNK_MIN + TARGET_CHUNK_MAX) // 2

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _estimate_tokens(para)
            if current_tokens + para_tokens > TARGET_CHUNK_MAX and current_parts:
                chunks.append("\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunks.append("\n".join(current_parts))

        result: list[str] = []
        for chunk in chunks:
            if _estimate_tokens(chunk) > TARGET_CHUNK_MAX * 2:
                result.extend(self._split_by_sentences(chunk))
            else:
                result.append(chunk)

        if not result:
            return [text]
        return result

    def _split_by_sentences(self, text: str) -> list[str]:
        """Fallback: split oversized chunk by sentence boundaries."""
        import re
        sentences = re.split(r"(?<=[.!?。！？\n])\s*", text)
        sentences = [s for s in sentences if s.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            st = _estimate_tokens(sent)
            if current_tokens + st > TARGET_CHUNK_MAX and current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sent)
            current_tokens += st

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]

    async def _summarize_with_chunking(
        self,
        module_name: str,
        raw_context: str,
        query: str,
        budget: int,
        level: int = 0,
    ) -> str:
        """Tree-of-agents synthesis: semantic chunking → leaf summarization → tree merge."""
        if not raw_context or not raw_context.strip():
            return ""
        budget = max(budget, 64)

        if _estimate_tokens(raw_context) <= budget:
            return await self._llm_summarize(
                module_name, raw_context, query, max_tokens=budget
            )

        chunks = self._split_text_semantic(raw_context)
        chunk_token_sizes = [_estimate_tokens(c) for c in chunks]
        logger.info(
            "toa_tree_chunking",
            module=module_name,
            num_chunks=len(chunks),
            level=level,
            raw_tokens=_estimate_tokens(raw_context),
            budget=budget,
            raw_chars=len(raw_context),
            chunk_token_sizes=chunk_token_sizes,
            chunk_token_min=min(chunk_token_sizes) if chunk_token_sizes else 0,
            chunk_token_max=max(chunk_token_sizes) if chunk_token_sizes else 0,
            leaf_budget=max(int(budget * 0.4 // max(len(chunks), 1)), 64),
        )

        if not chunks:
            return ""

        leaf_budget = max(int(budget * 0.4 // max(len(chunks), 1)), 64)

        async def _leaf_summarize(chunk):
            async with _SUMMARY_SEMAPHORE:
                return await self._llm_summarize(module_name, chunk, query, max_tokens=leaf_budget)

        raw_results = await asyncio.gather(
            *[_leaf_summarize(c) for c in chunks], return_exceptions=True,
        )

        summaries: list[str] = []
        failed_count = 0
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.warning("toa_leaf_failed", module=module_name, chunk=i, error=str(r))
                failed_count += 1
            else:
                summaries.append(r)

        logger.info(
            "toa_leaves_done",
            module=module_name,
            total_chunks=len(chunks),
            success=len(summaries),
            failed=failed_count,
            leaf_total_tokens=sum(_estimate_tokens(s) for s in summaries),
        )

        if not summaries:
            logger.error("tree_all_leaves_failed", module=module_name)
            return ""
        if len(summaries) == 1:
            return summaries[0]

        return await self._synthesize_tree(module_name, summaries, query, budget)

    async def _synthesize_tree(
        self, module_name: str, summaries: list[str], query: str, budget: int,
    ) -> str:
        """Recursive tree merge: group leaf summaries → intermediate synthesis → root."""
        merge_level = 0
        while len(summaries) > 1:
            groups = [
                summaries[i:i + TREE_GROUP_SIZE]
                for i in range(0, len(summaries), TREE_GROUP_SIZE)
            ]

            merge_budget = max(int(budget * 0.3 // max(len(groups), 1)), 64)
            logger.info(
                "toa_tree_merge_level",
                module=module_name,
                level=merge_level,
                input_count=len(summaries),
                group_count=len(groups),
                merge_budget=merge_budget,
            )

            async def _synthesize_group(partials: list[str]) -> str:
                if len(partials) == 1:
                    return partials[0]
                text = "\n\n---\n\n".join(partials)
                async with _SUMMARY_SEMAPHORE:
                    return await self._llm_summarize(
                        module_name, text, query, max_tokens=merge_budget,
                    )

            raw = await asyncio.gather(
                *[_synthesize_group(g) for g in groups], return_exceptions=True,
            )
            summaries = [r for r in raw if not isinstance(r, Exception)]
            if not summaries:
                return ""
            merge_level += 1

        return summaries[0]

    async def _build_personality_block(self) -> str:
        """Build compact personality description block for prompt."""
        if self.self_value is None:
            return ""

        try:
            from nan_agent.self_value.hexaco_descriptor import describe_hexaco
            from nan_agent.self_value.emotional_descriptor import values_priority

            ctx = self.self_value.get_personality_context()
            if not ctx:
                return ""

            hexaco = ctx.get("hexaco", {})
            self_desc = ctx.get("self_description", "")
            core_values = ctx.get("core_values", [])

            hexaco_line = describe_hexaco(hexaco)
            values_line = values_priority(core_values)

            lines = ["<personality>"]
            if self_desc:
                desc_cn = self._translate_to_cn(self_desc)
                lines.append(f"自我认知：{desc_cn}")
            if hexaco_line:
                lines.append(f"性格基调：{hexaco_line}")
            if values_line:
                lines.append(f"当前优先驱动力：{values_line}")
            lines.append("</personality>")

            return "\n".join(lines)
        except Exception as e:
            logger.debug("personality_block_failed", error=str(e))
            return ""

    @staticmethod
    def _translate_to_cn(text: str) -> str:
        """Simple keyword-based EN→CN translation for personality traits."""
        en_to_cn = {
            "introspective and internally focused": "内省驱动、内在聚焦",
            "intuitive and abstract thinker": "直觉型抽象思考者",
            "analytical and logic-driven": "分析型逻辑驱动",
            "structured and plan-oriented": "偏好结构和计划",
            "introspective": "内省",
            "intuitive": "直觉型",
            "abstract": "抽象",
            "analytical": "分析型",
            "logic-driven": "逻辑驱动",
            "structured": "结构化",
            "plan-oriented": "计划导向",
            "internally focused": "内在聚焦",
            "thinker": "思考者",
        }
        result = text
        for en, cn in sorted(en_to_cn.items(), key=lambda x: -len(x[0])):
            if en in result.lower():
                result = result.lower().replace(en, cn)
        return result

    async def infer_task_intent(self, user_input: str) -> str:
        """Pre-process user input into a short Chinese intent summary."""
        try:
            prompt = (
                "Output a one-sentence Chinese summary of the user's intent.\n\n"
                f"<user_message>\n{user_input}\n</user_message>\n\n"
                "Intent (one sentence in Chinese):"
            )
            mm = MultiModalInput()
            mm.add_text(prompt)
            request = InferenceRequest(input=mm, temperature=0.0, max_tokens=64)
            result = await self.provider.infer_small(request)
            intent = result.text.strip()
            return intent if intent else user_input
        except Exception as e:
            logger.debug("task_intent_failed", error=str(e))
            return user_input

    def _build_tools_block(self, tools_desc: str = "") -> str:
        """Build the tools description block for the reasoning prompt.

        The actual tool descriptions are generated by react_loop.get_tools_description()
        and passed to build_reasoning_prompt as the tools_desc parameter.
        """
        if tools_desc:
            return f"<tools>\n{tools_desc}\n</tools>"
        return ""

    async def _build_intent_block(self, user_input: str = "") -> str:
        """Build the task intent block for the reasoning prompt.

        Thin wrapper around infer_task_intent() that infers the user's intent
        and formats it as a prompt block.
        """
        if not user_input:
            return ""
        intent = await self.infer_task_intent(user_input)
        return f"<task_intent>\n{intent}\n</task_intent>" if intent else ""

    async def _fetch_personality_raw(self, query: str, _budget: int) -> str:
        return await self._build_personality_block()

    async def _fetch_memory_raw(self, query: str, _budget: int) -> str:
        if self.hard_memory is None:
            return ""
        try:
            memories = await self.hard_memory.recollect(query)
            if not memories:
                return ""
            memory_lines = "\n".join(
                m.content if hasattr(m, "content") else str(m)
                for m in memories
            )
            return f"<memory>\n{memory_lines}\n</memory>"
        except Exception:
            return ""

    async def _fetch_skills_raw(self, query: str, _budget: int) -> str:
        if self.skill_trees is None:
            return ""
        try:
            raw_skills = self.skill_trees.search_all(query)
            if not raw_skills:
                return ""
            skill_lines = "\n".join(
                f"- {s.name} ({int(s.proficiency * 100)}%): {s.description} (category: {s.category})"
                for s in raw_skills
            )
            return f"<skills>\n{skill_lines}\n</skills>"
        except Exception:
            return ""

    async def _fetch_adaptors_raw(self, query: str, _budget: int) -> str:
        if self.soft_memory is None:
            return ""
        try:
            adaptors = self.soft_memory.get_active_adaptors()
            if not adaptors:
                return ""
            adaptor_lines = "\n".join(f"- {a}" for a in adaptors)
            return f"<adaptors>\n{adaptor_lines}\n</adaptors>"
        except Exception:
            return ""

    async def _fetch_exp_raw(self, query: str, _budget: int) -> str:
        if self.hard_memory is None:
            return ""
        try:
            exps = await self.hard_memory.match_skills(query, top_k=5)
            if not exps:
                return ""
            exp_lines = "\n".join(
                f"- {e.content}" for e in exps
            )
            return f"<experience>\n{exp_lines}\n</experience>"
        except Exception:
            return ""

    async def _fetch_emotional_state(self, query: str, budget: int) -> str:
        """Fetch emotional state description for prompt enrichment."""
        if self.self_value is None:
            return ""
        try:
            from nan_agent.self_value.emotional_descriptor import format_emotional_state_for_prompt
            state = self.self_value.get_emotional_state()
            if not state:
                return ""
            description = format_emotional_state_for_prompt(state)
            if _estimate_tokens(description) > budget:
                return ""
            return description
        except Exception as e:
            logger.debug("emotional_state_fetch_failed", error=str(e))
            return ""

    async def health_check(self) -> bool:
        return await self.provider.health_check()
