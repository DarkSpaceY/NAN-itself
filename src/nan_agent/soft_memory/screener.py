import hashlib
import json

from dataclasses import dataclass

from nan_agent.logging.logger import get_logger
from nan_agent.model.types import MultiModalInput

logger = get_logger(__name__)


@dataclass
class ScreenedContent:
    source: str
    content: str
    novelty: float = 0.5
    importance: float = 0.5
    reliability: float = 0.5

    @property
    def learning_score(self) -> float:
        return self.novelty * 0.4 + self.importance * 0.4 + self.reliability * 0.2


class ContentScreener:
    def __init__(self, cognition):
        self.cognition = cognition

    async def screen(self, candidates: list[dict], top_k: int = 20) -> list[ScreenedContent]:
        if not candidates:
            return []

        limited = candidates[:100]
        formatted_lines = []
        for i, c in enumerate(limited):
            content_text = c.get("content", "")
            truncated = content_text if content_text else ""
            formatted_lines.append(f"[{i}] {c.get('id', f'item-{i}')}: {truncated}")

        candidates_text = "\n".join(formatted_lines)

        prompt_text = (
            "You are a content evaluator for an AI agent's learning system. "
            "Analyze the following content candidates and rate each on three dimensions "
            "(novelty, importance, reliability) on a scale from 0.0 to 1.0.\n\n"
            "Scoring guidelines:\n"
            "- novelty: How new, unexpected, or non-redundant this content is relative to existing knowledge (0.0 = completely known, 1.0 = entirely new)\n"
            "- importance: How significant or useful this content is for achieving goals (0.0 = trivial, 1.0 = critical)\n"
            "- reliability: How trustworthy and accurate the content appears to be (0.0 = highly dubious, 1.0 = completely reliable)\n\n"
            "Candidates to evaluate:\n\n"
            f"{candidates_text}\n\n"
            "Return your evaluation as a JSON array of objects, where each object has keys: "
            '"id" (the candidate index number), "novelty", "importance", "reliability". '
            "Only include candidates from the list. Do not include any other text."
        )

        user_input = MultiModalInput()
        user_input.add_text(prompt_text)

        try:
            result = await self.cognition.infer(
                user_input,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("screener_infer_failed", error=str(e))
            return []

        raw_text = result.text.strip() if result and result.text else ""

        if not raw_text:
            return []

        parsed = self._parse_json_response(raw_text)
        if not parsed:
            return []

        id_to_content: dict[str, dict] = {}
        for i, c in enumerate(limited):
            item_id = str(i)
            id_to_content[item_id] = {
                "source": c.get("source", f"candidate-{i}"),
                "content": c.get("content", ""),
            }

        scored: list[ScreenedContent] = []
        for item in parsed:
            item_id = str(item.get("id", ""))
            lookup = id_to_content.get(item_id)
            if lookup is None:
                continue
            scored.append(
                ScreenedContent(
                    source=lookup["source"],
                    content=lookup["content"],
                    novelty=self._clamp(item.get("novelty", 0.5)),
                    importance=self._clamp(item.get("importance", 0.5)),
                    reliability=self._clamp(item.get("reliability", 0.5)),
                )
            )

        if not scored:
            return []

        return self.rank(scored, top_k=top_k)

    @staticmethod
    def rank(items: list[ScreenedContent], top_k: int = 20) -> list[ScreenedContent]:
        sorted_items = sorted(items, key=lambda x: x.learning_score, reverse=True)

        seen_signatures: set[str] = set()
        deduped: list[ScreenedContent] = []
        for item in sorted_items:
            signature = hashlib.md5(item.content.encode()).hexdigest()
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                deduped.append(item)

        return deduped[:top_k]

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _parse_json_response(text: str) -> list[dict]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) > 1 and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("evaluations"), list):
                return data["evaluations"]
            return []
        except json.JSONDecodeError as e:
            logger.warning(
                "screener_json_parse_failed",
                error=str(e),
                raw_text=text[:500],
            )
            return []