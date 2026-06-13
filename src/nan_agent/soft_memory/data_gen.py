import json

from nan_agent.logging.logger import get_logger
from nan_agent.model.types import MultiModalInput, MultiModalOutput
from nan_agent.soft_memory.screener import ScreenedContent

logger = get_logger(__name__)


class TrainingDataGenerator:
    def __init__(self, cognition):
        self.cognition = cognition

    async def generate_questions(
        self, contents: list[ScreenedContent], n_questions: int = 10
    ) -> list[str]:
        if not contents:
            return []

        limited = contents[:10]
        formatted_lines = []
        for i, c in enumerate(limited):
            truncated = c.content if c.content else ""
            formatted_lines.append(f"[{i}] {c.source}: {truncated}")

        content_text = "\n".join(formatted_lines)

        prompt_text = (
            "You are a training data generator for an AI learning system. "
            "Based on the following content, generate questions that test "
            "understanding of the material.\n\n"
            "Content:\n"
            f"{content_text}\n\n"
            f"Generate exactly {n_questions} questions that cover different "
            "aspects of the content. Questions should be diverse, testing "
            "comprehension, application, and analysis.\n\n"
            "Return your response as a JSON object with a 'questions' key "
            "containing an array of question strings. "
            "Example: {\"questions\": [\"question 1\", \"question 2\"]}\n"
            "Return ONLY the JSON object, no other text."
        )

        user_input = MultiModalInput()
        user_input.add_text(prompt_text)

        try:
            result = await self.cognition.infer(
                user_input,
                temperature=0.7,
            )
        except Exception as e:
            logger.error("generate_questions_infer_failed", error=str(e))
            return []

        raw_text = result.text.strip() if result and result.text else ""

        if not raw_text:
            return []

        try:
            parsed = self._parse_json_response(raw_text)
            if isinstance(parsed, dict) and "questions" in parsed:
                return parsed["questions"]
            if isinstance(parsed, list):
                return [str(q) for q in parsed]
            return []
        except Exception as e:
            logger.warning(
                "generate_questions_parse_failed",
                error=str(e),
                raw_text=raw_text[:200],
            )
            return []

    async def generate_pairs(
        self, questions: list[str], contents: list[ScreenedContent]
    ) -> list[dict]:
        pairs: list[dict] = []

        if not questions or not contents:
            return pairs

        context_text = self._build_context_text(contents)

        for question in questions:
            try:
                chosen = await self._generate_chosen(question, context_text)
                reject = await self._generate_reject(question)

                if chosen and reject:
                    pairs.append({
                        "prompt": question,
                        "chosen": chosen,
                        "reject": reject,
                    })
            except Exception as e:
                logger.warning(
                    "generate_pair_failed",
                    question=question[:100],
                    error=str(e),
                )
                continue

        return pairs

    async def _generate_chosen(self, question: str, context_text: str) -> str:
        prompt_text = (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}"
        )

        user_input = MultiModalInput()
        user_input.add_text(prompt_text)

        result = await self.cognition.infer(
            user_input,
            temperature=0.3,
        )

        return result.text.strip() if result and result.text else ""

    async def _generate_reject(self, question: str) -> str:
        user_input = MultiModalInput()
        user_input.add_text(question)

        result = await self.cognition.infer(
            user_input,
            temperature=0.3,
        )

        return result.text.strip() if result and result.text else ""

    def _build_context_text(self, contents: list[ScreenedContent]) -> str:
        limited = contents[:10]
        lines = []
        for i, c in enumerate(limited):
            truncated = c.content if c.content else ""
            lines.append(f"[Source {i}] {c.source}\n{truncated}")
        return "\n\n".join(lines)

    @staticmethod
    def filter_quality(
        pairs: list[dict], min_chosen_length: int = 20
    ) -> list[dict]:
        filtered: list[dict] = []
        for pair in pairs:
            chosen = pair.get("chosen", "")
            reject = pair.get("reject", "")
            if (
                len(chosen) >= min_chosen_length
                and len(reject) > 0
                and len(chosen) > len(reject)
            ):
                filtered.append(pair)
        return filtered

    @staticmethod
    def augment(pairs: list[dict]) -> list[dict]:
        augmented: list[dict] = list(pairs)
        for pair in pairs:
            variant = {
                "prompt": f"Explain in detail: {pair['prompt']}",
                "chosen": pair["chosen"],
                "reject": pair["reject"],
            }
            augmented.append(variant)
        return augmented

    @staticmethod
    def _parse_json_response(text: str) -> dict | list:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) > 1 and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)