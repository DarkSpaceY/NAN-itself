"""
Qwen3-0.6B dialogue adapter (the voice module's surface composer).

Semantic / surface split: the main agent sends a semantic task
(intent + key points + tone); this small local SLM composes the
actual spoken utterance -- wording plus one whitelisted CosyVoice
directive. It also answers trivial turns autonomously (the fast
path) under a fixed, versioned allowlist prompt; anything
non-trivial is escalated to the main agent.

Non-thinking mode (enable_thinking=False) for latency; Qwen3's
recommended non-thinking sampling (temperature 0.7 / top_p 0.8 /
top_k 20). Output must be a single JSON object; parsing is
defensive -- a malformed reply yields None and the module falls
back to escalation instead of speaking garbage.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from nan_itself.utils import paths as _paths
from nan_itself.utils.tts import (
    ALLOWED_INSTRUCTS,
    sanitize_instruct,
)


DEFAULT_REPO_ID = "Qwen/Qwen3-0.6B"

# Versioned fast-path policy: bump when the allowlist prompt
# changes meaningfully.
FAST_PATH_PROMPT_VERSION = 1

_COMPOSE_SYSTEM = """\
你是语音助手的话术引擎。主智能体交给你一个任务，你要把它组织成\
一句自然的中文口语，交给 TTS 直接朗读。

规则：
1. 只输出一个 JSON 对象，不要输出任何其他内容：
   {{"instruct": "...", "text": "..."}}
2. instruct 必须从下面这些指令中逐字选择一个，不得改写、不得自创：
{allowlist}
3. text 是要朗读的中文口语：像说话，不要书面语，简洁自然，
   可以使用这些标记增加表现力：[laughter] [breath] <strong>；
   不要使用任何其他标记或 <|...|> 形式的特殊符号。
4. 只组织表达，不回答任务之外的问题，不解释。"""

_COMPOSE_USER = """\
任务：
意图：{intent}
要点：{key_points}
语气偏好：{tone}"""

_FAST_PATH_SYSTEM = f"""\
你是本机语音助手的快速应答器。用户刚对你说了一句话。只有当这句\
话属于下面几类琐碎话轮时，你才可以直接回答；否则必须升级给主智\
能体处理：

1. 打招呼、告别、感谢（你好 / 早上好 / 晚安 / 谢谢 / 再见）
2. 问时间、日期、星期（当前时间会提供给你）
3. 简单确认与应答（好的 / 嗯 / 可以 / 不用了）

硬性规则：
- 你没有观点，不做承诺，不执行任何动作，不谈论需要记忆或个性的\
内容。
- 只输出一个 JSON 对象，不要输出任何其他内容：
  可以直接回答时：{{"answer": "一句自然的中文口语"}}
  不该回答时：{{"escalate": true}}

[fast-path policy v{FAST_PATH_PROMPT_VERSION}]"""

_FAST_PATH_USER = """\
当前时间：{now}
用户说：{user_text}"""


def _allowlist_block() -> str:
    return "\n".join(
        f"   - {instruct}" for instruct in ALLOWED_INSTRUCTS
    )


class SmallDialogue:
    """
    Lazy Qwen3-0.6B adapter. Weights live under
    models/voice/slm/<snapshot>; a missing directory downloads
    from the Hub during provisioning (honors HF_ENDPOINT and
    proxy env vars) -- a failing download raises out of load(),
    so the Facade marks the module DOWN and retries.
    """

    def __init__(
        self,
        model_path: Path,
        repo_id: str | None = None,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ) -> None:
        self.model_path = Path(model_path)

        self.repo_id = repo_id or DEFAULT_REPO_ID

        self.max_new_tokens = max_new_tokens

        self.temperature = temperature

        self.top_p = top_p

        self.top_k = top_k

        self._tokenizer: Any = None

        self._model: Any = None

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    def _weights_present(self) -> bool:
        return self.model_path.is_dir() and any(
            self.model_path.iterdir()
        )

    def _download(self) -> None:
        logger.info(
            "slm weights missing at {}; downloading {} "
            "(~1.5 GB, honors HF_ENDPOINT and proxy envs)",
            self.model_path,
            self.repo_id,
        )

        import huggingface_hub

        huggingface_hub.snapshot_download(
            repo_id=self.repo_id,
            local_dir=self.model_path,
        )

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._weights_present():
            self._download()

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if torch.cuda.is_available():
            device = "cuda"

            dtype = torch.float16

        elif torch.backends.mps.is_available():
            device = "mps"

            dtype = torch.float16

        else:
            device = "cpu"

            dtype = torch.float32

        path = str(self.model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(path)

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )

        logger.info(
            "dialogue slm loaded from {} on {}",
            path,
            device,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int,
    ) -> str:
        self.load()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        text = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return self._strip_think(text)

    @staticmethod
    def _strip_think(text: str) -> str:
        """
        Defensive: non-thinking mode should never leak a think
        block, but a leaked one must not reach the JSON parser.
        """
        if "<think>" in text:
            return text.split("</think>", 1)[-1].strip()

        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        start = text.find("{")

        end = text.rfind("}")

        if start < 0 or end <= start:
            return None

        try:
            payload = json.loads(text[start : end + 1])

        except Exception:
            return None

        return payload if isinstance(payload, dict) else None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def compose(
        self,
        task: Mapping[str, Any],
    ) -> dict[str, str] | None:
        """
        Task -> surface utterance {"instruct", "text"}.

        The instruct is validated against the TTS whitelist here
        (illegal choices become ""), so the module can pass the
        result straight into CosyVoiceTTS.speak(). None means
        the SLM failed to produce a usable utterance.
        """
        raw = self._generate(
            _COMPOSE_SYSTEM.format(
                allowlist=_allowlist_block(),
            ),
            _COMPOSE_USER.format(
                intent=task.get("intent") or "narrate",
                key_points=task.get("key_points") or "",
                tone=task.get("tone") or "自然",
            ),
            self.max_new_tokens,
        )

        payload = self._extract_json(raw)

        if payload is None:
            return None

        text = str(payload.get("text") or "").strip()

        if not text:
            return None

        return {
            "instruct": sanitize_instruct(
                str(payload.get("instruct") or "")
            ),
            "text": text,
        }

    def fast_reply(
        self,
        user_text: str,
        now: str | None = None,
    ) -> str | None:
        """
        Trivial-turn fast path. Returns the spoken reply, or
        None when the turn must escalate to the main agent.
        """
        raw = self._generate(
            _FAST_PATH_SYSTEM,
            _FAST_PATH_USER.format(
                now=now or time.strftime("%Y-%m-%d %H:%M"),
                user_text=(user_text or "").strip(),
            ),
            self.max_new_tokens,
        )

        payload = self._extract_json(raw)

        if payload is None:
            return None

        if payload.get("escalate"):
            return None

        answer = str(payload.get("answer") or "").strip()

        return answer or None


def default_slm_path() -> Path:
    """
    Repo-conventional SLM weight path (env overridable at the
    module layer).
    """
    return _paths.models_dir() / "voice" / "slm" / "qwen3-0.6b"
