"""
CosyVoice2 TTS adapter (the voice module's speech surface).

Backed by an upstream FunAudioLLM/CosyVoice checkout (cloned to
models/voice/cosyvoice/ by convention, overridable via env at
the module layer) injected into sys.path here -- the repo itself
stays free of third-party code.

Contract:
    load()      heavy provisioning; raises loudly so the Facade
                can mark the module DOWN and retry with backoff
                (core principle 7). Missing checkout, missing
                weights or a missing reference wav are all loud.
    speak()     sentence-streamed: the upstream text normalizer
                splits long text and the generator yields float32
                mono chunks as they are synthesized.

The instruct directive is a whitelist-controlled control surface:
the SLM's choice is validated here, and anything hallucinated is
dropped before it can reach the synthesis model.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


# Directive used when the SLM emits no (or an illegal) instruct.
NEUTRAL_INSTRUCT = "用自然的语气说"

# The SLM's whitelist: instruct must be one of these, verbatim.
ALLOWED_INSTRUCTS: tuple[str, ...] = (
    NEUTRAL_INSTRUCT,
    "用开心的语气说",
    "用兴奋的语气说",
    "用温柔的语气说",
    "用认真的语气说",
    "用抱歉的语气说",
    "用平静的语气说",
    "用快速的语气说",
)


def sanitize_instruct(instruct: str) -> str:
    """
    Whitelist filter for the SLM's instruct choice.

    Returns the directive verbatim when legal, "" otherwise --
    callers substitute NEUTRAL_INSTRUCT. Hallucinated control
    text must never reach the synthesis model.
    """
    text = (instruct or "").strip()

    return text if text in ALLOWED_INSTRUCTS else ""


class CosyVoiceTTS:
    """
    Lazy CosyVoice2 adapter around an upstream checkout.

    Provisioning failures raise out of load(): the caller (the
    voice module's start()) propagates them, the Facade marks
    the module DOWN with the error and retries with backoff, and
    the module revives once the checkout and weights land.
    """

    def __init__(
        self,
        checkout_dir: Path,
        model_dir: Path,
        prompt_wav: Path,
        speed: float = 1.0,
    ) -> None:
        self.checkout_dir = Path(checkout_dir)

        self.model_dir = Path(model_dir)

        self.prompt_wav = Path(prompt_wav)

        self.speed = float(speed)

        self._model: Any = None

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    def _inject_sys_path(self) -> None:
        """
        The upstream package imports Matcha-TTS from
        third_party/, so both roots go on sys.path.
        """
        candidates = (
            self.checkout_dir,
            self.checkout_dir / "third_party" / "Matcha-TTS",
        )

        for path in candidates:
            resolved = str(path.resolve())

            if path.is_dir() and resolved not in sys.path:
                sys.path.insert(0, resolved)

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.checkout_dir.is_dir():
            raise RuntimeError(
                "CosyVoice checkout missing at "
                f"{self.checkout_dir}; clone FunAudioLLM/CosyVoice "
                "there (git clone https://github.com/FunAudioLLM/"
                "CosyVoice.git) and install its requirements"
            )

        if not (self.model_dir / "cosyvoice2.yaml").is_file():
            raise RuntimeError(
                "CosyVoice2 weights missing at "
                f"{self.model_dir} (no cosyvoice2.yaml); download "
                "them with: modelscope download --model "
                f"iic/CosyVoice2-0.5B --local_dir {self.model_dir}"
            )

        if not self.prompt_wav.is_file():
            raise RuntimeError(
                "reference wav missing at "
                f"{self.prompt_wav}; drop a >=5s 16kHz mono clip "
                "of the desired voice there"
            )

        self._inject_sys_path()

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2

        except ImportError as exc:
            raise RuntimeError(
                "cosyvoice runtime not importable "
                f"({exc}); install the checkout's requirements "
                "(pynini via conda-forge on macOS arm64, "
                "WeTextProcessing fallback elsewhere)"
            ) from exc

        self._model = CosyVoice2(
            str(self.model_dir),
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False,
        )

        logger.info(
            "cosyvoice tts ready (sr={}, checkout={})",
            self._model.sample_rate,
            self.checkout_dir,
        )

    @property
    def sample_rate(self) -> int:
        self.load()

        return int(self._model.sample_rate)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def speak(
        self,
        text: str,
        instruct: str = "",
        speed: float | None = None,
    ) -> Iterator[np.ndarray]:
        """
        Yield float32 mono speech chunks (native sample rate)
        as they are synthesized. Empty instruct resolves to the
        neutral directive; illegal instructs are dropped.
        """
        self.load()

        text = (text or "").strip()

        if not text:
            return

        directive = sanitize_instruct(instruct) or NEUTRAL_INSTRUCT

        effective = self.speed if speed is None else float(speed)

        outputs = self._model.inference_instruct2(
            text,
            directive,
            str(self.prompt_wav),
            stream=True,
            speed=effective,
        )

        for output in outputs:
            chunk = (
                output.get("tts_speech")
                if isinstance(output, dict)
                else None
            )

            if chunk is None:
                continue

            audio = (
                chunk.detach().cpu().float().numpy().reshape(-1)
            )

            if audio.size:
                yield audio
