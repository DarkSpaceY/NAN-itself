"""Cross-encoder reranker, torch fp16 on MPS.

The runtime was chosen by the full LoCoMo comparison in
`paper_search/memory/bench_ce_runtime.py` (1982 queries x 100 candidates,
identical pairs): torch fp16 matched onnxruntime int8 within 5 questions at
every cut, and ran 2.35x faster here (p50 675ms vs 1612ms per query). ONNX is
deliberately not carried: one runtime, no fallback path.
"""

from __future__ import annotations

import threading


def _resolve_device(torch, device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Reranker:
    """Lazily imported so service startup never pays the torch cost."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-base",
        *,
        device: str = "auto",
        fp16: bool = True,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._device = _resolve_device(torch, device)
        self._fp16 = bool(fp16 and self._device != "cpu")
        self._batch_size = batch_size
        self._max_length = max_length
        self._lock = threading.Lock()

        self._tokenizer = AutoTokenizer.from_pretrained(model)
        net = AutoModelForSequenceClassification.from_pretrained(model)
        if self._fp16:
            net.half()
        self._model = net.to(self._device).eval()

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Scores every text against the query; order is preserved."""
        if not texts:
            return []

        torch = self._torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [0.0] * len(texts)

        with self._lock, torch.inference_mode():
            for start in range(0, len(order), self._batch_size):
                idx = order[start : start + self._batch_size]
                enc = self._tokenizer(
                    [query] * len(idx),
                    [texts[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                if self._device == "mps":
                    torch.mps.synchronize()
                logits = self._model(**enc).logits.view(-1).float().cpu().tolist()
                if self._device == "mps":
                    torch.mps.synchronize()
                for i, value in zip(idx, logits, strict=True):
                    out[i] = float(value)

        return out
