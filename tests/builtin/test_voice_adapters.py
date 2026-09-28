"""
Voice adapter unit tests (no weights, no network).

Covers:
    - instruct whitelist (the SLM's control surface)
    - CosyVoiceTTS loud-failure provisioning paths (principle 7)
    - SmallDialogue defensive JSON parsing (compose + fast path)
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_voice_module():
    """
    Load builtin/modules/voice.py the way the loader does. The
    TTS/SLM adapters live inside the voice module now, so the
    adapter names are read off the loaded module object.
    """
    spec = importlib.util.spec_from_file_location(
        "voice_adapters_test",
        REPO / "builtin" / "modules" / "voice.py",
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


voice = _load_voice_module()

ALLOWED_INSTRUCTS = voice.ALLOWED_INSTRUCTS

NEUTRAL_INSTRUCT = voice.NEUTRAL_INSTRUCT

CosyVoiceTTS = voice.CosyVoiceTTS

SmallDialogue = voice.SmallDialogue

sanitize_instruct = voice.sanitize_instruct


# ============================================================================
# Instruct whitelist
# ============================================================================


def test_sanitize_instruct_accepts_verbatim_directive():
    for instruct in ALLOWED_INSTRUCTS:
        assert sanitize_instruct(instruct) == instruct


def test_sanitize_instruct_drops_hallucinated_directive():
    assert sanitize_instruct("用外星人的语气说") == ""

    assert sanitize_instruct("用开心的语气说啦") == ""

    # Near-misses are not tolerated: the whitelist is verbatim.
    assert sanitize_instruct("用 开心 的语气说") == ""


def test_sanitize_instruct_trims_and_empty():
    assert sanitize_instruct(f"  {ALLOWED_INSTRUCTS[1]}  ") == (
        ALLOWED_INSTRUCTS[1]
    )

    assert sanitize_instruct("") == ""

    assert sanitize_instruct(None) == ""  # type: ignore[arg-type]


# ============================================================================
# CosyVoiceTTS provisioning failures are loud and specific
# ============================================================================


def _tts(tmp_path: Path) -> CosyVoiceTTS:
    return CosyVoiceTTS(
        checkout_dir=tmp_path / "cosyvoice",
        model_dir=tmp_path / "tts" / "cosyvoice2-0.5b",
        prompt_wav=tmp_path / "tts" / "reference.wav",
    )


def test_tts_load_auto_clones_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    A missing checkout is fetched, not fatal: a successful clone
    lets load() advance to the next provisioning check.
    """

    def fake_run(cmd, **_kwargs):
        Path(cmd[-1]).mkdir(parents=True)

        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="CosyVoice2-0.5B"):
        _tts(tmp_path).load()

    assert (_tts(tmp_path).checkout_dir).is_dir()


def test_tts_load_clone_failure_is_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def failing_run(cmd, **_kwargs):
        raise subprocess.CalledProcessError(
            128,
            cmd,
            stderr="network unreachable",
        )

    monkeypatch.setattr(subprocess, "run", failing_run)

    with pytest.raises(RuntimeError, match="network unreachable"):
        _tts(tmp_path).load()


def test_tts_load_missing_weights(tmp_path: Path):
    tts = _tts(tmp_path)

    tts.checkout_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="CosyVoice2-0.5B"):
        tts.load()


def test_tts_load_missing_reference_wav(tmp_path: Path):
    tts = _tts(tmp_path)

    tts.checkout_dir.mkdir(parents=True)

    tts.model_dir.mkdir(parents=True)

    (tts.model_dir / "cosyvoice2.yaml").write_text("")

    with pytest.raises(RuntimeError, match="reference wav"):
        tts.load()


# ============================================================================
# SmallDialogue defensive parsing
# ============================================================================


def _slm(tmp_path: Path) -> SmallDialogue:
    return SmallDialogue(model_path=tmp_path / "slm")


def test_extract_json_takes_first_to_last_braces(tmp_path: Path):
    assert _slm(tmp_path)._extract_json(
        '前言 {"answer": "你好"} 后记'
    ) == {"answer": "你好"}

    assert _slm(tmp_path)._extract_json("没有大括号") is None

    assert _slm(tmp_path)._extract_json("{不合法") is None

    assert _slm(tmp_path)._extract_json("[1, 2]") is None


def test_strip_think_removes_leaked_block(tmp_path: Path):
    strip = SmallDialogue._strip_think

    assert strip("<think>\n思考\n</think>\n正文") == "正文"

    assert strip("  干净输出  ") == "干净输出"


def test_compose_parses_task_to_utterance(tmp_path: Path, monkeypatch):
    slm = _slm(tmp_path)

    monkeypatch.setattr(
        slm,
        "_generate",
        lambda *a, **k: '{"instruct": "用开心的语气说", "text": "好嘞，马上来！"}',
    )

    assert slm.compose(
        {"intent": "confirm", "key_points": "马上来", "tone": "轻快"}
    ) == {"instruct": "用开心的语气说", "text": "好嘞，马上来！"}


def test_compose_returns_instruct_verbatim(tmp_path: Path, monkeypatch):
    """Whitelist enforcement lives in the voice module, not here."""
    slm = _slm(tmp_path)

    monkeypatch.setattr(
        slm,
        "_generate",
        lambda *a, **k: '{"instruct": "用悲伤到极致的语气说", "text": "内容"}',
    )

    assert slm.compose({"intent": "narrate", "key_points": "x"}) == {
        "instruct": "用悲伤到极致的语气说",
        "text": "内容",
    }


def test_compose_returns_none_on_unusable_output(
    tmp_path: Path, monkeypatch
):
    slm = _slm(tmp_path)

    monkeypatch.setattr(
        slm, "_generate", lambda *a, **k: "我觉得应该 escalate"
    )

    assert slm.compose({"intent": "confirm"}) is None

    monkeypatch.setattr(
        slm, "_generate", lambda *a, **k: '{"instruct": "用开心的语气说"}'
    )

    # JSON fine but no text: nothing to speak.
    assert slm.compose({"intent": "confirm"}) is None


def test_fast_reply_answers_or_escalates(tmp_path: Path, monkeypatch):
    slm = _slm(tmp_path)

    monkeypatch.setattr(
        slm,
        "_generate",
        lambda *a, **k: '{"answer": "早上好！"}',
    )

    assert slm.fast_reply("早上好") == "早上好！"

    monkeypatch.setattr(
        slm,
        "_generate",
        lambda *a, **k: '{"escalate": true}',
    )

    assert slm.fast_reply("帮我订机票") is None

    monkeypatch.setattr(
        slm, "_generate", lambda *a, **k: "糟糕，输出坏了"
    )

    assert slm.fast_reply("你好") is None
