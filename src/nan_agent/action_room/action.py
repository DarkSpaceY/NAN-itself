"""
Action Room - Action Executors

提供各种动作执行器：
- TTS: 文本转语音
- GUI: 基于 OmniParser 的屏幕操作（截图、点击、输入等）
"""

import asyncio
import hashlib
import io
import os
import struct
import subprocess
import tempfile
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple

from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_TTS_FORMATS = frozenset({"wav", "mp3"})
DEFAULT_TTS_FORMAT = "wav"
DEFAULT_VOICE_ID = "nan_default"
DEFAULT_SPEED = 1.0
DEFAULT_PITCH = 1.0
DEFAULT_LANGUAGE = "zh-CN"

MAX_CACHE_SIZE = 128


@dataclass
class TTSOutput:
    """TTS 输出"""
    audio_data: bytes
    text: str
    format: str = DEFAULT_TTS_FORMAT
    duration_ms: int = 0
    voice_id: str = DEFAULT_VOICE_ID

    def __post_init__(self):
        if self.format not in SUPPORTED_TTS_FORMATS:
            raise ActionError(
                f"Unsupported TTS format: {self.format}",
                error_code="E510",
                details={"format": self.format},
            )


@dataclass
class TTSRequest:
    """TTS 请求"""
    text: str
    voice_id: str = DEFAULT_VOICE_ID
    speed: float = DEFAULT_SPEED
    pitch: float = DEFAULT_PITCH
    format: str = DEFAULT_TTS_FORMAT
    language: str = DEFAULT_LANGUAGE

    def __post_init__(self):
        if self.format not in SUPPORTED_TTS_FORMATS:
            raise ActionError(
                f"Unsupported TTS format: {self.format}",
                error_code="E510",
                details={"format": self.format},
            )
        if self.speed <= 0 or self.speed > 3.0:
            raise ActionError(
                f"Speed must be in (0, 3.0], got {self.speed}",
                error_code="E511",
                details={"speed": self.speed},
            )
        if self.pitch <= 0 or self.pitch > 2.0:
            raise ActionError(
                f"Pitch must be in (0, 2.0], got {self.pitch}",
                error_code="E512",
                details={"pitch": self.pitch},
            )


_EDGE_TTS_VOICE_MAP = {
    "nan_default": "zh-CN-XiaoxiaoNeural",
    "nan_female_01": "zh-CN-XiaoxiaoNeural",
    "nan_male_01": "zh-CN-YunxiNeural",
    "nan_female_en": "en-US-JennyNeural",
    "nan_male_en": "en-US-GuyNeural",
    "nan_child": "zh-CN-XiaoxiaoNeural",
}

# 基于 edge-tts voice map 构建声音列表
_TTS_VOICES = [
    {"id": vid, "name": vid.replace("_", " ").title(), "gender": "female" if "female" in vid else "male" if "male" in vid else "neutral", "language": vname.split("-")[0] + "-" + vname.split("-")[1]}
    for vid, vname in _EDGE_TTS_VOICE_MAP.items()
]


class TTSController:
    """TTS（文本转语音）控制器。

    集成 edge-tts 作为 TTS 后端。edge-tts 不可用时直接抛出 ActionError，
    不做 mock 降级，确保问题暴露而非静默隐藏。
    使用 LRU 缓存（默认 128 条）避免重复合成相同文本。支持流式输出和自定义声音管理。

    Attributes:
        _cache: LRU 缓存，key 为请求参数的 SHA256 哈希
        _voices: 可用声音列表
        _edge_tts_available: edge-tts 库是否可用
    """

    def __init__(self, cache_size: int = MAX_CACHE_SIZE):
        """初始化 TTS 控制器。

        Args:
            cache_size: 缓存最大条目数，默认 128

        Raises:
            ActionError: edge-tts 未安装时抛出
        """
        self._cache: OrderedDict[str, TTSOutput] = OrderedDict()
        self._cache_size = cache_size
        self._voices = _TTS_VOICES.copy()
        self._edge_tts_available = False

        try:
            import edge_tts  # noqa: F401
            self._edge_tts_available = True
        except ImportError:
            self._edge_tts_available = False

        if not self._edge_tts_available:
            raise ActionError(
                "edge-tts is not installed. Please install with: pip install edge-tts",
                error_code="E515",
            )

        logger.info(
            "tts_controller_initialized",
            cache_size=cache_size,
            edge_tts_available=self._edge_tts_available,
        )

    async def _synthesize_with_edge_tts(self, request: TTSRequest) -> bytes:
        """使用 edge-tts 合成语音"""
        import edge_tts

        voice = _EDGE_TTS_VOICE_MAP.get(request.voice_id, "zh-CN-XiaoxiaoNeural")

        # edge-tts 的 rate 参数格式为 "+0%", "+50%", "-50%"
        rate_str = f"{int((request.speed - 1.0) * 100):+d}%"
        # edge-tts 的 pitch 参数格式为 "+0Hz", "+50Hz", "-50Hz"
        pitch_str = f"{int((request.pitch - 1.0) * 50):+d}Hz"

        communicate = edge_tts.Communicate(
            text=request.text,
            voice=voice,
            rate=rate_str,
            pitch=pitch_str,
        )

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_data = audio_buffer.getvalue()

        if not audio_data:
            raise ActionError(
                "edge-tts produced empty audio",
                error_code="E514",
                details={"text": request.text[:50], "voice": voice},
            )

        return audio_data

    async def synthesize(self, request: TTSRequest) -> TTSOutput:
        """合成语音（非流式）。

        使用 edge-tts 合成语音，输出 mp3 格式。
        edge-tts 不可用或合成失败时直接抛出 ActionError。

        Args:
            request: TTS 请求参数（文本、声音、语速、音调、格式）

        Returns:
            TTSOutput 包含音频数据和元信息

        Raises:
            ActionError: edge-tts 不可用或合成失败
        """
        cache_key = hashlib.sha256(
            f"{request.text}:{request.voice_id}:{request.speed}:{request.pitch}:{request.format}".encode()
        ).hexdigest()

        if cache_key in self._cache:
            logger.info("tts_cache_hit", text=request.text[:30])
            return self._cache[cache_key]

        logger.info("tts_synthesize", text=request.text[:50], voice=request.voice_id)

        audio_data = await self._synthesize_with_edge_tts(request)

        # edge-tts 输出 mp3 格式
        output = TTSOutput(
            audio_data=audio_data,
            text=request.text,
            format="mp3",
            duration_ms=int(len(audio_data) / 32),
            voice_id=request.voice_id,
        )

        self._cache[cache_key] = output
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return output

    async def synthesize_stream(self, request: TTSRequest) -> AsyncGenerator[bytes, None]:
        """流式合成"""
        output = await self.synthesize(request)
        chunk_size = 1024
        for i in range(0, len(output.audio_data), chunk_size):
            yield output.audio_data[i : i + chunk_size]
            await asyncio.sleep(0.01)

    def get_voices(self) -> List[Dict[str, str]]:
        """获取可用声音列表"""
        return self._voices.copy()

    def add_voice(self, voice_info: Dict[str, str]) -> None:
        """添加自定义声音"""
        required = {"id", "name", "gender", "language"}
        if not required.issubset(voice_info.keys()):
            raise ActionError(
                f"Voice info must contain: {required}",
                error_code="E513",
                details={"provided": voice_info.keys(), "required": required},
            )
        self._voices.append(voice_info)
        logger.info("tts_voice_added", voice_id=voice_info["id"])



def get_gui_control(*args, **kwargs):
    """获取 GUIControl 实例（工厂函数）"""
    from nan_agent.action_room.gui_control import GUIControl
    return GUIControl(*args, **kwargs)



class ActionModule:
    """统一动作执行器，封装 TTSController 和 GUIControl。

    被 ActionRoom 用于提供 speak、take_screenshot、gui_click、gui_type 等工具。
    GUI 不可用时 self.gui 为 None，工具调用将抛出 ActionError。
    """

    def __init__(self, event_bus=None, config=None):
        self.tts = TTSController()
        try:
            self.gui = get_gui_control()
            logger.info("action_module_initialized", gui="available", tts="available")
        except Exception as e:
            logger.warning("action_module_gui_unavailable", error=str(e))
            self.gui = None

    async def shutdown(self) -> None:
        if self.gui is not None and hasattr(self.gui, "shutdown"):
            await self.gui.shutdown()
