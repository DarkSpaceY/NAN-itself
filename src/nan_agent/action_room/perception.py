"""
感知系统 - 多模态环境感知（视觉 + 听觉 + 语音识别）

提供摄像头图像采集、麦克风音频录制和语音识别（ASR）能力。
使用真实硬件后端（OpenCV cv2 / sounddevice / faster-whisper），
通过 Perception 协调器实现多模态同步观察。

核心组件：
- CameraCapture: 摄像头图像采集（cv2）
- MicrophoneCapture: 麦克风音频录制（sounddevice）
- SpeechRecognizer: 语音识别 ASR（faster-whisper）
- Perception: 多模态感知协调器
- CameraInput/AudioInput/ASRResult: 感知数据模型
"""

import asyncio
import io
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


class SensorState(Enum):
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    ERROR = "error"


@dataclass
class CameraInput:
    frames: List[Any]
    device_id: str = ""
    resolution: Tuple[int, int] = (640, 480)
    fps: float = 30.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AudioInput:
    audio_data: bytes
    sample_rate: int = 16000
    channels: int = 1
    duration_ms: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.duration_ms == 0.0 and self.sample_rate > 0 and self.channels > 0:
            bytes_per_sample = 2
            total_samples = len(self.audio_data) / (bytes_per_sample * self.channels)
            self.duration_ms = (total_samples / self.sample_rate) * 1000


@dataclass
class ASRResult:
    text: str
    language: str = "en"
    confidence: float = 1.0
    is_final: bool = True


class CameraCapture:
    """摄像头图像采集器。

    使用 OpenCV (cv2) 进行真实摄像头图像采集，提供单帧捕获和流式捕获两种模式。
    通过 SensorState 状态机管理摄像头生命周期。

    Attributes:
        state: 传感器状态（CLOSED/OPENING/OPEN/CLOSING/ERROR）
        resolution: 当前分辨率
        fps: 当前帧率
        device_id: 设备标识
    """

    def __init__(
        self,
        default_resolution: Tuple[int, int] = (640, 480),
        default_fps: float = 30.0,
    ):
        """初始化摄像头。

        Args:
            default_resolution: 默认分辨率（宽, 高）
            default_fps: 默认帧率
        """
        self._state = SensorState.CLOSED
        self._device_id: str = ""
        self._current_resolution = default_resolution
        self._current_fps = default_fps
        self._cap: Any = None
        self._streaming = False
        self._frame_counter = 0

    @property
    def state(self) -> SensorState:
        return self._state

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._current_resolution

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def device_id(self) -> str:
        return self._device_id

    @staticmethod
    async def list_cameras() -> List[Dict[str, Any]]:
        import cv2
        real_devices = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                real_devices.append({
                    "id": f"cam-{i}",
                    "name": f"Camera {i}",
                    "max_resolution": (int(w), int(h)),
                    "max_fps": fps if fps > 0 else 30,
                })
            cap.release()
        if not real_devices:
            logger.warning("no_real_cameras_found")
        return real_devices

    async def open(self, device_id: str = "cam-0", resolution: Optional[Tuple[int, int]] = None, fps: Optional[float] = None) -> None:
        if self._state == SensorState.OPEN:
            raise ActionError(
                "Camera is already open",
                error_code="E510",
                details={"device_id": self._device_id},
            )
        try:
            import cv2
        except ImportError:
            raise ActionError(
                "OpenCV (cv2) is not installed. Install it with: pip install opencv-python",
                error_code="E513",
                details={},
            )
        if not device_id.startswith("cam-"):
            raise ActionError(
                f"Invalid camera device_id '{device_id}'. Expected format: 'cam-<index>' (e.g. 'cam-0')",
                error_code="E514",
                details={"device_id": device_id},
            )
        try:
            index = int(device_id.split("-", 1)[1])
        except (ValueError, IndexError):
            raise ActionError(
                f"Invalid camera device_id '{device_id}'. Expected format: 'cam-<index>' (e.g. 'cam-0')",
                error_code="E514",
                details={"device_id": device_id},
            )
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise ActionError(
                f"Camera at index {index} could not be opened",
                error_code="E515",
                details={"device_id": device_id, "index": index},
            )
        self._state = SensorState.OPENING
        self._device_id = device_id
        self._cap = cap
        if resolution is not None:
            self._current_resolution = resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if fps is not None:
            self._current_fps = fps
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._frame_counter = 0
        self._state = SensorState.OPEN
        logger.info("camera_opened", device_id=device_id, resolution=self._current_resolution, fps=self._current_fps)

    async def close(self) -> None:
        if self._state != SensorState.OPEN:
            return
        self._state = SensorState.CLOSING
        self._streaming = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._state = SensorState.CLOSED
        logger.info("camera_closed", device_id=self._device_id)

    async def capture_frame(self) -> CameraInput:
        if self._state != SensorState.OPEN:
            raise ActionError(
                "Camera is not open",
                error_code="E511",
                details={"state": self._state.value},
            )
        ret, frame = self._cap.read()
        if not ret:
            raise ActionError(
                "Failed to capture frame from camera",
                error_code="E516",
                details={"device_id": self._device_id},
            )
        import cv2
        _, buffer = cv2.imencode('.jpg', frame)
        self._frame_counter += 1
        return CameraInput(
            frames=[buffer.tobytes()],
            device_id=self._device_id,
            resolution=self._current_resolution,
            fps=self._current_fps,
        )

    async def start_stream(self) -> AsyncGenerator[CameraInput, None]:
        if self._state != SensorState.OPEN:
            raise ActionError(
                "Camera is not open",
                error_code="E511",
                details={"state": self._state.value},
            )
        self._streaming = True
        try:
            while self._streaming:
                yield await self.capture_frame()
                interval = 1.0 / self._current_fps
                await asyncio.sleep(interval)
        finally:
            self._streaming = False

    async def stop_stream(self) -> None:
        self._streaming = False
        logger.info("camera_stream_stopped", device_id=self._device_id)


class MicrophoneCapture:
    """麦克风音频采集器。

    使用 sounddevice 进行真实麦克风音频采集，提供指定时长录制和流式录制两种模式。

    Attributes:
        state: 传感器状态
        sample_rate: 采样率
        channels: 声道数
        device_id: 设备标识
    """

    def __init__(
        self,
        default_sample_rate: int = 16000,
        default_channels: int = 1,
    ):
        """初始化麦克风。

        Args:
            default_sample_rate: 默认采样率（Hz），默认 16000
            default_channels: 默认声道数，默认 1（单声道）
        """
        self._state = SensorState.CLOSED
        self._device_id: str = ""
        self._current_sample_rate = default_sample_rate
        self._current_channels = default_channels
        self._stream: Any = None
        self._streaming = False
        self._audio_counter = 0

    @property
    def state(self) -> SensorState:
        return self._state

    @property
    def sample_rate(self) -> int:
        return self._current_sample_rate

    @property
    def channels(self) -> int:
        return self._current_channels

    @property
    def device_id(self) -> str:
        return self._device_id

    @staticmethod
    async def list_devices() -> List[Dict[str, Any]]:
        import sounddevice
        real_devices = []
        dev_list = sounddevice.query_devices()
        for i, dev in enumerate(dev_list):
            if dev.get("max_input_channels", 0) > 0:
                real_devices.append({
                    "id": f"mic-{i}",
                    "name": dev.get("name", f"Microphone {i}"),
                    "max_sample_rate": int(dev.get("default_samplerate", 44100)),
                    "max_channels": dev.get("max_input_channels", 1),
                })
        if not real_devices:
            logger.warning("no_real_microphones_found")
        return real_devices

    async def open(self, device_id: str = "mic-0", sample_rate: Optional[int] = None, channels: Optional[int] = None) -> None:
        if self._state == SensorState.OPEN:
            raise ActionError(
                "Microphone is already open",
                error_code="E520",
                details={"device_id": self._device_id},
            )
        try:
            import sounddevice
        except ImportError:
            raise ActionError(
                "sounddevice is not installed. Install it with: pip install sounddevice",
                error_code="E523",
                details={},
            )
        if not device_id.startswith("mic-"):
            raise ActionError(
                f"Invalid microphone device_id '{device_id}'. Expected format: 'mic-<index>' (e.g. 'mic-0')",
                error_code="E524",
                details={"device_id": device_id},
            )
        try:
            index = int(device_id.split("-", 1)[1])
        except (ValueError, IndexError):
            raise ActionError(
                f"Invalid microphone device_id '{device_id}'. Expected format: 'mic-<index>' (e.g. 'mic-0')",
                error_code="E524",
                details={"device_id": device_id},
            )
        self._state = SensorState.OPENING
        self._device_id = device_id
        if sample_rate is not None:
            self._current_sample_rate = sample_rate
        if channels is not None:
            self._current_channels = channels
        self._audio_counter = 0
        try:
            self._stream = sounddevice.InputStream(
                device=index,
                samplerate=self._current_sample_rate,
                channels=self._current_channels,
            )
            self._stream.start()
        except Exception as e:
            raise ActionError(
                f"Failed to open microphone at index {index}: {e}",
                error_code="E525",
                details={"device_id": device_id, "index": index, "error": str(e)},
            )
        self._state = SensorState.OPEN
        logger.info("microphone_opened", device_id=device_id, sample_rate=self._current_sample_rate, channels=self._current_channels)

    async def close(self) -> None:
        if self._state != SensorState.OPEN:
            return
        self._state = SensorState.CLOSING
        self._streaming = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._state = SensorState.CLOSED
        logger.info("microphone_closed", device_id=self._device_id)

    async def record(self, duration_ms: float = 1000.0) -> AudioInput:
        if self._state != SensorState.OPEN:
            raise ActionError(
                "Microphone is not open",
                error_code="E521",
                details={"state": self._state.value},
            )
        import sounddevice
        import numpy as np
        num_samples = int(self._current_sample_rate * duration_ms / 1000.0)
        recording = sounddevice.rec(
            frames=num_samples,
            samplerate=self._current_sample_rate,
            channels=self._current_channels,
            dtype='int16',
            blocking=True,
        )
        audio_data = np.int16(recording).tobytes()
        self._audio_counter += 1
        return AudioInput(
            audio_data=audio_data,
            sample_rate=self._current_sample_rate,
            channels=self._current_channels,
            duration_ms=duration_ms,
        )

    async def start_stream(self, chunk_duration_ms: float = 100.0) -> AsyncGenerator[AudioInput, None]:
        if self._state != SensorState.OPEN:
            raise ActionError(
                "Microphone is not open",
                error_code="E521",
                details={"state": self._state.value},
            )
        self._streaming = True
        try:
            while self._streaming:
                yield await self.record(duration_ms=chunk_duration_ms)
                await asyncio.sleep(chunk_duration_ms / 1000.0)
        finally:
            self._streaming = False

    async def stop_stream(self) -> None:
        self._streaming = False
        logger.info("microphone_stream_stopped", device_id=self._device_id)


class SpeechRecognizer:
    """语音识别器（ASR）。

    使用 faster-whisper 进行真实语音识别，提供单次转写和流式转写两种模式，
    以及语言检测功能。

    Attributes:
        model_name: faster-whisper 模型名称（如 "base", "small", "medium"）
        language: 目标语言代码
        is_loaded: 模型是否已加载
    """

    def __init__(
        self,
        model_name: str = "base",
        language: str = "en",
    ):
        """初始化语音识别器。

        Args:
            model_name: faster-whisper 模型名称，默认 "base"
            language: 目标语言代码（如 "en", "zh"）
        """
        self._model_name = model_name
        self._language = language
        self._loaded = False
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, value: str) -> None:
        self._language = value

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        if self._loaded:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ActionError(
                "faster-whisper is not installed. Install it with: pip install faster-whisper",
                error_code="E532",
                details={},
            )
        self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
        self._loaded = True
        logger.info("asr_model_loaded", model=self._model_name, language=self._language)

    async def unload(self) -> None:
        self._loaded = False
        self._model = None
        logger.info("asr_model_unloaded", model=self._model_name)

    async def transcribe(self, audio: AudioInput) -> ASRResult:
        if not self._loaded:
            await self.load()
        if self._model is None:
            raise ActionError(
                "ASR model not initialized",
                error_code="E530",
                details={"model": self._model_name},
            )
        import numpy as np
        audio_np = np.frombuffer(audio.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.channels > 1:
            audio_np = audio_np.reshape(-1, audio.channels).mean(axis=1)
        segments, info = self._model.transcribe(audio_np, language=self._language)
        full_text = " ".join(segment.text.strip() for segment in segments)
        return ASRResult(
            text=full_text,
            language=info.language if info.language else self._language,
            confidence=info.language_probability if info.language_probability else 0.0,
            is_final=True,
        )

    async def transcribe_stream(self, audio_stream: AsyncGenerator[AudioInput, None]) -> AsyncGenerator[ASRResult, None]:
        if not self._loaded:
            await self.load()
        async for chunk in audio_stream:
            yield await self.transcribe(chunk)

    async def detect_language(self, audio: AudioInput) -> str:
        result = await self.transcribe(audio)
        return result.language


class Perception:
    """多模态感知协调器。

    协调摄像头、麦克风和语音识别器的协同工作，提供同步多模态观察能力。
    支持构造统一的多模态输入结构供上层推理引擎使用。

    Attributes:
        camera: 摄像头实例
        microphone: 麦克风实例
        asr: 语音识别器实例
    """

    def __init__(
        self,
        camera: Optional[CameraCapture] = None,
        microphone: Optional[MicrophoneCapture] = None,
        asr: Optional[SpeechRecognizer] = None,
        event_bus: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        """初始化感知系统。

        Args:
            camera: 摄像头实例，None 需后续 initialize() 创建
            microphone: 麦克风实例，None 需后续 initialize() 创建
            asr: 语音识别器实例，None 需后续 initialize() 创建
            event_bus: 事件总线（预留）
            config: 配置字典（预留）
        """
        self._camera = camera
        self._microphone = microphone
        self._asr = asr
        self._streaming = False
        self._event_bus = event_bus
        self._config = config

    @property
    def camera(self) -> Optional[CameraCapture]:
        return self._camera

    @property
    def microphone(self) -> Optional[MicrophoneCapture]:
        return self._microphone

    @property
    def asr(self) -> Optional[SpeechRecognizer]:
        return self._asr

    async def initialize(
        self,
        camera_device_id: str = "cam-0",
        camera_resolution: Optional[Tuple[int, int]] = None,
        camera_fps: Optional[float] = None,
        mic_device_id: str = "mic-0",
        mic_sample_rate: Optional[int] = None,
        mic_channels: Optional[int] = None,
        asr_model: str = "base",
        asr_language: str = "en",
    ) -> None:
        if self._camera is None:
            self._camera = CameraCapture()
        if self._microphone is None:
            self._microphone = MicrophoneCapture()
        if self._asr is None:
            self._asr = SpeechRecognizer(model_name=asr_model, language=asr_language)

        await asyncio.gather(
            self._camera.open(device_id=camera_device_id, resolution=camera_resolution, fps=camera_fps),
            self._microphone.open(device_id=mic_device_id, sample_rate=mic_sample_rate, channels=mic_channels),
            self._asr.load(),
        )
        logger.info("perception_initialized")

    async def shutdown(self) -> None:
        self._streaming = False
        tasks = []
        if self._camera is not None:
            tasks.append(self._camera.close())
        if self._microphone is not None:
            tasks.append(self._microphone.close())
        if self._asr is not None:
            tasks.append(self._asr.unload())
        if tasks:
            await asyncio.gather(*tasks)
        logger.info("perception_shutdown")

    async def capture_visual(self) -> CameraInput:
        if self._camera is None:
            raise ActionError(
                "Camera not configured",
                error_code="E512",
                details={},
            )
        return await self._camera.capture_frame()

    async def capture_audio(self, duration_ms: float = 1000.0) -> AudioInput:
        if self._microphone is None:
            raise ActionError(
                "Microphone not configured",
                error_code="E522",
                details={},
            )
        return await self._microphone.record(duration_ms=duration_ms)

    async def recognize_speech(self, audio: AudioInput) -> ASRResult:
        if self._asr is None:
            raise ActionError(
                "ASR not configured",
                error_code="E531",
                details={},
            )
        return await self._asr.transcribe(audio)

    async def observe(self) -> Dict[str, Any]:
        visual = None
        audio = None
        speech = None

        try:
            if self._camera is not None and self._camera.state == SensorState.OPEN:
                visual = await self._camera.capture_frame()
        except Exception as e:
            logger.warning("visual_capture_failed", error=str(e))

        try:
            if self._microphone is not None and self._microphone.state == SensorState.OPEN:
                audio = await self._microphone.record(duration_ms=200.0)
        except Exception as e:
            logger.warning("audio_capture_failed", error=str(e))

        try:
            if audio is not None and self._asr is not None and self._asr.is_loaded:
                speech = await self._asr.transcribe(audio)
        except Exception as e:
            logger.warning("asr_failed", error=str(e))

        return {
            "timestamp": time.time(),
            "visual": visual,
            "audio": audio,
            "speech": speech,
        }

    async def perception_stream(self, interval_ms: float = 200.0) -> AsyncGenerator[Dict[str, Any], None]:
        self._streaming = True
        try:
            while self._streaming:
                yield await self.observe()
                await asyncio.sleep(interval_ms / 1000.0)
        finally:
            self._streaming = False

    async def stop_stream(self) -> None:
        self._streaming = False
        logger.info("perception_stream_stopped")

    def construct_multimodal_input(self, visual: Optional[CameraInput] = None, audio: Optional[AudioInput] = None, speech: Optional[ASRResult] = None) -> Dict[str, Any]:
        multimodal: Dict[str, Any] = {
            "timestamp": time.time(),
            "modalities": [],
        }

        if visual is not None:
            multimodal["visual"] = {
                "device_id": visual.device_id,
                "resolution": visual.resolution,
                "fps": visual.fps,
                "frame_count": len(visual.frames),
                "timestamp": visual.timestamp,
            }
            multimodal["modalities"].append("visual")

        if audio is not None:
            multimodal["audio"] = {
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "duration_ms": audio.duration_ms,
                "data_size_bytes": len(audio.audio_data),
                "timestamp": audio.timestamp,
            }
            multimodal["modalities"].append("audio")

        if speech is not None:
            multimodal["speech"] = {
                "text": speech.text,
                "language": speech.language,
                "confidence": speech.confidence,
                "is_final": speech.is_final,
            }
            multimodal["modalities"].append("speech")

        return multimodal

    @staticmethod
    async def list_available_sensors() -> Dict[str, List[Dict[str, Any]]]:
        cameras, mics = await asyncio.gather(
            CameraCapture.list_cameras(),
            MicrophoneCapture.list_devices(),
        )
        return {
            "cameras": cameras,
            "microphones": mics,
        }