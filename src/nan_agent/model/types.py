import base64
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    source_type: str
    source: str
    mime_type: str = "image/jpeg"

    def to_base64(self) -> str:
        if self.source_type == "path":
            with open(self.source, "rb") as f:
                raw = f.read()
            return base64.b64encode(raw).decode()
        return self.source

    def to_base64_data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.to_base64()}"


@dataclass
class AudioPart:
    source_type: str
    source: str
    mime_type: str = "audio/wav"

    def to_base64(self) -> str:
        if self.source_type == "path":
            with open(self.source, "rb") as f:
                raw = f.read()
            return base64.b64encode(raw).decode()
        return self.source


@dataclass
class VideoPart:
    source_type: str
    source: str
    mime_type: str = "video/mp4"


class MultiModalInput:
    def __init__(self):
        self.parts: list = []

    def add_text(self, text: str) -> None:
        self.parts.append(TextPart(text=text))

    def add_image_path(self, path: str, mime_type: str = "image/jpeg") -> None:
        self.parts.append(ImagePart(source_type="path", source=path, mime_type=mime_type))

    def add_image_base64(self, b64: str, mime_type: str = "image/jpeg") -> None:
        self.parts.append(ImagePart(source_type="base64", source=b64, mime_type=mime_type))

    def add_image_url(self, url: str) -> None:
        self.parts.append(ImagePart(source_type="url", source=url))

    def add_audio_path(self, path: str, mime_type: str = "audio/wav") -> None:
        self.parts.append(AudioPart(source_type="path", source=path, mime_type=mime_type))

    def add_audio_base64(self, b64: str, mime_type: str = "audio/wav") -> None:
        self.parts.append(AudioPart(source_type="base64", source=b64, mime_type=mime_type))

    def add_video_path(self, path: str, mime_type: str = "video/mp4") -> None:
        self.parts.append(VideoPart(source_type="path", source=path, mime_type=mime_type))

    def get_text(self) -> str:
        return "\n".join(p.text for p in self.parts if isinstance(p, TextPart))

    def get_images(self) -> list[ImagePart]:
        return [p for p in self.parts if isinstance(p, ImagePart)]

    def get_audios(self) -> list[AudioPart]:
        return [p for p in self.parts if isinstance(p, AudioPart)]

    def to_ollama_format(self) -> list[dict]:
        """转换为 Ollama Chat API 兼容的消息内容数组格式。

        文本转为 {"type": "text", "text": "..."}
        图片转为 {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}

        Returns:
            Ollama API messages content 列表
        """
        result: list[dict] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                result.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                result.append({
                    "type": "image_url",
                    "image_url": {"url": part.to_base64_data_url()},
                })
        return result


class MultiModalOutput:
    def __init__(self):
        self.text: str = ""
        self.audio_data: Optional[bytes] = None
        self.audio_mime: Optional[str] = None
        self.svg: Optional[str] = None

    def add_text(self, text: str) -> None:
        self.text = text

    def add_audio(self, data: bytes, mime: str) -> None:
        self.audio_data = data
        self.audio_mime = mime

    def add_svg(self, svg: str) -> None:
        self.svg = svg

    def is_empty(self) -> bool:
        return (
            not self.text
            and self.audio_data is None
            and self.svg is None
        )
