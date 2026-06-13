"""多模态附件数据结构和工具函数。

支持图片、音频、视频附件的序列化和从 MultiModalInput 提取。
"""

import base64
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from nan_agent.model.types import AudioPart, ImagePart, MultiModalInput, VideoPart


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MediaAttachment:
    """多模态附件数据类。

    Attributes:
        blob_key: BlobStore 中的键
        media_type: "image" / "audio" / "video"
        mime_type: MIME 类型，如 "image/png"
        description: 附件描述
        size_bytes: 数据大小（字节）
    """
    blob_key: str
    media_type: str
    mime_type: str = ""
    description: str = ""
    size_bytes: int = 0

    @property
    def is_image(self) -> bool:
        return self.media_type == "image"

    @property
    def is_audio(self) -> bool:
        return self.media_type == "audio"

    @property
    def is_video(self) -> bool:
        return self.media_type == "video"

    def to_dict(self) -> dict:
        return {
            "blob_key": self.blob_key,
            "media_type": self.media_type,
            "mime_type": self.mime_type,
            "description": self.description,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MediaAttachment":
        return cls(
            blob_key=data.get("blob_key", ""),
            media_type=data.get("media_type", ""),
            mime_type=data.get("mime_type", ""),
            description=data.get("description", ""),
            size_bytes=data.get("size_bytes", 0),
        )


def extract_attachments_from_multimodal(mm_input: MultiModalInput) -> list[MediaAttachment]:
    """从 MultiModalInput 中提取多模态附件列表。

    Args:
        mm_input: 多模态输入

    Returns:
        MediaAttachment 列表
    """
    attachments: list[MediaAttachment] = []
    for part in mm_input.parts:
        if isinstance(part, ImagePart):
            data = part.to_base64()
            blob_key = hashlib.sha256(data.encode()).hexdigest()[:16]
            attachments.append(MediaAttachment(
                blob_key=blob_key,
                media_type="image",
                mime_type=part.mime_type,
                description="Image attachment",
                size_bytes=len(data.encode()),
            ))
        elif isinstance(part, AudioPart):
            data = part.to_base64()
            blob_key = hashlib.sha256(data.encode()).hexdigest()[:16]
            attachments.append(MediaAttachment(
                blob_key=blob_key,
                media_type="audio",
                mime_type=part.mime_type,
                description="Audio attachment",
                size_bytes=len(data.encode()),
            ))
        elif isinstance(part, VideoPart):
            attachments.append(MediaAttachment(
                blob_key=_new_id(),
                media_type="video",
                mime_type=part.mime_type,
                description="Video attachment",
            ))
    return attachments


def attachments_to_metadata(attachments: list[MediaAttachment]) -> list[dict]:
    """将附件列表转换为可序列化的字典列表。

    Args:
        attachments: MediaAttachment 列表

    Returns:
        字典列表
    """
    return [att.to_dict() for att in attachments]


def attachments_from_metadata(data: list[dict]) -> list[MediaAttachment]:
    """从字典列表恢复附件列表。

    Args:
        data: 字典列表

    Returns:
        MediaAttachment 列表
    """
    return [MediaAttachment.from_dict(d) for d in data]
