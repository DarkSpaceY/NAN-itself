"""
NAN-Agent 会话管理系统

提供多会话的创建、加载、保存、切换、删除、重命名和导出功能。
会话数据以 JSON 文件持久化存储，支持消息记录和元数据管理。

主要组件：
- Session: 单个会话的数据模型，包含消息列表和元数据
- SessionManager: 会话管理器，负责会话 CRUD 和持久化
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Session:
    """单个会话的数据模型。

    Attributes:
        session_id: 唯一会话标识符（UUID）
        name: 会话名称，默认 "Session-{前8位UUID}"
        created_at: 创建时间（ISO 8601 格式）
        updated_at: 最后更新时间（ISO 8601 格式）
        messages: 消息列表，每条消息包含 timestamp、role、content
        metadata: 附加元数据字典
    """
    session_id: str
    name: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息并更新 updated_at 时间戳。"""
        self.messages.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "role": role,
                "content": content,
            }
        )
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """将会话序列化为字典，用于 JSON 持久化。"""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """从字典反序列化创建 Session 实例。"""
        return cls(
            session_id=data["session_id"],
            name=data["name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
        )


class SessionManager:
    """会话管理器，负责会话的 CRUD 操作和 JSON 文件持久化。

    数据存储结构：
    - 每个会话保存为 {sessions_dir}/{session_id}.json 文件
    - 内存中通过 _sessions 字典维护 session_id → Session 映射
    - _current_session_id 追踪当前活跃会话

    生命周期：
    - 初始化时自动调用 load_all() 加载所有已有会话文件
    - new_session() 创建并自动持久化
    - delete_session() 同时清理内存和文件

    Args:
        sessions_dir: 会话数据存储目录，默认 "./data/sessions"
    """

    def __init__(self, sessions_dir: str = "./data/sessions") -> None:
        self._sessions_dir = Path(sessions_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._current_session_id: Optional[str] = None
        self.load_all()

    def _session_path(self, session_id: str) -> Path:
        """返回会话 JSON 文件的完整路径。"""
        return self._sessions_dir / f"{session_id}.json"

    def create_session(self, name: Optional[str] = None) -> Session:
        """创建新会话并返回 Session 实例（不自动设为当前会话）。"""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        session = Session(
            session_id=session_id,
            name=name or f"Session-{session_id[:8]}",
            created_at=now,
            updated_at=now,
        )
        self._sessions[session_id] = session
        logger.info("session_created", session_id=session_id, name=session.name)
        return session

    def new_session(self, name: Optional[str] = None) -> Session:
        """创建新会话，设为当前会话，并持久化保存。"""
        session = self.create_session(name)
        self._current_session_id = session.session_id
        self.save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """按 ID 获取会话，不存在返回 None。"""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        """列出所有会话，按更新时间倒序排列。"""
        return sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )

    def get_current(self) -> Optional[Session]:
        """获取当前活跃会话，无则返回 None。"""
        if self._current_session_id is None:
            return None
        return self._sessions.get(self._current_session_id)

    def set_current(self, session_id: str) -> None:
        """设置当前活跃会话，若会话不存在则抛出 KeyError。"""
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        self._current_session_id = session_id

    def save_session(self, session: Session) -> None:
        """将会话持久化到 JSON 文件。"""
        session.updated_at = datetime.now(timezone.utc).isoformat()
        path = self._session_path(session.session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        logger.debug("session_saved", session_id=session.session_id)

    def save_current(self) -> None:
        """保存当前活跃会话，若无当前会话则抛出 RuntimeError。"""
        session = self.get_current()
        if session is None:
            raise RuntimeError("No current session to save")
        self.save_session(session)

    def load_session(self, session_id: str) -> Optional[Session]:
        """从 JSON 文件加载指定会话到内存，文件不存在返回 None。"""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = Session.from_dict(data)
        self._sessions[session_id] = session
        logger.debug("session_loaded", session_id=session_id)
        return session

    def load_all(self) -> None:
        """加载存储目录中的所有会话 JSON 文件到内存。"""
        for path in self._sessions_dir.glob("*.json"):
            session_id = path.stem
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            session = Session.from_dict(data)
            self._sessions[session_id] = session
        logger.info("sessions_loaded", count=len(self._sessions))

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话（内存和文件），若该会话为当前会话则清空当前会话引用。"""
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        if self._current_session_id == session_id:
            self._current_session_id = None
        logger.info("session_deleted", session_id=session_id)
        return True

    def rename_session(self, session_id: str, new_name: str) -> Optional[Session]:
        """重命名指定会话，返回更新后的 Session 或 None（会话不存在时）。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.name = new_name
        session.updated_at = datetime.now(timezone.utc).isoformat()
        logger.info("session_renamed", session_id=session_id, new_name=new_name)
        return session

    def export_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """导出指定会话的完整字典表示，不存在返回 None。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.to_dict()