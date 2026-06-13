"""
Action Room - 动作执行模块

提供：
- TTSController: 文本转语音
- GUIControl: 基于 OmniParser 的屏幕操作
- OmniParserPerception: 屏幕视觉感知
"""

from nan_agent.action_room.interface import (
    ActionRequest,
    ActionResult,
    ActionRoom,
    ComponentStatus,
)

# GUI 相关（基于 OmniParser）
from nan_agent.action_room.gui_control import GUIControl, GUIResult
from nan_agent.action_room.screen_perception import (
    OmniParserPerception,
    ParsedScreen,
    UIElement,
    get_perception,
)

# TTS 相关
from nan_agent.action_room.action import (
    TTSController,
    TTSOutput,
    TTSRequest,
)

__all__ = [
    # Interface
    "ActionRoom",
    "ActionRequest",
    "ActionResult",
    "ComponentStatus",
    # GUI Control (OmniParser-based)
    "GUIControl",
    "GUIResult",
    "OmniParserPerception",
    "ParsedScreen",
    "UIElement",
    "get_perception",
    # TTS
    "TTSController",
    "TTSOutput",
    "TTSRequest",
]
