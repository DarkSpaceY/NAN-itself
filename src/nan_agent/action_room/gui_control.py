"""
GUI Control Module - 基于 OmniParser 纯视觉的 GUI 控制

完全基于视觉解析，无需 Accessibility API。
支持：截图 → 解析元素 → 点击/输入/滚动等操作
"""

import base64
import io
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nan_agent.logging.logger import get_logger
from nan_agent.action_room.screen_perception import (
    OmniParserPerception,
    ParsedScreen,
    UIElement,
    get_perception,
)

logger = get_logger(__name__)


@dataclass
class GUIResult:
    """GUI 操作结果"""
    success: bool
    screenshot: Optional[bytes] = None
    parsed_screen: Optional[ParsedScreen] = None
    error: Optional[str] = None


class GUIControl:
    """
    基于 OmniParser 的 GUI 控制器
    
    工作流程：
    1. capture_screenshot() → 获取屏幕截图
    2. perception.parse() → 解析 UI 元素
    3. click/type/scroll... → 基于元素 ID 或描述执行操作
    """
    
    def __init__(
        self,
        screenshot_enabled: bool = True,
        perception: Optional[OmniParserPerception] = None,
    ):
        self._screenshot_enabled = screenshot_enabled
        self._is_macos = platform.system() == "Darwin"
        
        # 屏幕感知引擎
        self._perception = perception or get_perception()
        
        # 缓存最后一次解析结果
        self._last_parsed_screen: Optional[ParsedScreen] = None
        
        logger.info("gui_control_initialized", 
                   screenshot_enabled=screenshot_enabled,
                   macos=self._is_macos)
    
    @property
    def screenshot_enabled(self) -> bool:
        return self._screenshot_enabled
    
    @screenshot_enabled.setter
    def screenshot_enabled(self, value: bool) -> None:
        self._screenshot_enabled = value

    # ═══════════════════════════════════════════════════════════
    # 屏幕感知
    # ═══════════════════════════════════════════════════════════
    
    def capture_screenshot(self, fast: bool = False) -> GUIResult:
        """截取屏幕并 OmniParser 解析。fast=True 只跑 YOLO 不跑 Florence2。"""
        if not self._screenshot_enabled:
            return GUIResult(success=False, error="Screenshot disabled")
        
        screenshot_bytes = self._take_screenshot()
        if screenshot_bytes is None:
            return GUIResult(success=False, error="Failed to capture screenshot")
        
        try:
            parsed = self._perception.parse_screenshot_bytes(screenshot_bytes, fast=fast)
            self._last_parsed_screen = parsed
            logger.info("screenshot_parsed",
                       size=len(screenshot_bytes),
                       elements=len(parsed.elements),
                       yolo_ms=int(parsed.yolo_time_ms),
                       caption_ms=int(parsed.caption_time_ms))
            return GUIResult(success=True, screenshot=screenshot_bytes, parsed_screen=parsed)
        except Exception as e:
            logger.error("screenshot_parse_failed", error=str(e))
            return GUIResult(
                success=True,  # 截图成功
                screenshot=screenshot_bytes,
                error=f"Parse failed: {e}",
            )
    
    def get_screen_size(self) -> str:
        """获取当前屏幕分辨率。

        Raises:
            ActionError: 无法获取屏幕分辨率时抛出
        """
        try:
            import pyautogui
            w, h = pyautogui.size()
            return f"{w}x{h}"
        except ImportError as e:
            logger.debug("gui_pyautogui_not_available", error=str(e))
        if self._is_macos:
            try:
                result = subprocess.run(
                    ["osascript", "-e",
                     "tell application \"System Events\" to get {screen resolution's width, screen resolution's height}"],
                    capture_output=True, text=True, timeout=5,
                )
                parts = result.stdout.strip().split(", ")
                if len(parts) == 2:
                    return f"{parts[0]}x{parts[1]}"
            except Exception as e:
                logger.debug("gui_macos_resolution_failed", error=str(e))
        raise ActionError(
            "Cannot determine screen resolution. Install pyautogui: pip install pyautogui",
            error_code="E540",
        )

    def _take_screenshot(self) -> bytes:
        """底层截图实现

        Raises:
            ActionError: 截图失败时抛出
        """
        if not self._is_macos:
            raise ActionError(
                "Screenshot is only supported on macOS. Install pyautogui for cross-platform support: pip install pyautogui",
                error_code="E541",
            )

        try:
            fd, path = tempfile.mkstemp(suffix=".png", prefix="nan_screenshot_")
            os.close(fd)
            subprocess.run(
                ["screencapture", "-x", "-t", "png", path],
                check=True, capture_output=True, timeout=10,
            )
            with open(path, "rb") as f:
                data = f.read()
            os.unlink(path)
            return data
        except Exception as e:
            raise ActionError(
                f"Screenshot capture failed: {e}",
                error_code="E542",
            ) from e
    
    def parse_current_screen(self) -> GUIResult:
        """仅解析当前屏幕（不返回截图字节）"""
        return self.capture_screenshot()
    
    # ═══════════════════════════════════════════════════════════
    # 元素查找
    # ═══════════════════════════════════════════════════════════
    
    def find_element(self, description: str) -> Optional[UIElement]:
        """
        根据描述查找元素
        
        Args:
            description: 元素描述关键词，如 "Save button", "File menu"
        
        Returns:
            最匹配的元素，未找到返回 None
        """
        if self._last_parsed_screen is None:
            result = self.capture_screenshot()
            if not result.success or result.parsed_screen is None:
                return None
        
        # 优先查找可交互元素
        matches = self._last_parsed_screen.find_by_content(description, interactive_only=True)
        if matches:
            return matches[0]
        
        # 如果没有，查找所有元素
        matches = self._last_parsed_screen.find_by_content(description, interactive_only=False)
        return matches[0] if matches else None
    
    def find_elements(self, description: str) -> List[UIElement]:
        """查找所有匹配描述的元素"""
        if self._last_parsed_screen is None:
            result = self.capture_screenshot()
            if not result.success or result.parsed_screen is None:
                return []
        
        return self._last_parsed_screen.find_by_content(description, interactive_only=True)
    
    def get_element_by_id(self, element_id: int) -> Optional[UIElement]:
        """根据 ID 获取元素"""
        if self._last_parsed_screen is None:
            return None
        return self._last_parsed_screen.get_element_by_id(element_id)
    
    def list_interactive_elements(self) -> List[UIElement]:
        """列出所有可交互元素"""
        if self._last_parsed_screen is None:
            result = self.capture_screenshot()
            if not result.success or result.parsed_screen is None:
                return []
        
        return self._last_parsed_screen.get_interactive_elements()
    
    # ═══════════════════════════════════════════════════════════
    # 鼠标操作
    # ═══════════════════════════════════════════════════════════
    
    def click(self, target: str) -> GUIResult:
        """
        点击元素
        
        Args:
            target: 元素描述（如 "Save button"）或元素 ID（如 "#5"）
        """
        element = self._resolve_target(target)
        if element is None:
            return GUIResult(success=False, error=f"Element not found: {target}")
        
        x, y = element.center_px()
        success, error = self._execute_click(x, y)
        
        return GUIResult(
            success=success,
            error=error,
        )

    def double_click(self, target: str) -> GUIResult:
        """双击元素"""
        element = self._resolve_target(target)
        if element is None:
            return GUIResult(success=False, error=f"Element not found: {target}")
        
        x, y = element.center_px()
        success, error = self._execute_double_click(x, y)
        
        return GUIResult(
            success=success,
            error=error,
        )

    def right_click(self, target: str) -> GUIResult:
        """右键点击元素"""
        element = self._resolve_target(target)
        if element is None:
            return GUIResult(success=False, error=f"Element not found: {target}")
        
        x, y = element.center_px()
        success, error = self._execute_right_click(x, y)
        
        return GUIResult(
            success=success,
            error=error,
        )

    def move_to(self, target: str) -> GUIResult:
        """移动鼠标到元素"""
        element = self._resolve_target(target)
        if element is None:
            return GUIResult(success=False, error=f"Element not found: {target}")
        
        x, y = element.center_px()
        success, error = self._execute_move(x, y)
        
        return GUIResult(
            success=success,
            error=error,
        )

    def drag(self, from_target: str, to_target: str) -> GUIResult:
        """拖拽元素"""
        from_element = self._resolve_target(from_target)
        to_element = self._resolve_target(to_target)
        
        if from_element is None:
            return GUIResult(success=False, error=f"Source element not found: {from_target}")
        if to_element is None:
            return GUIResult(success=False, error=f"Target element not found: {to_target}")
        
        x1, y1 = from_element.center_px()
        x2, y2 = to_element.center_px()
        success, error = self._execute_drag(x1, y1, x2, y2)
        
        return GUIResult(
            success=success,
            error=error,
        )

    def scroll(self, direction: str = "down", amount: int = 3) -> GUIResult:
        """
        滚动屏幕
        
        Args:
            direction: "up" | "down" | "left" | "right"
            amount: 滚动次数
        """
        success, error = self._execute_scroll(direction, amount)
        
        return GUIResult(
            success=success,
            error=error,
        )

    # ═══════════════════════════════════════════════════════════
    # 键盘操作
    # ═══════════════════════════════════════════════════════════
    
    def type_text(self, text: str, target: Optional[str] = None) -> GUIResult:
        """
        输入文本
        
        Args:
            text: 要输入的文本
            target: 可选，目标输入框描述
        """
        # 如果指定了目标，先点击
        if target:
            click_result = self.click(target)
            if not click_result.success:
                return click_result
            time.sleep(0.1)  # 等待焦点
        
        success, error = self._execute_type(text)
        
        return GUIResult(
            success=success,
            error=error,
        )

    def press_key(self, key: str) -> GUIResult:
        """
        按下单个按键
        
        Args:
            key: 按键名，如 "return", "escape", "tab", "up", "down", "f1", etc.
        """
        success, error = self._execute_press(key)
        
        return GUIResult(
            success=success,
            error=error,
        )

    def hotkey(self, *keys: str) -> GUIResult:
        """
        按下组合键
        
        Args:
            *keys: 按键列表，如 "command", "c"
        """
        success, error = self._execute_hotkey(keys)
        
        return GUIResult(
            success=success,
            error=error,
        )

    # ═══════════════════════════════════════════════════════════
    # 窗口操作
    # ═══════════════════════════════════════════════════════════
    
    def focus_window(self, window_title: str) -> GUIResult:
        """激活窗口"""
        success, error = self._execute_osascript(
            f'tell application "{window_title}" to activate'
        )
        
        return GUIResult(
            success=success,
            error=error,
        )

    def list_windows(self) -> List[Dict[str, str]]:
        """列出所有窗口"""
        try:
            result = subprocess.run(
                ["osascript", "-e", 
                 'tell application "System Events" to get name of every application process whose background only is false'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                apps = [name.strip() for name in result.stdout.split(",")]
                return [{"name": name} for name in apps if name]
        except Exception as e:
            logger.warning("list_windows_failed", error=str(e))
        
        return []
    
    # ═══════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════
    
    def _resolve_target(self, target: str) -> Optional[UIElement]:
        """解析目标为元素"""
        # 检查是否是 ID 格式（如 "#5"）
        if target.startswith("#"):
            try:
                element_id = int(target[1:])
                return self.get_element_by_id(element_id)
            except ValueError as e:
                logger.debug("gui_element_id_parse_failed", target=target, error=str(e))
        
        # 作为描述查找
        return self.find_element(target)
    
    def _require_macos(self) -> None:
        """检查是否为 macOS，非 macOS 抛出 ActionError"""
        if not self._is_macos:
            raise ActionError(
                "GUI operations are only supported on macOS. "
                "Install pyautogui for cross-platform support: pip install pyautogui",
                error_code="E543",
            )

    def _execute_click(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """执行点击"""
        self._require_macos()
        return self._execute_osascript(f'tell app "System Events" to click at {{{x}, {y}}}')
    
    def _execute_double_click(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """执行双击"""
        self._require_macos()
        return self._execute_osascript(f'tell app "System Events" to double click at {{{x}, {y}}}')
    
    def _execute_right_click(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """执行右键"""
        self._require_macos()
        return self._execute_osascript(
            f'tell app "System Events" to key down control',
            f'tell app "System Events" to click at {{{x}, {y}}}',
            f'tell app "System Events" to key up control',
        )
    
    def _execute_move(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """执行移动"""
        self._require_macos()
        return self._execute_osascript(f'tell app "System Events" to set {{mouselocation}} to {{{x}, {y}}}')
    
    def _execute_drag(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[bool, Optional[str]]:
        """执行拖拽"""
        self._require_macos()
        return self._execute_osascript(
            f'tell app "System Events" to mouse down at {{{x1}, {y1}}}',
            f'tell app "System Events" to set {{mouselocation}} to {{{x2}, {y2}}}',
            f'tell app "System Events" to mouse up at {{{x2}, {y2}}}',
        )
    
    def _execute_scroll(self, direction: str, amount: int) -> Tuple[bool, Optional[str]]:
        """执行滚动"""
        self._require_macos()
        
        # macOS 使用方向键模拟滚动
        key = "down" if direction == "down" else "up" if direction == "up" else None
        if key is None:
            return False, f"Unsupported scroll direction: {direction}"
        
        keycode = 125 if key == "down" else 126
        scripts = [f'tell app "System Events" to key code {keycode}' for _ in range(amount)]
        return self._execute_osascript(*scripts)
    
    def _execute_type(self, text: str) -> Tuple[bool, Optional[str]]:
        """执行输入"""
        self._require_macos()
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return self._execute_osascript(f'tell app "System Events" to keystroke "{escaped}"')
    
    def _execute_press(self, key: str) -> Tuple[bool, Optional[str]]:
        """执行按键"""
        self._require_macos()
        
        keycode_map = {
            "return": 36, "enter": 36, "tab": 48, "space": 49,
            "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
            "left": 123, "right": 124, "down": 125, "up": 126,
            "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96,
            "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
            "f11": 103, "f12": 111, "home": 115, "end": 119,
            "pageup": 116, "pagedown": 121,
        }
        
        code = keycode_map.get(key.lower())
        if code:
            return self._execute_osascript(f'tell app "System Events" to key code {code}')
        else:
            # 直接输入字符
            return self._execute_osascript(f'tell app "System Events" to keystroke "{key}"')
    
    def _execute_hotkey(self, keys: Tuple[str, ...]) -> Tuple[bool, Optional[str]]:
        """执行组合键"""
        self._require_macos()
        
        modifiers = []
        main_key = None
        
        for k in keys:
            kl = k.lower()
            if kl in ("command", "cmd"):
                modifiers.append("command down")
            elif kl == "shift":
                modifiers.append("shift down")
            elif kl in ("option", "alt"):
                modifiers.append("option down")
            elif kl in ("control", "ctrl"):
                modifiers.append("control down")
            else:
                main_key = k
        
        if not main_key:
            return False, "No main key in hotkey"
        
        using = ", ".join(modifiers)
        return self._execute_osascript(
            f'tell app "System Events" to keystroke "{main_key}" using {{{using}}}'
        )
    
    def _execute_osascript(self, *lines: str) -> Tuple[bool, Optional[str]]:
        """执行 AppleScript"""
        script = "\n".join(lines)
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return True, None
            return False, result.stderr.strip()
        except Exception as e:
            return False, str(e)
    
    # ═══════════════════════════════════════════════════════════
    # 健康检查
    # ═══════════════════════════════════════════════════════════
    
    def health_check(self) -> bool:
        """检查控制器是否正常"""
        perception_ok = self._perception.health_check()
        screenshot_ok = self._take_screenshot() is not None
        
        logger.info("gui_control_health_check",
                   perception=perception_ok,
                   screenshot=screenshot_ok)
        
        return perception_ok and screenshot_ok
