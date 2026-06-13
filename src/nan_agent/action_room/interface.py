import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from nan_agent.action_room.code_executor import CodeExecutor, ExecutionResult
from nan_agent.action_room.filesystem import AgentFileSystem, FileInfo
from nan_agent.action_room.registry import Tool, ToolRegistry, ToolResult
from nan_agent.action_room.skill_trees import SkillTreeManager, SkillNode
from nan_agent.action_room.skill_loader import SkillLoader
from nan_agent.action_room.sub_agent_dispatcher import SubAgentDispatcher
from nan_agent.action_room.web_search import WebSearch, SearchResult
from nan_agent.event_bus import EventBus
from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import Timer, get_logger, log_event

logger = get_logger(__name__)

DEFAULT_WORKSPACE_ROOT = os.path.expanduser("~/.nan-agent/workspace")


# ── 仿真工具辅助函数 ──────────────────────────────────────────────────


def _sim_run_handler(
    math_eng,
    engine,
    state_schema: list,
    initial_state: dict,
    derivatives: dict,
    duration: float,
    method: Optional[str] = None,
    record_interval: int = 10,
) -> str:
    """sim_run 工具的 handler：将 JSON 参数转换为 DynamicalSystem 并运行仿真。"""
    from nan_agent.action_room.simulation import DynamicalSystem, State, SimulationEngine

    # 构建导数函数：用 MathEngine 安全求值导数表达式
    def derivatives_fn(t: float, state: State) -> State:
        variables = state.to_dict()
        variables["t"] = t
        result = {}
        for var_name, expr in derivatives.items():
            result[var_name] = float(math_eng.evaluate(expr, variables))
        return State(result)

    # 构建初始状态
    init = State({k: float(v) for k, v in initial_state.items()})

    sys = DynamicalSystem(
        state_schema=state_schema,
        initial_state=init,
        derivatives_fn=derivatives_fn,
    )

    # 如果指定了不同的 method，临时创建引擎
    run_engine = engine
    if method and method != engine.method:
        run_engine = SimulationEngine(dt=engine.dt, method=method)

    result = run_engine.run(sys, duration=duration, record_interval=record_interval)
    return str(result.summary())


def _sim_hybrid_handler(
    math_eng,
    engine,
    state_schema: list,
    initial_state: dict,
    derivatives: dict,
    duration: float,
    scheduled_events: Optional[list] = None,
    conditions: Optional[list] = None,
    method: Optional[str] = None,
    record_interval: int = 10,
) -> str:
    """sim_run_hybrid 工具的 handler：构建 HybridSystem 并运行仿真。"""
    from nan_agent.action_room.simulation import DynamicalSystem, HybridSystem, State, SimulationEngine

    def derivatives_fn(t: float, state: State) -> State:
        variables = state.to_dict()
        variables["t"] = t
        result = {}
        for var_name, expr in derivatives.items():
            result[var_name] = float(math_eng.evaluate(expr, variables))
        return State(result)

    init = State({k: float(v) for k, v in initial_state.items()})

    sys = DynamicalSystem(
        state_schema=state_schema,
        initial_state=init,
        derivatives_fn=derivatives_fn,
    )

    hybrid = HybridSystem(continuous=sys)

    # 注册定时事件
    if scheduled_events:
        for evt in scheduled_events:
            var = evt.get("variable", "")
            val = evt.get("value", 0.0)
            name = evt.get("name", "")
            t_fire = evt.get("time", 0.0)

            def make_cb(v, val):
                def cb(t, state):
                    state[v] = val
                return cb

            hybrid.schedule_event(t_fire, make_cb(var, val), name=name)

    # 注册条件触发器
    if conditions:
        for cond in conditions:
            expr = cond.get("expression", "")
            var = cond.get("variable", "")
            val = cond.get("value", 0.0)
            name = cond.get("name", "")

            def make_condition_fn(expr_str):
                def condition_fn(t, state):
                    variables = state.to_dict()
                    variables["t"] = t
                    return float(math_eng.evaluate(expr_str, variables)) > 0
                return condition_fn

            def make_cb(v, val):
                def cb(t, state):
                    state[v] = val
                return cb

            hybrid.on_condition(
                condition=make_condition_fn(expr),
                callback=make_cb(var, val),
                name=name,
            )

    run_engine = engine
    if method and method != engine.method:
        run_engine = SimulationEngine(dt=engine.dt, method=method)

    result = run_engine.run(hybrid, duration=duration, record_interval=record_interval)
    return str(result.summary())


class ComponentStatus(str, Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


@dataclass
class ActionRequest:
    action_type: str = "tool"
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    action_type: str = ""
    tool_name: str = ""
    observations: List[Dict[str, Any]] = field(default_factory=list)


class ActionRoom:
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[Dict[str, Any]] = None,
        lazy_init: bool = False,
        skills_engine: Optional[Any] = None,
    ):
        self._event_bus = event_bus
        self._config = config or {}
        self._lazy_init = lazy_init
        self._initialized = False
        self._skills_engine = skills_engine

        ar_config = self._config.get("action_room", {})

        self._component_status: Dict[str, ComponentStatus] = {
            "registry": ComponentStatus.UNINITIALIZED,
            "filesystem": ComponentStatus.UNINITIALIZED,
            "code_executor": ComponentStatus.UNINITIALIZED,
            "web_search": ComponentStatus.UNINITIALIZED,
            "skills": ComponentStatus.UNINITIALIZED,
            "perception": ComponentStatus.UNINITIALIZED,
            "action_module": ComponentStatus.UNINITIALIZED,
            "simulation": ComponentStatus.UNINITIALIZED,
            "mcp_adapter": ComponentStatus.UNINITIALIZED,
        }

        self._registry = ToolRegistry(event_bus=event_bus)
        self._component_status["registry"] = ComponentStatus.HEALTHY

        self._filesystem: Optional[AgentFileSystem] = None
        self._code_executor: Optional[CodeExecutor] = None
        self._web_search: Optional[WebSearch] = None
        self._skill_manager: Optional[SkillTreeManager] = None
        self._perception: Optional[Any] = None
        self._action_module: Optional[Any] = None
        self._simulation: Optional[Any] = None
        self._mcp_adapter: Optional[Any] = None
        self._skill_loader: Optional[SkillLoader] = None
        self._sub_agent_dispatcher: Optional[SubAgentDispatcher] = None

        self._workspace_root: str = ar_config.get(
            "workspace_dir",
            self._config.get("action_room", {}).get("workspace_dir", DEFAULT_WORKSPACE_ROOT),
        )

        if not lazy_init:
            self._initialize_all()

    def _initialize_all(self) -> None:
        self._init_filesystem()
        self._init_code_executor()
        self._init_web_search()
        self._init_skill_manager()
        self._init_perception()
        self._init_action_module()
        self._init_simulation()
        self._init_mcp_adapter()
        self._initialized = True
        logger.info("action_room_initialized", component_count=len(self._component_status))

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_filesystem(self) -> AgentFileSystem:
        if self._filesystem is None:
            self._init_filesystem()
        return self._filesystem

    def _ensure_code_executor(self) -> CodeExecutor:
        if self._code_executor is None:
            self._init_code_executor()
        return self._code_executor

    def _ensure_web_search(self) -> WebSearch:
        if self._web_search is None:
            self._init_web_search()
        return self._web_search

    def _ensure_skill_manager(self) -> SkillTreeManager:
        if self._skill_manager is None:
            self._init_skill_manager()
        return self._skill_manager

    def _init_filesystem(self) -> None:
        ar_config = self._config.get("action_room", {})
        self._component_status["filesystem"] = ComponentStatus.INITIALIZING
        try:
            self._filesystem = AgentFileSystem(workspace_root=self._workspace_root)
            self._register_filesystem_tools()
            self._component_status["filesystem"] = ComponentStatus.HEALTHY
            logger.info("filesystem_initialized", workspace_root=self._workspace_root)
        except Exception as e:
            self._component_status["filesystem"] = ComponentStatus.UNHEALTHY
            logger.error("filesystem_init_failed", error=str(e))
            raise ActionError(
                f"Failed to initialize filesystem: {e}",
                error_code="E550",
                details={"workspace_root": self._workspace_root},
            ) from e

    def _init_code_executor(self) -> None:
        ar_config = self._config.get("action_room", {})
        self._component_status["code_executor"] = ComponentStatus.INITIALIZING
        try:
            timeout = ar_config.get("code_timeout", 30.0)
            max_output = ar_config.get("max_output_chars", 100_000)
            self._code_executor = CodeExecutor(
                default_timeout=float(timeout),
                max_output_chars=int(max_output),
            )
            self._register_code_executor_tools()
            self._component_status["code_executor"] = ComponentStatus.HEALTHY
            logger.info("code_executor_initialized", timeout=timeout)
        except Exception as e:
            self._component_status["code_executor"] = ComponentStatus.UNHEALTHY
            logger.error("code_executor_init_failed", error=str(e))
            raise ActionError(
                f"Failed to initialize code executor: {e}",
                error_code="E551",
            ) from e

    def _init_web_search(self) -> None:
        ar_config = self._config.get("action_room", {})
        self._component_status["web_search"] = ComponentStatus.INITIALIZING
        try:
            search_config = ar_config.get("search", {})
            self._web_search = WebSearch(
                max_results=search_config.get("max_results", 10),
                rate_limit=search_config.get("rate_limit", 1.0),
                timeout=search_config.get("timeout", 30.0),
                proxy=search_config.get("proxy"),
            )
            self._register_web_search_tools()
            self._component_status["web_search"] = ComponentStatus.HEALTHY
            logger.info("web_search_initialized", backend=self._web_search.backend)
        except Exception as e:
            self._component_status["web_search"] = ComponentStatus.UNHEALTHY
            logger.error("web_search_init_failed", error=str(e))
            raise ActionError(
                f"Failed to initialize web search: {e}",
                error_code="E552",
            ) from e

    def _init_skill_manager(self) -> None:
        self._component_status["skills"] = ComponentStatus.INITIALIZING
        try:
            self._skill_loader = SkillLoader()
            self._skill_manager = SkillTreeManager(loader=self._skill_loader)
            self._skill_manager.initialize()

            # Create SubAgentDispatcher (LLM client set later via set_llm_client)
            self._sub_agent_dispatcher = SubAgentDispatcher(
                skill_loader=self._skill_loader,
                skill_tree_manager=self._skill_manager,
                tool_registry=self._registry,
                code_executor=self._code_executor or CodeExecutor(),
                llm_client=None,
            )

            self._register_skill_tools()
            self._component_status["skills"] = ComponentStatus.HEALTHY
            logger.info("skill_manager_initialized")
        except Exception as e:
            self._component_status["skills"] = ComponentStatus.UNHEALTHY
            logger.error("skill_manager_init_failed", error=str(e))
            raise ActionError(
                f"Failed to initialize skill manager: {e}",
                error_code="E553",
            ) from e

    def set_llm_client(self, llm_client: Any) -> None:
        """Set LLM client for Sub-Agent dispatch."""
        if self._sub_agent_dispatcher is not None:
            self._sub_agent_dispatcher._llm = llm_client

    def _init_perception(self) -> None:
        try:
            from nan_agent.action_room.perception import (
                CameraCapture,
                MicrophoneCapture,
                Perception,
                SpeechRecognizer,
            )
            self._component_status["perception"] = ComponentStatus.INITIALIZING

            camera = CameraCapture()
            microphone = MicrophoneCapture()
            asr = SpeechRecognizer(model_name="base", language="zh")

            self._perception = Perception(
                camera=camera,
                microphone=microphone,
                asr=asr,
                event_bus=self._event_bus,
                config=self._config,
            )
            self._component_status["perception"] = ComponentStatus.HEALTHY
            logger.info("perception_initialized", camera="cv2", mic="sounddevice", asr="faster-whisper")
        except ImportError:
            self._component_status["perception"] = ComponentStatus.UNINITIALIZED
            logger.debug("perception_not_available")
        except Exception as e:
            self._component_status["perception"] = ComponentStatus.UNHEALTHY
            logger.warning("perception_init_failed", error=str(e))

    def _ensure_action_module(self):
        if self._action_module is None:
            self._init_action_module()
        return self._action_module

    def _init_action_module(self) -> None:
        try:
            from nan_agent.action_room.action import ActionModule
            self._component_status["action_module"] = ComponentStatus.INITIALIZING
            self._action_module = ActionModule(event_bus=self._event_bus, config=self._config)
            self._component_status["action_module"] = ComponentStatus.HEALTHY
            self._register_action_module_tools()
            logger.info("action_module_initialized")
        except ImportError:
            self._component_status["action_module"] = ComponentStatus.UNINITIALIZED
            logger.debug("action_module_not_available")
        except Exception as e:
            self._component_status["action_module"] = ComponentStatus.UNHEALTHY
            logger.warning("action_module_init_failed", error=str(e))

    def _register_action_module_tools(self) -> None:
        am = self._ensure_action_module()
        if am is None:
            return

        def _require_gui():
            """GUI 不可用时抛出 ActionError"""
            if am.gui is None:
                raise ActionError(
                    "GUI is not available. Ensure pyautogui is installed and a display is connected.",
                    error_code="E550",
                )

        @self._registry.register_tool(
            name="speak",
            description="Convert text to speech and play audio. Use this to speak to the user.",
            category="action",
            tags=["tts", "speech", "audio", "output"],
        )
        async def speak(text: str, voice_id: str = "nan_default") -> str:
            from nan_agent.action_room.action import TTSRequest
            result = await am.tts.synthesize(TTSRequest(text=text, voice_id=voice_id))
            return str({
                "format": result.format,
                "duration_ms": result.duration_ms,
                "audio_size": len(result.audio_data),
                "text": result.text,
            })

        @self._registry.register_tool(
            name="take_screenshot",
            description="Capture a screenshot of the current screen",
            category="action",
            tags=["gui", "screenshot", "screen"],
        )
        async def take_screenshot(fast: bool = False) -> str:
            _require_gui()
            result = am.gui.capture_screenshot(fast=fast)
            if result.success:
                elements = result.parsed_screen.elements if result.parsed_screen else []
                return f"OK: screenshot captured, {len(elements)} elements found"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_click",
            description="Click on a UI element by description or ID (e.g. 'Save button' or '#5')",
            category="action",
            tags=["gui", "click", "mouse"],
        )
        async def gui_click(target: str) -> str:
            _require_gui()
            result = am.gui.click(target)
            if result.success:
                return f"OK: clicked '{target}'"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_type",
            description="Type text into the active window, optionally targeting a specific input element",
            category="action",
            tags=["gui", "keyboard", "type"],
        )
        async def gui_type(text: str, target: str = "") -> str:
            _require_gui()
            result = am.gui.type_text(text, target or None)
            if result.success:
                desc = f" into '{target}'" if target else ""
                return f"OK: typed text{desc}"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_double_click",
            description="Double-click on a UI element by description or ID",
            category="action",
            tags=["gui", "click"],
        )
        async def gui_double_click(target: str) -> str:
            _require_gui()
            result = am.gui.double_click(target)
            if result.success:
                return f"OK: double-clicked '{target}'"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_scroll",
            description="Scroll the screen in a direction",
            category="action",
            tags=["gui", "scroll"],
        )
        async def gui_scroll(direction: str = "down", amount: int = 3) -> str:
            _require_gui()
            result = am.gui.scroll(direction, amount)
            if result.success:
                return f"OK: scrolled {direction} x{amount}"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_drag",
            description="Drag from one element to another by description or ID",
            category="action",
            tags=["gui", "drag"],
        )
        async def gui_drag(from_target: str, to_target: str) -> str:
            _require_gui()
            result = am.gui.drag(from_target, to_target)
            if result.success:
                return f"OK: dragged '{from_target}' to '{to_target}'"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_find_elements",
            description="Find UI elements matching a description",
            category="action",
            tags=["gui", "find"],
        )
        async def gui_find_elements(description: str) -> str:
            _require_gui()
            elements = am.gui.find_elements(description)
            if not elements:
                return f"No elements found matching '{description}'"
            serialized = [
                {"id": el.id, "type": el.type, "content": el.content,
                 "interactivity": el.interactivity, "bbox": list(el.bbox)}
                for el in elements
            ]
            return json.dumps(serialized, ensure_ascii=False)

        @self._registry.register_tool(
            name="gui_right_click",
            description="Right-click on a UI element by description or ID",
            category="action",
            tags=["gui", "click"],
        )
        async def gui_right_click(target: str) -> str:
            _require_gui()
            result = am.gui.right_click(target)
            if result.success:
                return f"OK: right-clicked '{target}'"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_get_screen_size",
            description="Get current screen resolution",
            category="action",
            tags=["gui", "screen"],
        )
        async def gui_get_screen_size() -> str:
            _require_gui()
            return str(am.gui.get_screen_size())

        @self._registry.register_tool(
            name="speak_stream",
            description="Convert text to speech with streaming audio output",
            category="action",
            tags=["tts", "streaming", "audio"],
        )
        async def speak_stream(text: str, voice_id: str = "nan_default") -> str:
            from nan_agent.action_room.action import TTSRequest
            return am.tts.synthesize_stream(TTSRequest(text=text, voice_id=voice_id))

        @self._registry.register_tool(
            name="list_voices",
            description="List all available TTS voices",
            category="action",
            tags=["tts", "voice"],
        )
        async def list_voices() -> str:
            return str(am.tts.get_voices())

        @self._registry.register_tool(
            name="detect_language",
            description="Detect the language of audio input. Requires a recorded audio file path.",
            category="action",
            tags=["audio", "language"],
        )
        async def detect_language(audio_path: str) -> str:
            if hasattr(am.tts, 'detect_language'):
                return am.tts.detect_language(audio_path)
            if self._perception and self._perception.asr:
                return self._perception.asr.detect_language(audio_path)
            return "ASR not available"

        @self._registry.register_tool(
            name="list_sensors",
            description="List all available sensors (cameras, microphones)",
            category="action",
            tags=["sensors", "perception"],
        )
        async def list_sensors() -> str:
            if self._perception:
                return str(self._perception.list_available_sensors())
            return "No sensors available"

        @self._registry.register_tool(
            name="gui_mouse_move",
            description="Move mouse cursor to a UI element by description or ID",
            category="action",
            tags=["gui", "mouse"],
        )
        async def gui_mouse_move(target: str) -> str:
            _require_gui()
            result = am.gui.move_to(target)
            if result.success:
                return f"OK: moved to '{target}'"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_press_key",
            description="Press a single keyboard key",
            category="action",
            tags=["gui", "keyboard"],
        )
        async def gui_press_key(key: str) -> str:
            _require_gui()
            result = am.gui.press_key(key)
            if result.success:
                return f"OK: pressed '{key}'"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_hotkey",
            description="Press a hotkey combination like ctrl+c",
            category="action",
            tags=["gui", "keyboard"],
        )
        async def gui_hotkey(keys: str) -> str:
            _require_gui()
            key_list = [k.strip() for k in keys.split(",")]
            result = am.gui.hotkey(*key_list)
            if result.success:
                return f"OK: pressed hotkey '{keys}'"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_focus_window",
            description="Focus a window by title",
            category="action",
            tags=["gui", "window"],
        )
        async def gui_focus_window(title: str) -> str:
            _require_gui()
            result = am.gui.focus_window(title)
            if result.success:
                return f"OK: focused window '{title}'"
            return f"Error: {result.error}"

        @self._registry.register_tool(
            name="gui_list_windows",
            description="List all open window titles",
            category="action",
            tags=["gui", "window"],
        )
        async def gui_list_windows() -> str:
            _require_gui()
            return str(am.gui.list_windows())

        @self._registry.register_tool(
            name="list_interactive_elements",
            description="List all interactive UI elements on the current screen",
            category="action",
            tags=["gui", "elements", "screen"],
        )
        async def list_interactive_elements() -> str:
            _require_gui()
            elements = am.gui.list_interactive_elements()
            if not elements:
                return "No interactive elements found"
            serialized = [
                {"id": el.id, "type": el.type, "content": el.content,
                 "bbox": list(el.bbox)}
                for el in elements
            ]
            return json.dumps(serialized, ensure_ascii=False)

        @self._registry.register_tool(
            name="find_element",
            description="Find the best matching UI element by description. Returns a single element.",
            category="action",
            tags=["gui", "find"],
        )
        async def find_element(description: str) -> str:
            _require_gui()
            el = am.gui.find_element(description)
            if el is None:
                return f"No element found matching '{description}'"
            return json.dumps({
                "id": el.id, "type": el.type, "content": el.content,
                "interactivity": el.interactivity, "bbox": list(el.bbox),
            }, ensure_ascii=False)

        @self._registry.register_tool(
            name="get_element_by_id",
            description="Get a UI element by its numeric ID",
            category="action",
            tags=["gui", "element"],
        )
        async def get_element_by_id(element_id: int) -> str:
            _require_gui()
            el = am.gui.get_element_by_id(element_id)
            if el is None:
                return f"No element with ID {element_id}"
            return json.dumps({
                "id": el.id, "type": el.type, "content": el.content,
                "interactivity": el.interactivity, "bbox": list(el.bbox),
            }, ensure_ascii=False)

        @self._registry.register_tool(
            name="parse_screen",
            description="Parse the current screen and return structured UI elements without taking a screenshot. Lighter than take_screenshot.",
            category="action",
            tags=["gui", "screen", "parse"],
        )
        async def parse_screen() -> str:
            _require_gui()
            result = am.gui.parse_current_screen()
            if result.parsed_screen:
                formatted = result.parsed_screen.format_for_agent()
                return formatted
            return "Failed to parse screen"

    def _ensure_simulation(self):
        if self._simulation is None:
            self._init_simulation()
        return self._simulation

    def _init_simulation(self) -> None:
        try:
            from nan_agent.action_room.simulation import MathEngine, SimulationEngine
            ar_config = self._config.get("action_room", {})
            sim_config = ar_config.get("simulation", {})
            if sim_config.get("enabled", True):
                self._component_status["simulation"] = ComponentStatus.INITIALIZING
                self._simulation = {
                    "math": MathEngine(),
                    "engine": SimulationEngine(
                        dt=sim_config.get("dt", 0.01),
                        method=sim_config.get("method", "rk4"),
                    ),
                }
                self._component_status["simulation"] = ComponentStatus.HEALTHY
                self._register_simulation_tools()
                logger.info("simulation_initialized")
        except ImportError:
            self._component_status["simulation"] = ComponentStatus.UNINITIALIZED
            logger.debug("simulation_not_available")
        except Exception as e:
            self._component_status["simulation"] = ComponentStatus.UNHEALTHY
            logger.warning("simulation_init_failed", error=str(e))

    def _register_simulation_tools(self) -> None:
        sim = self._ensure_simulation()
        if sim is None:
            return

        math_eng = sim["math"]
        engine = sim["engine"]

        # ── Math 工具 ──────────────────────────────────────────────

        @self._registry.register_tool(
            name="math_evaluate",
            description="Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, log, etc.",
            category="simulation",
            tags=["math", "calculate", "evaluate"],
        )
        def math_evaluate(expression: str, variables: dict = None) -> str:
            return str(math_eng.evaluate(expression, variables=variables))

        @self._registry.register_tool(
            name="math_solve",
            description="Solve equation f(x)=0 numerically. Provide expression and initial guess.",
            category="simulation",
            tags=["math", "solve"],
        )
        def math_solve(expression: str, guess: float = 1.0) -> str:
            return str(math_eng.solve_equation(expression, guess))

        @self._registry.register_tool(
            name="math_derivative",
            description="Compute numerical derivative of expression at a point",
            category="simulation",
            tags=["math", "calculus"],
        )
        def math_derivative(expression: str, point: float) -> str:
            return str(math_eng.derivative(expression, point))

        @self._registry.register_tool(
            name="math_integral",
            description="Compute numerical definite integral from a to b",
            category="simulation",
            tags=["math", "calculus"],
        )
        def math_integral(expression: str, a: float, b: float) -> str:
            return str(math_eng.integral(expression, a, b))

        @self._registry.register_tool(
            name="math_matrix",
            description="Matrix operations: multiply/determinant/eigenvalues",
            category="simulation",
            tags=["math", "matrix"],
        )
        def math_matrix(a: list, operation: str, b: list = None) -> str:
            if operation == "multiply" and b:
                return str(math_eng.matrix_multiply(a, b))
            if operation == "determinant":
                return str(math_eng.determinant(a))
            return str(math_eng.eigenvalues(a))

        @self._registry.register_tool(
            name="math_statistics",
            description="Statistical analysis: mean/median/std/correlation/regression",
            category="simulation",
            tags=["math", "statistics"],
        )
        def math_statistics(data: list, operation: str, data2: list = None) -> str:
            if operation == "median":
                return str(math_eng.median(data))
            if operation == "std":
                return str(math_eng.std(data))
            if operation == "correlation" and data2:
                return str(math_eng.correlation(data, data2))
            if operation == "regression" and data2:
                return str(math_eng.linear_regression(data, data2))
            return "Unknown operation"

        # ── Simulation 工具（通用接口） ────────────────────────────

        @self._registry.register_tool(
            name="sim_run",
            description=(
                "Run a simulation of a dynamical system. "
                "Define state variables, their derivatives, and initial values. "
                "The engine integrates dy/dt = f(t, y) over the given duration."
            ),
            category="simulation",
            tags=["simulation", "ode", "dynamical_system"],
            parameters={
                "type": "object",
                "properties": {
                    "state_schema": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of state variables, e.g. ['x', 'v']",
                    },
                    "initial_state": {
                        "type": "object",
                        "description": "Initial values for each state variable, e.g. {'x': 1.0, 'v': 0.0}",
                    },
                    "derivatives": {
                        "type": "object",
                        "description": (
                            "Derivative expressions for each state variable. "
                            "Each key matches a state variable name; each value is a math expression "
                            "using the state variable names. "
                            "E.g. {'x': 'v', 'v': '-10*x - 0.5*v'} for a damped spring."
                        ),
                    },
                    "duration": {
                        "type": "number",
                        "description": "Simulation duration (seconds)",
                    },
                    "method": {
                        "type": "string",
                        "description": "Integration method: euler, rk4 (default), or rk45",
                    },
                    "record_interval": {
                        "type": "integer",
                        "description": "Record every N steps (default 10 to reduce output size)",
                    },
                },
                "required": ["state_schema", "initial_state", "derivatives", "duration"],
            },
        )
        def sim_run(state_schema: list, initial_state: dict, derivatives: dict, duration: float, method: str = None, record_interval: int = 10) -> str:
            return _sim_run_handler(math_eng, engine, state_schema, initial_state, derivatives, duration, method, record_interval)

        @self._registry.register_tool(
            name="sim_run_hybrid",
            description=(
                "Run a hybrid simulation with discrete events. "
                "Like sim_run but supports scheduled events and condition triggers "
                "that can modify state during simulation."
            ),
            category="simulation",
            tags=["simulation", "hybrid", "discrete_event"],
            parameters={
                "type": "object",
                "properties": {
                    "state_schema": {"type": "array", "items": {"type": "string"}},
                    "initial_state": {"type": "object"},
                    "derivatives": {"type": "object"},
                    "duration": {"type": "number"},
                    "scheduled_events": {
                        "type": "array",
                        "description": "Events to fire at specific times. Each: {time, variable, value, name}",
                        "items": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "number"},
                                "variable": {"type": "string"},
                                "value": {"type": "number"},
                                "name": {"type": "string"},
                            },
                        },
                    },
                    "conditions": {
                        "type": "array",
                        "description": "Condition triggers. Each: {expression, variable, value, name}. "
                                       "expression is a math expression using state vars that triggers when > 0.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                                "variable": {"type": "string"},
                                "value": {"type": "number"},
                                "name": {"type": "string"},
                            },
                        },
                    },
                    "method": {"type": "string"},
                    "record_interval": {"type": "integer"},
                },
                "required": ["state_schema", "initial_state", "derivatives", "duration"],
            },
        )
        def sim_run_hybrid(state_schema: list, initial_state: dict, derivatives: dict, duration: float, scheduled_events: list = None, conditions: list = None, method: str = None, record_interval: int = 10) -> str:
            return _sim_hybrid_handler(math_eng, engine, state_schema, initial_state, derivatives, duration, scheduled_events, conditions, method, record_interval)

        @self._registry.register_tool(
            name="sim_step",
            description="Advance a simulation by one step. Create a system first with sim_run, then use sim_step to advance interactively. Returns current time and state.",
            category="simulation",
            tags=["simulation", "interactive", "step"],
        )
        def sim_step(system_id: str) -> str:
            # Retrieve stored system by ID
            system = self._simulation.get("systems", {}).get(system_id)
            if system is None:
                return json.dumps({"error": f"System '{system_id}' not found. Use sim_run first."})
            t, state = engine.step_once(system)
            return json.dumps({
                "system_id": system_id,
                "time": t,
                "state": {k: v for k, v in zip(state.variables, state.values)},
            }, ensure_ascii=False)

    def _init_mcp_adapter(self) -> None:
        try:
            from nan_agent.action_room.mcp_adapter import MCPAdapter
            self._component_status["mcp_adapter"] = ComponentStatus.INITIALIZING
            self._mcp_adapter = MCPAdapter(config=self._config)
            self._register_mcp_management_tools()
            if self._mcp_adapter.has_configured_servers:
                self._component_status["mcp_adapter"] = ComponentStatus.INITIALIZING
                logger.info("mcp_adapter_initialized",
                            pending_count=len(self._mcp_adapter._pending_configs))
            else:
                self._component_status["mcp_adapter"] = ComponentStatus.UNINITIALIZED
                logger.debug("mcp_adapter_no_servers_configured")
        except ImportError:
            self._component_status["mcp_adapter"] = ComponentStatus.UNINITIALIZED
            logger.debug("mcp_adapter_not_available")
        except Exception as e:
            self._component_status["mcp_adapter"] = ComponentStatus.UNHEALTHY
            logger.warning("mcp_adapter_init_failed", error=str(e))

    async def connect_mcp(self) -> None:
        """Connect to configured MCP servers and register their tools."""
        with Timer(logger, "action_connect_mcp", warn_threshold_ms=30000):
            if self._mcp_adapter is None:
                return
            if not self._mcp_adapter.has_configured_servers:
                return

            self._component_status["mcp_adapter"] = ComponentStatus.INITIALIZING
            try:
                await self._mcp_adapter.initialize()
                self._register_mcp_tools_from_servers()
                self._component_status["mcp_adapter"] = ComponentStatus.HEALTHY
                logger.info("mcp_adapter_connected",
                            connected=self._mcp_adapter.connected_count)
            except Exception as e:
                self._component_status["mcp_adapter"] = ComponentStatus.UNHEALTHY
                logger.warning("mcp_adapter_connect_failed", error=str(e))

    def _register_mcp_tools_from_servers(self) -> None:
        """Register tools from all connected MCP servers."""
        if self._mcp_adapter is None:
            return
        for server_name, server in self._mcp_adapter.servers.items():
            if not server.connected:
                continue
            for tool_name, mcp_tool in server.tools.items():
                async def make_handler(s=server_name, t=tool_name):
                    async def handler(params=None, **_kw):
                        return await self._mcp_adapter.call_tool(s, t, params or {})
                    return handler

                self._registry.register(Tool(
                    name=f"mcp_{server_name}_{tool_name}",
                    description=f"[MCP:{server_name}] {mcp_tool.description or tool_name}",
                    parameters={
                        "type": "object",
                        "properties": {
                            "params": {
                                "type": "object",
                                "description": f"Tool parameters: {mcp_tool.input_schema}" if mcp_tool.input_schema else "Tool parameters",
                            },
                        },
                        "required": ["params"],
                    },
                    handler=make_handler(),
                    category="mcp",
                    tags=["mcp", server_name],
                ))
        logger.info("mcp_tools_registered",
                    count=sum(len(s.tools) for s in self._mcp_adapter.servers.values() if s.connected))

    async def disconnect_mcp(self):
        if self._mcp_adapter is not None:
            try:
                await self._mcp_adapter.disconnect_all()
            except Exception as e:
                logger.warning("mcp_disconnect_all_failed", error=str(e))

    def _register_mcp_management_tools(self) -> None:
        """Register MCP server management tools for dynamic discovery and control."""

        @self._registry.register_tool(
            name="connect_mcp_server",
            description="Connect to an MCP server by command or URL. The server's tools will become available.",
            category="mcp",
            tags=["mcp", "connect", "server"],
        )
        async def connect_mcp_server(name: str, command: str = "", url: str = "",
                                     transport_type: str = "stdio", args: list = None,
                                     env: dict = None) -> str:
            if self._mcp_adapter is None:
                return "MCP adapter not available"
            from nan_agent.action_room.mcp_adapter import MCPServerConfig
            config = MCPServerConfig(
                name=name,
                command=command or None,
                url=url or None,
                transport_type=transport_type,
                args=args or [],
                env=env or {},
            )
            try:
                await self._mcp_adapter.connect_server(config)
                self._register_mcp_tools_from_servers()
                return f"Connected to MCP server '{name}'"
            except Exception as e:
                return f"Failed to connect: {e}"

        @self._registry.register_tool(
            name="disconnect_mcp_server",
            description="Disconnect from an MCP server by name",
            category="mcp",
            tags=["mcp", "disconnect"],
        )
        async def disconnect_mcp_server(server_name: str) -> str:
            if self._mcp_adapter is None:
                return "MCP adapter not available"
            disconnected = await self._mcp_adapter.disconnect_server(server_name)
            return f"Disconnected '{server_name}'" if disconnected else f"Server '{server_name}' not found"

        @self._registry.register_tool(
            name="discover_mcp_server",
            description="Discover tools provided by an MCP server without permanently connecting. Returns list of available tools.",
            category="mcp",
            tags=["mcp", "discover"],
        )
        async def discover_mcp_server(command: str = "", url: str = "",
                                      transport_type: str = "stdio", args: list = None,
                                      env: dict = None) -> str:
            if self._mcp_adapter is None:
                return "MCP adapter not available"
            try:
                tools = await self._mcp_adapter.discover_server(
                    command=command or None, args=args, env=env,
                )
                return json.dumps({
                    "tools": [
                        {"name": t.name, "description": t.description}
                        for t in tools
                    ],
                    "count": len(tools),
                }, ensure_ascii=False)
            except Exception as e:
                return f"Discovery failed: {e}"

        @self._registry.register_tool(
            name="list_mcp_tools",
            description="List all tools from all connected MCP servers",
            category="mcp",
            tags=["mcp", "list", "tools"],
        )
        async def list_mcp_tools() -> str:
            if self._mcp_adapter is None:
                return "MCP adapter not available"
            all_tools = await self._mcp_adapter.list_all_tools()
            return json.dumps({
                "tools": [
                    {"qualified_name": qn, "description": t.description}
                    for qn, t in all_tools.items()
                ],
                "count": len(all_tools),
            }, ensure_ascii=False)

        @self._registry.register_tool(
            name="health_check_mcp",
            description="Check health status of all or a specific MCP server",
            category="mcp",
            tags=["mcp", "health"],
        )
        async def health_check_mcp(server_name: str = "") -> str:
            if self._mcp_adapter is None:
                return "MCP adapter not available"
            if server_name:
                result = await self._mcp_adapter.health_check_server(server_name)
                if result is None:
                    return f"Server '{server_name}' not found"
                return json.dumps({server_name: result})
            else:
                results = await self._mcp_adapter.health_check_all()
                return json.dumps(results)

    def _register_filesystem_tools(self) -> None:
        fs = self._ensure_filesystem()

        @self._registry.register_tool(
            name="read_file",
            description="Read the contents of a file in the workspace",
            category="filesystem",
            tags=["file", "read", "io"],
        )
        async def read_file(path: str, mode: str = "r") -> str:
            return await fs.read_file(path, mode=mode)

        @self._registry.register_tool(
            name="write_file",
            description="Write content to a file in the workspace",
            category="filesystem",
            tags=["file", "write", "io"],
        )
        async def write_file(path: str, content: str, mode: str = "w", create_parents: bool = True) -> str:
            return await fs.write_file(path, content, mode=mode, create_parents=create_parents)

        @self._registry.register_tool(
            name="list_dir",
            description="List directory contents",
            category="filesystem",
            tags=["file", "directory", "list"],
        )
        async def list_dir(path: str = ".", recursive: bool = False) -> list:
            return await fs.list_directory(path, recursive=recursive)

        @self._registry.register_tool(
            name="search_files",
            description="Search for files in the workspace by name pattern or content",
            category="filesystem",
            tags=["file", "search", "find"],
        )
        async def search_files(pattern: str = "*", content_pattern: str = "", recursive: bool = True, max_results: int = 100) -> list:
            return await fs.search(pattern, content_pattern=content_pattern, recursive=recursive, max_results=max_results)

        @self._registry.register_tool(
            name="create_directory",
            description="Create a new directory",
            category="filesystem",
            tags=["file", "directory"],
        )
        async def create_directory(path: str, parents: bool = True) -> str:
            return await fs.create_directory(path, parents=parents)

        @self._registry.register_tool(
            name="delete_file",
            description="Delete a file or directory",
            category="filesystem",
            tags=["file", "delete"],
        )
        async def delete_file(path: str, recursive: bool = False) -> str:
            return await fs.delete(path, recursive=recursive)

        @self._registry.register_tool(
            name="copy_file",
            description="Copy a file or directory",
            category="filesystem",
            tags=["file", "copy"],
        )
        async def copy_file(src: str, dst: str, overwrite: bool = False) -> str:
            return await fs.copy(src, dst, overwrite=overwrite)

        @self._registry.register_tool(
            name="move_file",
            description="Move a file or directory",
            category="filesystem",
            tags=["file", "move"],
        )
        async def move_file(src: str, dst: str, overwrite: bool = False) -> str:
            return await fs.move(src, dst, overwrite=overwrite)

        @self._registry.register_tool(
            name="get_file_info",
            description="Get metadata for a file",
            category="filesystem",
            tags=["file", "metadata"],
        )
        async def get_file_info(path: str) -> dict:
            return await fs.get_file_info(path)

        @self._registry.register_tool(
            name="check_exists",
            description="Check if a file or directory exists",
            category="filesystem",
            tags=["file", "check"],
        )
        async def check_exists(path: str) -> bool:
            return await fs.exists(path)

        @self._registry.register_tool(
            name="get_quota_info",
            description="Get workspace disk quota information (used bytes, quota limit, file count)",
            category="filesystem",
            tags=["file", "quota", "disk"],
        )
        async def get_quota_info() -> str:
            info = await fs.get_quota_info()
            return json.dumps({
                "used_bytes": info.used_bytes,
                "quota_bytes": info.quota_bytes,
                "file_count": info.file_count,
                "usage_pct": round(info.used_bytes / info.quota_bytes * 100, 1) if info.quota_bytes > 0 else 0,
            })

        @self._registry.register_tool(
            name="cleanup_temp_files",
            description="Clean up temporary files older than max_age seconds. Returns number of files removed.",
            category="filesystem",
            tags=["file", "cleanup", "temp"],
        )
        async def cleanup_temp_files(max_age: int = 86400) -> str:
            removed = await fs.cleanup_temp_files(max_age=max_age)
            return f"Removed {removed} temporary files"

    def _register_code_executor_tools(self) -> None:
        ce = self._ensure_code_executor()

        @self._registry.register_tool(
            name="execute_python",
            description="Execute Python code in a sandboxed environment",
            category="code_execution",
            tags=["code", "python", "execution"],
        )
        def execute_python(code: str, timeout: float = 30.0, session_id: str = "",
                          input_vars: dict = None, capture_vars: list = None) -> str:
            return ce.execute(
                code, timeout=timeout,
                session_id=session_id or None,
                input_vars=input_vars,
                capture_vars=capture_vars,
            )

        @self._registry.register_tool(
            name="execute_bash",
            description="Execute a bash/shell script",
            category="code_execution",
            tags=["code", "bash", "shell", "execution"],
        )
        def execute_bash(code: str, language: str = "bash", timeout: float = 30.0) -> str:
            return ce.execute(code, language=language, timeout=timeout)

        @self._registry.register_tool(
            name="create_session",
            description="Create a persistent code execution session. Variables persist across executions within the same session.",
            category="code_execution",
            tags=["code", "session", "state"],
        )
        def create_session() -> str:
            session_id = ce.create_session()
            return json.dumps({"session_id": session_id})

        @self._registry.register_tool(
            name="list_sessions",
            description="List all active code execution session IDs",
            category="code_execution",
            tags=["code", "session"],
        )
        def list_sessions() -> str:
            sessions = ce.list_sessions()
            return json.dumps({"sessions": sessions, "count": len(sessions)})

        @self._registry.register_tool(
            name="delete_session",
            description="Delete a code execution session and its persisted variables",
            category="code_execution",
            tags=["code", "session"],
        )
        def delete_session(session_id: str) -> str:
            deleted = ce.delete_session(session_id)
            return f"Session {session_id} deleted" if deleted else f"Session {session_id} not found"

    def _register_web_search_tools(self) -> None:
        ws = self._ensure_web_search()

        @self._registry.register_tool(
            name="search_web",
            description="Search the web for information",
            category="web_search",
            tags=["search", "web", "internet"],
        )
        async def search_web(query: str, max_results: int = 10, region: str = "us-en") -> list:
            return await ws.search(query, max_results=max_results, region=region)

        @self._registry.register_tool(
            name="fetch_content",
            description="Fetch content from a URL and extract readable text",
            category="web_search",
            tags=["web", "fetch", "http"],
        )
        async def fetch_content(url: str, max_length: int = 0) -> str:
            html = await ws.fetch_content(url)
            return ws.extract_text(html, max_length=max_length)

    def _register_skill_tools(self) -> None:
        """注册 delegate_task 工具 — Main Agent 唯一的 skill 入口。

        Sub-Agent 自主搜索/加载/执行技能，Main Agent 只需派发自然语言任务。
        """

        @self._registry.register_tool(
            name="delegate_task",
            description="Delegate a natural language task to a specialized Sub-Agent. The Sub-Agent will autonomously search the skill tree, find the best matching skill, activate it, and execute the task. Use this for any task that might benefit from a specialized skill.",
            category="agent",
            tags=["delegate", "sub-agent", "skill", "task"],
        )
        async def delegate_task(task: str, context: str = "") -> str:
            if self._sub_agent_dispatcher is None:
                raise ActionError("Skill dispatcher not available", error_code="E540")
            result = await self._sub_agent_dispatcher.delegate(
                task=task,
                context=context if context else None,
            )
            if result.success:
                return result.summary
            else:
                return f"Task delegation failed: {result.error}"

    def _read_skill_content(self, name: str) -> dict:
        sm = self._skill_manager
        node = sm.get_node(name)
        if node is None:
            return {"error": f"Skill '{name}' not found"}
        if node.skill_path:
            skill_md = Path(node.skill_path) / "SKILL.md"
            if skill_md.exists():
                return {"name": node.name, "content": skill_md.read_text(), "path": node.skill_path}
        return node.to_dict()

    def _get_skill_summary(self) -> dict:
        sm = self._skill_manager
        trees = {}
        for cat in sm.categories:
            tree = sm.get_tree(cat)
            if tree:
                trees[cat] = {
                    "total_nodes": tree.total_nodes,
                    "unlocked_count": tree.unlocked_count,
                }
        return {"trees": trees, "total_nodes": sm.total_nodes, "categories": sm.categories}

    async def execute(self, action_request: ActionRequest) -> ActionResult:
        with Timer(logger, "action_execute", warn_threshold_ms=10000, tool_name=action_request.action_type):
            start_time = time.perf_counter()

            try:
                if action_request.action_type == "tool":
                    return await self._execute_tool(action_request, start_time)
                elif action_request.action_type == "observe":
                    return await self._execute_observe(action_request, start_time)
                elif action_request.action_type == "health_check":
                    return await self._execute_health_check(start_time)
                else:
                    execution_time = (time.perf_counter() - start_time) * 1000
                    return ActionResult(
                        success=False,
                        error=f"Unknown action type: {action_request.action_type}",
                        execution_time_ms=execution_time,
                        action_type=action_request.action_type,
                    )
            except ActionError:
                raise
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.exception("action_execution_error", action_type=action_request.action_type, error=str(e))
                return ActionResult(
                    success=False,
                    error=f"Action execution error: {str(e)}",
                    execution_time_ms=execution_time,
                    action_type=action_request.action_type,
                )

    async def _execute_tool(self, request: ActionRequest, start_time: float) -> ActionResult:
        if request.tool_name is None:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ActionResult(
                success=False,
                error="Tool name is required for 'tool' action type",
                execution_time_ms=execution_time,
                action_type="tool",
            )

        if self._lazy_init and not self._initialized:
            self._ensure_component_for_tool(request.tool_name)

        result = await self._registry.execute(
            tool_name=request.tool_name,
            parameters=request.parameters,
            timeout=request.timeout,
        )

        execution_time = (time.perf_counter() - start_time) * 1000
        if result.success:
            logger.info(
                "tool_executed",
                tool=request.tool_name,
                success=True,
                elapsed_ms=round(execution_time),
            )
        else:
            logger.warning(
                "tool_executed",
                tool=request.tool_name,
                success=False,
                error=result.error[:100] if result.error else "",
                elapsed_ms=round(execution_time),
            )
        return ActionResult(
            success=result.success,
            data=result.data,
            error=result.error,
            execution_time_ms=execution_time,
            action_type="tool",
            tool_name=result.tool_name,
        )

    def _ensure_component_for_tool(self, tool_name: str) -> None:
        fs_tools = {"read_file", "write_file", "list_dir", "search_files", "create_directory",
                     "delete_file", "copy_file", "move_file", "get_file_info", "check_exists",
                     "get_quota_info", "cleanup_temp_files"}
        code_tools = {"execute_python", "execute_bash",
                      "create_session", "list_sessions", "delete_session"}
        search_tools = {"search_web", "fetch_content"}
        skill_tools = {"delegate_task"}
        action_tools = {"speak", "take_screenshot", "gui_click", "gui_type",
                        "gui_double_click", "gui_scroll", "gui_drag", "gui_find_elements",
                        "gui_right_click", "gui_get_screen_size",
                        "gui_mouse_move", "gui_press_key", "gui_hotkey",
                        "gui_focus_window", "gui_list_windows",
                        "speak_stream", "list_voices",
                        "detect_language", "list_sensors",
                        "list_interactive_elements", "find_element",
                        "get_element_by_id", "parse_screen"}
        sim_tools = {"math_evaluate", "math_solve", "math_derivative", "math_integral",
                     "math_matrix", "math_statistics",
                     "sim_run", "sim_run_hybrid", "sim_step"}
        mcp_mgmt_tools = {"connect_mcp_server", "disconnect_mcp_server",
                          "discover_mcp_server", "list_mcp_tools", "health_check_mcp"}

        if tool_name in fs_tools:
            self._ensure_filesystem()
        elif tool_name in code_tools:
            self._ensure_code_executor()
        elif tool_name in search_tools:
            self._ensure_web_search()
        elif tool_name in skill_tools:
            self._ensure_skill_manager()
        elif tool_name in action_tools:
            self._ensure_action_module()
        elif tool_name in sim_tools:
            self._ensure_simulation()
        elif tool_name in mcp_mgmt_tools:
            self._ensure_mcp_adapter()

    async def _observe_filesystem(self) -> Optional[Dict[str, Any]]:
        if self._ensure_filesystem() is None:
            return None
        try:
            root_files = await self._filesystem.list_directory(".", recursive=False)
            return {
                "type": "filesystem_view",
                "workspace_root": str(self._filesystem.workspace_root),
                "file_count": len(root_files),
                "files": [{"name": f.name, "type": f.type, "size": f.size} for f in root_files[:20]],
            }
        except Exception as e:
            return {"type": "filesystem_view", "error": str(e)}

    async def _observe_screenshot(self):
        import base64
        if self._action_module is None:
            return None, []
        try:
            screenshot = self._action_module.gui.capture_screenshot()
            images = []
            if screenshot and screenshot.success and screenshot.screenshot:
                images.append({
                    "type": "screenshot",
                    "source": "gui",
                    "data": base64.b64encode(screenshot.screenshot).decode("utf-8"),
                    "mime_type": "image/png",
                })
            gui_info = {"type": "screenshot", "status": "captured"}
            if screenshot and screenshot.parsed_screen and screenshot.parsed_screen.elements:
                elements = [
                    {"label": getattr(el, "label", ""),
                     "bbox": getattr(el, "bbox", None),
                     "confidence": round(getattr(el, "confidence", 0), 2)}
                    for el in screenshot.parsed_screen.elements[:20]
                ]
                gui_info["ui_elements"] = elements
                gui_info["element_count"] = len(screenshot.parsed_screen.elements)
                gui_info["parsed_screen"] = screenshot.parsed_screen
            return gui_info, images
        except Exception as e:
            logger.debug("screenshot_capture_failed", error=str(e))
            return None, []

    async def _observe_camera(self):
        import base64
        if self._perception is None or self._perception.camera is None:
            return None, []
        try:
            visual = await self._perception.capture_visual()
            images = []
            if visual and visual.frames:
                for frame in visual.frames:
                    frame_data = None
                    if isinstance(frame, dict):
                        frame_data = frame.get("data")
                        frame_w = frame.get("width", visual.resolution[0])
                        frame_h = frame.get("height", visual.resolution[1])
                    elif hasattr(frame, "data") and isinstance(frame.data, bytes):
                        frame_data = frame.data
                        frame_w = getattr(frame, "width", visual.resolution[0])
                        frame_h = getattr(frame, "height", visual.resolution[1])
                    if frame_data and isinstance(frame_data, bytes):
                        images.append({
                            "type": "camera_frame",
                            "source": "camera",
                            "data": base64.b64encode(frame_data).decode("utf-8"),
                            "mime_type": "image/jpeg",
                            "width": frame_w,
                            "height": frame_h,
                        })
                return {
                    "type": "camera",
                    "status": "captured",
                    "resolution": list(visual.resolution),
                    "frame_count": len(visual.frames),
                }, images
            return None, []
        except Exception as e:
            logger.debug("camera_capture_failed", error=str(e))
            return None, []

    async def _observe_asr(self) -> Optional[Dict[str, Any]]:
        if self._perception is None or self._perception.microphone is None:
            return None
        try:
            audio = await self._perception.capture_audio(duration_ms=200.0)
            if audio is not None and self._perception.asr is not None:
                speech = await self._perception.recognize_speech(audio)
                if speech and speech.text:
                    return {
                        "type": "speech",
                        "text": speech.text,
                        "confidence": speech.confidence,
                        "language": speech.language,
                    }
            return None
        except Exception as e:
            logger.debug("audio_asr_failed", error=str(e))
            return None

    async def _execute_observe(self, request: ActionRequest, start_time: float) -> ActionResult:
        observations: List[Dict[str, Any]] = []
        images: List[Dict[str, str]] = []

        if (fs := await self._observe_filesystem()):
            observations.append(fs)

        gui_info, gui_images = await self._observe_screenshot()
        if gui_info:
            observations.append(gui_info)
        images.extend(gui_images)

        cam_info, cam_images = await self._observe_camera()
        if cam_info:
            observations.append(cam_info)
        images.extend(cam_images)

        if (asr := await self._observe_asr()):
            observations.append(asr)

        # 合并为统一多模态结构
        if self._perception is not None and hasattr(self._perception, 'construct_multimodal_input'):
            try:
                unified = await self._perception.observe()
                mm = self._perception.construct_multimodal_input(
                    visual=unified.get("visual"),
                    audio=unified.get("audio"),
                    speech=unified.get("speech"),
                )
                observations.append(mm)
            except Exception as e:
                logger.warning("observe_multimodal_create_failed", error=str(e))

        execution_time = (time.perf_counter() - start_time) * 1000
        return ActionResult(
            success=True,
            data={"observations": observations, "images": images},
            observations=observations,
            execution_time_ms=execution_time,
            action_type="observe",
        )

    async def _execute_health_check(self, start_time: float) -> ActionResult:
        status = await self.health_check()
        execution_time = (time.perf_counter() - start_time) * 1000
        all_healthy = all(s == ComponentStatus.HEALTHY or s == ComponentStatus.UNINITIALIZED
                         for s in status.values())
        return ActionResult(
            success=all_healthy,
            data=status,
            execution_time_ms=execution_time,
            action_type="health_check",
        )

    def list_tools(self, include_disabled: bool = False) -> List[Tool]:
        return self._registry.list_all_tools(include_disabled=include_disabled)

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        return self._registry.get_tool(tool_name)

    def list_tools_by_category(self, category: str) -> List[Tool]:
        return self._registry.list_by_category(category)

    def enable_tool(self, name: str) -> bool:
        return self._registry.enable_tool(name) if hasattr(self._registry, 'enable_tool') else False

    def disable_tool(self, name: str) -> bool:
        return self._registry.disable_tool(name) if hasattr(self._registry, 'disable_tool') else False

    def get_filesystem_quota(self):
        if self._filesystem is None:
            return {}
        return self._filesystem.get_quota_info()

    async def cleanup_temp_files(self):
        if self._filesystem is not None and hasattr(self._filesystem, 'cleanup_temp_files'):
            await self._filesystem.cleanup_temp_files()

    def list_all_tools_with_metadata(self) -> List[Dict[str, Any]]:
        tools = self._registry.list_all_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "tags": t.tags,
                "version": t.version,
                "enabled": t.enabled,
            }
            for t in tools
        ]

    async def health_check(self) -> Dict[str, ComponentStatus]:
        status: Dict[str, ComponentStatus] = {}
        status["registry"] = ComponentStatus.HEALTHY

        if self._filesystem is not None:
            try:
                fs_healthy = await self._filesystem.health_check()
                status["filesystem"] = ComponentStatus.HEALTHY if fs_healthy else ComponentStatus.UNHEALTHY
            except Exception:
                status["filesystem"] = ComponentStatus.UNHEALTHY
        else:
            status["filesystem"] = ComponentStatus.UNINITIALIZED

        if self._code_executor is not None:
            try:
                test_result = self._code_executor.execute("print('ok')", language="python")
                status["code_executor"] = (
                    ComponentStatus.HEALTHY if test_result.exit_code == 0 else ComponentStatus.DEGRADED
                )
            except Exception:
                status["code_executor"] = ComponentStatus.UNHEALTHY
        else:
            status["code_executor"] = ComponentStatus.UNINITIALIZED

        if self._web_search is not None:
            status["web_search"] = ComponentStatus.HEALTHY
        else:
            status["web_search"] = ComponentStatus.UNINITIALIZED

        if self._skill_manager is not None:
            status["skills"] = ComponentStatus.HEALTHY
        else:
            status["skills"] = ComponentStatus.UNINITIALIZED

        status["perception"] = self._component_status.get("perception", ComponentStatus.UNINITIALIZED)
        status["action_module"] = self._component_status.get("action_module", ComponentStatus.UNINITIALIZED)
        status["simulation"] = self._component_status.get("simulation", ComponentStatus.UNINITIALIZED)

        if self._mcp_adapter is not None:
            if self._mcp_adapter.has_configured_servers:
                status["mcp_adapter"] = self._component_status.get("mcp_adapter", ComponentStatus.UNINITIALIZED)
            else:
                status["mcp_adapter"] = ComponentStatus.UNINITIALIZED
        else:
            status["mcp_adapter"] = ComponentStatus.UNINITIALIZED

        self._component_status.update(status)
        logger.info("health_check_completed", status={k: v.value for k, v in status.items()})
        return status

    def get_component_status(self) -> Dict[str, str]:
        return {k: v.value for k, v in self._component_status.items()}

    async def shutdown(self) -> None:
        logger.info("action_room_shutdown_start")

        if self._web_search is not None:
            try:
                await self._web_search.close()
                self._component_status["web_search"] = ComponentStatus.SHUTDOWN
            except Exception as e:
                logger.warning("web_search_shutdown_error", error=str(e))

        if self._filesystem is not None:
            try:
                await self._filesystem.close()
                self._component_status["filesystem"] = ComponentStatus.SHUTDOWN
            except Exception as e:
                logger.warning("filesystem_shutdown_error", error=str(e))

        self._component_status["registry"] = ComponentStatus.SHUTDOWN
        self._component_status["code_executor"] = ComponentStatus.SHUTDOWN
        self._component_status["skills"] = ComponentStatus.SHUTDOWN

        if self._simulation is not None:
            try:
                self._component_status["simulation"] = ComponentStatus.SHUTDOWN
            except Exception as e:
                logger.warning("simulation_shutdown_error", error=str(e))

        if self._mcp_adapter is not None:
            try:
                self._component_status["mcp_adapter"] = ComponentStatus.SHUTDOWN
            except Exception as e:
                logger.warning("mcp_adapter_shutdown_error", error=str(e))

        self._filesystem = None
        self._code_executor = None
        self._web_search = None
        self._skill_manager = None
        self._perception = None
        self._action_module = None
        self._simulation = None
        self._mcp_adapter = None
        self._initialized = False

        logger.info("action_room_shutdown_complete")

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def filesystem(self) -> Optional[AgentFileSystem]:
        return self._filesystem

    @property
    def code_executor(self) -> Optional[CodeExecutor]:
        return self._code_executor

    @property
    def web_search(self) -> Optional[WebSearch]:
        return self._web_search

    @property
    def skill_manager(self) -> Optional[SkillTreeManager]:
        return self._skill_manager

    @property
    def perception(self) -> Optional[Any]:
        return self._perception

    @property
    def workspace_root(self) -> str:
        return self._workspace_root