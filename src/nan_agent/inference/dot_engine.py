"""
DotEngine - Draw-of-Thought Engine
高性能多模态推理引擎，基于 DwT (Drawing-with-Thought) 范式

公共 API 只有两个方法：
  reason(requirement) → DotEngineResult   # 画图辅助推理
  visualize(svg_code) → MultiModalInput   # 把图转回多模态输入，注入推理链
"""

import base64
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from nan_agent.inference.draw_of_thought import DrawOfThought, SVGResult
from nan_agent.logging.logger import get_logger
from nan_agent.model.types import ImagePart, MultiModalInput

logger = get_logger(__name__)


@dataclass
class DotEngineResult:
    """DotEngine 推理结果"""
    svg_code: str
    reasoning_chain: list[dict] = field(default_factory=list)
    visual_feedback: Optional[ImagePart] = None
    metadata: dict = field(default_factory=dict)
    execution_time_ms: float = 0.0


class DotEngine:
    """
    Draw-of-Thought Engine - 把思维画出来辅助推理。

    使用方式（TA/GoT 通用）：
        result = await dot.reason("帮我理解用户模块和订单模块之间的依赖关系")
        # result.svg_code 就是可用于 visual feedback 的 SVG

        multi_input = dot.visualize(result.svg_code, "模块依赖关系图")
        # 把 multi_input 注入下一轮推理，形成视觉反馈闭环

    基于 DwT (Drawing-with-Thought) 六阶段推理链：
    1. Concept Sketching → 分析需求，选择最佳视觉形式
    2. Canvas Planning   → 规划布局和空间结构
    3. Shape Decomposition → 映射概念到 SVG 元素
    4. Coordinate Calculation → 精确计算坐标
    5. Styling and Coloring → 配色和样式
    6. Final Assembly → 组装完整 SVG
    """

    def __init__(self, cognition, config: Optional[dict] = None):
        self.cognition = cognition
        self.config = config or {}
        self.dwt = DrawOfThought(cognition=cognition)

        # ── 渲染器 ──
        self._renderer = _SVGRenderer()

        # ── 统计 ──
        self._stats = {"calls": 0, "total_ms": 0}

    # ═════════════════════════════════════════════════════════════════
    # 公共 API（只有这两个）
    # ═════════════════════════════════════════════════════════════════

    async def reason(self, requirement: str) -> DotEngineResult:
        """画图辅助推理。TA/GoT 只管描述需求，引擎自主决定画什么、怎么画。

        Args:
            requirement: 推理需求，自然语言描述即可
        Returns:
            DotEngineResult(.svg_code, .reasoning_chain, .visual_feedback)
        """
        t0 = time.time()
        svg_result = await self.dwt.generate_multi_stage(requirement)

        reasoning_chain = [
            {"stage": name, "output": output}
            for name, output in svg_result.stages_output.items()
        ]

        visual_feedback = self._renderer.svg_to_image_part(svg_result.svg_code)

        elapsed = (time.time() - t0) * 1000
        self._stats["calls"] += 1
        self._stats["total_ms"] += elapsed

        return DotEngineResult(
            svg_code=svg_result.svg_code,
            reasoning_chain=reasoning_chain,
            visual_feedback=visual_feedback,
            metadata=svg_result.metadata,
            execution_time_ms=elapsed,
        )

    def visualize(self, svg_code: str, description: str = "") -> MultiModalInput:
        """把 SVG 转回多模态输入，注入推理链形成视觉反馈闭环。

        Args:
            svg_code: reason() 返回的 SVG
            description: 可选描述
        Returns:
            MultiModalInput，可直接作为下一轮推理的输入
        """
        inp = MultiModalInput()
        text = "[Visual Reasoning]\n"
        if description:
            text += f"{description}\n\n"
        text += "Previous reasoning visualized as SVG. Use this to inform the next step:\n"
        inp.add_text(text)

        image_part = self._renderer.svg_to_image_part(svg_code)
        if image_part:
            inp.parts.append(image_part)

        return inp

    # ═════════════════════════════════════════════════════════════════
    # 统计 & 健康检查
    # ═════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        return {
            "calls": self._stats["calls"],
            "avg_ms": self._stats["total_ms"] / max(self._stats["calls"], 1),
        }

    async def health_check(self) -> bool:
        try:
            cog_ok = await self.cognition.health_check()
            return cog_ok and self.dwt is not None and self._renderer is not None
        except Exception as e:
            logger.error("dot_health_check_failed", error=str(e))
            return False


# ═════════════════════════════════════════════════════════════════
# 内部：SVG 渲染器
# ═════════════════════════════════════════════════════════════════

class _SVGRenderer:
    """SVG → PNG/ImagePart 转换（内部使用）"""

    def __init__(self):
        self._has_cairo = False
        self._has_pil = False
        try:
            import cairosvg  # noqa: F401
            self._has_cairo = True
        except ImportError as e:
            logger.debug("dot_engine_cairosvg_not_available", error=str(e))
            self._has_cairo = False
        try:
            from PIL import Image  # noqa: F401
            self._has_pil = True
        except ImportError as e:
            logger.debug("dot_engine_pil_not_available", error=str(e))

    def render(self, svg_code: str, width: int = 512, height: int = 512) -> Optional[bytes]:
        if not svg_code or not self._has_pil:
            return None
        if self._has_cairo:
            try:
                import cairosvg
                return cairosvg.svg2png(bytestring=svg_code.encode(), output_width=width, output_height=height)
            except Exception as e:
                logger.warning("cairosvg_failed", error=str(e))
        return None

    def svg_to_image_part(self, svg_code: str) -> Optional[ImagePart]:
        if not svg_code:
            return None
        png = self.render(svg_code)
        if png:
            return ImagePart(source_type="base64", source=base64.b64encode(png).decode(), mime_type="image/png")
        encoded = base64.b64encode(svg_code.encode()).decode()
        return ImagePart(source_type="base64", source=encoded, mime_type="image/svg+xml")