import base64
import re
from dataclasses import dataclass, field
from enum import Enum

from nan_agent.logging.logger import get_logger
from nan_agent.model.types import MultiModalInput, MultiModalOutput

logger = get_logger(__name__)


class DrawOfThoughtStage(Enum):
    CONCEPT_SKETCH = "concept_sketch"
    CANVAS_PLANNING = "canvas_planning"
    SHAPE_DECOMPOSITION = "shape_decomposition"
    COORDINATE_CALCULATION = "coordinate_calculation"
    STYLE_COLORING = "style_coloring"
    FINAL_ASSEMBLY = "final_assembly"


STAGE_DISPLAY_NAMES = {
    DrawOfThoughtStage.CONCEPT_SKETCH: "Concept Sketch",
    DrawOfThoughtStage.CANVAS_PLANNING: "Canvas Planning",
    DrawOfThoughtStage.SHAPE_DECOMPOSITION: "Shape Decomposition",
    DrawOfThoughtStage.COORDINATE_CALCULATION: "Coordinate Calculation",
    DrawOfThoughtStage.STYLE_COLORING: "Style Coloring",
    DrawOfThoughtStage.FINAL_ASSEMBLY: "Final Assembly",
}

STAGE_ORDER = [
    DrawOfThoughtStage.CONCEPT_SKETCH,
    DrawOfThoughtStage.CANVAS_PLANNING,
    DrawOfThoughtStage.SHAPE_DECOMPOSITION,
    DrawOfThoughtStage.COORDINATE_CALCULATION,
    DrawOfThoughtStage.STYLE_COLORING,
    DrawOfThoughtStage.FINAL_ASSEMBLY,
]


@dataclass
class SVGResult:
    svg_code: str
    stages_output: dict[str, str] = field(default_factory=dict)
    reasoning_context: str = ""
    metadata: dict = field(default_factory=dict)


class DrawOfThought:
    def __init__(self, cognition):
        self.cognition = cognition

    def get_stages(self) -> list[str]:
        return [STAGE_DISPLAY_NAMES[stage] for stage in STAGE_ORDER]

    async def generate(self, reasoning_prompt: str, context: str = "") -> SVGResult:
        full_prompt = self._build_combined_prompt(reasoning_prompt, context)
        inp = MultiModalInput()
        inp.add_text(full_prompt)

        try:
            output = await self.cognition.infer(inp)
            response_text = output.text if output else ""
        except Exception as e:
            logger.error("dwot_generate_failed", error=str(e), prompt=reasoning_prompt[:200])
            raise RuntimeError(f"Draw-of-Thought generation failed: {e}") from e

        svg_code = self._extract_svg(response_text)
        if not svg_code:
            svg_code = response_text

        svg_code = self.sanitize_svg(svg_code)

        return SVGResult(
            svg_code=svg_code,
            metadata={
                "prompt": reasoning_prompt,
                "stages": self.get_stages(),
                "mode": "single_pass",
            },
        )

    async def generate_multi_stage(self, reasoning_prompt: str) -> SVGResult:
        stages_output = {}
        previous_output = ""
        combined_responses = []

        for stage in STAGE_ORDER:
            stage_prompt = self.get_prompt_for_stage(stage, reasoning_prompt, previous_output)
            inp = MultiModalInput()
            inp.add_text(stage_prompt)

            try:
                output = await self.cognition.infer(inp)
                response_text = output.text if output else ""
            except Exception as e:
                logger.error(
                    "dwot_multi_stage_failed",
                    error=str(e),
                    stage=stage.value,
                    prompt=reasoning_prompt[:200],
                )
                raise RuntimeError(
                    f"Draw-of-Thought multi-stage generation failed at {stage.value}: {e}"
                ) from e

            stage_name = STAGE_DISPLAY_NAMES[stage]
            stages_output[stage_name] = response_text
            combined_responses.append(f"[{stage_name}]\n{response_text}")
            previous_output = response_text

        full_response = "\n\n".join(combined_responses)
        svg_code = self._extract_svg(full_response)
        if not svg_code:
            svg_code = self._extract_svg(previous_output)
        svg_code = self.sanitize_svg(svg_code)

        return SVGResult(
            svg_code=svg_code,
            stages_output=stages_output,
            metadata={
                "prompt": reasoning_prompt,
                "stages": self.get_stages(),
                "mode": "multi_stage",
            },
        )

    def _build_combined_prompt(self, reasoning_prompt, context=""):
        stage_descriptions = "\n".join(
            f"{i + 1}. {STAGE_DISPLAY_NAMES[s]}: {self._get_stage_description(s)}"
            for i, s in enumerate(STAGE_ORDER)
        )

        prompt_parts = [
            "You are a visual reasoning engine that uses Draw-of-Thought to generate SVG.",
            "",
            "CRITICAL: You are NOT limited to flowcharts. SVG can represent ANYTHING:",
            "- Concept maps, mind maps, knowledge graphs",
            "- System architecture diagrams, component relationships",
            "- Data visualizations: bar charts, line charts, scatter plots, heat maps",
            "- Mathematical graphs, geometric proofs, coordinate systems",
            "- State machines, decision trees, process flows",
            "- Timelines, comparison tables, hierarchical trees",
            "- Abstract representations of ideas, relationships, or structures",
            "- ANY visual form that helps understand or reason about the content",
            "",
            "YOU decide the best visual form based on the task. Do NOT default to flowcharts.",
            "",
            "Follow these 6 stages to produce your visual reasoning:",
            stage_descriptions,
            "",
            "After all stages, output the final SVG code enclosed in ```svg ... ``` tags.",
            "The SVG must be self-contained, responsive (viewBox), and visually clear.",
        ]

        if context:
            prompt_parts.insert(0, f"[Additional Context]\n{context}\n")

        prompt_parts.append(f"\n[Reasoning Task]\n{reasoning_prompt}")
        prompt_parts.append("\nNow work through each stage and produce the final SVG:")

        return "\n".join(prompt_parts)

    @staticmethod
    def _get_stage_description(stage):
        descriptions = {
            DrawOfThoughtStage.CONCEPT_SKETCH: "Analyze the task. Decide what visual form best serves it (concept map, architecture, chart, graph, etc.). Identify core entities and their relationships.",
            DrawOfThoughtStage.CANVAS_PLANNING: "Plan the layout — viewBox dimensions, spatial organization, compositional structure. How will the chosen visual form be arranged?",
            DrawOfThoughtStage.SHAPE_DECOMPOSITION: "Map each concept/entity/data point to concrete SVG elements — whatever shapes best express the idea.",
            DrawOfThoughtStage.COORDINATE_CALCULATION: "Calculate precise coordinates, sizes, and positions for all elements within the viewBox.",
            DrawOfThoughtStage.STYLE_COLORING: "Apply colors, fonts, strokes, and styling. Use visual properties to encode meaning (e.g., color for categories, size for importance).",
            DrawOfThoughtStage.FINAL_ASSEMBLY: "Assemble everything into a complete, valid SVG document with proper viewBox and xmlns.",
        }
        return descriptions[stage]

    def _extract_svg(self, text):
        if not text:
            return ""

        code_block_pattern = r"```(?:svg|html|xml)\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if "<svg" in match.lower():
                return match.strip()

        svg_tag_pattern = r"(<svg\b.*?</svg>)"
        matches = re.findall(svg_tag_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        return ""

    def is_valid_svg(self, svg):
        if not svg or not isinstance(svg, str):
            return False

        has_open = bool(re.search(r"<svg\b", svg, re.IGNORECASE))
        has_close = "</svg>" in svg
        if not (has_open and has_close):
            return False

        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(svg)
            return True
        except Exception:
            return bool(
                re.match(r"^\s*<svg\b", svg.strip(), re.IGNORECASE)
                and svg.strip().endswith("</svg>")
            )

    def sanitize_svg(self, svg_code):
        if not svg_code:
            return svg_code

        sanitized = re.sub(
            r"<script\b[^>]*>.*?</script>",
            "",
            svg_code,
            flags=re.DOTALL | re.IGNORECASE,
        )

        event_handlers = [
            "onload", "onerror", "onclick", "onmouseover", "onmouseout",
            "onfocus", "onblur", "onchange", "onsubmit", "onkeydown",
            "onkeyup", "onkeypress", "ondblclick", "onmousedown",
            "onmouseup", "onmousemove", "onscroll", "onresize",
        ]
        for handler in event_handlers:
            sanitized = re.sub(
                rf'\s+{handler}\s*=\s*"[^"]*"',
                "",
                sanitized,
                flags=re.IGNORECASE,
            )
            sanitized = re.sub(
                rf"\s+{handler}\s*=\s*'[^']*'",
                "",
                sanitized,
                flags=re.IGNORECASE,
            )

        sanitized = re.sub(
            r'xlink:href\s*=\s*"(?:https?://|javascript:)',
            'xlink:href="#"',
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r"xlink:href\s*=\s*'(?:https?://|javascript:)",
            "xlink:href='#'",
            sanitized,
            flags=re.IGNORECASE,
        )

        return sanitized

    def get_prompt_for_stage(self, stage, reasoning_prompt, previous_output=""):
        stage_name = STAGE_DISPLAY_NAMES[stage]
        stage_desc = self._get_stage_description(stage)

        prompt_parts = [
            f"You are executing Stage {STAGE_ORDER.index(stage) + 1} of 6 in Draw-of-Thought.",
            f"Stage: {stage_name}",
            f"Goal: {stage_desc}",
            f"\nReasoning task: {reasoning_prompt}",
        ]

        if previous_output:
            prompt_parts.append(f"\nOutput from the previous stage:\n{previous_output}")

        if stage == DrawOfThoughtStage.CONCEPT_SKETCH:
            prompt_parts.append(
                "\nAnalyze the task deeply. What is the user trying to understand or reason about? "
                "What visual form would BEST help them? Choose freely from: "
                "concept map, mind map, architecture diagram, data chart, mathematical graph, "
                "state machine, timeline, comparison table, relationship graph, abstract visualization, "
                "or any other form. Explain WHY this form was chosen. "
                "Identify the key entities, concepts, data points, or relationships to represent."
            )
        elif stage == DrawOfThoughtStage.CANVAS_PLANNING:
            prompt_parts.append(
                "\nPlan the visual layout based on the chosen form. "
                "Define viewBox dimensions appropriate for the content. "
                "Describe the spatial organization: positions of elements, grouping, hierarchy, flow. "
                "How will the viewer's eye move through the visualization?"
            )
        elif stage == DrawOfThoughtStage.SHAPE_DECOMPOSITION:
            prompt_parts.append(
                "\nMap each concept/entity/data point to specific SVG elements. "
                "Choose shapes that best express the meaning: rect for containers/blocks, "
                "circle for nodes/entities, path for connections/curves, "
                "text for labels, line for axes/connections, polygon for custom shapes, etc. "
                "List each element with its purpose."
            )
        elif stage == DrawOfThoughtStage.COORDINATE_CALCULATION:
            prompt_parts.append(
                "\nCalculate exact coordinates for all elements within the viewBox. "
                "Specify x, y, width, height, cx, cy, r, path data (d attribute), "
                "transform attributes, and any positioning values. "
                "Ensure proper spacing, alignment, and no overlapping unless intentional."
            )
        elif stage == DrawOfThoughtStage.STYLE_COLORING:
            prompt_parts.append(
                "\nDefine the visual styling. Choose a color palette that encodes meaning "
                "(e.g., different colors for different categories, saturation for intensity). "
                "Define stroke widths, font families, font sizes, opacity, gradients if useful. "
                "Ensure high contrast and readability. Make it visually polished."
            )
        elif stage == DrawOfThoughtStage.FINAL_ASSEMBLY:
            prompt_parts.append(
                "\nAssemble everything into a complete SVG document. "
                "Output ONLY the SVG code enclosed in ```svg ... ``` tags. "
                "Must include: proper viewBox, xmlns='http://www.w3.org/2000/svg', "
                "and all elements from previous stages. The SVG must be self-contained."
            )

        return "\n".join(prompt_parts)

    def render_svg_to_image(self, svg_code):
        svg_bytes = svg_code.encode("utf-8")
        encoded = base64.b64encode(svg_bytes).decode()
        return {
            "format": "svg",
            "mime_type": "image/svg+xml",
            "encoding": "base64",
            "data_size": len(svg_bytes),
            "preview": f"data:image/svg+xml;base64,{encoded}",
            "note": "Full SVG rendering to raster requires cairosvg or resvg-py. This returns SVG as data URL.",
        }

    def feedback_as_multimodal_input(self, svg_code):
        inp = MultiModalInput()
        inp.add_text(
            "[Draw-of-Thought Visual Feedback]\n"
            "The following SVG diagram was generated as part of the reasoning process. "
            "Use it to inform your next reasoning steps.\n"
        )

        svg_bytes = svg_code.encode("utf-8")
        encoded = base64.b64encode(svg_bytes).decode()
        inp.add_image_base64(encoded, mime_type="image/svg+xml")

        return inp