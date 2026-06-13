"""
Screen Perception Module - 基于 OmniParser 的纯视觉屏幕感知

YOLO 检测图标位置 → Florence2 生成图标描述 → 输出结构化元素列表
无 OCR（由主视觉模型 Gamma 自行理解文字），跨平台通用。
"""

import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

# YOLO 推理用的最大边长（大幅压缩提速）
YOLO_MAX_DIM = 800
# Florence2 图标裁剪尺寸
CAPTION_CROP_SIZE = 48
# Florence2 生成最大 token 数
CAPTION_MAX_NEW_TOKENS = 12


@dataclass
class UIElement:
    """OmniParser 解析出的 UI 元素"""
    id: int
    type: str  # "icon" | "text"
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2] 归一化坐标
    interactivity: bool
    content: str
    source: str

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def center_px(self, screen_width: int = 1920, screen_height: int = 1080) -> Tuple[int, int]:
        cx, cy = self.center
        return (int(cx * screen_width), int(cy * screen_height))


@dataclass
class ParsedScreen:
    """解析后的屏幕结果"""
    elements: List[UIElement] = field(default_factory=list)
    labeled_image_base64: str = ""
    original_image_size: Tuple[int, int] = (0, 0)
    yolo_time_ms: float = 0.0
    caption_time_ms: float = 0.0

    def find_by_content(self, keyword: str, interactive_only: bool = True) -> List[UIElement]:
        results = []
        kw = keyword.lower()
        for el in self.elements:
            if interactive_only and not el.interactivity:
                continue
            if kw in el.content.lower():
                results.append(el)
        return results

    def get_interactive_elements(self) -> List[UIElement]:
        return [el for el in self.elements if el.interactivity]

    def get_element_by_id(self, element_id: int) -> Optional[UIElement]:
        for el in self.elements:
            if el.id == element_id:
                return el
        return None

    def format_for_agent(self, max_elements: int = 50) -> str:
        """格式化元素列表供 Agent（Gamma）阅读"""
        lines = [f"Screen: {self.original_image_size[0]}x{self.original_image_size[1]} "
                 f"({len(self.elements)} elements detected)\n"]
        for el in self.elements[:max_elements]:
            cx, cy = el.center
            tag = "[✦]" if el.interactivity else "[ ]"
            lines.append(f"  #{el.id:03d} {tag} "
                        f"({cx:.3f}, {cy:.3f}) {el.content}")
        return "\n".join(lines)


class OmniParserPerception:
    """
    OmniParser 屏幕感知引擎（优化版）

    YOLO 检测图标 → Florence2 生成描述 → 结构化输出
    无 OCR，文字理解交给主视觉模型 Gamma。
    """

    def __init__(
        self,
        model_path: str = "weights/icon_detect/model.pt",
        caption_model_path: str = "weights/icon_caption_florence",
        device: str = "auto",
        box_threshold: float = 0.01,
        iou_threshold: float = 0.7,
    ):
        self.model_path = model_path
        self.caption_model_path = caption_model_path
        self.device = self._resolve_device(device)
        self.box_threshold = box_threshold
        self.iou_threshold = iou_threshold

        self._som_model = None
        self._caption_model = None
        self._caption_processor = None

        logger.info("omniparser_init", device=self.device)

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    # ═══════════════════════════════════════════════════════════
    # 模型加载
    # ═══════════════════════════════════════════════════════════

    def _load_models(self):
        if self._som_model is not None:
            return
        logger.info("loading_models")

        from ultralytics import YOLO
        self._som_model = YOLO(self.model_path)
        # YOLO 自动管理设备，不要手动 to(device)

        try:
            from transformers import (
                AutoProcessor, AutoModelForCausalLM, AutoConfig, RobertaTokenizer
            )
            import torch

            if not hasattr(RobertaTokenizer, 'additional_special_tokens'):
                RobertaTokenizer.additional_special_tokens = []
            if not hasattr(RobertaTokenizer, 'additional_special_tokens_ids'):
                RobertaTokenizer.additional_special_tokens_ids = []

            config = AutoConfig.from_pretrained(
                self.caption_model_path, trust_remote_code=True
            )
            if hasattr(config, 'language_config') and config.language_config is not None:
                if not hasattr(config.language_config, 'forced_bos_token_id'):
                    config.language_config.forced_bos_token_id = None

            self._caption_processor = AutoProcessor.from_pretrained(
                self.caption_model_path, trust_remote_code=True
            )
            self._caption_model = AutoModelForCausalLM.from_pretrained(
                self.caption_model_path,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(self.device)

            if not hasattr(self._caption_model, '_supports_sdpa'):
                self._caption_model._supports_sdpa = False

            logger.info("florence2_loaded")
        except Exception as e:
            logger.warning("florence2_load_failed", error=str(e))
            self._caption_model = None
            self._caption_processor = None

        logger.info("models_loaded")

    # ═══════════════════════════════════════════════════════════
    # 核心解析
    # ═══════════════════════════════════════════════════════════

    def parse(
        self,
        image: Any,
        fast: bool = False,
        caption_batch_size: int = 128,
        max_captions: int = 0,
    ) -> ParsedScreen:
        """
        解析屏幕截图。

        Args:
            image: PIL.Image 或文件路径
            fast: True = 只跑 YOLO 不跑 Florence2
            caption_batch_size: Florence2 批处理大小
            max_captions: 最多描述多少个图标（0=全部，CPU 建议设 20 以内）
        """
        self._load_models()

        from PIL import Image
        import numpy as np

        t0 = time.time()

        if isinstance(image, str):
            image = Image.open(image)
        image_rgb = image.convert("RGB")
        original_size = image.size

        # ── 1. YOLO：检测图标 ──
        # 不预缩放（YOLO 内部有 imgsz 参数自己处理），传原图。
        # 但超大连屏可以适当缩小以加快。
        t_yolo_start = time.time()

        # 超大屏（>1600px）压缩到 1600 提速，YOLO 仍能正常检测
        if max(original_size) > 1600:
            work_image = self._resize_for_yolo(image_rgb, 1600)
        else:
            work_image = image_rgb

        yolo_results = self._som_model(work_image, verbose=False, conf=self.box_threshold)

        scale_x = original_size[0] / work_image.size[0]
        scale_y = original_size[1] / work_image.size[1]

        boxes = []
        for result in yolo_results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes.append([
                        xyxy[0] * scale_x / original_size[0],
                        xyxy[1] * scale_y / original_size[1],
                        xyxy[2] * scale_x / original_size[0],
                        xyxy[3] * scale_y / original_size[1],
                    ])

        yolo_time = (time.time() - t_yolo_start) * 1000

        # ── 2. Florence2：图标描述（可跳过） ──
        icon_elements = []
        caption_time = 0.0

        if boxes and not fast and self._caption_model is not None:
            t_cap = time.time()
            logger.info("caption_start", icon_count=len(boxes))
            n_cap = max_captions if max_captions > 0 else len(boxes)
            try:
                icon_elements = self._caption_all(
                    image_rgb, boxes[:n_cap], original_size, caption_batch_size
                )
                # 剩余未描述图标用 plain 模式
                if n_cap < len(boxes):
                    icon_elements += self._plain_icons(boxes[n_cap:], start_id=n_cap)
            except Exception as e:
                logger.warning("caption_failed", error=str(e))
                icon_elements = self._plain_icons(boxes)
            caption_time = (time.time() - t_cap) * 1000
        elif boxes:
            icon_elements = self._plain_icons(boxes)

        elements = icon_elements

        # ── 3. 生成标注图 ──
        labeled_image_base64 = self._draw_labeled_image(image_rgb, elements)

        total_time = (time.time() - t0) * 1000
        logger.info("screen_parsed",
                     elements=len(elements),
                     interactive=sum(1 for e in elements if e.interactivity),
                     yolo_ms=int(yolo_time),
                     caption_ms=int(caption_time),
                     total_ms=int(total_time))

        return ParsedScreen(
            elements=elements,
            labeled_image_base64=labeled_image_base64,
            original_image_size=original_size,
            yolo_time_ms=yolo_time,
            caption_time_ms=caption_time,
        )

    # ═══════════════════════════════════════════════════════════
    # 内部
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _resize_for_yolo(image, max_dim=YOLO_MAX_DIM):
        """压缩到 max_dim 以内，保持宽高比"""
        from PIL import Image
        w, h = image.size
        if max(w, h) <= max_dim:
            return image
        ratio = max_dim / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, resample=Image.LANCZOS)

    def _caption_all(self, image, boxes, original_size, batch_size):
        """Florence2 批量生成图标描述"""
        from PIL import Image
        import torch

        crops = []
        valid_boxes = []
        for box in boxes[:batch_size]:
            xmin = int(box[0] * original_size[0])
            xmax = int(box[2] * original_size[0])
            ymin = int(box[1] * original_size[1])
            ymax = int(box[3] * original_size[1])
            xmin, xmax = max(0, xmin), min(original_size[0], xmax)
            ymin, ymax = max(0, ymin), min(original_size[1], ymax)
            if xmax > xmin and ymax > ymin:
                crop = image.crop((xmin, ymin, xmax, ymax))
                crop = crop.resize((CAPTION_CROP_SIZE, CAPTION_CROP_SIZE),
                                   resample=Image.LANCZOS)
                crops.append(crop)
                valid_boxes.append(box)

        if not crops:
            return []

        logger.info("caption_inference", crop_count=len(crops))

        prompt = "<CAPTION>"
        inputs = self._caption_processor(
            images=crops,
            text=[prompt] * len(crops),
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self._caption_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=CAPTION_MAX_NEW_TOKENS,
                num_beams=1,
                do_sample=False,
            )

        texts = self._caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        texts = [t.strip() for t in texts]

        return [
            UIElement(
                id=idx, type="icon", bbox=tuple(b),
                interactivity=True, content=c, source="box_yolo_content_yolo",
            )
            for idx, (b, c) in enumerate(zip(valid_boxes, texts))
        ]

    @staticmethod
    def _plain_icons(boxes, start_id=0):
        """纯 YOLO 模式：无描述、只标位置"""
        return [
            UIElement(
                id=start_id + idx, type="icon", bbox=tuple(b),
                interactivity=True, content=f"icon_{start_id + idx}",
                source="box_yolo_content_yolo",
            )
            for idx, b in enumerate(boxes)
        ]

    def _draw_labeled_image(self, image, elements):
        """在原图上绘制编号标注框"""
        try:
            from PIL import ImageDraw

            labeled = image.copy()
            draw = ImageDraw.Draw(labeled)
            w, h = image.size

            for el in elements:
                x1 = int(el.bbox[0] * w)
                y1 = int(el.bbox[1] * h)
                x2 = int(el.bbox[2] * w)
                y2 = int(el.bbox[3] * h)
                color = (0, 255, 0) if el.interactivity else (255, 0, 0)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label = f"#{el.id}"
                draw.text((x1, max(0, y1 - 12)), label, fill=color)

            buf = io.BytesIO()
            labeled.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning("draw_failed", error=str(e))
            return ""

    def parse_screenshot_bytes(self, screenshot_bytes: bytes, **kwargs) -> ParsedScreen:
        from PIL import Image
        return self.parse(Image.open(io.BytesIO(screenshot_bytes)), **kwargs)

    def health_check(self) -> bool:
        try:
            self._load_models()
            return self._som_model is not None
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return False


_perception_instance: Optional[OmniParserPerception] = None


def get_perception(
    model_path: Optional[str] = None,
    caption_model_path: Optional[str] = None,
    **kwargs
) -> OmniParserPerception:
    global _perception_instance
    if _perception_instance is None:
        _perception_instance = OmniParserPerception(
            model_path=model_path or "weights/icon_detect/model.pt",
            caption_model_path=caption_model_path or "weights/icon_caption_florence",
            **kwargs
        )
    return _perception_instance
