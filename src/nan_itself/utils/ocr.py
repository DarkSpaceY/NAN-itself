from PIL import Image
from loguru import logger
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel
import numpy as np

import cv2
from easyocr import Reader

class OCRResult(BaseModel):
    """OCR 识别结果"""
    text: str
    bbox: List[Tuple[float, float]]  # 四个角坐标
    confidence: float

class OCRProvider:
    """EasyOCR 封装，支持中文/英文/数字"""
    
    def __init__(
        self,
        lang_list: List[str],
        gpu: bool,
        model_storage_directory: str,
    ):
        """
        Args:
            lang_list: 语言列表
            gpu: 是否使用 GPU
            model_storage_directory: 模型存储目录
        """
        logger.info(f"Initializing OCRProvider: lang={lang_list}, gpu={gpu}")
        self.reader = Reader(
            lang_list=lang_list,
            gpu=gpu,
            model_storage_directory=model_storage_directory,
        )
        
        self._closed = False
    
    def parser(
        self,
        image: Image.Image,
        detail: int,
        paragraph: bool = False,
        min_size: int = 10,
    ) -> List[OCRResult]:
        """
        从图片中提取文字
        
        Args:
            image: PIL Image 对象
            detail: 0=只返回文字, 1=返回文字+坐标, 2=返回更多细节
            paragraph: 是否合并为段落
            min_size: 最小文字尺寸
        
        Returns:
            OCRResult 列表
        """
        # 转换为 numpy 数组
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # EasyOCR 返回: list of [bbox, text, confidence]
        results = self.reader.readtext(
            img,
            detail=detail,
            paragraph=paragraph,
            min_size=min_size,
        )
        
        ocr_results = []
        for item in results:
            if detail == 0:
                # 只返回文字时, 格式是 [text]
                ocr_results.append(OCRResult(
                    text=item,
                    bbox=[],
                    confidence=0.0,
                ))
            else:
                # 格式: [bbox, text, confidence]
                bbox, text, confidence = item
                ocr_results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                ))
        
        logger.info(f"OCR extracted {len(ocr_results)} text blocks")
        return ocr_results
    
    def close(self) -> None:
        if not self._closed:
            logger.info("Closing OCRProvider")
            if self.reader:
                del self.reader
                self.reader = None
            self._closed = True