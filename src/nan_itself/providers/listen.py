from loguru import logger
from pathlib import Path
from collections import deque
import threading

from pywhispercpp.examples.assistant import Assistant

class SpeechProvider:
    """pywhispercpp 封装（Whisper 语音识别）"""
    
    def __init__(
        self,
        models_dir: Path,
        queue_maxsize: int,
        model_size: str = "base",
        n_threads: int = 4,
    ):
        """
        Args:
            models_dir: 模型文件目录
            queue_maxsize: 队列最大大小
            model_size: 模型大小 tiny/base/small/medium/large
            n_threads: 线程数
        """
        logger.info(f"Initializing SpeechProvider: model_size={model_size}")
        
        self._messages = deque(maxlen=queue_maxsize)
        self.assistant = Assistant(
            model=model_size,
            commands_callback=self.callback,
            n_threads=n_threads,
            models_dir=models_dir,
        )
        self._running = False

    def callback(self, text: str) -> None:
        """回调函数，用于接收识别结果"""
        logger.debug(f"Speech recognition callback: {text}")
        self._messages.append(text)

    def get_messages(self) -> list[str]:
        """获取识别结果消息"""
        messages = list(self._messages)
        return messages

    def start(self) -> None:
        """开始语音识别"""
        self._thread = threading.Thread(
            target=self.assistant.start,
            daemon=True,
            name="NANAgentWhisperAssistantThread",
        )
        self._running = True
        self._thread.start()
        logger.info("Starting speech recognition")

    def close(self) -> None: 
        """停止语音识别"""
        if self._running:
            self.assistant.running = False
            self._thread.join(timeout=3.0)
            self.assistant = None
            self._thread = None
            self._messages = None
            logger.info("Stopping speech recognition")
            self._running = False
            