from collections import deque
from RealtimeSTT import AudioToTextRecorder
import multiprocessing as mp
from pathlib import Path
from typing import List
from loguru import logger


def _worker_process(model_path: Path, messages_queue, running_event, maxsize: int) -> None:
    """
    子进程运行的函数
    
    Args:
        model_path: 模型路径
        messages_queue: 共享的 deque 队列
        running_event: 运行状态 Event
        maxsize: 队列最大长度
    """
    # 在子进程中创建 AudioToTextRecorder
    recorder = AudioToTextRecorder(model="base", download_root=model_path)
    recorder.start()
    
    try:
        # 持续运行直到 running_event 被清除
        while running_event.is_set():
            text = recorder.text()
            if text:
                messages_queue.append(text)
                # 手动维护队列长度
                if len(messages_queue) > maxsize:
                    messages_queue.pop(0)  # 移除最早的元素
    finally:
        recorder.shutdown()


class STTProvider:
    def __init__(self, model_path:Path, maxsize=10):
        self.model_path = model_path

        self.manager = mp.Manager()
        self._messages = self.manager.list()
        self._maxsize = maxsize

        self._running = mp.Event()
        self._running.set()  # 初始为运行状态

        logger.info(f"STTProvider initialized with model_path: {model_path}, maxsize: {maxsize}")

        self._process = None

    def start(self) -> None:
        """启动子进程"""
        self._running.set()
        self._process = mp.Process(
            target=_worker_process,
            args=(
                self.model_path,      # Path 可序列化
                self._messages,       # 共享队列（Manager 代理）
                self._running,        # Event 可序列化
                self._maxsize         # int 可序列化
            )
        )
        self._process.start()

    def get_messages(self) -> List[str]:
        return list(self._messages)

    def close(self) -> None:
        """清理资源"""
        self._running.clear()
        
        # 先关闭进程
        if self._process:
            self._process.join()
            self._process = None
        
        # 再关闭 manager
        if self.manager:
            self.manager.shutdown()

        self._messages = None
        logger.info("STTProvider closed")