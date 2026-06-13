"""
事件总线模块 - EventBus

本模块提供了一个基于异步模式的事件总线实现，是 NAN-Agent 各组件间
松耦合通信的核心基础设施。

核心功能：
- 支持 glob 风格的事件模式匹配（如 "lifecycle.*" 匹配所有生命周期事件）
- 异步事件分发：所有 handler 通过 asyncio.gather 并发执行
- 静默错误处理：单个 handler 的异常不会影响其他 handler 的执行

使用示例：
    bus = EventBus()

    @bus.on("lifecycle.*")
    async def on_lifecycle(event_name, **kwargs):
        print(f"Lifecycle event: {event_name}")

    await bus.emit("lifecycle.running", state="running")

设计说明：
- EventBus 是 NAN-Agent 中"神经系统"的实现，类似生物体的神经信号传递机制
- 各个组件（HardMemory、SoftMemory、SelfValue、ActionRoom 等）通过事件总线
  感知彼此的状态变化，而无需直接持有对方的引用
- 这种设计降低了组件间的耦合度，便于独立测试和替换

相关文件：
- main.py：NANAgent 在 initialize() 中创建 EventBus 实例并传递给各组件
- lifecycle.py：LifecycleManager 通过 EventBus 发出生命周期状态变更事件
- action_room/interface.py：ActionRoom 通过 EventBus 接收和发送事件
"""

import asyncio
import fnmatch
from collections import defaultdict


class EventBus:
    """
    异步事件总线

    提供基于 glob 模式匹配的事件订阅与分发机制。支持多个 handler 订阅
    同一事件模式，事件触发时所有匹配的 handler 并发执行。

    Attributes:
        _handlers (defaultdict[str, set]): 事件模式到 handler 集合的映射。
            键为 glob 模式字符串（如 "lifecycle.*"），值为可调用对象的集合。
    """

    def __init__(self):
        """初始化事件总线，创建空的 handler 注册表。"""
        self._handlers = defaultdict(set)

    def on(self, event_pattern: str):
        """
        装饰器：将函数注册为指定事件模式的 handler。

        使用 glob 风格的模式匹配：
        - "*"  匹配除点号外的任意字符
        - "?"  匹配单个字符
        - "[seq]" 匹配序列中的任意字符

        Args:
            event_pattern: 事件匹配模式，如 "lifecycle.*"、"*.error"

        Returns:
            callable: 装饰器函数，将原始 handler 注册后原样返回

        Example:
            @bus.on("lifecycle.*")
            async def handle_lifecycle(event_name, **kwargs):
                ...
        """

        def decorator(handler):
            self._handlers[event_pattern].add(handler)
            return handler

        return decorator

    async def emit(self, event_name: str, *args, **kwargs):
        """
        触发事件，通知所有匹配的 handler。

        遍历所有已注册的事件模式，使用 fnmatch 进行模式匹配，
        将匹配的 handler 通过 asyncio.gather 并发执行。
        单个 handler 的异常会被静默捕获，不会影响其他 handler。

        Args:
            event_name: 事件名称，如 "lifecycle.running"、"memory.consolidated"
            *args: 传递给 handler 的位置参数
            **kwargs: 传递给 handler 的关键字参数
        """
        handlers = set()
        for pattern, subs in self._handlers.items():
            if fnmatch.fnmatch(event_name, pattern):
                handlers.update(subs)

        if not handlers:
            return

        tasks = [_dispatch(handler, event_name, *args, **kwargs) for handler in handlers]
        await asyncio.gather(*tasks)


async def _dispatch(handler, event_name, *args, **kwargs):
    """
    内部函数：执行单个事件 handler 并静默捕获异常。

    设计上，单个 handler 的失败不应影响其他 handler 的执行，
    因此所有异常在此处被静默捕获。

    Args:
        handler: 可调用的事件处理函数
        event_name: 触发的事件名称（传递给 handler 的第一个参数）
        *args: 额外位置参数
        **kwargs: 额外关键字参数
    """
    try:
        await handler(event_name, *args, **kwargs)
    except Exception:
        pass