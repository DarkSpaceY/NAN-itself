"""
异常定义模块 - NAN-Agent 异常体系

本模块定义了 NAN-Agent 项目中使用的所有自定义异常类，采用层级化设计，
便于精确的异常捕获和错误溯源。

异常层次结构：
    NANBaseException (E000)
    ├── ConfigError (E100)     —— 配置相关错误
    ├── ModelError (E200)      —— 模型/推理相关错误
    ├── StorageError (E300)    —— 存储相关错误
    │   └── MemoryError (E310) —— 记忆系统特定错误
    ├── InferenceError (E400)  —— 图推理引擎错误
    ├── ActionError (E500)     —— 动作执行错误
    ├── SelfValueError (E600)  —— 自我价值/人格系统错误
    ├── SoftMemoryError (E700) —— 软记忆/微调相关错误
    └── LifecycleError (E800)  —— 生命周期管理错误

错误码设计原则：
- 格式：[E + 3位数字]，如 E100、E310
- 百位数表示异常大类（1=配置, 2=模型, 3=存储, 4=推理, 5=动作,
  6=自我价值, 7=软记忆, 8=生命周期）
- 基本错误码可在实例化时覆盖（如 LifecycleError 使用 E802 表示
  具体的非法状态转换错误）

每个异常实例包含三个属性：
- message: 人类可读的错误描述
- error_code: 错误码（用于日志、监控和自动化处理）
- details: 附加上下文字典（如涉及的状态、参数值等）

使用示例：
    raise ConfigError(
        "Invalid model configuration",
        error_code="E101",
        details={"field": "model_name", "value": "unknown_model"}
    )

    try:
        ...
    except StorageError as e:  # 捕获所有存储相关错误（包括 MemoryError）
        logger.error(f"Storage failure: {e}")
"""


class NANBaseException(Exception):
    """
    NAN-Agent 异常基类

    所有自定义异常的公共父类，提供统一的错误信息结构。
    继承自 Python 内置 Exception。

    Attributes:
        message (str): 人类可读的错误描述信息
        error_code (str): 错误码，格式 [E + 3位数字]，默认 "E000"
        details (dict): 附加上下文信息字典，默认为空字典
    """

    def __init__(self, message, error_code="E000", details=None):
        """
        初始化异常实例。

        Args:
            message: 错误描述文本
            error_code: 错误码，默认 "E000"（基类默认码）
            details: 额外上下文信息字典，默认 None（将被设为空字典）
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

    def __str__(self):
        """返回格式化的错误字符串：[错误码] 错误信息"""
        return f"[{self.error_code}] {self.message}"


# ── 配置层异常 (E100-E199) ──────────────────────────────────────────

class ConfigError(NANBaseException):
    """
    配置错误

    用于配置文件加载失败、配置项验证不通过、配置格式错误等场景。

    Example:
        raise ConfigError("Missing required field: model.base_url",
                          error_code="E101")
    """

    def __init__(self, message, error_code="E100", details=None):
        super().__init__(message, error_code=error_code, details=details)


# ── 模型层异常 (E200-E299) ──────────────────────────────────────────

class ModelError(NANBaseException):
    """
    模型相关错误

    用于模型加载失败、推理请求超时、模型响应解析错误等场景。
    涵盖 Ollama 提供器、Cognition 推理过程中的所有异常。

    Example:
        raise ModelError("Ollama service unreachable",
                         error_code="E201",
                         details={"base_url": "http://localhost:11434"})
    """

    def __init__(self, message, error_code="E200", details=None):
        super().__init__(message, error_code=error_code, details=details)


# ── 存储层异常 (E300-E399) ──────────────────────────────────────────

class StorageError(NANBaseException):
    """
    存储相关错误

    用于向量数据库、图数据库、状态存储、Blob 存储等持久化操作中的异常。
    是 MemoryError 的父类。

    Example:
        raise StorageError("Failed to persist vector index",
                           error_code="E301")
    """

    def __init__(self, message, error_code="E300", details=None):
        super().__init__(message, error_code=error_code, details=details)


class MemoryError(StorageError):
    """
    记忆系统错误（存储子类）

    用于 HardMemory 组件的特定错误，如记忆单元创建失败、
    记忆检索异常、经验树蒸馏失败等场景。

    Example:
        raise MemoryError("Failed to consolidate memory cells",
                          error_code="E311")
    """

    def __init__(self, message, error_code="E310", details=None):
        super().__init__(message, error_code=error_code, details=details)


# ── 推理层异常 (E400-E499) ──────────────────────────────────────────

class InferenceError(NANBaseException):
    """
    图推理引擎错误

    用于 GoT（Graph of Thought）引擎的运行异常，如图节点创建失败、
    推理循环中断、思维图状态不一致等场景。

    Example:
        raise InferenceError("GoT node pool overflow",
                             error_code="E401")
    """

    def __init__(self, message, error_code="E400", details=None):
        super().__init__(message, error_code=error_code, details=details)


# ── 动作层异常 (E500-E599) ──────────────────────────────────────────

class ActionError(NANBaseException):
    """
    动作执行错误

    用于 ActionRoom 中工具调用失败、代码执行异常、
    网络搜索失败、文件系统操作错误等场景。

    Example:
        raise ActionError("Code execution timed out",
                          error_code="E501",
                          details={"timeout": 30.0})
    """

    def __init__(self, message, error_code="E500", details=None):
        super().__init__(message, error_code=error_code, details=details)


# ── 自我价值层异常 (E600-E699) ──────────────────────────────────────

class SelfValueError(NANBaseException):
    """
    自我价值系统错误

    用于 SelfValue 组件的异常，如人格评估失败、情绪动力学计算错误、
    价值观冲突检测异常等场景。

    Example:
        raise SelfValueError("Failed to evaluate self-worth",
                             error_code="E601")
    """

    def __init__(self, message, error_code="E600", details=None):
        super().__init__(message, error_code=error_code, details=details)


# ── 软记忆层异常 (E700-E799) ────────────────────────────────────────

class SoftMemoryError(NANBaseException):
    """
    软记忆系统错误

    用于 SoftMemory 组件的异常，如 LoRA 适配器训练失败、
    学习周期异常、人格适配器热插拔错误等场景。

    Example:
        raise SoftMemoryError("LoRA adapter training diverged",
                              error_code="E701")
    """

    def __init__(self, message, error_code="E700", details=None):
        super().__init__(message, error_code=error_code, details=details)


# ── 生命周期层异常 (E800-E899) ──────────────────────────────────────

class LifecycleError(NANBaseException):
    """
    生命周期管理错误

    用于 LifecycleManager 的状态转换异常，如非法状态跳转、
    生命周期钩子超时、启动/关闭流程异常等场景。

    常用错误码：
    - E801: 钩子执行错误
    - E802: 非法状态转换

    Example:
        raise LifecycleError(
            f"Invalid state transition: RUNNING -> INITIALIZING",
            error_code="E802",
            details={"current": "RUNNING", "target": "INITIALIZING"}
        )
    """

    def __init__(self, message, error_code="E800", details=None):
        super().__init__(message, error_code=error_code, details=details)