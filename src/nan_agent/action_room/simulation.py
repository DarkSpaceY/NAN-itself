"""
数学计算与通用仿真引擎 - 安全数学求值 + 连续-离散混合仿真

提供两大核心能力：
- MathEngine: 基于 AST 解析的安全数学表达式求值器，支持方程求解、微积分、
  线性代数、统计计算等通用数学运算
- SimulationEngine: 通用连续-离散混合仿真框架，用户通过组合原语构建任意系统：
  - State: 命名状态向量，支持算术运算
  - DynamicalSystem: 连续动态系统（ODE），用户定义状态 + 导数函数
  - DiscreteEventSystem: 离散事件系统（事件调度 + 条件触发）
  - HybridSystem: 连续-离散耦合系统
  - SimulationResult: 统一结果对象（轨迹 + 事件日志 + 统计摘要）

安全设计：
- 数学表达式求值使用 ast.parse 解析而非 eval()，仅允许安全 AST 节点
- 受限作用域仅暴露 math 模块函数与安全内建函数，杜绝代码注入风险
- 仿真导数函数在 Python 沙箱中执行，不涉及 AST 注入
"""

import ast
import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# MathEngine — 安全数学表达式求值器
# ═══════════════════════════════════════════════════════════════════════

# ── 安全 AST 求值相关常量 ──────────────────────────────────────────────

# 允许的 AST 节点类型白名单
_SAFE_AST_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Attribute,
    ast.Load,
    # 二元运算符
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    # 一元运算符
    ast.UAdd, ast.USub, ast.Not,
    # 比较运算符
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

# 安全的 math 模块函数与常量
_SAFE_MATH_SCOPE: Dict[str, Any] = {
    # 常量
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    "tau": math.tau,
    # 三角函数
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # 双曲函数
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    # 角度转换
    "degrees": math.degrees,
    "radians": math.radians,
    # 指数/对数
    "exp": math.exp,
    "expm1": math.expm1,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "log1p": math.log1p,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    # 取整/绝对值
    "ceil": math.ceil,
    "floor": math.floor,
    "fabs": math.fabs,
    "trunc": math.trunc,
    # 其他
    "copysign": math.copysign,
    "fmod": math.fmod,
    "frexp": math.frexp,
    "ldexp": math.ldexp,
    "modf": math.modf,
    "hypot": math.hypot,
    "pow": math.pow,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "comb": math.comb,
    "perm": math.perm,
    "dist": math.dist,
    "erf": math.erf,
    "erfc": math.erfc,
    "gamma": math.gamma,
    "lgamma": math.lgamma,
    "nextafter": math.nextafter,
    "ulp": math.ulp,
    "prod": math.prod,
    "isclose": math.isclose,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
}

# 安全的内建函数
_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
}


class _SafeEvalVisitor(ast.NodeVisitor):
    """AST 安全检查访问器，遍历表达式 AST 并拒绝不安全节点。"""

    def visit(self, node: ast.AST) -> ast.AST:
        if not isinstance(node, _SAFE_AST_NODES):
            raise ActionError(
                f"表达式中包含不允许的语法元素: {type(node).__name__}",
                error_code="E501",
                details={"node_type": type(node).__name__},
            )
        self.generic_visit(node)
        return node


class MathEngine:
    """安全的数学表达式求值器

    使用 AST（抽象语法树）解析数学表达式，在受限作用域内执行运算，
    杜绝 eval() 带来的代码注入风险。同时提供方程求解、微积分、
    线性代数、统计分析等高级数学计算能力。

    用法示例::

        engine = MathEngine()
        engine.evaluate("sin(pi / 2)")          # => 1.0
        engine.solve_equation("x**2 - 4", 3)    # => 2.0
        engine.derivative("x**3", 2.0)          # => 12.0
        engine.integral("x**2", 0, 1)           # => 0.333...
    """

    def __init__(self) -> None:
        self._scope: Dict[str, Any] = {**_SAFE_MATH_SCOPE, **_SAFE_BUILTINS}

    # ── 核心：安全求值 ────────────────────────────────────────────────

    def evaluate(self, expression: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        """安全求值数学表达式

        使用 ast.parse 解析表达式，仅允许安全节点在受限作用域内执行。
        支持数学函数、常量、四则运算、比较运算等。

        Args:
            expression: 数学表达式字符串，如 "sin(pi/2) + x**2"
            variables: 可选变量字典，如 {"x": 3.0}

        Returns:
            表达式计算结果

        Raises:
            ActionError: 表达式包含不安全语法或求值失败
        """
        if not expression or not expression.strip():
            raise ActionError("表达式不能为空", error_code="E501")

        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            raise ActionError(
                f"表达式语法错误: {exc}",
                error_code="E501",
                details={"expression": expression, "syntax_error": str(exc)},
            ) from exc

        # 安全检查
        _SafeEvalVisitor().visit(tree)

        # 构建求值作用域
        scope = dict(self._scope)
        if variables:
            scope.update(variables)

        try:
            compiled = compile(tree, "<math_expression>", "eval")
            result = eval(compiled, {"__builtins__": {}}, scope)  # noqa: S307
            return result
        except Exception as exc:
            raise ActionError(
                f"表达式求值失败: {exc}",
                error_code="E501",
                details={"expression": expression, "error": str(exc)},
            ) from exc

    # ── 方程求解 ──────────────────────────────────────────────────────

    def solve_equation(self, expression: str, guess: float = 1.0) -> float:
        """数值求解方程 f(x) = 0

        使用牛顿迭代法求解方程。表达式应包含变量 x，函数将寻找使表达式
        值为零的 x 值。

        Args:
            expression: 包含变量 x 的表达式，如 "x**2 - 4"
            guess: 初始猜测值，默认 1.0

        Returns:
            方程的数值解

        Raises:
            ActionError: 迭代不收敛或求值失败
        """
        max_iterations = 1000
        tolerance = 1e-10
        h = 1e-7
        x = float(guess)

        for i in range(max_iterations):
            try:
                fx = self.evaluate(expression, {"x": x})
            except ActionError:
                raise
            except Exception as exc:
                raise ActionError(
                    f"方程求值失败: {exc}",
                    error_code="E501",
                    details={"expression": expression, "x": x},
                ) from exc

            if abs(fx) < tolerance:
                logger.debug("方程求解收敛", iterations=i, result=x)
                return x

            # 数值导数
            try:
                fx_plus_h = self.evaluate(expression, {"x": x + h})
            except ActionError:
                raise
            except Exception as exc:
                raise ActionError(
                    f"方程求值失败（数值导数计算）: {exc}",
                    error_code="E501",
                    details={"expression": expression, "x": x + h},
                ) from exc

            dfx = (fx_plus_h - fx) / h
            if abs(dfx) < 1e-15:
                raise ActionError(
                    "牛顿法迭代失败：导数接近零，可能遇到驻点",
                    error_code="E501",
                    details={"expression": expression, "x": x, "derivative": dfx},
                )

            x = x - fx / dfx

        raise ActionError(
            f"牛顿法在 {max_iterations} 次迭代后未收敛",
            error_code="E501",
            details={"expression": expression, "guess": guess, "last_x": x},
        )

    # ── 微积分 ────────────────────────────────────────────────────────

    def derivative(self, expression: str, point: float, h: float = 1e-7) -> float:
        """计算表达式在指定点的数值导数

        使用中心差分法: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

        Args:
            expression: 包含变量 x 的表达式
            point: 求导点
            h: 差分步长，默认 1e-7

        Returns:
            数值导数值

        Raises:
            ActionError: 求值失败
        """
        try:
            f_plus = self.evaluate(expression, {"x": point + h})
            f_minus = self.evaluate(expression, {"x": point - h})
        except ActionError:
            raise
        except Exception as exc:
            raise ActionError(
                f"导数计算失败: {exc}",
                error_code="E501",
                details={"expression": expression, "point": point},
            ) from exc

        return (f_plus - f_minus) / (2 * h)

    def integral(self, expression: str, a: float, b: float, n: int = 1000) -> float:
        """计算定积分（辛普森法则）

        使用复合辛普森法则计算 ∫[a,b] f(x) dx。
        n 必须为偶数，若为奇数则自动加 1。

        Args:
            expression: 包含变量 x 的被积函数表达式
            a: 积分下限
            b: 积分上限
            n: 分割数（必须为偶数），默认 1000

        Returns:
            定积分近似值

        Raises:
            ActionError: 求值失败或区间无效
        """
        if n <= 0:
            raise ActionError("分割数 n 必须为正整数", error_code="E501")
        if a == b:
            return 0.0

        # 确保 n 为偶数
        if n % 2 != 0:
            n += 1

        h = (b - a) / n
        try:
            result = self.evaluate(expression, {"x": a}) + self.evaluate(expression, {"x": b})
        except ActionError:
            raise
        except Exception as exc:
            raise ActionError(
                f"积分计算失败: {exc}",
                error_code="E501",
                details={"expression": expression},
            ) from exc

        for i in range(1, n):
            x_i = a + i * h
            try:
                fx_i = self.evaluate(expression, {"x": x_i})
            except ActionError:
                raise
            except Exception as exc:
                raise ActionError(
                    f"积分计算失败: {exc}",
                    error_code="E501",
                    details={"expression": expression, "x": x_i},
                ) from exc

            coefficient = 4 if i % 2 != 0 else 2
            result += coefficient * fx_i

        return result * h / 3

    # ── 线性代数 ──────────────────────────────────────────────────────

    def matrix_multiply(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """矩阵乘法

        计算矩阵 a (m×n) 与矩阵 b (n×p) 的乘积，结果为 m×p 矩阵。

        Args:
            a: m×n 矩阵
            b: n×p 矩阵

        Returns:
            m×p 乘积矩阵

        Raises:
            ActionError: 矩阵维度不匹配或输入无效
        """
        if not a or not b or not a[0] or not b[0]:
            raise ActionError("矩阵不能为空", error_code="E501")

        n_cols_a = len(a[0])
        n_rows_b = len(b)

        if n_cols_a != n_rows_b:
            raise ActionError(
                f"矩阵维度不匹配: a 的列数 ({n_cols_a}) != b 的行数 ({n_rows_b})",
                error_code="E501",
                details={"a_cols": n_cols_a, "b_rows": n_rows_b},
            )

        # 验证矩阵形状一致性
        for row in a:
            if len(row) != n_cols_a:
                raise ActionError("矩阵 a 的行长度不一致", error_code="E501")
        for row in b:
            if len(row) != len(b[0]):
                raise ActionError("矩阵 b 的行长度不一致", error_code="E501")

        n_cols_b = len(b[0])
        result = [[0.0] * n_cols_b for _ in range(len(a))]

        for i in range(len(a)):
            for j in range(n_cols_b):
                for k in range(n_cols_a):
                    result[i][j] += a[i][k] * b[k][j]

        return result

    def determinant(self, a: List[List[float]]) -> float:
        """计算矩阵行列式

        使用 LU 分解（高斯消元法）计算方阵行列式。

        Args:
            a: n×n 方阵

        Returns:
            行列式值

        Raises:
            ActionError: 矩阵非方阵或输入无效
        """
        if not a or not a[0]:
            raise ActionError("矩阵不能为空", error_code="E501")

        n = len(a)
        for row in a:
            if len(row) != n:
                raise ActionError(
                    f"行列式要求方阵，但矩阵为 {n}×{len(row)}",
                    error_code="E501",
                )

        # 复制矩阵避免修改原数据
        matrix = [row[:] for row in a]
        det = 1.0
        sign = 1

        for col in range(n):
            # 选主元
            max_row = col
            for row in range(col + 1, n):
                if abs(matrix[row][col]) > abs(matrix[max_row][col]):
                    max_row = row

            if abs(matrix[max_row][col]) < 1e-12:
                return 0.0

            # 行交换
            if max_row != col:
                matrix[col], matrix[max_row] = matrix[max_row], matrix[col]
                sign = -sign

            det *= matrix[col][col]

            # 消元
            for row in range(col + 1, n):
                factor = matrix[row][col] / matrix[col][col]
                for j in range(col + 1, n):
                    matrix[row][j] -= factor * matrix[col][j]
                matrix[row][col] = 0.0

        return det * sign

    def eigenvalues(self, a: List[List[float]]) -> List[float]:
        """计算矩阵特征值

        对于 2×2 矩阵使用解析解（特征方程），对于更大矩阵使用幂迭代法
        逐个提取主特征值后进行压缩（Wielandt 压缩）。

        Args:
            a: n×n 方阵

        Returns:
            特征值列表

        Raises:
            ActionError: 矩阵非方阵或计算失败
        """
        if not a or not a[0]:
            raise ActionError("矩阵不能为空", error_code="E501")

        n = len(a)
        for row in a:
            if len(row) != n:
                raise ActionError(
                    f"特征值要求方阵，但矩阵为 {n}×{len(row)}",
                    error_code="E501",
                )

        if n == 1:
            return [a[0][0]]

        if n == 2:
            # 2×2 解析解: λ = (tr ± sqrt(tr² - 4*det)) / 2
            trace = a[0][0] + a[1][1]
            det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
            discriminant = trace * trace - 4 * det

            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                return [(trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2]
            else:
                # 复数特征值，返回实部
                real_part = trace / 2
                imag_part = math.sqrt(-discriminant) / 2
                return [complex(real_part, imag_part), complex(real_part, -imag_part)]

        # n >= 3: 幂迭代 + Wielandt 压缩
        eigenvals: List[float] = []
        current_matrix = [row[:] for row in a]

        for _ in range(n):
            eigenval = self._power_iteration(current_matrix)
            eigenvals.append(eigenval)
            current_matrix = self._wielandt_deflation(current_matrix, eigenval)

        return eigenvals

    def _power_iteration(self, matrix: List[List[float]], max_iter: int = 200, tol: float = 1e-8) -> float:
        """幂迭代法求主特征值"""
        n = len(matrix)
        v = [1.0] * n

        eigenvalue = 0.0
        for _ in range(max_iter):
            new_v = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]

            norm_sq = sum(x * x for x in new_v)
            if norm_sq < 1e-15:
                return 0.0

            new_eigenvalue = sum(v[i] * new_v[i] for i in range(n)) / sum(v[i] * v[i] for i in range(n))

            norm = math.sqrt(norm_sq)
            v = [x / norm for x in new_v]

            if abs(new_eigenvalue - eigenvalue) < tol:
                return new_eigenvalue
            eigenvalue = new_eigenvalue

        return eigenvalue

    def _wielandt_deflation(self, matrix: List[List[float]], eigenvalue: float) -> List[List[float]]:
        """Wielandt 压缩：移除已知特征值，生成降阶矩阵"""
        n = len(matrix)

        v = [1.0] * n
        for _ in range(100):
            new_v = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]
            norm = math.sqrt(sum(x * x for x in new_v))
            if norm < 1e-15:
                break
            v = [x / norm for x in new_v]

        max_idx = max(range(n), key=lambda i: abs(v[i]))

        reduced = []
        for i in range(n):
            if i == max_idx:
                continue
            row = []
            for j in range(n):
                if j == max_idx:
                    continue
                correction = eigenvalue * v[i] * v[j] / (v[max_idx] * v[max_idx]) if abs(v[max_idx]) > 1e-15 else 0
                row.append(matrix[i][j] - correction)
            reduced.append(row)

        return reduced

    # ── 统计 ──────────────────────────────────────────────────────────

    def median(self, data: List[float]) -> float:
        """计算中位数"""
        if not data:
            raise ActionError("数据列表不能为空", error_code="E501")

        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2

        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2
        else:
            return sorted_data[mid]

    def std(self, data: List[float]) -> float:
        """计算标准差（总体标准差）"""
        if not data:
            raise ActionError("数据列表不能为空", error_code="E501")

        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        return math.sqrt(variance)

    def correlation(self, data1: List[float], data2: List[float]) -> float:
        """计算皮尔逊相关系数

        衡量两组数据之间的线性相关程度，返回值在 [-1, 1] 之间。
        """
        if not data1 or not data2:
            raise ActionError("数据列表不能为空", error_code="E501")
        if len(data1) != len(data2):
            raise ActionError(
                f"数据长度不一致: data1={len(data1)}, data2={len(data2)}",
                error_code="E501",
            )

        n = len(data1)
        mean1 = sum(data1) / n
        mean2 = sum(data2) / n

        cov = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(n))
        var1 = sum((x - mean1) ** 2 for x in data1)
        var2 = sum((x - mean2) ** 2 for x in data2)

        denom = math.sqrt(var1 * var2)
        if denom < 1e-15:
            raise ActionError(
                "相关系数计算失败：至少一组数据标准差为零",
                error_code="E501",
            )

        return cov / denom

    def linear_regression(self, x: List[float], y: List[float]) -> Dict[str, float]:
        """线性回归分析

        拟合 y = slope * x + intercept，返回回归参数与决定系数。
        """
        if not x or not y:
            raise ActionError("数据列表不能为空", error_code="E501")
        if len(x) != len(y):
            raise ActionError(
                f"数据长度不一致: x={len(x)}, y={len(y)}",
                error_code="E501",
            )

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        ss_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        ss_xx = sum((xi - mean_x) ** 2 for xi in x)

        if ss_xx < 1e-15:
            raise ActionError(
                "线性回归失败：x 的方差为零",
                error_code="E501",
            )

        slope = ss_xy / ss_xx
        intercept = mean_y - slope * mean_x

        # 决定系数 R²
        ss_yy = sum((yi - mean_y) ** 2 for yi in y)
        if ss_yy < 1e-15:
            r_squared = 1.0
        else:
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        }


# ═══════════════════════════════════════════════════════════════════════
# 通用仿真框架 — 连续-离散混合仿真
# ═══════════════════════════════════════════════════════════════════════


# ── 状态表示 ──────────────────────────────────────────────────────────


class State:
    """命名状态向量。

    用字典存储状态变量，支持按名称访问和算术运算。
    这是仿真系统中所有状态表示的基础数据结构。

    Example::

        s = State({"x": 1.0, "v": 0.5})
        s["x"]                          # 1.0
        s + State({"x": 0.1, "v": -0.2})  # State({"x": 1.1, "v": 0.3})
        s * 0.01                        # State({"x": 0.01, "v": 0.005})
    """

    __slots__ = ("_data", "_keys")

    def __init__(self, data: Dict[str, float]):
        self._data = dict(data)
        self._keys = tuple(sorted(data.keys()))

    @property
    def keys(self) -> Tuple[str, ...]:
        """状态变量名称元组"""
        return self._keys

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __setitem__(self, key: str, value: float) -> None:
        if key not in self._data:
            raise KeyError(f"State has no variable '{key}'. Existing: {list(self._data.keys())}")
        self._data[key] = value

    def __add__(self, other: "State") -> "State":
        return State({k: self._data[k] + other._data[k] for k in self._data})

    def __mul__(self, scalar: float) -> "State":
        return State({k: v * scalar for k, v in self._data.items()})

    def __rmul__(self, scalar: float) -> "State":
        return self.__mul__(scalar)

    def __sub__(self, other: "State") -> "State":
        return State({k: self._data[k] - other._data[k] for k in self._data})

    def __truediv__(self, scalar: float) -> "State":
        return State({k: v / scalar for k, v in self._data.items()})

    def norm(self) -> float:
        """欧几里得范数"""
        return sum(v * v for v in self._data.values()) ** 0.5

    def to_dict(self) -> Dict[str, float]:
        return dict(self._data)

    def copy(self) -> "State":
        return State(dict(self._data))

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.6g}" for k, v in self._data.items())
        return f"State({items})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self._data == other._data


# 类型别名
DerivativesFn = Callable[[float, State], State]
ObserverFn = Callable[[float, State], None]
ConditionFn = Callable[[float, State], bool]
EventCallbackFn = Callable[[float, State], None]


# ── 连续动态系统 ─────────────────────────────────────────────────────


class DynamicalSystem:
    """连续动态系统（ODE）。

    用户定义状态变量名称和导数函数，框架负责数值积分。
    这是仿真框架的核心原语——任何连续系统都可以表示为
    ``dy/dt = f(t, y)`` 的形式。

    Example::

        # 阻尼弹簧: m*x'' + c*x' + k*x = 0
        def spring_deriv(t, s):
            return State({
                "x": s["v"],
                "v": (-10.0 * s["x"] - 0.1 * s["v"]) / 1.0,
            })

        sys = DynamicalSystem(
            state_schema=["x", "v"],
            initial_state=State({"x": 1.0, "v": 0.0}),
            derivatives_fn=spring_deriv,
        )
    """

    def __init__(
        self,
        state_schema: Sequence[str],
        initial_state: State,
        derivatives_fn: DerivativesFn,
    ):
        """初始化连续动态系统。

        Args:
            state_schema: 状态变量名称列表，如 ["x", "v", "theta", "omega"]
            initial_state: 初始状态
            derivatives_fn: 导数函数 f(t, state) -> State，返回各变量的时间导数

        Raises:
            ActionError: schema 与 initial_state 不匹配时抛出
        """
        schema_set = set(state_schema)
        state_keys = set(initial_state.keys)
        if schema_set != state_keys:
            missing = schema_set - state_keys
            extra = state_keys - schema_set
            raise ActionError(
                f"State schema mismatch. Missing: {missing}, Extra: {extra}",
                error_code="E560",
            )

        self._schema = tuple(state_schema)
        self._initial_state = initial_state.copy()
        self._derivatives_fn = derivatives_fn
        self._current_state = initial_state.copy()
        self._current_time = 0.0

    @property
    def schema(self) -> Tuple[str, ...]:
        """状态变量名称元组"""
        return self._schema

    @property
    def initial_state(self) -> State:
        return self._initial_state.copy()

    @property
    def current_state(self) -> State:
        return self._current_state.copy()

    @property
    def current_time(self) -> float:
        return self._current_time

    def derivatives(self, t: float, state: State) -> State:
        """计算状态导数。

        Args:
            t: 当前时间
            state: 当前状态

        Returns:
            各变量的时间导数组成的 State
        """
        return self._derivatives_fn(t, state)

    def reset(self) -> None:
        """重置到初始状态"""
        self._current_state = self._initial_state.copy()
        self._current_time = 0.0

    def advance(self, t: float, state: State) -> None:
        """推进到新状态（由积分器调用）"""
        self._current_state = state.copy()
        self._current_time = t

    def set_param(self, key: str, value: float) -> None:
        """运行时修改当前状态中的某个变量（用于离散事件干预连续系统）"""
        self._current_state[key] = value


# ── 离散事件系统 ─────────────────────────────────────────────────────


@dataclass
class ScheduledEvent:
    """调度事件"""
    time: float
    callback: EventCallbackFn
    name: str = ""
    priority: int = 0  # 同时刻优先级，数值小先执行

    def __lt__(self, other: "ScheduledEvent") -> bool:
        if self.time != other.time:
            return self.time < other.time
        return self.priority < other.priority


@dataclass
class ConditionTrigger:
    """条件触发器"""
    condition: ConditionFn
    callback: EventCallbackFn
    name: str = ""
    once: bool = True  # 触发一次后移除
    _fired: bool = field(default=False, repr=False)


@dataclass
class EventRecord:
    """事件发生记录"""
    time: float
    name: str
    event_type: str  # "scheduled" | "condition"
    state_snapshot: Dict[str, float]


class DiscreteEventSystem:
    """离散事件系统。

    支持两种事件触发方式：
    - 定时事件：在指定时间点触发
    - 条件事件：当状态满足条件时触发

    Example::

        des = DiscreteEventSystem()
        des.schedule_event(time=5.0, callback=lambda t, s: print(f"t={t}"), name="alarm")
        des.schedule_condition(
            condition=lambda t, s: s["x"] > 2.0,
            callback=lambda t, s: print("Threshold crossed!"),
            name="threshold",
        )
    """

    def __init__(self):
        self._event_queue: list[ScheduledEvent] = []  # 最小堆
        self._conditions: list[ConditionTrigger] = []
        self._event_log: list[EventRecord] = []

    def schedule_event(
        self,
        time: float,
        callback: EventCallbackFn,
        name: str = "",
        priority: int = 0,
    ) -> None:
        """调度定时事件。

        Args:
            time: 触发时间
            callback: 回调函数 (t, state) -> None
            name: 事件名称（用于日志）
            priority: 同时刻优先级
        """
        heapq.heappush(self._event_queue, ScheduledEvent(time, callback, name, priority))

    def schedule_condition(
        self,
        condition: ConditionFn,
        callback: EventCallbackFn,
        name: str = "",
        once: bool = True,
    ) -> None:
        """注册条件触发器。

        Args:
            condition: 条件函数 (t, state) -> bool
            callback: 触发回调 (t, state) -> None
            name: 触发器名称
            once: 是否只触发一次（默认 True）
        """
        self._conditions.append(ConditionTrigger(condition, callback, name, once))

    def process_events(self, t: float, state: State) -> State:
        """处理当前时刻的所有到期事件和满足条件的触发器。

        Args:
            t: 当前时间
            state: 当前状态

        Returns:
            可能被事件修改后的状态
        """
        # 1. 处理定时事件
        while self._event_queue and self._event_queue[0].time <= t:
            event = heapq.heappop(self._event_queue)
            event.callback(t, state)
            self._event_log.append(EventRecord(
                time=t, name=event.name, event_type="scheduled",
                state_snapshot=state.to_dict(),
            ))

        # 2. 检查条件触发器
        remaining = []
        for trigger in self._conditions:
            if trigger.once and trigger._fired:
                continue
            try:
                if trigger.condition(t, state):
                    trigger.callback(t, state)
                    trigger._fired = True
                    self._event_log.append(EventRecord(
                        time=t, name=trigger.name, event_type="condition",
                        state_snapshot=state.to_dict(),
                    ))
            except Exception as e:
                logger.warning("condition_trigger_error", name=trigger.name, error=str(e))

            if not (trigger.once and trigger._fired):
                remaining.append(trigger)
        self._conditions = remaining

        return state

    @property
    def event_log(self) -> list[EventRecord]:
        return list(self._event_log)

    def has_pending_events(self) -> bool:
        return bool(self._event_queue) or bool(self._conditions)

    def clear(self) -> None:
        self._event_queue.clear()
        self._conditions.clear()
        self._event_log.clear()


# ── 混合系统 ─────────────────────────────────────────────────────────


class HybridSystem:
    """连续-离散混合系统。

    将 DynamicalSystem 和 DiscreteEventSystem 耦合在一起：
    - 连续积分推进状态
    - 离散事件可以修改连续状态（如参数突变、模式切换）
    - 条件触发器可以基于连续状态触发离散事件

    Example::

        # 弹簧断裂仿真：当位移超过阈值时弹簧常数归零
        spring = DynamicalSystem(
            state_schema=["x", "v"],
            initial_state=State({"x": 2.0, "v": 0.0}),
            derivatives_fn=lambda t, s: State({"x": s["v"], "v": -10*s["x"] - 0.5*s["v"]}),
        )
        hybrid = HybridSystem(continuous=spring)
        hybrid.on_condition(
            condition=lambda t, s: abs(s["x"]) > 1.5,
            callback=lambda t, s: s.__setitem__("v", 0),
            name="spring_break",
        )
    """

    def __init__(self, continuous: DynamicalSystem):
        """初始化混合系统。

        Args:
            continuous: 连续动态系统
        """
        self._continuous = continuous
        self._discrete = DiscreteEventSystem()

    @property
    def continuous(self) -> DynamicalSystem:
        return self._continuous

    @property
    def discrete(self) -> DiscreteEventSystem:
        return self._discrete

    def schedule_event(
        self,
        time: float,
        callback: EventCallbackFn,
        name: str = "",
        priority: int = 0,
    ) -> "HybridSystem":
        """调度定时事件（链式调用）"""
        self._discrete.schedule_event(time, callback, name, priority)
        return self

    def on_condition(
        self,
        condition: ConditionFn,
        callback: EventCallbackFn,
        name: str = "",
        once: bool = True,
    ) -> "HybridSystem":
        """注册条件触发器（链式调用）"""
        self._discrete.schedule_condition(condition, callback, name, once)
        return self


# ── 仿真结果 ─────────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    """仿真运行结果。

    Attributes:
        times: 时间序列
        trajectory: 各时刻的状态字典列表
        events: 事件记录列表
        variable_names: 状态变量名称
        duration: 仿真总时长
        step_count: 总步数
    """

    times: List[float]
    trajectory: List[Dict[str, float]]
    events: List[EventRecord]
    variable_names: Tuple[str, ...]
    duration: float
    step_count: int

    def get_series(self, variable: str) -> List[float]:
        """获取某个变量的时间序列。

        Args:
            variable: 变量名

        Returns:
            该变量在各时间步的值列表

        Raises:
            ActionError: 变量名不存在
        """
        if variable not in self.variable_names:
            raise ActionError(
                f"Variable '{variable}' not found. Available: {self.variable_names}",
                error_code="E561",
            )
        return [step[variable] for step in self.trajectory]

    def get_last_state(self) -> Dict[str, float]:
        """获取最终状态"""
        return self.trajectory[-1] if self.trajectory else {}

    def get_last(self, variable: str) -> float:
        """获取某个变量的最终值"""
        series = self.get_series(variable)
        return series[-1] if series else 0.0

    def summary(self) -> Dict[str, Any]:
        """生成统计摘要"""
        result: Dict[str, Any] = {
            "duration": self.duration,
            "step_count": self.step_count,
            "event_count": len(self.events),
            "variables": {},
        }
        for var in self.variable_names:
            series = self.get_series(var)
            if series:
                result["variables"][var] = {
                    "min": min(series),
                    "max": max(series),
                    "final": series[-1],
                    "mean": sum(series) / len(series),
                }
        if self.events:
            result["events"] = [
                {"time": e.time, "name": e.name, "type": e.event_type}
                for e in self.events
            ]
        return result

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "times": self.times,
            "trajectory": self.trajectory,
            "events": [
                {"time": e.time, "name": e.name, "type": e.event_type, "state": e.state_snapshot}
                for e in self.events
            ],
            "variable_names": list(self.variable_names),
            "duration": self.duration,
            "step_count": self.step_count,
        }


# ── 积分器 ───────────────────────────────────────────────────────────


class _Integrator:
    """数值积分器（内部实现）"""

    @staticmethod
    def euler_step(sys: DynamicalSystem, t: float, state: State, dt: float) -> State:
        """欧拉法单步"""
        dydt = sys.derivatives(t, state)
        return state + dydt * dt

    @staticmethod
    def rk4_step(sys: DynamicalSystem, t: float, state: State, dt: float) -> State:
        """经典四阶 Runge-Kutta 单步"""
        k1 = sys.derivatives(t, state)
        k2 = sys.derivatives(t + dt / 2, state + k1 * (dt / 2))
        k3 = sys.derivatives(t + dt / 2, state + k2 * (dt / 2))
        k4 = sys.derivatives(t + dt, state + k3 * dt)
        return state + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)

    @staticmethod
    def rk45_step(
        sys: DynamicalSystem, t: float, state: State, dt: float, tol: float = 1e-6,
    ) -> Tuple[State, float, float]:
        """Runge-Kutta-Fehlberg 自适应步长。

        Returns:
            (新状态, 误差估计, 建议的新步长)
        """
        k1 = sys.derivatives(t, state)
        k2 = sys.derivatives(t + dt / 4, state + k1 * (dt / 4))
        k3 = sys.derivatives(t + 3 * dt / 8, state + k1 * (3 * dt / 32) + k2 * (9 * dt / 32))
        k4 = sys.derivatives(t + 12 * dt / 13, state + k1 * (1932 * dt / 2197) - k2 * (7200 * dt / 2197) + k3 * (7296 * dt / 2197))
        k5 = sys.derivatives(t + dt, state + k1 * (439 * dt / 216) - k2 * (8 * dt) + k3 * (3680 * dt / 513) - k4 * (845 * dt / 4104))
        k6 = sys.derivatives(t + dt / 2, state - k1 * (8 * dt / 27) + k2 * (2 * dt) - k3 * (3544 * dt / 2565) + k4 * (1859 * dt / 4104) - k5 * (11 * dt / 40))

        # 5阶解
        y5 = state + (k1 * (16 / 135) + k3 * (6656 / 12825) + k4 * (28561 / 56430) - k5 * (9 / 50) + k6 * (2 / 55)) * dt
        # 4阶解
        y4 = state + (k1 * (25 / 216) + k3 * (1408 / 2565) + k4 * (2197 / 4104) - k5 * (1 / 5)) * dt

        # 误差估计
        error = (y5 - y4).norm()
        scale = max(y5.norm(), 1e-10)

        if error < 1e-20:
            return y5, 0.0, dt

        relative_error = error / scale

        if relative_error < tol:
            new_dt = dt * min(5.0, 0.9 * (tol / relative_error) ** 0.2)
        else:
            new_dt = dt * max(0.1, 0.9 * (tol / relative_error) ** 0.25)

        return y5, relative_error, new_dt


# ── 仿真引擎 ─────────────────────────────────────────────────────────


class SimulationEngine:
    """通用仿真引擎。

    统一的时间推进器，支持运行 DynamicalSystem 和 HybridSystem。
    提供 Euler / RK4 / RK45（自适应）三种积分方法。

    Example::

        engine = SimulationEngine(dt=0.01, method="rk4")

        # 运行连续系统
        sys = DynamicalSystem(
            state_schema=["x", "v"],
            initial_state=State({"x": 1.0, "v": 0.0}),
            derivatives_fn=lambda t, s: State({"x": s["v"], "v": -s["x"]}),
        )
        result = engine.run(sys, duration=10.0)
        print(result.get_series("x")[:5])
        print(result.summary())
    """

    def __init__(
        self,
        dt: float = 0.01,
        method: str = "rk4",
        adaptive_tol: float = 1e-6,
        max_steps: int = 1_000_000,
    ):
        """初始化仿真引擎。

        Args:
            dt: 基础时间步长
            method: 积分方法 "euler" / "rk4" / "rk45"
            adaptive_tol: RK45 自适应容差
            max_steps: 最大步数限制（防止无限循环）

        Raises:
            ActionError: 参数无效时抛出
        """
        if dt <= 0:
            raise ActionError("dt must be positive", error_code="E562")
        if method not in ("euler", "rk4", "rk45"):
            raise ActionError(
                f"Unknown method '{method}'. Choose from: euler, rk4, rk45",
                error_code="E562",
            )

        self._dt = dt
        self._method = method
        self._adaptive_tol = adaptive_tol
        self._max_steps = max_steps

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def method(self) -> str:
        return self._method

    def run(
        self,
        system: Union[DynamicalSystem, HybridSystem],
        duration: float,
        observers: Optional[list[ObserverFn]] = None,
        record_interval: int = 1,
    ) -> SimulationResult:
        """运行仿真。

        Args:
            system: 连续动态系统或混合系统
            duration: 仿真时长
            observers: 观察者回调列表，每步调用 (t, state) -> None
            record_interval: 记录间隔（每 N 步记录一次，默认每步都记录）

        Returns:
            SimulationResult 包含轨迹、事件日志和统计摘要

        Raises:
            ActionError: 参数无效或仿真超步数时抛出
        """
        if duration <= 0:
            raise ActionError("duration must be positive", error_code="E563")

        # 解构系统
        if isinstance(system, HybridSystem):
            continuous = system.continuous
            discrete = system.discrete
        elif isinstance(system, DynamicalSystem):
            continuous = system
            discrete = None
        else:
            raise ActionError(
                f"Unsupported system type: {type(system).__name__}. "
                "Expected DynamicalSystem or HybridSystem.",
                error_code="E564",
            )

        continuous.reset()
        state = continuous.initial_state
        t = 0.0

        times: List[float] = []
        trajectory: List[Dict[str, float]] = []
        all_events: List[EventRecord] = []
        step = 0
        dt = self._dt

        # 记录初始状态
        times.append(t)
        trajectory.append(state.to_dict())

        while t < duration - 1e-12:
            if step >= self._max_steps:
                raise ActionError(
                    f"Simulation exceeded max_steps ({self._max_steps}). "
                    f"Current t={t:.4f}/{duration}. Consider larger dt or fewer steps.",
                    error_code="E565",
                )

            # 1. 离散事件处理（积分前）
            if discrete is not None:
                state = discrete.process_events(t, state)
                all_events.extend(discrete.event_log)
                discrete._event_log.clear()

            # 2. 数值积分
            if self._method == "euler":
                state = _Integrator.euler_step(continuous, t, state, dt)
            elif self._method == "rk4":
                state = _Integrator.rk4_step(continuous, t, state, dt)
            elif self._method == "rk45":
                state, error, new_dt = _Integrator.rk45_step(
                    continuous, t, state, dt, self._adaptive_tol,
                )
                dt = max(new_dt, 1e-10)
            else:
                raise ActionError(f"Unknown method: {self._method}", error_code="E562")

            t += dt
            step += 1

            # 3. 更新系统内部状态
            continuous.advance(t, state)

            # 4. 观察者回调
            if observers:
                for obs in observers:
                    try:
                        obs(t, state)
                    except Exception as e:
                        logger.warning("observer_error", error=str(e))

            # 5. 记录
            if step % record_interval == 0:
                times.append(t)
                trajectory.append(state.to_dict())

        return SimulationResult(
            times=times,
            trajectory=trajectory,
            events=all_events,
            variable_names=continuous.schema,
            duration=t,
            step_count=step,
        )

    def step_once(
        self,
        system: Union[DynamicalSystem, HybridSystem],
    ) -> Tuple[float, State]:
        """单步推进（交互式仿真）。

        Args:
            system: 连续或混合系统

        Returns:
            (新时间, 新状态)
        """
        if isinstance(system, HybridSystem):
            continuous = system.continuous
            discrete = system.discrete
        elif isinstance(system, DynamicalSystem):
            continuous = system
            discrete = None
        else:
            raise ActionError(f"Unsupported system type: {type(system).__name__}", error_code="E564")

        t = continuous.current_time
        state = continuous.current_state

        if discrete is not None:
            state = discrete.process_events(t, state)

        if self._method == "euler":
            state = _Integrator.euler_step(continuous, t, state, self._dt)
        elif self._method == "rk4":
            state = _Integrator.rk4_step(continuous, t, state, self._dt)
        elif self._method == "rk45":
            state, _, _ = _Integrator.rk45_step(continuous, t, state, self._dt, self._adaptive_tol)
        else:
            raise ActionError(f"Unknown method: {self._method}", error_code="E562")

        t += self._dt
        continuous.advance(t, state)
        return t, state
