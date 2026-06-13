"""
代码执行引擎 - 安全的沙盒代码执行

提供多语言代码执行能力，包含危险代码检测、会话持久化、资源限制等安全机制。
支持 Python、Bash、Node.js、Ruby 等多种语言的子进程隔离执行。

核心组件：
- CodeValidator: 代码安全检查器
- CodeExecutor: 代码执行器（主入口）
- Session: 跨执行会话（变量持久化）
- ExecutionResult: 执行结果
"""

import ast
import io
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


_SAFE_BUILTIN_MODULES = frozenset({
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii", "binhex",
    "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk",
    "cmath", "cmd", "code", "codecs", "codeop", "collections", "colorsys",
    "compileall", "concurrent", "configparser", "contextlib", "contextvars",
    "copy", "copyreg", "cProfile", "crypt", "csv", "ctypes",
    "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
    "distutils", "doctest", "email", "encodings", "enum", "errno",
    "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch",
    "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "graphlib", "grp", "gzip", "hashlib", "heapq",
    "hmac", "html", "http", "idlelib", "imaplib", "imghdr", "imp",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "linecache", "locale", "logging", "lzma", "mailbox",
    "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder",
    "multiprocessing", "netrc", "nis", "nntplib", "numbers", "operator",
    "optparse", "os.path", "parser", "pathlib", "pdb", "pickle",
    "pickletools", "pipes", "pkgutil", "platform", "plistlib", "poplib",
    "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
    "pyclbr", "pydoc", "queue", "quopri", "random", "re",
    "readline", "reprlib", "resource", "rlcompleter", "runpy",
    "sched", "secrets", "select", "selectors", "shelve", "shlex",
    "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr",
    "socket", "socketserver", "sqlite3", "ssl", "stat", "statistics",
    "string", "stringprep", "struct", "sunau", "symtable", "sysconfig",
    "tabnanny", "tarfile", "telnetlib", "tempfile", "termios",
    "textwrap", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "trace", "traceback", "tracemalloc", "tty",
    "turtle", "types", "typing", "unicodedata", "unittest", "urllib",
    "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib",
    "_thread", "__future__",
    "numpy", "scipy", "sympy", "pandas",
    "PIL", "matplotlib", "sklearn",
    "requests", "yaml", "toml", "orjson",
})

_DANGEROUS_MODULES = frozenset({
    "os", "subprocess", "shutil", "ctypes",
    "multiprocessing", "signal", "ptrace", "fcntl",
    "posix", "popen2", "commands",
})

_DANGEROUS_FUNCS = frozenset({
    "eval", "exec", "compile", "__import__", "open",
    "breakpoint", "input", "memoryview",
})


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    truncated: bool = False


@dataclass
class Session:
    session_id: str
    variables: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_code(self) -> str:
        lines = []
        for name, value in self.variables.items():
            lines.append(f"{name} = {repr(value)}")
        return "\n".join(lines)


_DANGEROUS_PATTERNS = [
    (re.compile(r'\bos\.system\s*\('), "os.system() is not allowed"),
    (re.compile(r'\bsubprocess\.'), "subprocess is not allowed"),
    (re.compile(r'\beval\s*\('), "eval() is not allowed"),
    (re.compile(r'\bexec\s*\('), "exec() is not allowed"),
    (re.compile(r'\bcompile\s*\('), "compile() is not allowed"),
    (re.compile(r'\b__import__\s*\('), "__import__() is not allowed"),
    (re.compile(r'\bopen\s*\('), "open() is not allowed"),
    (re.compile(r'\bshutil\.'), "shutil is not allowed"),
    (re.compile(r'\bos\.remove\s*\('), "os.remove() is not allowed"),
    (re.compile(r'\bos\.rmdir\s*\('), "os.rmdir() is not allowed"),
    (re.compile(r'\bos\.unlink\s*\('), "os.unlink() is not allowed"),
    (re.compile(r'\bglobals\s*\(\s*\)'), "globals() is not allowed"),
    (re.compile(r'\blocals\s*\(\s*\)'), "locals() is not allowed"),
    (re.compile(r'\bgetattr\s*\([^)]*__'), "getattr with dunder is not allowed"),
    (re.compile(r'\bsetattr\s*\('), "setattr() is not allowed"),
    (re.compile(r'\bdelattr\s*\('), "delattr() is not allowed"),
]


class CodeValidator:
    """Python 代码安全检查器。

    通过正则模式匹配检测潜在的危险操作，包括：
    - 系统调用（os.system, subprocess）
    - 动态代码执行（eval, exec, compile）
    - 文件操作（open, os.remove, shutil）
    - 内部属性访问（getattr with __dunder__, setattr, delattr）
    """

    @staticmethod
    def validate_python(code: str) -> list[str]:
        """验证 Python 代码安全性。

        Args:
            code: 待验证的 Python 源代码

        Returns:
            安全问题列表，空列表表示代码安全
        """
        issues: list[str] = []
        for pattern, message in _DANGEROUS_PATTERNS:
            if pattern.search(code):
                issues.append(message)
        if not code.strip():
            issues.append("Empty code block")
        return issues


class CodeExecutor:
    """安全的代码执行器。

    支持 Python、Bash、Node.js、Ruby 等多种语言的子进程隔离执行。
    Python 执行提供会话持久化（变量跨执行保留）和变量捕获能力。
    安全保障包括：危险代码检测、内存限制（256MB）、超时控制、输出截断。

    Attributes:
        DEFAULT_TIMEOUT_SECONDS: 默认超时时间（30秒）
        DEFAULT_MAX_OUTPUT_CHARS: 默认最大输出字符数（100,000）
    """

    DEFAULT_TIMEOUT_SECONDS = 30.0
    DEFAULT_MAX_OUTPUT_CHARS = 100_000

    def __init__(
        self,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ):
        """初始化代码执行器。

        Args:
            default_timeout: 默认执行超时（秒）
            max_output_chars: stdout/stderr 最大输出字符数，超出部分截断
        """
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[float] = None,
        session_id: Optional[str] = None,
        input_vars: Optional[dict[str, Any]] = None,
        capture_vars: Optional[list[str]] = None,
    ) -> ExecutionResult:
        """执行代码。

        Args:
            code: 源代码字符串
            language: 编程语言（python/bash/node/ruby 等）
            timeout: 超时时间（秒），None 使用默认值
            session_id: 会话 ID，用于 Python 变量跨执行持久化
            input_vars: 注入到执行环境的变量字典
            capture_vars: 需要捕获返回的变量名列表

        Returns:
            ExecutionResult 包含 stdout、stderr、exit_code、耗时等信息
        """
        if language == "python":
            return self._execute_python(
                code=code,
                timeout=timeout,
                session_id=session_id,
                input_vars=input_vars,
                capture_vars=capture_vars,
            )
        else:
            return self._execute_subprocess(
                code=code,
                language=language,
                timeout=timeout,
            )

    def _build_python_script(
        self, code: str, session_id=None, input_vars=None, capture_vars=None,
    ) -> str | None:
        issues = CodeValidator.validate_python(code)
        if issues:
            return None
        lines = ["import sys, json, time, tracemalloc, gc"]
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if session.variables:
                lines.append(session.to_code())
        if input_vars:
            for name, value in input_vars.items():
                lines.append(f"{name} = {repr(value)}")
        lines.extend([
            "tracemalloc.start()", "gc.collect()",
            "_mem_start = tracemalloc.get_traced_memory()[0]",
            "_t0 = time.perf_counter()", "try:",
        ])
        for line in code.splitlines():
            lines.append(f"    {line}")
        lines.extend([
            "finally:", "    _t1 = time.perf_counter()",
            "    _mem_end, _mem_peak = tracemalloc.get_traced_memory()",
            "    tracemalloc.stop()",
        ])
        if capture_vars:
            lines.append("    _captured = {}")
            for var in capture_vars:
                lines.append(f"    try:\n        _captured[{repr(var)}] = repr(locals().get({repr(var)}, globals().get({repr(var)})))\n    except Exception:\n        _captured[{repr(var)}] = None")
            lines.append('    print("__CAPTURE__" + json.dumps(_captured) + "__END_CAPTURE__", file=sys.stderr)')
        lines.append("    print(json.dumps({'execution_time_ms': (_t1 - _t0) * 1000, 'memory_used_mb': (_mem_end - _mem_start) / (1024 * 1024), 'memory_peak_mb': (_mem_peak - _mem_start) / (1024 * 1024)}), file=sys.stderr)")
        if session_id and session_id in self._sessions:
            lines.append("    import types")
            lines.append("    _session_vars = {k: v for k, v in locals().items() if not k.startswith('_') and not isinstance(v, types.ModuleType) and k != 'types'}")
            lines.append('    print("__SESSION__" + json.dumps({k: repr(v) for k, v in _session_vars.items()}, default=str) + "__END_SESSION__", file=sys.stderr)')
        return "\n".join(lines)

    def _parse_execution_result(
        self, stdout, stderr, exit_code, wall_time_ms, capture_vars, session_id,
    ):
        truncated_out = False
        truncated_err = False

        if len(stdout) > self._max_output_chars:
            stdout = stdout[:self._max_output_chars]
            truncated_out = True
        if len(stderr) > self._max_output_chars:
            stderr = stderr[:self._max_output_chars]
            truncated_err = True

        import json as _json
        session_data = None
        capture_data = None

        stderr_lines = stderr.splitlines()
        clean_stderr_lines = []
        for line in stderr_lines:
            if line.startswith("__CAPTURE__") and line.endswith("__END_CAPTURE__"):
                try:
                    capture_json = line[len("__CAPTURE__"):-len("__END_CAPTURE__")]
                    capture_data = _json.loads(capture_json)
                except Exception as e:
                    logger.debug("code_executor_capture_parse_failed", error=str(e))
                continue
            if line.startswith("__SESSION__") and line.endswith("__END_SESSION__"):
                try:
                    session_json = line[len("__SESSION__"):-len("__END_SESSION__")]
                    session_data = _json.loads(session_json)
                except Exception as e:
                    logger.debug("code_executor_session_parse_failed", error=str(e))
                continue
            if line.startswith("{"):
                try:
                    meta = _json.loads(line)
                    if "execution_time_ms" not in meta:
                        clean_stderr_lines.append(line)
                except Exception:
                    clean_stderr_lines.append(line)
            else:
                clean_stderr_lines.append(line)

        stderr = "\n".join(clean_stderr_lines)

        if session_id and session_id in self._sessions and session_data:
            session = self._sessions[session_id]
            for k, v in session_data.items():
                try:
                    session.variables[k] = eval(v)
                except Exception:
                    session.variables[k] = v

        result = ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_ms=wall_time_ms,
            truncated=truncated_out or truncated_err,
        )

        if capture_data and capture_vars:
            if not hasattr(result, "captured_vars"):
                setattr(result, "captured_vars", {})
            for var in capture_vars:
                if var in capture_data:
                    try:
                        result.captured_vars[var] = eval(capture_data[var])
                    except Exception:
                        result.captured_vars[var] = capture_data[var]

        return result

    def _execute_python(
        self,
        code: str,
        timeout: Optional[float] = None,
        session_id: Optional[str] = None,
        input_vars: Optional[dict[str, Any]] = None,
        capture_vars: Optional[list[str]] = None,
    ) -> ExecutionResult:
        full_code = self._build_python_script(code, session_id, input_vars, capture_vars)
        if full_code is None:
            return ExecutionResult(
                exit_code=1,
                stderr="\n".join(CodeValidator.validate_python(code)),
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout

        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(full_code)
                script_path = f.name

            start_time = time.perf_counter()

            kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
            if platform.system() != "Windows":
                kwargs["preexec_fn"] = _set_process_limits

            process = subprocess.Popen([sys.executable, script_path], **kwargs)

            try:
                stdout, stderr = process.communicate(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return ExecutionResult(
                    exit_code=-1,
                    stderr=f"Execution timed out after {effective_timeout}s",
                    execution_time_ms=effective_timeout * 1000,
                    truncated=False,
                )

            wall_time_ms = (time.perf_counter() - start_time) * 1000

            return self._parse_execution_result(
                stdout, stderr, process.returncode, wall_time_ms,
                capture_vars, session_id,
            )

        except Exception as e:
            return ExecutionResult(
                exit_code=1,
                stderr=str(e),
            )
        finally:
            if script_path:
                try:
                    os.unlink(script_path)
                except OSError as e:
                    logger.debug("code_executor_cleanup_failed", path=script_path, error=str(e))

    def _execute_subprocess(
        self,
        code: str,
        language: str,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        effective_timeout = timeout if timeout is not None else self._default_timeout

        interpreter = self._get_interpreter(language)
        if interpreter is None:
            return ExecutionResult(
                exit_code=1,
                stderr=f"Unsupported language: {language}",
            )

        script_path = None
        try:
            suffix = self._get_suffix(language)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                script_path = f.name

            start_time = time.perf_counter()

            if platform.system() == "Windows":
                process = subprocess.Popen(
                    [interpreter, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                process = subprocess.Popen(
                    [interpreter, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=_set_process_limits,
                )

            try:
                stdout, stderr = process.communicate(timeout=effective_timeout)
                exit_code = process.returncode
                truncated = False

                if len(stdout) > self._max_output_chars:
                    stdout = stdout[:self._max_output_chars]
                    truncated = True
                if len(stderr) > self._max_output_chars:
                    stderr = stderr[:self._max_output_chars]
                    truncated = True

            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return ExecutionResult(
                    exit_code=-1,
                    stderr=f"Execution timed out after {effective_timeout}s",
                    execution_time_ms=effective_timeout * 1000,
                )

            wall_time_ms = (time.perf_counter() - start_time) * 1000

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time_ms=wall_time_ms,
                truncated=truncated,
            )

        except FileNotFoundError:
            return ExecutionResult(
                exit_code=1,
                stderr=f"Interpreter not found for language: {language} ({interpreter})",
            )
        except Exception as e:
            return ExecutionResult(
                exit_code=1,
                stderr=str(e),
            )
        finally:
            if script_path:
                try:
                    os.unlink(script_path)
                except OSError as e:
                    logger.debug("code_executor_cleanup_failed", path=script_path, error=str(e))

    @staticmethod
    def _get_interpreter(language: str) -> Optional[str]:
        interpreters = {
            "python": sys.executable,
            "python3": sys.executable,
            "bash": "/bin/bash",
            "sh": "/bin/sh",
            "node": "node",
            "ruby": "ruby",
            "perl": "perl",
            "lua": "lua",
            "r": "Rscript",
        }
        return interpreters.get(language.lower())

    @staticmethod
    def _get_suffix(language: str) -> str:
        suffixes = {
            "python": ".py",
            "python3": ".py",
            "bash": ".sh",
            "sh": ".sh",
            "node": ".js",
            "ruby": ".rb",
            "perl": ".pl",
            "lua": ".lua",
            "r": ".r",
        }
        return suffixes.get(language.lower(), ".txt")


def _set_process_limits():
    try:
        import resource
        # RLIMIT_AS is not functional on macOS; skip to avoid preexec_fn crash
        if sys.platform == "darwin":
            return
        mem_bytes = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception as e:
        logger.debug("set_process_limits_failed", error=str(e))