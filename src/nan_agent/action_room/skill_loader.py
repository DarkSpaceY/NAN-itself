"""
技能加载器 - 扫描 SKILL.md 目录、解析定义、构建技能树

扫描 nan_agent/skills/ 目录下的 SKILL.md 文件，解析 YAML frontmatter
与 markdown body，构建分层技能树，支持渐进式加载：
  advertise → activate → execute

核心组件：
- SkillAdvertise: 轻量级广告数据（名称、描述）
- LoadedSkill: 完整技能定义（含指令、脚本、引用）
- _ScanNode: 内部扫描节点（用于树构建）
- SkillLoader: 主加载器类
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

# 尝试导入 yaml，不可用时回退到简单解析
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# 扫描时跳过的目录名
_SKIP_DIRS = {"scripts", "references", "assets"}


@dataclass
class SkillAdvertise:
    """轻量级技能广告，用于 advertise 阶段。"""

    name: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "description": self.description}


@dataclass
class LoadedSkill:
    """完整技能定义，用于 activate 阶段。"""

    name: str
    description: str
    skill_path: str
    allowed_tools: List[str] = field(default_factory=list)
    compatibility: List[str] = field(default_factory=list)
    instructions: str = ""
    scripts: Dict[str, str] = field(default_factory=dict)
    references: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点（有脚本或允许的工具）。"""
        return bool(self.scripts) or bool(self.allowed_tools)


class _ScanNode:
    """内部扫描节点，用于树构建和渐进式加载。"""

    def __init__(
        self,
        name: str,
        description: str,
        level: int,
        skill_path: Path,
        is_leaf: bool = False,
        allowed_tools: Optional[List[str]] = None,
        compatibility: Optional[List[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ):
        self.name = name
        self.description = description
        self.level = level
        self.skill_path = skill_path
        self.is_leaf = is_leaf
        self.allowed_tools: List[str] = allowed_tools or []
        self.compatibility: List[str] = compatibility or []
        self.metadata: Dict[str, object] = metadata or {}
        self.children: List[_ScanNode] = []
        self._raw_body: Optional[str] = None
        self._scripts: Dict[str, str] = {}
        self._scripts_loaded: bool = False
        self._refs: Dict[str, str] = {}
        self._refs_loaded: bool = False

    def _ensure_body_loaded(self) -> None:
        """延迟加载 SKILL.md body。"""
        if self._raw_body is not None:
            return
        skill_md = self.skill_path / "SKILL.md"
        if skill_md.exists():
            try:
                text = skill_md.read_text(encoding="utf-8")
                _, body = SkillLoader._parse_skill_md_static(skill_md, text)
                self._raw_body = body
            except Exception as e:
                logger.warning("skill_body_load_failed", path=str(skill_md), error=str(e))
                self._raw_body = ""
        else:
            self._raw_body = ""

    def _ensure_scripts_loaded(self) -> None:
        """延迟加载 scripts/ 目录内容。"""
        if self._scripts_loaded:
            return
        self._scripts_loaded = True
        scripts_dir = self.skill_path / "scripts"
        if scripts_dir.is_dir():
            for f in scripts_dir.iterdir():
                if f.is_file():
                    try:
                        self._scripts[f.name] = f.read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning("script_load_failed", file=str(f), error=str(e))

    def _ensure_refs_loaded(self) -> None:
        """延迟加载 references/ 目录内容。"""
        if self._refs_loaded:
            return
        self._refs_loaded = True
        refs_dir = self.skill_path / "references"
        if refs_dir.is_dir():
            for f in refs_dir.iterdir():
                if f.is_file():
                    try:
                        self.references[f.name] = f.read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning("ref_load_failed", file=str(f), error=str(e))

    @property
    def instructions(self) -> str:
        self._ensure_body_loaded()
        return self._raw_body or ""

    @property
    def scripts(self) -> Dict[str, str]:
        self._ensure_scripts_loaded()
        return self._scripts

    @property
    def references(self) -> Dict[str, str]:
        self._ensure_refs_loaded()
        return self._refs


class SkillLoader:
    """技能加载器 — 扫描、解析、构建技能树。

    扫描指定目录下的 SKILL.md 文件，解析 YAML frontmatter 与 markdown
    body，构建分层技能树。支持渐进式加载（advertise → activate → execute）。

    Usage::

        loader = SkillLoader()
        roots = loader.scan()
        adv = loader.advertise("pytest")
        skill = loader.activate("pytest")
        script = loader.load_script("pytest", "run.sh")
    """

    def __init__(self, skills_dir: Optional[Path] = None):
        if skills_dir is None:
            # 默认使用 nan_agent 包目录下的 skills/
            package_dir = Path(__file__).resolve().parent.parent  # nan_agent/
            skills_dir = package_dir / "skills"
        self.skills_dir = skills_dir
        self._nodes: Dict[str, _ScanNode] = {}
        self._scanned: bool = False

    def scan(self) -> List[_ScanNode]:
        """扫描目录，解析 SKILL.md 文件，构建技能树。

        Returns:
            顶层 _ScanNode 列表（每个代表一个顶级类别）。
        """
        self._nodes.clear()
        self._scanned = False

        if not self.skills_dir.exists():
            logger.warning("skills_dir_not_found", path=str(self.skills_dir))
            self._scanned = True
            return []

        roots: List[_ScanNode] = []
        try:
            entries = sorted(self.skills_dir.iterdir())
        except PermissionError:
            logger.warning("skills_dir_permission_denied", path=str(self.skills_dir))
            self._scanned = True
            return []

        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                continue
            node = self._scan_directory(entry, level=0)
            if node is not None:
                roots.append(node)

        self._scanned = True
        logger.info("skill_scan_complete", roots=len(roots), total_nodes=len(self._nodes))
        return roots

    def _scan_directory(self, dir_path: Path, level: int) -> Optional[_ScanNode]:
        """递归扫描目录，构建 _ScanNode 树。

        Args:
            dir_path: 当前扫描目录
            level: 当前层级（0 = 顶级类别）

        Returns:
            _ScanNode 或 None（目录无效时）
        """
        skill_md = dir_path / "SKILL.md"
        frontmatter: Dict[str, object] = {}
        raw_body: str = ""

        if skill_md.exists():
            try:
                text = skill_md.read_text(encoding="utf-8")
                frontmatter, raw_body = self._parse_skill_md(skill_md, text)
            except Exception as e:
                logger.warning("skill_md_parse_error", path=str(skill_md), error=str(e))

        name = str(frontmatter.get("name", dir_path.name))
        description = str(frontmatter.get("description", ""))

        # 解析 allowed-tools（逗号分隔字符串）
        allowed_tools_raw = frontmatter.get("allowed-tools", "")
        allowed_tools: List[str] = []
        if allowed_tools_raw:
            allowed_tools = [t.strip() for t in str(allowed_tools_raw).split(",") if t.strip()]

        # 解析 compatibility（列表或逗号分隔字符串）
        compatibility: List[str] = []
        compat_raw = frontmatter.get("compatibility", frontmatter.get("requires", []))
        if isinstance(compat_raw, list):
            compatibility = [str(c) for c in compat_raw]
        elif isinstance(compat_raw, str) and compat_raw:
            compatibility = [c.strip() for c in compat_raw.split(",") if c.strip()]

        # 判断是否为叶子节点
        has_scripts_dir = (dir_path / "scripts").is_dir()
        is_leaf = has_scripts_dir or bool(allowed_tools)

        # 收集额外 metadata
        skip_keys = {"name", "description", "allowed-tools", "compatibility", "requires"}
        metadata = {k: v for k, v in frontmatter.items() if k not in skip_keys}

        node = _ScanNode(
            name=name,
            description=description,
            level=level,
            skill_path=dir_path,
            is_leaf=is_leaf,
            allowed_tools=allowed_tools,
            compatibility=compatibility,
            metadata=metadata,
        )
        node._raw_body = raw_body

        # 注册到全局查找表
        self._nodes[name] = node

        # 递归扫描子目录
        try:
            child_entries = sorted(dir_path.iterdir())
        except PermissionError:
            child_entries = []

        for child_entry in child_entries:
            if not child_entry.is_dir():
                continue
            if child_entry.name.startswith(".") or child_entry.name in _SKIP_DIRS:
                continue
            child_node = self._scan_directory(child_entry, level + 1)
            if child_node is not None:
                node.children.append(child_node)

        return node

    def _parse_skill_md(self, path: Path, text: str) -> tuple[dict, str]:
        """解析 SKILL.md 的 YAML frontmatter 与 body。

        Args:
            path: SKILL.md 文件路径（用于日志）
            text: SKILL.md 文件内容

        Returns:
            (frontmatter_dict, body_string) 元组
        """
        return self._parse_skill_md_static(path, text)

    @staticmethod
    def _parse_skill_md_static(path: Path, text: str) -> tuple[dict, str]:
        """静态方法：解析 SKILL.md 的 YAML frontmatter 与 body。"""
        if not text.startswith("---"):
            return {}, text

        # 找到第二个 ---
        rest = text[3:]
        end_idx = rest.find("\n---")
        if end_idx == -1:
            # 尝试没有换行的情况
            end_idx = rest.find("---")
            if end_idx == -1:
                return {}, text

        frontmatter_text = rest[:end_idx].strip()
        body = rest[end_idx + 4:].strip()

        if _YAML_AVAILABLE:
            try:
                frontmatter = yaml.safe_load(frontmatter_text) or {}
                if not isinstance(frontmatter, dict):
                    logger.warning("skill_md_frontmatter_not_dict", path=str(path))
                    return {}, text
                return frontmatter, body
            except Exception as e:
                logger.warning("skill_md_yaml_parse_error", path=str(path), error=str(e))
                return SkillLoader._parse_simple_frontmatter(frontmatter_text), body
        else:
            return SkillLoader._parse_simple_frontmatter(frontmatter_text), body

    @staticmethod
    def _parse_simple_frontmatter(text: str) -> dict:
        """当 yaml 不可用时的简单 frontmatter 解析。

        仅支持 key: value 格式，不支持嵌套结构。

        Args:
            text: frontmatter 文本

        Returns:
            解析后的字典
        """
        result: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result

    def advertise(self, name: str) -> Optional[SkillAdvertise]:
        """获取技能的轻量级广告信息。

        Args:
            name: 技能名称

        Returns:
            SkillAdvertise 或 None（技能不存在时）
        """
        node = self._nodes.get(name)
        if node is None:
            return None
        return SkillAdvertise(name=node.name, description=node.description)

    def activate(self, name: str) -> Optional[LoadedSkill]:
        """激活技能，加载完整定义（指令、脚本、引用）。

        Args:
            name: 技能名称

        Returns:
            LoadedSkill 或 None（技能不存在时）
        """
        node = self._nodes.get(name)
        if node is None:
            return None

        return LoadedSkill(
            name=node.name,
            description=node.description,
            skill_path=str(node.skill_path),
            allowed_tools=node.allowed_tools,
            compatibility=node.compatibility,
            instructions=node.instructions,
            scripts=dict(node.scripts),
            references=dict(node.references),
            metadata=dict(node.metadata),
        )

    def load_script(self, name: str, script_name: str) -> Optional[str]:
        """加载指定技能的脚本内容。

        Args:
            name: 技能名称
            script_name: 脚本文件名

        Returns:
            脚本内容字符串，或 None（技能/脚本不存在时）
        """
        node = self._nodes.get(name)
        if node is None:
            return None
        scripts = node.scripts
        return scripts.get(script_name)

    def validate_compatibility(self, name: str) -> bool:
        """验证技能的兼容性要求。

        检查 compatibility 列表中的要求（如 python >= 3.10）是否满足。

        Args:
            name: 技能名称

        Returns:
            True 表示兼容，False 表示不兼容或技能不存在
        """
        node = self._nodes.get(name)
        if node is None:
            return False

        if not node.compatibility:
            return True

        for req in node.compatibility:
            req_lower = req.lower().strip()
            if req_lower.startswith("python"):
                # 解析 "python >= 3.10" 等格式
                if not self._check_python_requirement(req_lower):
                    return False

        return True

    def _check_python_requirement(self, req: str) -> bool:
        """检查 Python 版本要求。

        支持格式：python >= 3.10, python >=3.10, python>=3.10 等。
        """
        import re

        match = re.search(r'>=\s*(\d+)\.(\d+)', req)
        if match:
            req_major = int(match.group(1))
            req_minor = int(match.group(2))
            return sys.version_info >= (req_major, req_minor)

        match = re.search(r'>\s*(\d+)\.(\d+)', req)
        if match:
            req_major = int(match.group(1))
            req_minor = int(match.group(2))
            return sys.version_info > (req_major, req_minor)

        # 无法解析的要求，默认通过
        return True

    def get_node(self, name: str) -> Optional[_ScanNode]:
        """获取指定名称的扫描节点。

        Args:
            name: 技能名称

        Returns:
            _ScanNode 或 None
        """
        return self._nodes.get(name)

    def list_all_names(self) -> List[str]:
        """列出所有已扫描技能的名称。"""
        return list(self._nodes.keys())

    def list_leaf_names(self) -> List[str]:
        """列出所有叶子技能的名称。"""
        return [name for name, node in self._nodes.items() if node.is_leaf]
