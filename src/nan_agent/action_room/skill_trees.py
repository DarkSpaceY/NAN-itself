"""
技能树系统 - Agent 技能的分层管理与导航

技能树从 SkillLoader 扫描 SKILL.md 文件构建，无需预定义。
支持语义搜索、导航和推荐。

核心组件：
- SkillNode: 技能节点（名称、描述、层级、子节点）
- SkillTree: 单棵技能树（按类别组织）
- SkillTreeManager: 技能树管理器（多树管理、搜索、推荐）
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SkillNode:
    name: str
    description: str
    prerequisites: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    level: int = 0
    category: str = ""
    handler: Optional[Callable] = None
    unlocked: bool = False
    proficiency: float = 0.0
    usage_count: int = 0
    skill_path: Optional[str] = None
    is_leaf: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    compatibility: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "prerequisites": self.prerequisites,
            "children": self.children,
            "level": self.level,
            "category": self.category,
            "unlocked": self.unlocked,
            "proficiency": self.proficiency,
            "usage_count": self.usage_count,
            "skill_path": self.skill_path,
            "is_leaf": self.is_leaf,
            "allowed_tools": self.allowed_tools,
            "compatibility": self.compatibility,
        }


class SkillTree:
    """单棵技能树，按类别组织技能节点。

    支持节点的添加、查找、搜索和统计。根节点默认为解锁状态，
    无前置条件的节点自动解锁。

    Attributes:
        category: 技能树类别名称
        total_nodes: 节点总数
        unlocked_count: 已解锁节点数
    """

    def __init__(self, category: str):
        self.category = category
        self._nodes: dict[str, SkillNode] = {}
        logger.debug("skill_tree_created", category=category)

    def add_node(self, node: SkillNode):
        node.category = self.category
        self._nodes[node.name] = node
        if not node.prerequisites:
            node.unlocked = True

    def get_node(self, name: str) -> Optional[SkillNode]:
        return self._nodes.get(name)

    def get_children(self, name: str) -> list[SkillNode]:
        node = self._nodes.get(name)
        if not node or not node.children:
            return []
        return [
            self._nodes[c]
            for c in node.children
            if c in self._nodes
        ]

    def search_nodes(self, query: str) -> list[SkillNode]:
        q = query.lower()
        return [
            n for n in self._nodes.values()
            if q in n.name.lower() or q in n.description.lower()
        ]

    def list_all(self) -> list[SkillNode]:
        return list(self._nodes.values())

    @property
    def total_nodes(self) -> int:
        return len(self._nodes)

    @property
    def unlocked_count(self) -> int:
        return sum(1 for n in self._nodes.values() if n.unlocked)


class SkillTreeManager:
    """技能树管理器 - 管理多棵技能树。

    从 SkillLoader 扫描 SKILL.md 文件构建技能树。
    支持语义搜索、导航和推荐。
    """

    def __init__(self, loader):
        self._trees: dict[str, SkillTree] = {}
        self._loader = loader
        self._initialized = False

    def initialize(self) -> None:
        """Build skill trees from SkillLoader scan results."""
        if self._initialized:
            return
        self._build_from_loader()
        self._initialized = True

    def _build_from_loader(self) -> None:
        """Build trees from SkillLoader scan results."""
        from nan_agent.action_room.skill_loader import SkillLoader as _SL
        if not isinstance(self._loader, _SL):
            self._build_all_trees()
            return

        root_nodes = self._loader.scan()
        for root_node in root_nodes:
            tree = SkillTree(root_node.name)
            self._add_scan_node(tree, root_node, root_node.name)
            self._trees[root_node.name] = tree

        logger.info(
            "skill_tree_manager_initialized_from_loader",
            trees=len(self._trees),
            total_nodes=self.total_nodes,
        )
        self._initialized = True

    def _add_scan_node(self, tree: SkillTree, scan_node, category: str) -> None:
        """Recursively add _ScanNode and its children to a SkillTree."""
        node = SkillNode(
            name=scan_node.name,
            description=scan_node.description,
            level=scan_node.level,
            category=category,
            children=[c.name for c in scan_node.children],
            skill_path=str(scan_node.skill_path),
            is_leaf=scan_node.is_leaf,
            allowed_tools=scan_node.allowed_tools,
            compatibility=", ".join(scan_node.compatibility) if isinstance(scan_node.compatibility, list) else (scan_node.compatibility or None),
            metadata=dict(scan_node.metadata) if scan_node.metadata else {},
            unlocked=True,
        )
        tree.add_node(node)

        for child in scan_node.children:
            self._add_scan_node(tree, child, category)

    def get_tree(self, category: str) -> Optional[SkillTree]:
        return self._trees.get(category)

    def get_node(self, name: str) -> Optional[SkillNode]:
        for tree in self._trees.values():
            node = tree.get_node(name)
            if node:
                return node
        return None

    def get_children(self, parent_name: str) -> list[SkillNode]:
        for tree in self._trees.values():
            node = tree.get_node(parent_name)
            if node:
                return tree.get_children(parent_name)
        return []

    def search_all(self, query: str) -> list[SkillNode]:
        results: list[SkillNode] = []
        for tree in self._trees.values():
            results.extend(tree.search_nodes(query))
        return results

    def search_unlockable(self) -> list[SkillNode]:
        results: list[SkillNode] = []
        for tree in self._trees.values():
            for node in tree._nodes.values():
                if not node.unlocked and self._all_prereqs_met(node):
                    results.append(node)
        return results

    def _all_prereqs_met(self, node: SkillNode) -> bool:
        if not node.prerequisites:
            return True
        for prereq_name in node.prerequisites:
            prereq = self.get_node(prereq_name)
            if prereq is None or prereq.proficiency < 0.3:
                return False
        return True

    def recommend_skills(self, top_k: int = 5) -> list[SkillNode]:
        unlockable = self.search_unlockable()
        unlockable.sort(key=lambda n: n.proficiency, reverse=True)
        if len(unlockable) < top_k:
            unlocked = []
            for tree in self._trees.values():
                for n in tree._nodes.values():
                    if n.unlocked and n.proficiency < 0.5:
                        unlocked.append(n)
            unlocked.sort(key=lambda n: n.proficiency)
            needed = top_k - len(unlockable)
            unlockable.extend(unlocked[:needed])
        return unlockable[:top_k]

    def register_skill_from_disk(self, name: str, category: str, skill_dir: str):
        tree = self._trees.get(category)
        if not tree:
            logger.warning("unknown_skill_category", name=name, category=category)
            return None

        node = tree.get_node(name)
        if node is None:
            parts = skill_dir.rstrip("/").split("/")
            parent_name = parts[-2] if len(parts) >= 2 else category
            parent = tree.get_node(parent_name)
            node = SkillNode(
                name=name,
                description=f"Skill: {name}",
                level=3,
                category=category,
                skill_path=skill_dir,
                unlocked=True,
            )
            if parent:
                parent.children.append(name)
            tree.add_node(node)
            logger.info("skill_registered", name=name, category=category, path=skill_dir)
        else:
            node.skill_path = skill_dir
            node.unlocked = True

        return node

    def load_skills_from_directory(self, base_dir: str):
        base = Path(base_dir)
        if not base.exists():
            logger.warning("skill_dir_not_found", path=base_dir)
            return 0

        count = 0
        for skill_md_path in base.rglob("SKILL.md"):
            try:
                content = skill_md_path.read_text()
                meta = self._parse_skill_metadata(content)
                name = meta.get("name", skill_md_path.parent.name)
                category_path = meta.get("category", "")
                category = category_path.split("/")[0] if category_path else ""
                if category not in self._trees:
                    continue
                skill_dir = str(skill_md_path.parent)
                node = self.register_skill_from_disk(name, category, skill_dir)
                if node:
                    count += 1
            except Exception as e:
                logger.warning("skill_load_error", path=str(skill_md_path), error=str(e))

        logger.info("skills_loaded_from_directory", count=count, base_dir=base_dir)
        return count

    def _parse_skill_metadata(self, content: str) -> dict:
        if not content.startswith("---"):
            return {}
        parts = content[3:].split("---", 1)
        if len(parts) < 2:
            return {}
        import yaml
        return yaml.safe_load(parts[0]) or {}

    async def agentic_search(
        self,
        task_intent: str,
        cognition,
        max_depth: int = 3,
    ) -> Optional[SkillNode]:
        from nan_agent.model.types import MultiModalInput

        candidates: list[SkillNode] = []
        for tree in self._trees.values():
            root = tree.get_node(tree.category)
            if root:
                candidates.append(root)

        for depth in range(max_depth):
            if not candidates:
                return None

            children: list[SkillNode] = []
            for c in candidates:
                children.extend(self.get_children(c.name))

            if not children:
                if len(candidates) == 1:
                    return candidates[0]
                return self._best_by_semantic(candidates, task_intent)

            candidate_info = [
                {
                    "name": ch.name,
                    "description": ch.description,
                    "level": ch.level,
                    "category": ch.category,
                }
                for ch in children
            ]

            prompt = (
                "You are navigating a skill tree to find the best category "
                "for a task. From the categories below, select the ONE most "
                "relevant match.\n\n"
                f"Task: {task_intent}\n\n"
                f"Categories:\n{json.dumps(candidate_info, ensure_ascii=False, indent=2)}\n\n"
                "Reply with ONLY the exact category name, nothing else."
            )

            user_input = MultiModalInput()
            user_input.add_text(prompt)

            try:
                output = await cognition.infer_small(user_input, temperature=0.3)
                response = output.text.strip() if output and output.text else ""
            except Exception as e:
                logger.warning("agentic_search_infer_failed", error=str(e))
                return candidates[0] if candidates else None

            selected = None
            response_lower = response.lower()
            for ch in children:
                if ch.name.lower() in response_lower or response_lower in ch.name.lower():
                    selected = ch
                    break

            if selected is None:
                return self._best_by_semantic(children, task_intent)

            candidates = [selected]

        return candidates[0] if candidates else None

    def _best_by_semantic(
        self,
        nodes: list[SkillNode],
        task: str,
    ) -> Optional[SkillNode]:
        task_words = set(task.lower().split())
        best_node = None
        best_score = 0
        for node in nodes:
            desc_words = set(node.description.lower().split())
            name_words = set(node.name.lower().replace("_", " ").split())
            score = len(task_words & desc_words) + 2 * len(task_words & name_words)
            if score > best_score:
                best_score = score
                best_node = node
        return best_node

    def get_category_overview(self) -> Dict[str, Any]:
        """返回所有顶级类别的概览信息。

        用于 Agent 初始了解有哪些技能领域，获取每个类别的名称、
        描述和直接子节点数量，便于后续导航决策。

        Returns:
            包含 categories 列表的字典，每个元素包含 name、description、child_count。
        """
        categories = []
        for cat_name, tree in self._trees.items():
            root = tree.get_node(cat_name)
            categories.append({
                "name": cat_name,
                "description": root.description if root else "",
                "child_count": len(root.children) if root else 0,
            })
        return {"categories": categories}

    def navigate(self, node_path: str) -> Dict[str, Any]:
        """通过节点路径导航技能树，返回当前节点信息和子节点列表。

        路径格式为 "category/level1_node/level2_node"，第一段为类别名称，
        后续段为子节点名称。逐层解析路径，定位到目标节点后返回其详细信息
        及子节点概览。

        Args:
            node_path: 以斜杠分隔的节点路径，如 "software_development/code_generation"。

        Returns:
            包含 current（当前节点信息）、children（子节点列表）、
            has_children（是否有子节点）、path（当前路径）的字典。
            若路径中的节点不存在，返回 {"error": "Node 'xxx' not found in path"}。
        """
        parts = node_path.strip("/").split("/")
        if not parts or not parts[0]:
            return {"error": "Empty path"}

        # 第一段是 category
        category = parts[0]
        tree = self._trees.get(category)
        if tree is None:
            return {"error": f"Node '{category}' not found in path"}

        current = tree.get_node(category)
        if current is None:
            return {"error": f"Node '{category}' not found in path"}

        # 逐层解析后续路径段
        for part in parts[1:]:
            if part not in current.children:
                return {"error": f"Node '{part}' not found in path"}
            child = tree.get_node(part)
            if child is None:
                return {"error": f"Node '{part}' not found in path"}
            current = child

        # 构建子节点列表
        children_info = []
        for child_name in current.children:
            child_node = tree.get_node(child_name)
            if child_node:
                children_info.append({
                    "name": child_node.name,
                    "description": child_node.description,
                    "level": child_node.level,
                })

        return {
            "current": {
                "name": current.name,
                "description": current.description,
                "level": current.level,
                "category": current.category,
                "children": current.children,
            },
            "children": children_info,
            "has_children": len(children_info) > 0,
            "path": node_path,
        }

    def get_node_details(self, name: str) -> Dict[str, Any]:
        """获取指定节点的详细信息，包括完整属性、父节点、兄弟节点等。

        通过遍历所有技能树的所有节点来查找目标节点及其父节点，
        返回节点的完整信息、父节点概览、兄弟节点列表和子节点列表。

        Args:
            name: 要查询的节点名称。

        Returns:
            包含 node（节点完整属性）、parent（父节点信息或 None）、
            siblings（兄弟节点列表）、children（子节点列表）、
            has_children（是否有子节点）的字典。
            若节点不存在，返回 {"error": "Node 'xxx' not found"}。
        """
        # 查找目标节点
        target_node = None
        parent_node = None
        target_tree = None

        for tree in self._trees.values():
            node = tree.get_node(name)
            if node:
                target_node = node
                target_tree = tree
                # 查找父节点：遍历该树所有节点，找到 children 中包含 name 的节点
                for candidate in tree._nodes.values():
                    if name in candidate.children:
                        parent_node = candidate
                        break
                break

        if target_node is None:
            return {"error": f"Node '{name}' not found"}

        # 构建子节点列表
        children_info = []
        for child_name in target_node.children:
            child_node = target_tree.get_node(child_name)
            if child_node:
                children_info.append({
                    "name": child_node.name,
                    "description": child_node.description,
                    "level": child_node.level,
                })

        # 构建兄弟节点列表
        siblings_info = []
        if parent_node:
            for sibling_name in parent_node.children:
                if sibling_name != name:
                    sibling_node = target_tree.get_node(sibling_name)
                    if sibling_node:
                        siblings_info.append({
                            "name": sibling_node.name,
                            "description": sibling_node.description,
                        })

        return {
            "node": target_node.to_dict(),
            "parent": {
                "name": parent_node.name,
                "description": parent_node.description,
            } if parent_node else None,
            "siblings": siblings_info,
            "children": children_info,
            "has_children": len(children_info) > 0,
        }

    @property
    def tree_structure(self) -> Dict[str, Any]:
        """返回整个技能树的精简结构视图。

        每棵树只包含名称和子节点名称（不包含详细描述），
        用于全局概览，帮助 Agent 快速了解技能树的整体结构。

        Returns:
            以类别名为键、包含 children 列表的字典为值的嵌套结构。
        """
        result: Dict[str, Any] = {}
        for cat_name, tree in self._trees.items():
            root = tree.get_node(cat_name)
            result[cat_name] = {
                "children": root.children if root else [],
            }
        return result

    @property
    def total_nodes(self) -> int:
        return sum(t.total_nodes for t in self._trees.values())

    @property
    def categories(self) -> list[str]:
        return list(self._trees.keys())
