"""硬记忆共享工具模块。

提取自 store.py / skill_store.py / retrieve.py 的公共组件，
消除三处重复定义。
"""

import math
import re
import uuid
from collections import defaultdict


def new_id() -> str:
    """生成 12 位十六进制唯一 ID。"""
    return uuid.uuid4().hex[:12]


def tokenize(text: str) -> list[str]:
    """简单分词：小写 + 按非字母数字分割。"""
    return [t for t in re.split(r'[^a-z0-9]+', text.lower()) if t]


class BM25Index:
    """BM25 关键词索引。

    为每条记忆内容建立倒排索引，支持基于关键词的精确匹配检索。
    与向量语义检索互补：BM25 擅长专有名词、数字、精确匹配。
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._inverted: dict[str, dict[str, int]] = defaultdict(dict)
        self._avg_dl: float = 0.0
        self._total_docs: int = 0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self._docs[doc_id] = tokens
        self._doc_len[doc_id] = len(tokens)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for t, f in tf.items():
            self._inverted[t][doc_id] = f
        self._total_docs += 1
        self._avg_dl = sum(self._doc_len.values()) / self._total_docs

    def remove(self, doc_id: str) -> None:
        """从索引中移除一条文档。"""
        if doc_id not in self._docs:
            return
        tokens = self._docs.pop(doc_id)
        self._doc_len.pop(doc_id, None)
        for t in set(tokens):
            self._inverted.get(t, {}).pop(doc_id, None)
            if t in self._inverted and not self._inverted[t]:
                del self._inverted[t]
        self._total_docs -= 1
        self._avg_dl = (
            sum(self._doc_len.values()) / self._total_docs
            if self._total_docs > 0 else 0.0
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """BM25 搜索，返回 (doc_id, score) 列表（按 score 降序）。"""
        query_tokens = tokenize(query)
        if not query_tokens or self._total_docs == 0:
            return []
        scores: dict[str, float] = defaultdict(float)
        for t in set(query_tokens):
            if t not in self._inverted:
                continue
            n = len(self._inverted[t])
            idf = math.log((self._total_docs - n + 0.5) / (n + 0.5) + 1.0)
            for doc_id, tf in self._inverted[t].items():
                dl = self._doc_len[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avg_dl)
                scores[doc_id] += idf * numerator / denominator
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
