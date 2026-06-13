import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from nan_agent.exceptions import MemoryError, ModelError
from nan_agent.logging.logger import get_logger
from nan_agent.model.provider import InferenceRequest, ModelProvider
from nan_agent.model.types import MultiModalOutput

logger = get_logger(__name__)

DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_EMBED_DIM = 2560
DEFAULT_SMALL_MODEL = "alibayram/smollm3:latest"
DEFAULT_KEEP_ALIVE = "5m"  # 5 分钟复用窗口；切换时主动卸载
DEFAULT_MAX_PARALLEL_PER_MODEL = 3


class OllamaProvider(ModelProvider):
    """Ollama 模型提供器，支持按模型并发和按需切换。

    调度策略：
    - 每个模型独立信号量，允许单模型并发（默认 3 路）
    - keep_alive=-1 持久保活，直到显式切换时主动卸载
    - 切换模型时：先发 /api/generate keep_alive=0 卸载旧模型 → 轮询 /api/ps 确认
    - embed 模型使用独立并发控制，不与 chat 模型互斥
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "gemma4:26b",
        timeout: float = 120.0,
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed_dim: int = DEFAULT_EMBED_DIM,
        small_chat_model: str | None = DEFAULT_SMALL_MODEL,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        max_parallel: int = DEFAULT_MAX_PARALLEL_PER_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.small_chat_model = small_chat_model
        self._keep_alive = keep_alive
        self._max_parallel = max_parallel
        self._client: httpx.AsyncClient | None = None

        # ── 模型调度状态 ──
        self._loaded_model: str | None = None  # 当前加载的 chat 模型
        self._model_lock = asyncio.Lock()  # 模型切换互斥锁
        self._per_model_semaphores: dict[str, asyncio.Semaphore] = {}
        # embed 模型独立信号量（不与 chat 模型竞争）
        self._embed_semaphore = asyncio.Semaphore(max_parallel)

        logger.info(
            "ollama_provider_initialized",
            model_name=self.model_name,
            small_model=self.small_chat_model,
            embed_model=self.embed_model,
            keep_alive=self._keep_alive,
            max_parallel=self._max_parallel,
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _get_chat_semaphore(self, model: str) -> asyncio.Semaphore:
        """获取指定 chat 模型的并发信号量。"""
        if model not in self._per_model_semaphores:
            self._per_model_semaphores[model] = asyncio.Semaphore(self._max_parallel)
            logger.debug(
                "semaphore_created",
                model=model,
                max_parallel=self._max_parallel,
            )
        return self._per_model_semaphores[model]

    # ═══════════════════════════════════════════════════════════
    # 模型切换
    # ═══════════════════════════════════════════════════════════

    async def _switch_model_if_needed(self, model: str) -> None:
        """需要切换模型时，等待旧模型卸载释放显存。

        仅在首次推理或模型变更时才轮询 /api/ps。
        同模型连续调用不触发任何等待。
        """
        if self._loaded_model is None:
            # 首次推理，无需等待
            self._loaded_model = model
            logger.info("model_first_load", model=model)
            return
        if self._loaded_model == model:
            # 同一模型，不切换
            logger.debug("model_reuse", model=model)
            return
        # 切换模型：主动卸载旧模型，再等待确认
        logger.info(
            "model_switch_start",
            from_model=self._loaded_model,
            to_model=model,
        )
        old_model = self._loaded_model
        self._loaded_model = model  # 提前标记，避免并发重复等待
        await self._unload_model(old_model)
        await self._wait_until_gone(old_model, timeout=30.0)

    async def _unload_model(self, model_name: str) -> None:
        """发送 keep_alive=0 请求强制 Ollama 卸载模型。"""
        t0 = time.monotonic()
        try:
            client = self._get_client()
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": 0,
                    "stream": False,
                },
            )
            elapsed = round((time.monotonic() - t0) * 1000)
            if resp.status_code == 200:
                logger.info("model_unload_requested", model=model_name, elapsed_ms=elapsed)
            else:
                logger.warning(
                    "model_unload_request_failed",
                    model=model_name,
                    status=resp.status_code,
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = round((time.monotonic() - t0) * 1000)
            logger.warning("model_unload_request_error", model=model_name, error=str(e), elapsed_ms=elapsed)

    async def _wait_until_gone(
        self, model_name: str, timeout: float = 30.0, interval: float = 0.5
    ) -> None:
        """轮询 /api/ps 直到 model_name 不在已加载列表中。"""
        deadline = time.monotonic() + timeout
        client = self._get_client()
        poll_count = 0
        start_time = time.monotonic()
        while time.monotonic() < deadline:
            poll_count += 1
            try:
                resp = await client.get(f"{self.base_url}/api/ps")
                if resp.status_code == 200:
                    loaded_models = resp.json().get("models", [])
                    loaded_names = [m.get("name", "") for m in loaded_models]
                    if not any(m.get("name", "") == model_name for m in loaded_models):
                        elapsed = round(time.monotonic() - start_time, 2)
                        logger.info(
                            "model_unloaded",
                            model=model_name,
                            elapsed_s=elapsed,
                            polls=poll_count,
                            still_loaded=loaded_names,
                        )
                        return
            except Exception:
                pass
            await asyncio.sleep(interval)
        elapsed = round(time.monotonic() - start_time, 2)
        logger.warning(
            "model_unload_timeout",
            model=model_name,
            timeout=timeout,
            elapsed_s=elapsed,
            polls=poll_count,
        )

    # ═══════════════════════════════════════════════════════════
    # Embedding
    # ═══════════════════════════════════════════════════════════

    async def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.embed_dim

        wait_start = time.monotonic()
        async with self._embed_semaphore:
            wait_ms = round((time.monotonic() - wait_start) * 1000)
            if wait_ms > 10:
                logger.debug("embed_semaphore_wait", model=self.embed_model, wait_ms=wait_ms)
            return await self._embed_impl(text)

    async def _embed_impl(self, text: str) -> list[float]:
        last_error = None
        client = self._get_client()

        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.embed_model, "input": [text], "keep_alive": self._keep_alive},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and len(embeddings) > 0:
                        vec = embeddings[0]
                        if len(vec) >= self.embed_dim:
                            return vec[:self.embed_dim]
                        padded = list(vec) + [0.0] * (self.embed_dim - len(vec))
                        return padded
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                last_error = str(e)
            if attempt < 2:
                wait = 2 ** attempt
                await asyncio.sleep(wait)

        raise MemoryError(
            f"Embedding API unavailable at {self.base_url} after retries. "
            f"Last error: {last_error}"
        )

    # ═══════════════════════════════════════════════════════════
    # Chat Inference (主模型，支持并发)
    # ═══════════════════════════════════════════════════════════

    async def infer(self, request: InferenceRequest) -> MultiModalOutput:
        model = self.model_name
        await self._switch_model_if_needed(model)
        sem = self._get_chat_semaphore(model)
        wait_start = time.monotonic()
        async with sem:
            wait_ms = round((time.monotonic() - wait_start) * 1000)
            if wait_ms > 10:
                logger.debug("infer_semaphore_wait", model=model, wait_ms=wait_ms)
            t0 = time.monotonic()
            result = await self._infer_impl(request, model=model)
            elapsed_ms = round((time.monotonic() - t0) * 1000)
            logger.debug("infer_done", model=model, elapsed_ms=elapsed_ms)
            return result

    async def infer_small(self, request: InferenceRequest) -> MultiModalOutput:
        """使用小型 chat 模型推理（独立并发）。"""
        model = self.small_chat_model or self.model_name
        await self._switch_model_if_needed(model)
        sem = self._get_chat_semaphore(model)
        wait_start = time.monotonic()
        async with sem:
            wait_ms = round((time.monotonic() - wait_start) * 1000)
            if wait_ms > 10:
                logger.debug("infer_small_semaphore_wait", model=model, wait_ms=wait_ms)
            t0 = time.monotonic()
            result = await self._infer_impl(request, model=model)
            elapsed_ms = round((time.monotonic() - t0) * 1000)
            logger.debug("infer_small_done", model=model, elapsed_ms=elapsed_ms)
            return result

    async def infer_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        model = self.model_name
        await self._switch_model_if_needed(model)
        sem = self._get_chat_semaphore(model)
        wait_start = time.monotonic()
        async with sem:
            wait_ms = round((time.monotonic() - wait_start) * 1000)
            if wait_ms > 10:
                logger.debug("infer_stream_semaphore_wait", model=model, wait_ms=wait_ms)
            t0 = time.monotonic()
            chunk_count = 0
            async for chunk in self._infer_stream_impl(request, model=model):
                chunk_count += 1
                yield chunk
            elapsed_ms = round((time.monotonic() - t0) * 1000)
            logger.debug("infer_stream_done", model=model, elapsed_ms=elapsed_ms, chunks=chunk_count)

    # ═══════════════════════════════════════════════════════════
    # 底层 HTTP 调用
    # ═══════════════════════════════════════════════════════════

    async def _infer_impl(
        self, request: InferenceRequest, model: str | None = None
    ) -> MultiModalOutput:
        payload = self._build_payload(request, stream=False, model=model)
        client = self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/api/chat", json=payload
            )
            logger.debug(
                "ollama_request",
                model=payload.get("model"),
                msg_count=len(payload.get("messages", [])),
                msg_preview=str(payload.get("messages", [{}])[0].get("content", ""))[:100] if payload.get("messages") else "",
                options=payload.get("options", {}),
            )
            if not response.is_success:
                cleaned = {k: v for k, v in payload.items() if k != "messages"}
                cleaned["messages_count"] = len(payload.get("messages", []))
                cleaned["messages_first_200"] = str(payload.get("messages", [{}])[0].get("content", ""))[:200] if payload.get("messages") else ""
                logger.error(
                    "ollama_http_error",
                    status=response.status_code,
                    body=response.text[:500],
                    payload=cleaned,
                )
                raise ModelError(
                    f"Ollama inference failed: HTTP {response.status_code}",
                    details={"response": response.text},
                )
            data = response.json()
            content = data["message"].get("content", "")
            if not content:
                thinking = data["message"].get("thinking", "")
                content = thinking if thinking else ""
            output = MultiModalOutput()
            if isinstance(content, str):
                output.add_text(content)
            elif isinstance(content, list):
                text_parts = [
                    item.get("text", "") for item in content if item.get("type") == "text"
                ]
                output.add_text(" ".join(text_parts))
            else:
                output.add_text(str(content))
            return output
        except httpx.HTTPError as e:
            raise ModelError(str(e)) from e

    async def _infer_stream_impl(
        self, request: InferenceRequest, model: str | None = None
    ) -> AsyncIterator[str]:
        payload = self._build_payload(request, stream=True, model=model)
        client = self._get_client()
        try:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                if not response.is_success:
                    body = await response.aread()
                    raise ModelError(
                        f"Ollama streaming failed: HTTP {response.status_code}",
                        details={"response": body.decode()},
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as e:
            raise ModelError(str(e)) from e

    # ═══════════════════════════════════════════════════════════
    # 健康检查 & 生命周期
    # ═══════════════════════════════════════════════════════════

    async def health_check(self) -> bool:
        client = self._get_client()
        try:
            response = await client.get(f"{self.base_url}/api/tags")
            return response.is_success
        except Exception:
            return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        await self.close()
        return False

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ═══════════════════════════════════════════════════════════
    # Payload 构建
    # ═══════════════════════════════════════════════════════════

    def _build_payload(
        self, request: InferenceRequest, stream: bool, model: str | None = None
    ) -> dict:
        options: dict = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens,
        }
        if request.stop:
            options["stop"] = request.stop

        message: dict = {"role": "user"}
        images = request.input.get_images()

        if images:
            message["content"] = request.input.get_text()
            message["images"] = [img.to_base64() for img in images]
        else:
            message["content"] = request.input.get_text()

        return {
            "model": model or self.model_name,
            "messages": [message],
            "stream": stream,
            "options": options,
            "keep_alive": self._keep_alive,
        }
