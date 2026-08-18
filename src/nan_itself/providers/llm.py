import json
from typing import AsyncIterator, Literal
from pydantic import BaseModel

from loguru import logger
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    provider: Literal["openai", "anthropic"]

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class LLMRequest(BaseModel):
    messages: list[Message]
    temperature: float
    max_tokens: int

class LLMProvider:
    USAGE_MARKER = "__USAGE__"  # 特殊标记，正常 token 不会包含

    def __init__(
        self,
        provider: Literal["openai", "anthropic"],
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        max_retries: int,
    ):
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        logger.info(f"Initializing LLMProvider: provider={provider}, model={model}, base_url={base_url}, timeout={timeout}, max_retries={max_retries}")

        if provider == "openai":
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            self.client = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

        self._closed = False

    async def generate(self, request: LLMRequest) -> AsyncIterator[str]:
        """永远返回流式迭代器"""
        if self.provider == "openai":
            async for chunk in self._stream_openai(request):
                yield chunk
        else:
            async for chunk in self._stream_anthropic(request):
                yield chunk

    async def _stream_openai(self, request: LLMRequest) -> AsyncIterator[str]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        async for chunk in await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if chunk.choices:
                yield chunk.choices[0].delta.content
            
            if chunk.usage:
                usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                }
                yield f"{self.USAGE_MARKER}{json.dumps(usage_data)}"

    async def _stream_anthropic(self, request: LLMRequest) -> AsyncIterator[str]:
        #默认使用第一个消息作为系统提示
        system_prompt = request.messages[0].content
        messages = [{"role": m.role, "content": m.content} for m in request.messages[1:]]

        async with self.client.messages.stream(
            model=self.model,
            messages=messages,
            system=system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text
            
            final_message = await stream.get_final_message()
            if final_message.usage:
                usage_data = {
                    "prompt_tokens": final_message.usage.input_tokens,
                    "completion_tokens": final_message.usage.output_tokens,
                }
                yield f"{self.USAGE_MARKER}{json.dumps(usage_data)}"

    async def generate_complete(self, request: LLMRequest) -> LLMResponse:
        """
        收集流式输出为完整响应，同时提取 usage 信息
        
        Returns:
            LLMResponse: 包含完整内容和 usage
        """
        content_parts: list[str] = []
        async for item in self.generate(request):
            # 检查是否是 usage 标记
            if item.startswith(self.USAGE_MARKER):
                usage_data = json.loads(item[len(self.USAGE_MARKER):])
                usage = Usage(
                    prompt_tokens=usage_data["prompt_tokens"],
                    completion_tokens=usage_data["completion_tokens"],
                )
            else:
                content_parts.append(item)
        
        content = "".join(content_parts)
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            provider=self.provider,
        )

    async def close(self) -> None:
        if not self._closed:
            logger.info(f"Closing LLMProvider: provider={self.provider}, model={self.model}")
            await self.client.close()
            self._closed = True