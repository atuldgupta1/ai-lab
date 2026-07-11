"""LLM Harness - Config Loading (Phase 0, piece 1)."""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from anthropic import Anthropic, AsyncAnthropic, APIConnectionError, APIStatusError, RateLimitError
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("llm")

# Quiet the noisy third-party loggers — only show their warnings/errors.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

load_dotenv()

class LLMConfig(BaseModel):
    """Typed, validated configuration for the LLM client."""
    api_key: str = Field(...,description="Anthropic API key, loaded from env") #... means required
    model:str = Field(default="claude-haiku-4-5-20251001")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024,gt= 0)

def load_config() -> LLMConfig:
    """Build an LLMConfig, reading the API key from the environment.

    Raises a clear error if the key is missing, instead of failing
    later with a confusing authentication error from the SDK.
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file as: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return LLMConfig(api_key=api_key)

def get_client(config:LLMConfig) -> Anthropic:
    """Create an Anthropic SDK client from our config."""
    return Anthropic(api_key=config.api_key)

def get_async_client(config:LLMConfig) -> AsyncAnthropic:
    """Create an async Anthropic SDK client from our config."""
    return AsyncAnthropic(api_key=config.api_key)

# Price per MILLION tokens, in USD. Update if rates change.
# (input, output)
PRICING = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost of a call from token counts."""
    in_rate, out_rate = PRICING.get(model, (0.0, 0.0))
    cost = (input_tokens / 1_000_000) * in_rate
    cost += (output_tokens / 1_000_000) * out_rate
    return cost

@retry(
        retry=retry_if_exception_type(
            (APIConnectionError,APIStatusError,RateLimitError)
        ),
        stop = stop_after_attempt(4),
        wait = wait_exponential(multiplier = 1, min=2, max = 30),
        before_sleep = before_sleep_log(logger, logging.WARNING),
        reraise = True,
)
def _call_with_retry(client: Anthropic, config:LLMConfig, prompt:str):
    """The raw API call, wrapped with automatic retry on transient errors."""
    return client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature= config.temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

@retry(
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APIStatusError)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _acall_with_retry(client:AsyncAnthropic, config:LLMConfig, prompt:str):
    """The raw async API call, wrapped with automatic retry."""
    return await client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}]
    )

def complete(prompt:str, config:LLMConfig | None = None) -> str:
    """Send a prompt to Claude, with retries, logging, and cost tracking.

    If no config is passed, load the default one.
    """
    if config is None:
        config = load_config()
    
    client = get_client(config)

    response = _call_with_retry(client,config, prompt)

    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cost = estimate_cost(config.model, input_tokens, output_tokens)

    logger.info(
        "model=%s | in=%d tok | out=%d tok | cost=$%.6f",
        config.model, input_tokens, output_tokens, cost,
    )

    # response.content is a LIST of content blocks, not a string.
    # For a plain text reply, the text lives in the first block.
    return response.content[0].text

async def acomplete(prompt:str, config:LLMConfig | None = None) -> str:
    """Send a prompt to Claude, with retries, logging, and cost tracking.

    If no config is passed, load the default one.
    """
    if config is None:
        config = load_config()
    
    client = get_async_client(config)

    response = await _acall_with_retry(client,config, prompt)

    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cost = estimate_cost(config.model, input_tokens, output_tokens)

    logger.info(
        "model=%s | in=%d tok | out=%d tok | cost=$%.6f",
        config.model, input_tokens, output_tokens, cost,
    )

    # response.content is a LIST of content blocks, not a string.
    # For a plain text reply, the text lives in the first block.
    return response.content[0].text

