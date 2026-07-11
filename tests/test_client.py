"""Tests for the LLM harness."""

import pytest
from unittest.mock import patch, MagicMock

from src.llm.client import load_config, estimate_cost, complete, LLMConfig


# --- Pure logic tests (no API calls, free, instant) ---

def test_estimate_cost_haiku():
    """Cost math should match hand-calculated values for Haiku."""
    # 1,000,000 input tokens at $1/M = $1.00; 1,000,000 output at $5/M = $5.00
    cost = estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
    assert cost == pytest.approx(6.0)


def test_estimate_cost_unknown_model_is_zero():
    """Unknown models fall back to zero cost, not a crash."""
    cost = estimate_cost("not-a-real-model", 1000, 1000)
    assert cost == 0.0


def test_config_rejects_bad_temperature():
    """Pydantic should reject a temperature above 1.0."""
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        LLMConfig(api_key="fake", temperature=5.0)


def test_load_config_missing_key_raises():
    """A missing API key should raise a clear ValueError."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            load_config()


# --- Harness test with a MOCKED API (no real call, no cost) ---

def test_complete_returns_text():
    """complete() should return the text from the response, mocked."""
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="hello world")]
    fake_response.usage.input_tokens = 5
    fake_response.usage.output_tokens = 2

    config = LLMConfig(api_key="fake-key")

    with patch("src.llm.client._call_with_retry", return_value=fake_response):
        result = complete("any prompt", config=config)

    assert result == "hello world"