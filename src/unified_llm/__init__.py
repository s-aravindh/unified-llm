"""Unified LLM Interface - Provider-agnostic LLM interface."""

from .openai_like import OpenAILike
from .tool_executor import ToolExecutor
from .models import ChatResponse, ChatStreamResponse
from .exceptions import (
    UnifiedLLMError,
    ProviderError,
    ToolExecutionError,
    ValidationError,
    ConfigurationError
)

__version__ = "0.1.0"

__all__ = [
    "OpenAILike",
    "ToolExecutor",
    "ChatResponse", 
    "ChatStreamResponse",
    "UnifiedLLMError",
    "ProviderError",
    "ToolExecutionError", 
    "ValidationError",
    "ConfigurationError",
]
