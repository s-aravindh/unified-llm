"""Data models for unified LLM interface."""

from typing import Iterator, Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Response from a chat completion."""
    content: str                                  # Final response text
    reasoning_content: Optional[str] = None       # Reasoning process (if available)
    reasoning_tokens: Optional[int] = None        # Token count for reasoning (if provided)
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Tool calls made by the model (standardized format)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Provider/model-specific metadata


class ChatStreamResponse(BaseModel):
    """Response chunk from a streaming chat completion."""
    delta: str                                    # Incremental final content
    reasoning_delta: Optional[str] = None         # Incremental reasoning content
    is_reasoning_complete: bool = False           # Reasoning phase completion
    is_complete: bool = False                     # Stream completion flag
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Tool calls in this chunk (standardized format)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Provider/model-specific metadata 