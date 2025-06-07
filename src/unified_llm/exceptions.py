"""Exception classes for unified LLM interface."""


class UnifiedLLMError(Exception):
    """Base exception for all framework errors."""
    pass


class ProviderError(UnifiedLLMError):
    """Provider-specific API errors."""
    
    def __init__(self, message: str, provider: str = None, status_code: int = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class ToolExecutionError(UnifiedLLMError):
    """Tool execution failures."""
    
    def __init__(self, message: str, tool_name: str = None, original_error: Exception = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


class ValidationError(UnifiedLLMError):
    """Input validation errors."""
    pass


class ConfigurationError(UnifiedLLMError):
    """Configuration-related errors."""
    pass 