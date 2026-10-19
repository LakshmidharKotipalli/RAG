"""LLM provider errors with actionable messages."""


class LLMError(RuntimeError):
    """Base class for provider failures."""


class ProviderNotConfiguredError(LLMError):
    """Missing or rejected credentials, or an unknown provider/model mapping."""


class ProviderUnavailableError(LLMError):
    """The provider could not be reached (e.g. Ollama is not running)."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout after all retries."""
