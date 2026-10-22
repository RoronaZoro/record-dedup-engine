"""Domain-specific errors raised by the library."""


class DeduplicationError(Exception):
    """Base class for expected deduplication failures."""


class InputValidationError(DeduplicationError, ValueError):
    """Raised when caller-provided options or inputs are invalid."""


class SchemaValidationError(InputValidationError):
    """Raised when requested key columns are absent or unsupported."""


class PersistenceError(DeduplicationError):
    """Raised when the optional persistence adapter cannot write records."""
