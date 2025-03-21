"""Custom exceptions for the connections package."""


class ConnectionError(Exception):
    """Base exception for all connection-related errors."""

    pass


class ParsingError(ConnectionError):
    """Raised when parsing input fails."""

    pass


class ValidationError(ConnectionError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(ConnectionError):
    """Raised when configuration is invalid."""

    pass


class UnificationError(ConnectionError):
    """Raised when unification fails."""

    pass
