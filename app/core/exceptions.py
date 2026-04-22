class AppError(Exception):
    """Base application error."""


class ConfigurationError(AppError):
    """Raised when application configuration is invalid."""


class SessionError(AppError):
    """Raised when session handling fails."""


class MWSFetchError(AppError):
    """Raised when source pages cannot be fetched from MWS."""


class MWSParseError(AppError):
    """Raised when source pages cannot be parsed into structured data."""


class CatalogBuildError(AppError):
    """Raised when model and pricing data cannot be merged into a catalog."""


class ProfileBuildError(AppError):
    """Raised when the user case profile cannot be built."""


class EstimationError(AppError):
    """Raised when cost estimation fails."""


class RecommendationError(AppError):
    """Raised when recommendations cannot be produced."""


class ReportBuildError(AppError):
    """Raised when the final structured report cannot be created."""
