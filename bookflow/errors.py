class BookFlowError(RuntimeError):
    """Expected, user-actionable failure in the BookFlow pipeline."""


class ExternalToolError(BookFlowError):
    pass


class QualityError(BookFlowError):
    pass
