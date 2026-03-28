"""
Custom exceptions for the experiment framework.
"""

class FatalModelError(Exception):
    """
    Raised when a model error occurs that should terminate the experiment immediately.
    Examples: Model ID not found (404), Invalid API key (401).
    """
    pass
