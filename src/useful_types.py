"""
Useful types for the annotations
"""

from typing import Protocol


class ReprPretty(Protocol):
    """
    This can be used in _repr_pretty as the p argument
    """
    def text(self, string: str) -> None:
        """
        The method called in _repr_pretty on p
        """
