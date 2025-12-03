"""
Useful types for the annotations
"""

from typing import Protocol

import numpy as np


class ReprPretty(Protocol):
    """
    This can be used in _repr_pretty as the p argument
    """

    def text(self, string: str) -> None:
        """
        The method called in _repr_pretty on p
        """


type U8Matrix = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
type U8Vector = np.ndarray[tuple[int], np.dtype[np.uint8]]

type IntMatrix = np.ndarray[tuple[int, int], np.dtype[np.integer]]
type IntVector = np.ndarray[tuple[int], np.dtype[np.integer]]

type BoolIntVector = np.ndarray[tuple[int], np.dtype[np.integer | np.bool]]

type BoolMatrix = np.ndarray[tuple[int, int], np.dtype[np.bool]]
type BoolVector = np.ndarray[tuple[int], np.dtype[np.bool]]
