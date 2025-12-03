"""
Useful types for the annotations
"""

from typing import Protocol

import numpy as np


# pylint:disable=too-few-public-methods
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

type IntMatrix = np.ndarray[tuple[int, int], np.dtype[np.integer | np.signedinteger]]
type IntVector = np.ndarray[tuple[int], np.dtype[np.integer | np.signedinteger]]

type BoolIntVector = np.ndarray[
    tuple[int], np.dtype[np.integer | np.bool | np.signedinteger]
]
type BoolIntMatrix = np.ndarray[
    tuple[int, int], np.dtype[np.integer | np.bool | np.signedinteger]
]

type BoolMatrix = np.ndarray[tuple[int, int], np.dtype[np.bool]]
type BoolVector = np.ndarray[tuple[int], np.dtype[np.bool]]
