"""
Abstract away the details of
storing tableau's as np arrays
for safety of manipulations
and flexibility to modification
"""

from __future__ import annotations

from typing import List, Optional, Self, Tuple, Union, cast

import numpy as np

from src.tableau_helper_functions import beta, symplectic_inner_product
from src.useful_types import BoolIntMatrix, BoolIntVector, BoolMatrix, BoolVector


class CNCSet:
    """
    Everything in this CNC set is expressible as a binary tree
    where at each internal vertex of the tree the element of Omega
    is the sum of the elements associated to the two children
    which commute
    and the leaves are decorated with the vector
    in the relevant row of _set_elements
    They are all in E_n = BoolVector of length 2*self._n
    self._preomega_size is the number of generating decorations of leaves
    self._is_full_set is if we don't need any trees and everything in Omega
    can be expressed as a single leaf
    """

    __slots__ = [
        "_set_elements",
        "_n",
        "_preomega_size",
        "_value_assignment",
        "_is_maximal",
        "_is_full_set",
        "_is_isotropic",
    ]

    def __init__(
        self,
        set_elements: Union[List[BoolIntVector], BoolIntMatrix],
        value_assignment: Optional[List[bool]] = None,
        check_repeated_rows: bool = False,
    ):
        self._is_maximal: Optional[bool] = None
        self._is_full_set: Optional[bool] = None
        self._is_isotropic: Optional[bool] = None
        if not isinstance(set_elements, List):
            (self._preomega_size, two_n) = set_elements.shape
            assert (
                two_n % 2 == 0
            ), "The elements provided were not in E_n because they were not even length"
            self._n = two_n // 2
            self._set_elements = (set_elements % 2).astype(np.bool)
            if value_assignment is not None:
                assert (
                    len(value_assignment) == self._preomega_size
                ), "If you are providing a value assignment, you must provide it on all elements"
            self._value_assignment = value_assignment
            self.__obvious_is_flags()
            if check_repeated_rows:
                assert (
                    self.__validate_is_set()
                ), "There were repeated entries in what was supposed to be a set"
            return
        self._preomega_size = 0
        expected_2n = None
        self._set_elements = cast(BoolMatrix, np.zeros((0, 0), dtype=np.bool))
        for cur_element in set_elements:
            self._preomega_size += 1
            cur_element = cast(BoolIntVector, cur_element)
            should_be_2n = cast(Tuple[int], cur_element.shape)
            assert len(should_be_2n) == 1, (
                "There was a provided element of the set,"
                "which was not in E_n because it was not a boolean vector"
            )
            if expected_2n is None:
                expected_2n = should_be_2n[0]
                assert expected_2n % 2 == 0, (
                    "There was a provided element of the set,"
                    "which was not in E_n because it was not even length"
                )
                self._n = expected_2n // 2
                self._set_elements = cast(
                    BoolMatrix, np.zeros((0, expected_2n), dtype=np.bool)
                )
                cur_element = (cur_element % 2).astype(np.bool)
                self._set_elements = np.vstack((self._set_elements, cur_element))
                self._set_elements = cast(BoolMatrix, self._set_elements)
            else:
                should_be_2n = should_be_2n[0]
                assert should_be_2n == expected_2n, (
                    "There was a provided element of the set,"
                    f"which was not in E_n because it was not the correct size {expected_2n}"
                )
                cur_element = (cur_element % 2).astype(np.bool)
                self._set_elements = np.vstack((self._set_elements, cur_element))
                self._set_elements = cast(BoolMatrix, self._set_elements)
        if value_assignment is not None:
            assert (
                len(value_assignment) == self._preomega_size
            ), "If you are providing a value assignment, you must provide it on all elements"
            self._value_assignment = value_assignment
        self.__obvious_is_flags()
        if check_repeated_rows:
            assert (
                self.__validate_is_set()
            ), "There were repeated entries in what was supposed to be a set"

    def __validate_is_set(self) -> bool:
        """
        Check if there are no repeats
        """
        for idx in range(self._preomega_size):
            ith_element = cast(BoolVector, self._set_elements[idx])
            for jdx in range(idx + 1, self._preomega_size):
                jth_element = cast(BoolVector, self._set_elements[jdx])
                if (ith_element == jth_element).all():
                    return False
        return True

    @property
    def is_empty(self) -> bool:
        """
        The empty set Omega is vacuously closed.
        Here we just query whether or not
        we are dealing with this vacuous case.
        """
        return self._preomega_size == 0

    #pylint:disable=protected-access
    @staticmethod
    def empty_set() -> CNCSet:
        """
        The empty set Omega is vacuously closed.
        Decorating it with a value assignment is
        also vacuous.
        """
        to_return = CNCSet([], [])
        to_return._declare_maximality(False)
        to_return._declare_full_set(True)
        to_return._is_isotropic = True
        return to_return

    #pylint:disable=protected-access
    @staticmethod
    def zero_set(n_val: int) -> CNCSet:
        """
        The singleton set Omega which
        contains only the 0 vector
        is closed.
        Decorating it with a value assignment is
        trivial.
        """
        zero_vector = cast(BoolVector, np.zeros((1, 2 * n_val), dtype=np.bool))
        to_return = CNCSet([zero_vector], [False])
        to_return._declare_maximality(False)
        to_return._declare_full_set(True)
        to_return._is_isotropic = True
        return to_return

    def __obvious_is_flags(self):
        """
        Set is_full_set, is_maximal, is_isotropic
        if we can do so without any computation
        """
        if self.is_empty:
            self._declare_full_set(True)
            self._declare_maximality(False)
            self._is_isotropic = True
        elif self._preomega_size == 1:
            if (self._set_elements == np.False_).all():
                self._declare_maximality(False)
                self._declare_full_set(True)
            elif self._n > 1:
                # It is a nonzero vector so Omega is {0,v}
                # and we can easily add others
                # keeping it isotropic subspace
                # a fortiori CNC
                self._declare_maximality(False)
                self._declare_full_set(False)
            else:
                # It is a nonzero vector so Omega is {0,v}
                self._declare_full_set(False)
            self._is_isotropic = True

    def _declare_full_set(self, is_full_set: bool):
        """
        Declare that the stored vectors are or are not all of Omega
        """
        if self._is_full_set is None:
            self._is_full_set = is_full_set
        elif self._is_full_set != is_full_set:
            raise ValueError(
                "We already declared that we know whether we have listed all of Omega. "
                "This would be changing it to be inconsistent with that."
            )

    def _declare_maximality(self, is_maximal: bool):
        """
        Declare the maximality to be known
        Either say yes it is definitely maximal
        or that no it is definitely not maximal
        """
        if self._is_maximal is None:
            self._is_maximal = is_maximal
        elif self._is_maximal != is_maximal:
            raise ValueError(
                "We already declared that we know it's maximality property. "
                "This would be changing it to be inconsistent with that."
            )

    def associated_vector(
        self, which_in_preomega: Union[int, Tuple[int, int]]
    ) -> BoolVector:
        """
        For a single element stored in the table
        or the sum of two commuting elements in the table
        give the associated element of E_n
        """
        if isinstance(which_in_preomega, int):
            assert (
                0 <= which_in_preomega < self._preomega_size
            ), "The desired element does not exist as in index of what is stored about Omega"
            return self._set_elements[which_in_preomega]
        (i, j) = which_in_preomega
        assert (
            0 <= i < self._preomega_size
        ), "The desired element does not exist as in index of what is stored about Omega"
        assert (
            0 <= j < self._preomega_size
        ), "The desired element does not exist as in index of what is stored about Omega"
        if symplectic_inner_product(self._set_elements[i], self._set_elements[j]) != 0:
            raise ValueError(
                "You are asking for the value assignment of two indices which are stored in Omega,"
                "but they do not commute so the sum does not have to be in Omega"
            )
        return self.associated_vector(i) + self.associated_vector(j)

    def value_assignment(
        self, which_in_preomega: Union[int, Tuple[int, int]]
    ) -> Optional[bool]:
        """
        For a single element stored in the table
        or the sum of two commuting elements in the table
        if we have a value assignment, then what is the associated
        v(a) or v(a+b) as appropriate
        """
        if self.value_assignment is None:
            return None
        self._value_assignment = cast(List[bool], self._value_assignment)
        if isinstance(which_in_preomega, int):
            assert (
                0 <= which_in_preomega < self._preomega_size
            ), "The desired element does not exist as in index of what is stored about Omega"
            return self._value_assignment[which_in_preomega]
        (i, j) = which_in_preomega
        assert (
            0 <= i < self._preomega_size
        ), "The desired element does not exist as in index of what is stored about Omega"
        assert (
            0 <= j < self._preomega_size
        ), "The desired element does not exist as in index of what is stored about Omega"
        ith_vector = cast(BoolVector, self._set_elements[i])
        jth_vector = cast(BoolVector, self._set_elements[j])
        if symplectic_inner_product(ith_vector, jth_vector) != 0:
            raise ValueError(
                "You are asking for the value assignment of two indices which are stored in Omega,"
                "but they do not commute so the sum does not have to be in Omega"
            )
        cur_sum = (
            self._value_assignment[i]
            + self._value_assignment[j]
            + beta(ith_vector, jth_vector)
        )
        return (cur_sum % 2) == 1

    def value_assignment_b_shift(self, b: int):
        """
        Replace the value assignment gamma for this `CNCSet`
        with gamma + [v_b, - ]
        where v_b is the vector associated with the b'th stored element of the set
        """
        if self._value_assignment is not None:
            self._value_assignment = [
                (
                    gamma_idx
                    + symplectic_inner_product(
                        self._set_elements[b], self._set_elements[idx]
                    )
                )
                % 2
                == 1
                for (idx, gamma_idx) in enumerate(self._value_assignment)
            ]

    def value_assignment_b_complement(self, b: BoolIntVector, _r: bool) -> Self:
        """
        Given a b that is not in Omega
        produce the CNC set with value assignment
        Omega(b) = <b> + <b>^perp bigcap Omega
        and value assignment is self.value_assignment star r
        """
        assert b.shape == (2 * self._n,), f"The provided b was not in E_{self._n}"
        raise NotImplementedError
