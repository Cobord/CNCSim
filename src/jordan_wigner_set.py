"""
Subsets of E_n
such that they pairwise anticommute.
"""

# pylint:disable=duplicate-code
from typing import List, Optional, Set, Tuple, Union, cast

import numpy as np

from src.tableau_helper_functions import symplectic_inner_product
from src.useful_types import BoolIntMatrix, BoolIntVector, BoolMatrix, BoolVector


class AntiCommutingSet:
    """
    A subset of E_n
    such that elements pairwise anticommute.
    """

    __slots__ = ["_n", "_set_elements", "_set_size", "_is_maximal"]

    def __init__(
        self,
        set_elements: Union[List[BoolIntVector], BoolIntMatrix],
        check_anticommuting: bool = False,
    ):
        self._is_maximal: Optional[bool] = None
        if not isinstance(set_elements, List):
            (self._set_size, two_n) = set_elements.shape
            assert (
                two_n % 2 == 0
            ), "The elements provided were not in E_n because they were not even length"
            self._n = two_n // 2
            self._set_elements = (set_elements % 2).astype(np.bool)
            if check_anticommuting:
                self.__validate()
            return
        self._set_size = 0
        expected_2n = None
        self._set_elements = cast(BoolMatrix, np.zeros((0, 0), dtype=np.bool))
        for cur_element in set_elements:
            self._set_size += 1
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
        if check_anticommuting:
            self.__validate()

    def remove_idx(self, idx: int):
        """
        Removing a single element from the set
        still retains the property of being anticommuting
        """
        self._set_elements = np.vstack(
            (self._set_elements[0::idx], self._set_elements[idx + 1 :])
        )
        self._is_maximal = False
        self._set_size -= 1

    def __validate(self):
        """
        For every distinct idx, jdx
        the associated elements of E_n
        anticommute
        """
        for idx in range(self._set_size):
            ith_element = cast(BoolVector, self._set_elements[idx])
            for jdx in range(idx + 1, self._set_size):
                jth_element = cast(BoolVector, self._set_elements[jdx])
                assert (
                    symplectic_inner_product(ith_element, jth_element) == 1
                ), f"{idx} and {jdx} did not anticommute"

    def is_maximal(self) -> bool:
        """
        Can we append another element of E_n to this set
        and still be pairwise anticommuting?
        """
        if self._is_maximal is not None:
            return self._is_maximal
        if self._set_size % 2 == 0:
            self._is_maximal = False
            return False
        sum_all = cast(BoolVector, (self._set_elements.sum(axis=0) % 2).astype(np.bool))
        zero_vec = np.zeros((self._n * 2,), dtype=np.bool)
        sum_all_is_zero: np.bool = (sum_all == zero_vec).all()
        if sum_all_is_zero:
            self._is_maximal = True
            return True
        self._is_maximal = False
        return False

    def basis_to_maximal(self):
        """
        If we have an anticommuting set of even size
        we can append another element corresponding
        to the sum of them.
        """
        assert self._set_size % 2 == 0, (
            "Turning an anticommuting set into a maximal one by "
            "adding the nonzero sum of the others requires an even size"
        )
        assert not self.is_maximal()
        self._set_elements = cast(BoolMatrix, self._set_elements)
        sum_all = cast(BoolVector, (self._set_elements.sum(axis=0) % 2).astype(np.bool))
        self._set_elements = np.vstack((self._set_elements, sum_all))
        self._set_elements = cast(BoolMatrix, self._set_elements)
        self._is_maximal = True
        self._set_size += 1

    @property
    def is_jordan_wigner(self) -> bool:
        """
        Is it specifically a Jordan-Wigner
        AntiCommutingSet of the correct size?
        """
        return self._set_size == 2 * self._n + 1

    def jordan_wigner_extend(self):
        """
        Extend this non-maximal AntiCommutingSet to a Jordan-Wigner set
        but if it is already Jordan-Wigner do not change it.
        If it is maximal but not already Jordan-Wigner, then it cannot be extended
        to a Jordan-Wigner set by just adding more elements to the set
        """
        if self.is_jordan_wigner:
            return
        if self.is_maximal():
            raise ValueError("It is maximal, but not already a Jordan-Wigner set.")
        if self._set_size == 2 * self._n:
            self.basis_to_maximal()
            return
        if self._set_size % 2 == 0:
            # just find any a_(N+1) that anticommutes with everything
            any_anticommuting = self.anticommutes_with_everything()
            assert (
                any_anticommuting is not None
            ), "There does exist something which anticommutes with everything"
            self._set_elements = np.vstack((self._set_elements, any_anticommuting))
            self._set_elements = cast(BoolMatrix, self._set_elements)
            # we still have a non-maximal anticommuting set, keep going
            # This recursion can easily be turned into a loop instead
            # and do so if this is causing any problems with RecursionDepth
            self.jordan_wigner_extend()
            return
        c = self.something_not_in_span()
        assert (
            c is not None
        ), f"The set is of size {self._set_size} which is too small to span everything"
        (ac, cc) = self.ac_cc_decomposition(c)
        if len(cc) % 2 == 1:
            new_vec = c + (sum(self._set_elements[idx] for idx in ac) % 2)
            new_vec = cast(BoolVector, (new_vec % 2).astype(np.bool))
        elif len(cc) != 0:
            new_vec = c + (sum(self._set_elements[idx] for idx in cc) % 2)
            new_vec = cast(BoolVector, (new_vec % 2).astype(np.bool))
        else:
            new_vec = c
        self._set_elements = np.vstack((self._set_elements, new_vec))
        self._set_elements = cast(BoolMatrix, self._set_elements)
        self._set_size += 1
        # we still have a non-maximal anticommuting set, keep going
        # This recursion can easily be turned into a loop instead
        # and do so if this is causing any problems with RecursionDepth
        self.jordan_wigner_extend()

    def anticommutes_with_everything(self) -> Optional[BoolVector]:
        """
        Find some element of E_n such that
        it anticommutes with everything in this set.
        """
        raise NotImplementedError

    def something_not_in_span(self) -> Optional[BoolVector]:
        """
        Any element in E_n setminus Span<a_1 .. a_n>
        """
        raise NotImplementedError

    def ac_cc_decomposition(self, c: BoolIntVector) -> Tuple[Set[int], Set[int]]:
        """
        Decompose this set as a disjoint union
        of those that anticommute with c
        and those that commute with c
        """
        ac_set: Set[int] = set()
        cc_set: Set[int] = set()
        for idx in range(self._set_size):
            ith_element = cast(BoolVector, self._set_elements[idx])
            if symplectic_inner_product(ith_element, c) == 1:
                ac_set.add(idx)
            else:
                cc_set.add(idx)
        return (ac_set, cc_set)

    def jordan_wigner_coefficients(
        self, _b: BoolIntVector, omitted_vector: Optional[int] = None
    ) -> BoolIntVector:
        """
        In a Jordan-Wigner set
        omitting omitted_vector
        leaves 2*n anticommuting vectors and we can expand b
        in as a linear combination thereof.
        Return the coefficients of that linear combination.
        """
        assert self.is_jordan_wigner, "We are assuming this is a Jordan-Wigner set"
        if omitted_vector is None:
            omitted_vector = self._set_size - 1
        assert 0 <= omitted_vector < self._set_size
        raise NotImplementedError
