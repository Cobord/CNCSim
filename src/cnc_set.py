"""
Abstract away the details of
storing tableau's as np arrays
for safety of manipulations
and flexibility to modification
"""

from typing import List, Optional, Self, Tuple, Union, cast

import numpy as np

from src.tableau_helper_functions import symplectic_inner_product
from src.useful_types import BoolIntMatrix, BoolIntVector, BoolMatrix


class CNCSet:
    __slots__ = ["_set_elements", "_n", "_preomega_size", "_value_assignment", "_is_maximal", "_is_full_set"]
    
    def __init__(self,
                 set_elements: Union[List[BoolIntVector],BoolIntMatrix],
                 value_assignment: Optional[List[bool]] = None
                 ):
        self._is_maximal : Optional[bool] = None
        self._is_full_set: Optional[bool] = None
        if not isinstance(set_elements, List):
            (self._preomega_size, two_n) = set_elements.shape
            assert two_n % 2 == 0, \
                "The elements provided were not in E_n because they were not even length"
            self._n = two_n // 2
            self._set_elements = set_elements.astype(np.bool)
            if value_assignment is not None:
                assert len(value_assignment) == self._preomega_size, \
                    "If you are providing a value assignment, you must provide it on all elements"
            self._value_assignment = value_assignment
            return
        self._preomega_size = 0
        expected_2n = None
        self._set_elements = cast(BoolMatrix,np.zeros((0,0),dtype=np.bool))
        for cur_element in set_elements:
            self._preomega_size += 1
            cur_element = cast(BoolIntVector, cur_element)
            should_be_2n = cast(Tuple[int],cur_element.shape)
            assert len(should_be_2n) == 1, \
                "There was a provided element of the set, which was not in E_n because it was not a boolean vector"
            if expected_2n is None:
                expected_2n = should_be_2n[0]
                assert expected_2n % 2 == 0, \
                    "There was a provided element of the set, which was not in E_n because it was not even length"
                self._n = expected_2n // 2
                self._set_elements = cast(BoolMatrix,np.zeros((0,expected_2n),dtype=np.bool))
                cur_element = cur_element.astype(np.bool)
                self._set_elements = np.vstack((self._set_elements, cur_element))
                self._set_elements = cast(BoolMatrix,self._set_elements)
            else:
                should_be_2n = should_be_2n[0]
                assert should_be_2n == expected_2n,\
                    f"There was a provided element of the set, which was not in E_n because it was not the correct size {expected_2n}"
                cur_element = cur_element.astype(np.bool)
                self._set_elements = np.vstack((self._set_elements, cur_element))
                self._set_elements = cast(BoolMatrix,self._set_elements)
        if value_assignment is not None:
            assert len(value_assignment) == self._preomega_size, \
                "If you are providing a value assignment, you must provide it on all elements"
            self._value_assignment = value_assignment

    def declare_full_set(self, is_full_set: bool):
        """
        Declare that the stored vectors are or are not all of Omega
        """
        if self._is_full_set is None:
            self._is_full_set = is_full_set
        elif self._is_full_set != is_full_set:
            raise ValueError("We already declared that we know whether we have listed all of Omega. "
                             "This would be changing it to be inconsistent with that.")

    def declare_maximality(self, is_maximal: bool):
        """
        Declare the maximality to be known
        Either say yes it is definitely maximal
        or that no it is definitely not maximal
        """
        if self._is_maximal is None:
            self._is_maximal = is_maximal
        elif self._is_maximal != is_maximal:
            raise ValueError("We already declared that we know it's maximality property. "
                             "This would be changing it to be inconsistent with that.")

    def value_assignment(self, which_in_preomega: Union[int,Tuple[int,int]]) -> Optional[bool]:
        if self.value_assignment is None:
            return None
        self._value_assignment = cast(List[bool], self._value_assignment)
        if isinstance(which_in_preomega, int):
            assert 0 <= which_in_preomega < self._preomega_size, \
                "The desired element does not exist as in index of what is stored about Omega"
            return self._value_assignment[which_in_preomega]
        (i,j) = which_in_preomega
        assert 0 <= i < self._preomega_size, \
                "The desired element does not exist as in index of what is stored about Omega"
        assert 0 <= j < self._preomega_size, \
                "The desired element does not exist as in index of what is stored about Omega"
        if symplectic_inner_product(self._set_elements[i], self._set_elements[j]) != 0:
            raise ValueError("You are asking for the value assignment of two indices which are stored in Omega," \
                "but they do not commute so the sum does not have to be in Omega")
        return (self._value_assignment[i] + self._value_assignment[j]) % 2 == 1

    def valuation_b_shift(self, b: int):
        """
        Replace the valuation gamma for this `CNCSet`
        with gamma + [v_b, - ]
        where v_b is the vector associated with the b'th stored element of the set
        """
        if self._value_assignment is not None:
            self._value_assignment = [
                gamma_idx + symplectic_inner_product(self._set_elements[b], self._set_elements[idx]) % 2 == 1
                for (idx,gamma_idx) in enumerate(self._value_assignment)
            ]

    def valuation_b_complement(self, b: BoolIntVector) -> Self:
        """
        Given a b that is not in Omega
        produce the CNC set with valuation
        Omega(b) = <b> + <b>^perp bigcap Omega
        """
        assert b.shape == (2*self._n,), f"The provided b was not in E_{self._n}"
        raise NotImplementedError