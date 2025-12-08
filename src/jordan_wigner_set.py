from typing import List, Optional, Tuple, Union, cast

import numpy as np

from src.tableau_helper_functions import symplectic_inner_product
from src.useful_types import BoolIntMatrix, BoolIntVector, BoolMatrix, BoolVector


class AntiCommutingSet:
    __slots__ = ["_n", "_set_elements", "_set_size", "_is_maximal"]
    
    def __init__(self,
                 set_elements: Union[List[BoolIntVector],BoolIntMatrix],
                 check_anticommuting: bool = False
                 ):
        self._is_maximal : Optional[bool] = None
        if not isinstance(set_elements, List):
            (self._set_size, two_n) = set_elements.shape
            assert two_n % 2 == 0, \
                "The elements provided were not in E_n because they were not even length"
            self._n = two_n // 2
            self._set_elements = set_elements.astype(np.bool)
            if check_anticommuting:
                self.__validate()
            return
        self._set_size = 0
        expected_2n = None
        self._set_elements = cast(BoolMatrix,np.zeros((0,0),dtype=np.bool))
        for cur_element in set_elements:
            self._set_size += 1
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
        if check_anticommuting:
            self.__validate()

    def __validate(self):
        for idx in range(self._set_size):
            ith_element = cast(BoolVector, self._set_elements[idx])
            for jdx in range(idx+1,self._set_size):
                jth_element = cast(BoolVector, self._set_elements[jdx])
                assert symplectic_inner_product(ith_element, jth_element) == 1, f"{idx} and {jdx} did not anticommute"

    def is_maximal(self) -> bool:
        if self._is_maximal is not None:
            return self._is_maximal
        if self._set_size % 2 == 0:
            self._is_maximal = False
            return False
        raise NotImplementedError

    def basis_to_maximal(self):
        assert self._set_size % 2 == 0, \
            "Turning an anticommuting set into a maximal one by adding the nonzero sum of the others requires an even size"
        raise NotImplementedError

    @property
    def is_jordan_wigner(self) -> bool:
        return self._set_size == 2*self._n + 1
    
    def jordan_wigner_extend(self):
        """
        Extend this non-maximal anticommuting set to a Jordan-Wigner set
        but if it is already Jordan-Wigner do not change it
        if it is maximal but not already Jordan-Wigner, then it cannot be extended
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
            # and it will be maximal
            any_anticommuting = self.anticommutes_with_everything()
            assert any_anticommuting is not None, \
                "There does exist something which anticommutes with everything"
            self._set_elements = np.vstack((self._set_elements, any_anticommuting))
            self._set_elements = cast(BoolMatrix,self._set_elements)
            self.jordan_wigner_extend()
            return
        else:
            raise NotImplementedError

    def anticommutes_with_everything(self) -> Optional[BoolVector]:
        raise NotImplementedError

    def jordan_wigner_coefficients(self, b: BoolIntVector, omitted_vector: Optional[int] = None) -> BoolIntVector:
        assert self.is_jordan_wigner, "We are assuming this is a Jordan-Wigner set"
        if omitted_vector is None:
            omitted_vector = self._set_size - 1
        assert 0 <= omitted_vector < self._set_size
        raise NotImplementedError