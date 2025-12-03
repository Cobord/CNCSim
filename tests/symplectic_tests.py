"""
Tests of the tableau helper functions
"""

from typing import List
from hypothesis import given, strategies as st
import numpy as np

from src.tableau_helper_functions import (
    find_commuting_elements,
    find_independent_subset,
    symplectic_inner_product,
)
from src.useful_types import BoolVector


@given(st.lists(st.booleans()).filter(lambda lst: len(lst) % 2 == 0))
def test_self_pairings(x_raw: List[bool]):
    """
    The symplectic inner product of any vector with itself is 0
    """
    x = np.array(x_raw)
    assert symplectic_inner_product(x, x) == 0


@st.composite
def all_in_x_part(draw: st.DrawFn) -> List[BoolVector]:
    """
    Produce many vectors all in the X only subspace possibly with repeats
    """
    n = draw(st.integers(min_value=0, max_value=4))
    num_vecs = draw(st.integers(min_value=0, max_value=2 * n))
    to_return = []
    for _ in range(num_vecs):
        raw_bool_vec = draw(st.lists(st.booleans(), min_size=n, max_size=n))
        to_return.append(np.block([np.array(raw_bool_vec), np.zeros(n, dtype=np.bool)]))
    return to_return


@st.composite
def all_in_z_part(draw: st.DrawFn) -> List[BoolVector]:
    """
    Produce many vectors all in the Z only subspace possibly with repeats
    """
    n = draw(st.integers(min_value=0, max_value=4))
    num_vecs = draw(st.integers(min_value=0, max_value=2 * n))
    to_return = []
    for _ in range(num_vecs):
        raw_bool_vec = draw(st.lists(st.booleans(), min_size=n, max_size=n))
        to_return.append(np.block([np.zeros(n, dtype=np.bool), np.array(raw_bool_vec)]))
    return to_return


@given(all_in_x_part(), all_in_z_part())
def test_in_isotropic(many_all_xs: List[BoolVector], many_all_zs: List[BoolVector]):
    """
    When manifestly generating vectors which are all in the first or last GF(2)^n of GF(2)^{2n}
    it is isotropic
    """
    many_all_xs_set = set((tuple(an_all_xs.tolist()) for an_all_xs in many_all_xs))
    count_unique = len(many_all_xs_set)
    many_all_xs_2 = [x.astype(np.int64) for x in many_all_xs]
    isotropic, non_isotropic = find_commuting_elements(many_all_xs_2)
    assert len(non_isotropic) == 0
    assert len(isotropic) == count_unique
    for idx in range(count_unique):
        for jdx in range(idx, count_unique):
            assert symplectic_inner_product(isotropic[idx], isotropic[jdx]) == 0

    many_all_zs_set = set((tuple(an_all_zs.tolist()) for an_all_zs in many_all_zs))
    count_unique = len(many_all_zs_set)
    many_all_zs_2 = [z.astype(np.int64) for z in many_all_zs]
    isotropic, non_isotropic = find_commuting_elements(many_all_zs_2)
    assert len(non_isotropic) == 0
    assert len(isotropic) == count_unique
    for idx in range(count_unique):
        for jdx in range(idx, count_unique):
            assert symplectic_inner_product(isotropic[idx], isotropic[jdx]) == 0


def test_small_symplectic_pairings():
    """
    Small symplectic inner products
    """
    x = np.array([1, 0, 0, 1])
    y = np.array([1, 0, 1, 1])
    assert symplectic_inner_product(x, y) == 1


def test_small_independence():
    """
    when 3 vectors are clearly dependent
    find_independent_subset should
    see that
    """
    x = np.array([1, 0, 0, 1], dtype=np.uint8)
    # the (possibly excessive) flexibility in how dtype is specified
    y = np.array([True, False, True, True], dtype=bool)
    z = (x + y) % 2
    w = np.zeros(shape=4, dtype=np.int32)
    independent_matrix = find_independent_subset([x, y, z, w, w, w])
    assert independent_matrix.shape[0] == 2
    assert independent_matrix.shape[1] == 4
