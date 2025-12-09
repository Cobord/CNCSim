"""
Tests of the anticommuting sets
"""
#pylint:disable=invalid-name, protected-access

import numpy as np
from src.jordan_wigner_set import AntiCommutingSet


def test_empty():
    """
    The empty set is an anticommuting set vacuously
    """
    N = 4
    empty_set = AntiCommutingSet(np.zeros((0, 2 * N), dtype=np.bool))
    assert not empty_set._is_maximal
    assert not empty_set.is_maximal()
    assert empty_set._set_size == 0
    assert empty_set._n == N
    # if we try to Jordan-Wigner extend, we should get
    # an error only because it isn't implemented
    try:
        empty_set.jordan_wigner_extend()
        assert False
    except NotImplementedError:
        pass


def test_zero():
    """
    The set consisting of only the 0 vector
    """
    N = 4
    zero_set = AntiCommutingSet(np.zeros((1, 2 * N), dtype=np.bool))
    assert zero_set._is_maximal is None
    assert zero_set.is_maximal()
    assert zero_set._set_size == 1
    assert zero_set._n == N
    # if we try to Jordan-Wigner extend, we should get
    # an error because there is no way to extend as an anticommuting
    # set into a Jordan-Wigner set
    try:
        zero_set.jordan_wigner_extend()
        assert False
    except ValueError as e:
        assert str(e) == "It is maximal, but not already a Jordan-Wigner set."


def test_singleton():
    """
    The set consisting of only the 0 vector
    """
    N = 4
    element = np.random.randint(low=0, high=2, size=(1, 2 * N), dtype=np.bool)
    if element.any():
        nonzero_set = AntiCommutingSet(element)
        assert nonzero_set._is_maximal is None
        assert not nonzero_set.is_maximal()
        assert nonzero_set._set_size == 1
        assert nonzero_set._n == N
        # if we try to Jordan-Wigner extend, we should get
        # an error only because it isn't implemented
        try:
            nonzero_set.jordan_wigner_extend()
            assert False
        except NotImplementedError:
            pass
