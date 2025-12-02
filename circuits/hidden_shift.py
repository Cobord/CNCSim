"""
Build n-qubit hidden-shift circuits with specifications on oracle types
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Iterable
from qiskit import QuantumCircuit
try:
    from .ccz_7T_decomposition import apply_ccz_via_7t_decomposition
except ImportError:
    from ccz_7T_decomposition import apply_ccz_via_7t_decomposition

# Add the parent directory to sys.path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(Path("circuits", "hidden_shift.py")), "..")
    )
)

# suppress qiskit deprecation warnings
warnings.filterwarnings("ignore")


def to_bit_string(bit_list: Iterable[int | str]):
    """
    Convert a list of bits to its integer representation.
    """
    return int("".join(str(b) for b in bit_list), 2)


def clifford_oracle(qc: QuantumCircuit, ni: int, nf: int):
    """
    Default Clifford-phase oracle using CZ (via H–CX–H) and S gates.

    Args:
        qc (QuantumCircuit): The quantum circuit upon which to append the Clifford-phase oracle
        ni (int): The first ni qubits of qc are left as is
        nf (int): The last relevant qubit, this function will modify what happens on qubits [ni,nf]
    """
    # nearest-neighbor CZ + S padding
    for q in range(ni, nf):
        qc.h(q + 1)
        qc.cx(q, q + 1)
        qc.h(q + 1)
        qc.s(q)
        qc.s(q)
    # final Z layer on last qubit
    qc.s(nf)
    qc.s(nf)


def non_clifford_oracle(qc: QuantumCircuit, ni: int, g: int):
    """
    Non-Clifford oracle: plant exactly g CCZ gates on triples starting at ni.

    Args:
        qc (QuantumCircuit): The quantum circuit upon which to append the Non-Clifford oracle
        ni (int): The first ni qubits of qc are left as is
        g (int): How many CCZ gates to apply
    """
    for k in range(g):
        q = ni + k
        qc.ccz(q, q + 1, q + 2)


def hidden_shift(
    n: int, g: int, shift: list[int] | list[bool], is_clifford: bool = False
) -> QuantumCircuit:
    """
    Build the n-qubit hidden-shift circuit with g CCZs per non-Clifford oracle.
    Requires n even and 3*g <= n.

    Args:
        n (int): How many qubits for the circuit
        g (int): How many CCZs per non-Clifford oracle
        shift (List[int] | List[bool]): A list of n which on truthy values applies the hidden
            shift upon those qubits
        is_clifford (bool): Should the oracles appended be Clifford-phase using `clifford_oracle`
            or Non-Clifford using `non_clifford_oracle`
    """
    if n % 2 != 0:
        raise ValueError("n must be even to split into two halves")
    if 3 * g > n // 2:
        raise ValueError(f"Need 3·g ≤ n; got 3·{g} > {n}")

    m = n // 2
    qc = QuantumCircuit(n, n)

    # 1) initial H on all qubits
    qc.h(range(n))

    # 2) cross-CZ linking halves
    for q in range(m):
        qc.h(q + m)
        qc.cx(q, q + m)
        qc.h(q + m)

    # 3) first oracle on [0…m-1]
    if is_clifford:
        clifford_oracle(qc, 0, m - 1)
    else:
        non_clifford_oracle(qc, 0, g)

    # 4) mid Hadamards on all qubits
    qc.h(range(n))

    # 5) repeat cross-CZ linking halves
    for q in range(m):
        qc.h(q + m)
        qc.cx(q, q + m)
        qc.h(q + m)

    # 6) second oracle on [m…n-1]
    if is_clifford:
        clifford_oracle(qc, m, n - 1)
    else:
        non_clifford_oracle(qc, m, g)

    # 7) apply the hidden shift as Z (S^2)
    for q in range(n):
        if shift[q]:
            qc.s(q)
            qc.s(q)

    # 8) final H and measurement
    qc.h(range(n))
    qc.measure(range(n), range(n))

    qc_w_ccz_decomposition = apply_ccz_via_7t_decomposition(qc)

    return qc_w_ccz_decomposition
