"""
Toffoli decompositions
"""

# pylint:disable=invalid-name, duplicate-code

# suppress warnings from qiskit 1.0
import warnings
from typing import cast, List

from qiskit import QuantumCircuit, Aer, execute, ClassicalRegister
from qiskit.circuit.quantumcircuit import (
    QubitSpecifier,
    Clbit,
    Qubit,
    Operation,
    CircuitInstruction,
)

# Suppress all warnings
warnings.filterwarnings("ignore")

# Example code that raises warnings
warnings.warn("This is a warning!")


def toff_7t_decomposition(
    qc: QuantumCircuit, c1: QubitSpecifier, c2: QubitSpecifier, target: QubitSpecifier
):
    """
    Append the toff_7t decomposition onto `qc`

    Args:
        qc (QuantumCircuit): The circuit upon which to append.
        c1 (QubitSpecifier): Specifier for the first control qubit of Toffoli.
        c2 (QubitSpecifier): Specifier for the second control qubit of Toffoli.
        target (QubitSpecifier): Specifier for the target qubit of Toffoli

    """
    # layer 1:
    qc.h(target)
    # layer 2
    qc.cnot(c2, target)
    # layer 3: tdg = sdg * t = s*s*s*t
    qc.s(target)
    qc.s(target)
    qc.s(target)
    qc.t(target)
    # layer 4
    qc.cnot(c1, target)
    # layer 5
    qc.t(target)
    # layer 6
    qc.cnot(c2, target)
    # layer 7: tdg
    qc.s(target)
    qc.s(target)
    qc.s(target)
    qc.t(target)
    # layer 8
    qc.cnot(c1, target)
    # layer 9
    qc.s(c2)
    qc.s(c2)
    qc.s(c2)
    qc.t(c2)
    qc.t(target)
    # qc.h(target)
    # layer 10
    qc.cnot(c1, c2)
    # layer 11: tdg
    qc.s(c2)
    qc.s(c2)
    qc.s(c2)
    qc.t(c2)
    # layer 12
    qc.cnot(c1, c2)
    # layer 13
    qc.t(c1)
    qc.s(c2)
    qc.h(target)


def apply_toff_via_7t_decomposition(original_qc: QuantumCircuit) -> QuantumCircuit:
    """
    Replace all ccx in `original_qc: QuantumCircuit`
    with the decomposition of `toff_7t_decomposition`

    Args:
        original_qc (QuantumCircuit): The circuit which has Toff's that we wish to decompose.

    Returns:
        QuantumCircuit: A copy of `original_qc` but with all Toff's decomposed.
    """
    new_qc = QuantumCircuit(original_qc.num_qubits, original_qc.num_clbits)

    qubit_mapping = {q: i for i, q in enumerate(original_qc.qubits)}
    clbit_mapping = {c: i for i, c in enumerate(original_qc.clbits)}

    for gate in original_qc.data:
        gate = cast(CircuitInstruction, gate)
        instruction, qargs, cargs = gate
        instruction = cast(Operation, instruction)
        qargs = cast(List[Qubit], qargs)
        cargs = cast(List[Clbit], cargs)

        if instruction.name == "ccx":
            c1 = qubit_mapping[qargs[0]]
            c2 = qubit_mapping[qargs[1]]
            target = qubit_mapping[qargs[2]]

            # Append custom decomposition
            toff_7t_decomposition(new_qc, c1, c2, target)
        else:
            mapped_qubits = [qubit_mapping[q] for q in qargs]
            mapped_clbits = [clbit_mapping[c] for c in cargs]
            new_qc.append(instruction, mapped_qubits, mapped_clbits)

    return new_qc


def main():
    """
    create a sample circuit and show how toff gets decomposed in it
    The simulated behavior after this transformation should match
    that of the original circuit
    """
    # Simple circuit to test state injection:
    n = 3
    qc = QuantumCircuit(n, n)
    # qc.x(range(n))
    qc.ccx(0, 1, 2)
    # qc.ccz(0+m,1+m,2+m)

    print("Original Circuit:")
    print(qc.draw())

    # inject ccz state:
    qc_with_tgates = apply_toff_via_7t_decomposition(qc)

    # Add measurements only for the original qubits (first 6)
    for qubit in range(n):
        qc_with_tgates.measure(
            qubit, qubit
        )  # Measure each original qubit into its corresponding classical bit
    print("\nState Injected Quantum Circuit with Measurements:")
    print(qc_with_tgates.draw())

    # Simulate the circuit
    simulator = Aer.get_backend("qasm_simulator")
    job = execute(qc_with_tgates, simulator, shots=1024)
    result = job.result()
    raw_counts = result.get_counts()
    print("\nSimulation Results:")
    print(raw_counts)


if __name__ == "__main__":
    main()


# pylint:disable=too-many-arguments, too-many-positional-arguments
def custom_ccz_decomposition(
    qc: QuantumCircuit,
    c1: QubitSpecifier,
    c2: QubitSpecifier,
    anc1: QubitSpecifier,
    anc2: QubitSpecifier,
    target: QubitSpecifier,
    clbit_index: Clbit | ClassicalRegister | int,
):
    """
    Append the custom decomposition onto `qc`
    This uses measurement unlike `toff_7t_decomposition`

    Args:
        qc (QuantumCircuit): The circuit upon which to append.
        c1 (QubitSpecifier): Specifier for the first control qubit.
        c2 (QubitSpecifier): Specifier for the second control qubit.
        anc1 (QubitSpecifier): Specifier for an ancilla
        anc2 (QubitSpecifier): Specifier for an ancilla
        target (QubitSpecifier): Specifier for the target qubit
        clbit_index: (Clbit | ClassicalRegister | int): The measurement will be put
            into this classical register
    """
    qc.h(target)
    qc.h(anc1)
    qc.cx(c1, anc2)
    qc.cx(anc1, c1)
    qc.cx(anc1, c2)
    qc.cx(c2, anc2)
    qc.t(anc1)
    qc.t(anc2)

    qc.tdg(c2)
    qc.tdg(c1)
    qc.cx(c2, anc2)
    qc.cx(anc1, c1)
    qc.cx(anc1, c2)
    qc.cx(c1, anc2)

    qc.h(anc1)
    qc.s(anc1)
    qc.cx(anc1, target)
    qc.h(anc1)

    # Measure ancilla to the specified classical bit
    qc.measure(anc1, clbit_index)

    qc.h(c2)
    qc.cx(c1, c2).c_if(clbit_index, 1)
    qc.h(c2)
    qc.h(target)
