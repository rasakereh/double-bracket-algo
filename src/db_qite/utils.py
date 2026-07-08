from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
import numpy as np

def create_evolution_gate(s, H, use_pauli=True):
    if use_pauli:
        reorder_synthesis = LieTrotter(reps=1, preserve_order=True)
        op = H.copy()
        commuting_groups = op.group_commuting(qubit_wise=True)
        H = sum(commuting_groups)
        return PauliEvolutionGate(H, time=-s**0.5, label='$e^{i\\sqrt{s}H}$', synthesis=reorder_synthesis)
    else:
        return HamiltonianGate(H, time=-s**0.5, label='$e^{i\\sqrt{s}H}$')

def create_zero_projection_gate(s, num_qubits, use_mcp=True, add_noise=False):
    if use_mcp:
        projection_circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            projection_circuit.x(i)
        projection_circuit.mcp(s**0.5, [0], range(1, num_qubits))  # Control qubits first, then target
        for i in range(num_qubits):
            projection_circuit.x(i)
        projection_circuit.name = f"projection_circuit"
        projection_gate = projection_circuit.to_gate(label='$e^{i\\sqrt{s}|0><0|}$')
        return projection_gate
    else:
        P0 = np.zeros((2**num_qubits, 2**num_qubits))
        if add_noise:
            np.fill_diagonal(P0, np.random.normal(0, 0.01, size=2**num_qubits))
        P0[0, 0] = 1
        projection_gate = HamiltonianGate(P0, time=-s**0.5, label='$e^{i\\sqrt{s}|0><0|}$')
        return projection_gate

def create_monotonic_diagonal(s, num_qubits, ascending=True):
    phases = [np.pi/2*s**0.5/(2**i) for i in range(num_qubits)]
    if ascending:
        phases = phases[::-1]
    monotonic_circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        monotonic_circuit.p(phases[i], i)
    monotonic_gate = monotonic_circuit.to_gate(label='MonotonicDiagonal')

    return monotonic_gate

def to_sparse_pauli(H, convert):
    if isinstance(H, (SparsePauliOp, Pauli)) or not convert:
        return H
    return SparsePauliOp.from_operator(Operator(H))
