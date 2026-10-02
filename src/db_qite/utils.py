from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
import numpy as np

def create_evolution_gate(s, H, use_pauli=True):
    if H is None:
        return None
    if use_pauli:
        reorder_synthesis = LieTrotter(reps=1, preserve_order=True)
        op = H.copy()
        commuting_groups = op.group_commuting(qubit_wise=True)
        H = sum(commuting_groups)
        return PauliEvolutionGate(H, time=-s**0.5, label='$e^{i\\sqrt{s}H}$', synthesis=reorder_synthesis)
    else:
        return HamiltonianGate(H, time=-s**0.5, label='$e^{i\\sqrt{s}H}$')

def create_zero_projection_gate(s, num_qubits, use_mcp=True, ascending=False, hadamard_basis=False, custom_basis=None):
    if use_mcp:
        if hadamard_basis:
            custom_basis = QuantumCircuit(num_qubits)
            custom_basis.h(range(num_qubits))
        projection_circuit = QuantumCircuit(num_qubits)
        if custom_basis:
            projection_circuit.compose(custom_basis.inverse(), qubits=range(num_qubits), inplace=True)
        for i in range(num_qubits):
            projection_circuit.x(i)
        time_step = s**0.5 if not ascending else -s**0.5
        projection_circuit.mcp(time_step, [0], range(1, num_qubits))  # Control qubits first, then target
        for i in range(num_qubits):
            projection_circuit.x(i)
        if custom_basis:
            projection_circuit.compose(custom_basis, qubits=range(num_qubits), inplace=True)
        projection_circuit.name = f"projection_circuit"
        projection_gate = projection_circuit.to_gate(label='$e^{i\\sqrt{s}|0><0|}$')
        return projection_gate
    else:
        P0 = np.zeros((2**num_qubits, 2**num_qubits))
        P0[0, 0] = 1 if not ascending else -1
        if hadamard_basis:
            P0[:, :] = (1 / np.sqrt(2**num_qubits)) * P0[0, 0]
        projection_gate = HamiltonianGate(P0, time=-s**0.5, label='$e^{i\\sqrt{s}|0><0|}$')
        return projection_gate

def create_monotonic_diagonal(s, num_qubits, ascending=True, hadamard_basis=False, custom_basis=None):
    if hadamard_basis:
        custom_basis = QuantumCircuit(num_qubits)
        custom_basis.h(range(num_qubits))
            
    sign = 1 if ascending else -1
    phases = [sign * np.pi/2 * s**0.5/(2**i) for i in range(num_qubits)]
    monotonic_circuit = QuantumCircuit(num_qubits)
    if custom_basis:
        monotonic_circuit.compose(custom_basis.inverse(), qubits=range(num_qubits), inplace=True)
    for i in range(num_qubits):
        monotonic_circuit.p(phases[i], i)
    if custom_basis:
        monotonic_circuit.compose(custom_basis, qubits=range(num_qubits), inplace=True)
    monotonic_gate = monotonic_circuit.to_gate(label='MonotonicDiagonal')

    return monotonic_gate

def create_select_k(s, num_qubits, k=0, ascending=True, hadamard_basis=False, custom_basis=None):
    assert k < 2**num_qubits, "k should be smaller than 2^num_qubits"
    
    if hadamard_basis:
        custom_basis = QuantumCircuit(num_qubits)
        custom_basis.h(range(num_qubits))

    sign = 1 if ascending else -1
    binary_k = list(reversed(bin(k)[2:]))
    if len(binary_k) < num_qubits:
        binary_k.extend(['0' for _ in range(len(binary_k), num_qubits)])
    k_sign = [sign if k_i == '0' else -sign for k_i in binary_k]
    phases = [k_i * np.pi * s**0.5/(num_qubits+1) for k_i in k_sign]
    select_k_circuit = QuantumCircuit(num_qubits)
    if custom_basis:
        select_k_circuit.compose(custom_basis.inverse(), qubits=range(num_qubits), inplace=True)
    for i in range(num_qubits):
        select_k_circuit.p(phases[i], i)
    if custom_basis:
        select_k_circuit.compose(custom_basis, qubits=range(num_qubits), inplace=True)
    select_k_gate = select_k_circuit.to_gate(label='SelectKDiagonal')

    return select_k_gate


def to_sparse_pauli(H, convert):
    if isinstance(H, (SparsePauliOp, Pauli)) or not convert:
        return H
    return SparsePauliOp.from_operator(Operator(H))
