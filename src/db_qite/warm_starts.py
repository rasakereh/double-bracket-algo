from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

import numpy as np


def vqe_callback(eval_count, parameters, mean, metadata):
    print(f"Iteration {eval_count:3d} | Energy: {mean:.6f}\r", end="")

def run_vqe_and_get_circuit(hamiltonian: SparsePauliOp, backend_name: str, prefix_circuit: QuantumCircuit = None, suffix_circuit: QuantumCircuit = None) -> QuantumCircuit:
    """
    Runs VQE optimization on a Hamiltonian and returns the ansatz circuit
    bound with the optimal parameters.

    Args:
        hamiltonian (SparsePauliOp): The target Hamiltonian.
        backend_name (str): "simulator" for local Aer, or an IBM Quantum backend name.
        prefix_circuit (QuantumCircuit, optional): A circuit to prepend to the ansatz.
        suffix_circuit (QuantumCircuit, optional): A circuit to append to the ansatz.

    Returns:
        QuantumCircuit: The VQE ansatz circuit assigned with optimal parameters.
    """
    num_qubits = hamiltonian.num_qubits

    # 1. Select backend
    if backend_name.lower() == "simulator":
        backend = AerSimulator()
    else:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

    # 2. Define hardware-efficient ansatz
    raw_ansatz = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=['ry'],
        entanglement_blocks=['cz'],
        entanglement='linear',
        reps=2,
        insert_barriers=False,
    )
    if prefix_circuit:
        raw_ansatz = prefix_circuit.compose(raw_ansatz)
    if suffix_circuit:
        raw_ansatz = raw_ansatz.compose(suffix_circuit)
    ansatz = transpile(raw_ansatz, backend=backend)

    # 3. Instantiate Estimator primitive with backend
    estimator = Estimator(mode=backend)

    # 4. Set up and run VQE
    num_params = ansatz.num_parameters
    bounds = [(-np.pi, np.pi) for _ in range(num_params)]
    ansatz.parameter_bounds = bounds
    initial_point = np.random.uniform(-np.pi, np.pi, size=num_params)
    optimizer = SLSQP(maxiter=20, ftol=1e-4)

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point, callback=vqe_callback)
    
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    print(f"\nOptimal Energy: {result.optimal_value:.6f}")

    # 5. Bind optimal parameters to ansatz circuit
    optimal_circuit = ansatz.assign_parameters(result.optimal_parameters)

    return optimal_circuit

