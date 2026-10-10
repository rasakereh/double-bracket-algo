from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal, hamiltonian_variational_ansatz
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from typing import Literal

import numpy as np


def vqe_callback(eval_count, parameters, mean, metadata):
    print(f"Iteration {eval_count:3d} | Energy: {mean:.6f}\r", end="")

def get_ansatz(
    hamiltonian: SparsePauliOp,
    prefix_circuit: QuantumCircuit = None,
    suffix_circuit: QuantumCircuit = None,
    num_reps: int = 2,
    method: Literal["mixed", "hva", "hardware_efficient"] = "hardware_efficient",
    ) -> QuantumCircuit:
    if method == "hva":
        ansatz = hamiltonian_variational_ansatz(
            hamiltonian=hamiltonian,
            reps=num_reps,
            insert_barriers=False,
        )
    elif method == "hardware_efficient":
        ansatz = TwoLocal(
            num_qubits=hamiltonian.num_qubits,
            rotation_blocks=['ry'],
            entanglement_blocks=['cz'],
            entanglement='linear',
            reps=num_reps,
            insert_barriers=False,
        )
    elif method == "mixed":
            ansatz = TwoLocal(
                num_qubits=hamiltonian.num_qubits,
                rotation_blocks=['ry'],
                entanglement_blocks=['cz'],
                entanglement='linear',
                reps=1,
                insert_barriers=False,
                parameter_prefix=f"he_pre"
            )
            curr_ansatz = hamiltonian_variational_ansatz(
                hamiltonian=hamiltonian,
                reps=max(num_reps-2, 1),
                insert_barriers=False,
                # parameter_prefix=f"hve{rep}_"
            ).compose(
                TwoLocal(
                    num_qubits=hamiltonian.num_qubits,
                    rotation_blocks=['ry'],
                    entanglement_blocks=['cz'],
                    entanglement='linear',
                    reps=1,
                    insert_barriers=False,
                    parameter_prefix=f"he_post"
                )
            )
            
            ansatz = ansatz.compose(curr_ansatz)

    if prefix_circuit:
        ansatz = prefix_circuit.compose(ansatz)
    if suffix_circuit:
        ansatz = ansatz.compose(suffix_circuit)

    return ansatz

def run_vqe_and_get_circuit(
    hamiltonian: SparsePauliOp,
    backend_name: str = "simulator",
    prefix_circuit: QuantumCircuit = None,
    suffix_circuit: QuantumCircuit = None,
    maxiter: int = 20,
    ftol: float = 1e-4,
    num_reps: int = 2,
    method: Literal["mixed", "hva", "hardware_efficient"] = "hardware_efficient",
    ) -> QuantumCircuit:
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

    # 2. Define ansatz
    raw_ansatz = get_ansatz(
        hamiltonian=hamiltonian,
        prefix_circuit=prefix_circuit,
        suffix_circuit=suffix_circuit,
        num_reps=num_reps,
        method=method,
    )
    ansatz = transpile(raw_ansatz, backend=backend)

    # 3. Instantiate Estimator primitive with backend
    estimator = Estimator(mode=backend)

    # 4. Set up and run VQE
    num_params = ansatz.num_parameters
    bounds = [(-np.pi, np.pi) for _ in range(num_params)]
    ansatz.parameter_bounds = bounds
    initial_point = np.random.uniform(-np.pi, np.pi, size=num_params)
    optimizer = SLSQP(maxiter=maxiter, ftol=ftol)

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point, callback=vqe_callback)
    
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    print(f"\nOptimal Energy: {result.optimal_value:.6f}")

    # 5. Bind optimal parameters to ansatz circuit
    optimal_circuit = ansatz.assign_parameters(result.optimal_parameters)

    return optimal_circuit


import numpy as np
from qiskit_algorithms import TimeEvolutionProblem, VarQITE
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp

def var_qite_ws(hamiltonian, prefix_circuit: QuantumCircuit = None, suffix_circuit: QuantumCircuit = None):
    problem = TimeEvolutionProblem(hamiltonian, time=1.0)

    # 2. Choose a variational ansatz
    num_qubits = hamiltonian.num_qubits
    ansatz = efficient_su2(num_qubits, reps=3)
    if prefix_circuit:
        ansatz = prefix_circuit.compose(ansatz)
    if suffix_circuit:
        ansatz = ansatz.compose(suffix_circuit)
    init_param_values = np.random.uniform(-np.pi, np.pi, size=len(ansatz.parameters))

    # 3. Configure the variational principle and VarQITE
    var_principle = ImaginaryMcLachlanPrinciple()
    var_qite = VarQITE(
        ansatz=ansatz,
        initial_parameters=init_param_values,
        variational_principle=var_principle,
        num_timesteps=5,
        # Additional parameters like estimator or ode_solver can be passed here
    )

    # 4. Evolve the system
    evolution_result = var_qite.evolve(problem)

    # evolution_result.parameter_values
    return evolution_result.evolved_state


