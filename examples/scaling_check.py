# you need `pip install  qiskit-nature pyscf` for this
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector, state_fidelity
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

import matplotlib.pyplot as plt
import numpy as np
import gc
import pathlib

from db_qite import DB_Sorter
from db_qite.utils import create_zero_projection_gate, create_select_k, create_monotonic_diagonal




def create_TFIM_hamiltonian(num_qubits, J=.5):
    return SparsePauliOp.from_sparse_list(
        [('ZZ', [i, i+1], J) for i in range(num_qubits-1)] +
        [('X', [i], J) for i in range(num_qubits)],
        num_qubits=num_qubits
    )

def create_XXZ_hamiltonian(num_qubits, delta=.5):
    return SparsePauliOp.from_sparse_list(
        [('XX', [i, i+1], .5) for i in range(num_qubits-1)] +
        [('YY', [i, i+1], .5) for i in range(num_qubits-1)] +
        [('ZZ', [i, i+1], .5*delta) for i in range(num_qubits-1)],
        num_qubits=num_qubits
    )

def create_J1_J2_hamiltonian(num_qubits, J2_J1=.5):
    return SparsePauliOp.from_sparse_list(
        [('XX', [i, i+1], .5) for i in range(num_qubits-1)] +
        [('YY', [i, i+1], .5) for i in range(num_qubits-1)] +
        [('ZZ', [i, i+1], .5) for i in range(num_qubits-1)] +
        [('XX', [i, i+2], .5*J2_J1) for i in range(num_qubits-2)] +
        [('YY', [i, i+2], .5*J2_J1) for i in range(num_qubits-2)] +
        [('ZZ', [i, i+2], .5*J2_J1) for i in range(num_qubits-2)],
        num_qubits=num_qubits
    )



def vqe_callback(eval_count, parameters, mean, metadata):
    print(f"Iteration {eval_count:3d} | Energy: {mean:.6f}\r", end="")

def run_vqe_and_get_circuit(hamiltonian: SparsePauliOp, backend_name: str) -> QuantumCircuit:
    """
    Runs VQE optimization on a Hamiltonian and returns the ansatz circuit
    bound with the optimal parameters.

    Args:
        hamiltonian (SparsePauliOp): The target Hamiltonian.
        backend_name (str): "simulator" for local Aer, or an IBM Quantum backend name.

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
        reps=1,
        insert_barriers=True,
    )
    ansatz = transpile(raw_ansatz, backend=backend)

    # 3. Instantiate Estimator primitive with backend
    estimator = Estimator(mode=backend)

    # 4. Set up and run VQE
    num_params = ansatz.num_parameters
    bounds = [(-np.pi, np.pi) for _ in range(num_params)]
    initial_point = np.random.uniform(-np.pi, np.pi, size=num_params)
    optimizer = SLSQP(maxiter=20, bounds=bounds)

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point, callback=vqe_callback)
    
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

    # 5. Bind optimal parameters to ansatz circuit
    optimal_circuit = ansatz.assign_parameters(result.optimal_parameters)

    return optimal_circuit




def evaluate_ground_state_fidelity(h_gen, qc_gen, qubit_counts=[2, 4, 6, 8, 10], output_dir="examples/scaling"):
    """Calculates fidelity errors and plots them against the number of qubits.

    Parameters:
    - h_gen: Function with signature h_gen(n_qubit) -> SparsePauliOp / Operator
    - qc_gen: Function with signature qc_gen(hamiltonian) -> QuantumCircuit
    - qubit_counts: List of qubit counts to evaluate
    """
    fidelity_errors = []
    energy_errors = []

    for n in qubit_counts:
        # 1. Generate the Hamiltonian for n qubits
        hamiltonian = h_gen(n)

        # 2. Compute the exact ground state (eigenvector corresponding to min eigenvalue)
        # Convert Hamiltonian to Operator matrix if necessary
        hamiltonian_matrix = Operator(hamiltonian).data
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
        E_0 = eigenvalues[0]
        ground_state_vec = eigenvectors[:, 0]
        ground_state = Statevector(ground_state_vec)

        # 3. Generate the ansatz circuit and apply it to initial state |0...0>
        qc = qc_gen(hamiltonian)
        initial_state = Statevector.from_int(0, dims=2**n)
        prepared_state = initial_state.evolve(qc)

        # 4. Calculate fidelity and fidelity error: 1 - F(prepared_state, ground_state)
        F = state_fidelity(prepared_state, ground_state)
        error = 1.0 - F
        fidelity_errors.append(error)

        E_prepared = prepared_state.expectation_value(hamiltonian).real
        energy_error = 1.0 - (E_0 / E_prepared)
        energy_errors.append(energy_error)
    
    print(energy_errors)

    # 5. Plot the results
    plt.figure(figsize=(9, 5))
    # plt.plot(
    #     qubit_counts,
    #     fidelity_errors,
    #     marker="o",
    #     linestyle="-",
    #     color="crimson",
    #     linewidth=2,
    #     label=r"Fidelity Error (\(1 - F\))",
    # )
    plt.plot(
        qubit_counts,
        energy_errors,
        marker="s",
        linestyle="--",
        color="royalblue",
        linewidth=2,
        label=r"Relative Energy Error (\(1 - E_0 / E_{\text{prepared}}\))",
    )

    # plt.yscale("log")
    plt.xlabel("Number of Qubits (\(n\))", fontsize=12)
    plt.ylabel("Error Scale (Log Scale)", fontsize=12)
    plt.title("Ground State Preparation Errors", fontsize=14)
    plt.xticks(qubit_counts)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{str(output_dir)}/fidelity_error_plot.png", dpi=300)
    plt.close()

    return fidelity_errors



def db_sorter_circuit_generator(hamiltonian, s=0.5, num_steps=3, trotterization=True, d_oracle=None):
    """Generates a DB_Sorter circuit for a given Hamiltonian.

    Parameters:
    - hamiltonian: SparsePauliOp / Operator representing the Hamiltonian
    - s: Time step for evolution
    - num_steps: Number of steps in the DB_Sorter algorithm
    - trotterization: Whether to use Trotterization for evolution

    Returns:
    - QuantumCircuit implementing the DB_Sorter algorithm
    """

    num_qubits = hamiltonian.num_qubits

    custom_basis = QuantumCircuit(num_qubits)
    # np.random.seed(42)
    for i in range(num_qubits):
        custom_basis.h(i)
        custom_basis.p(
            np.random.choice([1/3, 1/4, 1/5, 1/6]) * np.pi * np.random.choice([-1, 1]),
            i
        )
        if i>1 and np.random.choice([True, False], p=[.25, .75]):
            custom_basis.cx(i-1, i)
    
    zero_projection_oracle = create_zero_projection_gate(s=s, num_qubits=num_qubits, use_mcp=True, ascending=True, custom_basis=custom_basis)
    monotonic_oracle = create_monotonic_diagonal(s=s, num_qubits=num_qubits, ascending=True, custom_basis=custom_basis)
    select_k_oracle = create_select_k(s=s, num_qubits=num_qubits, ascending=True, custom_basis=custom_basis)
    
    d_oracle = select_k_oracle if d_oracle is None else d_oracle
    warm_start_circuit = run_vqe_and_get_circuit(hamiltonian, backend_name="simulator")

    db_sorter = DB_Sorter(
        hamiltonian,
        s,
        trotterization=trotterization,
        measure=False,
        custom_basis=custom_basis,
        warm_start=warm_start_circuit,
        diagonal_oracle=d_oracle
    )
    return db_sorter.create_circuit(num_steps)


##########################################
hamiltonian_families = {
    "H_TFIM": create_TFIM_hamiltonian,
    "H_XXZ": create_XXZ_hamiltonian,
    "H_J1_J2": create_J1_J2_hamiltonian,
}

s = .5
backend = "simulator"

for h_family, h_generator in hamiltonian_families.items():
    print(f"processing {h_family}")
    output_dir = f"examples/scaling/{h_family}"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluate_ground_state_fidelity(h_generator, db_sorter_circuit_generator, qubit_counts=[4, 5, 6, 7, 8, 9], output_dir=output_dir)
    

