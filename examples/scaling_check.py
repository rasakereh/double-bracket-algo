# you need `pip install  qiskit-nature pyscf` for this
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector, state_fidelity

import matplotlib.pyplot as plt
import numpy as np
import gc
import pathlib

from db_qite import DB_Sorter
from db_qite.utils import create_zero_projection_gate, create_select_k, create_monotonic_diagonal
from db_qite.warm_starts import run_vqe_and_get_circuit



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


def evaluate_ground_state_fidelity(h_gen, qc_gen, qubit_counts=[2, 4, 6, 8, 10], output_dir="examples/scaling"):
    """Calculates fidelity errors and plots them against the number of qubits.

    Parameters:
    - h_gen: Function with signature h_gen(n_qubit) -> SparsePauliOp / Operator
    - qc_gen: Function with signature qc_gen(hamiltonian) -> QuantumCircuit
    - qubit_counts: List of qubit counts to evaluate
    """
    fidelity_errors = []
    fidelity_scaled_vals = []
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

        scaled_fidelity = F * (2**n)
        fidelity_scaled_vals.append(scaled_fidelity)

        E_prepared = prepared_state.expectation_value(hamiltonian).real
        energy_error = np.abs((E_prepared - E_0) / E_0)
        energy_errors.append(energy_error)
    
    # 5. Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(14, 5))

    # Left: fidelity error
    axs[0, 0].plot(
        qubit_counts,
        fidelity_errors,
        marker="o",
        linestyle="-",
        color="crimson",
        linewidth=2,
        label=r"Fidelity Error ($1 - F$)",
    )
    axs[0, 0].set_xlabel(r"Number of Qubits ($n$)", fontsize=12)
    axs[0, 0].set_ylabel("Fidelity Error", fontsize=12)
    axs[0, 0].set_title("Fidelity Error", fontsize=14)
    axs[0, 0].set_xticks(qubit_counts)
    axs[0, 0].grid(True, which="both", linestyle="--", alpha=0.6)
    axs[0, 0].legend(fontsize=11)
    axs[0, 0].set_yscale("log")

    # Right: fidelity scaled
    axs[0, 1].plot(
        qubit_counts,
        fidelity_scaled_vals,
        marker="o",
        linestyle="-",
        color="crimson",
        linewidth=2,
        label=r"Scaled Fidelity ($2^nF$)",
    )
    axs[0, 1].set_xlabel(r"Number of Qubits ($n$)", fontsize=12)
    axs[0, 1].set_ylabel("Fidelity Scaled", fontsize=12)
    axs[0, 1].set_title("Fidelity Scaled", fontsize=14)
    axs[0, 1].set_xticks(qubit_counts)
    axs[0, 1].grid(True, which="both", linestyle="--", alpha=0.6)
    axs[0, 1].legend(fontsize=11)
    axs[0, 1].set_yscale("log")

    # Right: relative energy error
    axs[1, 0].plot(
        qubit_counts,
        energy_errors,
        marker="s",
        linestyle="--",
        color="royalblue",
        linewidth=2,
        label=r"Relative Energy Error ($|\frac{E_{\mathrm{prepared}} - E_0}{E_0}|$)",
    )
    axs[1, 0].set_xlabel(r"Number of Qubits ($n$)", fontsize=12)
    axs[1, 0].set_ylabel("Relative Energy Error", fontsize=12)
    axs[1, 0].set_title("Relative Energy Error", fontsize=14)
    axs[1, 0].set_xticks(qubit_counts)
    axs[1, 0].grid(True, which="both", linestyle="--", alpha=0.6)
    axs[1, 0].legend(fontsize=11)
    # axs[1, 0].set_yscale("log")

    fig.suptitle("Ground State Preparation Errors", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fidelity_error_plot.png", dpi=300)
    plt.close(fig)

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
    np.random.seed(42)
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
    # warm_start_circuit = run_vqe_and_get_circuit(hamiltonian, backend_name="simulator")
    warm_start_circuit = None

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
    

