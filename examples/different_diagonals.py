# you need `pip install  qiskit-nature pyscf` for this
from qiskit import QuantumCircuit

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
import gc

from db_qite import db_range_runner, DB_Insight
from db_qite.utils import create_zero_projection_gate, create_select_k, create_monotonic_diagonal
from db_qite.warm_starts import run_vqe_and_get_circuit

# Define the hamiltonian (H2 molecule)
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
problem = driver.run()
hamiltonian = problem.hamiltonian
mapper = JordanWignerMapper()
H2 = mapper.map(hamiltonian.second_q_op())





##########################################
from pyscf import gto

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import (
    FreezeCoreTransformer,
    ActiveSpaceTransformer,
)
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import HartreeFock

from qiskit.quantum_info import Statevector
import numpy as np


geometry = (
    "N 0.000000 0.000000 0.117790; "
    "H 0.000000 0.937700 -0.274840; "
    "H 0.811900 -0.468850 -0.274840; "
    "H -0.811900 -0.468850 -0.274840"
)

driver = PySCFDriver(
    atom=geometry,
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)
problem = driver.run()

print(problem.num_particles, problem.num_spatial_orbitals)

transformer = FreezeCoreTransformer()
problem = transformer.transform(problem)

print(problem.num_particles, problem.num_spatial_orbitals)

active_space = ActiveSpaceTransformer(
    num_electrons=6,
    num_spatial_orbitals=4,
)
problem = active_space.transform(problem)

print(problem.num_particles, problem.num_spatial_orbitals)


fermionic_hamiltonian = problem.hamiltonian.second_q_op()

mapper = JordanWignerMapper()
NH3 = mapper.map(fermionic_hamiltonian)
print(NH3.size, NH3.num_qubits)

# exit()

num_qubits = 8
J = .5
H_TFIM = SparsePauliOp.from_sparse_list(
    [('ZZ', [i, i+1], .5) for i in range(num_qubits-1)] + \
    [('X', [i], .5) for i in range(num_qubits)],
    num_qubits=num_qubits
)


num_qubits = 8
delta = .5
H_XXZ = SparsePauliOp.from_sparse_list(
    [('XX', [i, i+1], .5) for i in range(num_qubits-1)] + \
    [('YY', [i, i+1], .5) for i in range(num_qubits-1)] + \
    [('ZZ', [i, i+1], .5*delta) for i in range(num_qubits-1)],
    num_qubits=num_qubits
)

num_qubits = 8
J2_J1 = .5
H_J1_J2 = SparsePauliOp.from_sparse_list(
    [('XX', [i, i+1], .5) for i in range(num_qubits-1)] + \
    [('YY', [i, i+1], .5) for i in range(num_qubits-1)] + \
    [('ZZ', [i, i+1], .5) for i in range(num_qubits-1)] + \
    [('XX', [i, i+2], .5*J2_J1) for i in range(num_qubits-2)] + \
    [('YY', [i, i+2], .5*J2_J1) for i in range(num_qubits-2)] + \
    [('ZZ', [i, i+2], .5*J2_J1) for i in range(num_qubits-2)],
    num_qubits=num_qubits
)

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
print(custom_basis)


##########################################

for toy_model, H in zip(["H_TFIM", "H_XXZ", "H_J1_J2", "H2", "NH3"], [H_TFIM, H_XXZ, H_J1_J2, H2, NH3]):
    warm_start_circuit = run_vqe_and_get_circuit(H, backend_name="simulator")
    for warm_start in [None, warm_start_circuit]:
        gc.collect()
        print(f"processing {toy_model} {'with warm start' if warm_start else 'with cold start'}")
        n_qubit = H.num_qubits
        s = .5

        hadamard_basis = toy_model in ["H2", "NH3"]

        zero_projection_oracle = create_zero_projection_gate(
            s=s,
            num_qubits=n_qubit,
            use_mcp=True,
            ascending=True,
            hadamard_basis=hadamard_basis if (not warm_start) else False,
            custom_basis=custom_basis if (not hadamard_basis) and (not warm_start) else None
        )
        monotonic_oracle = create_monotonic_diagonal(
            s=s,
            num_qubits=n_qubit,
            ascending=True,
            hadamard_basis=hadamard_basis if (not warm_start) else False,
            custom_basis=custom_basis if (not hadamard_basis) and (not warm_start) else None
        )
        select_k_oracle = create_select_k(
            s=s,
            num_qubits=n_qubit,
            ascending=True,
            hadamard_basis=hadamard_basis if (not warm_start) else False,
            custom_basis=custom_basis if (not hadamard_basis) and (not warm_start) else None
        )

        diagonal_oracles = {
            "I - 2|0><0|": zero_projection_oracle,
            "Monotonic": monotonic_oracle,
            "Select-K": select_k_oracle,
        }

        # initial_state = QuantumCircuit(n_qubit)
        # initial_state.x(0)
        initial_state = None

        runner, results = db_range_runner(
            hamiltonian=H,
            initial_state=initial_state,
            time_step=s,
            diagonal_oracle=diagonal_oracles,
            hadamard_basis=hadamard_basis if (not warm_start) else False,
            custom_basis=custom_basis if (not hadamard_basis) and (not warm_start) else None,
            warm_start=warm_start,
            num_steps_range=[0, 1, 2, 3] if n_qubit > 4 else [0, 1, 2, 3, 4],
            backend="simulator",
            estimate_energy=True,
            shots=1024,
            method="db_sorter",
            output_dir=f"examples/diagonal_zoo/{toy_model}_{'ws' if warm_start else 'cs'}",
        )


