# you need `pip install  qiskit-nature pyscf` for this

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from db_qite import db_range_runner, DB_Insight
from db_qite.utils import create_zero_projection_gate, create_select_k, create_monotonic_diagonal

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

##########################################

for molecule, H in zip(["H2", "NH3"], [H2, NH3]):
    print(f"processing {molecule}")
    n_qubit = H.num_qubits
    s = .5

    zero_projection_oracle = create_zero_projection_gate(s=s, num_qubits=n_qubit, use_mcp=True, ascending=True, hadamard_basis=True)
    monotonic_oracle = create_monotonic_diagonal(s=s, num_qubits=n_qubit, ascending=True, hadamard_basis=True)
    select_k_oracle = create_select_k(s=s, num_qubits=n_qubit, ascending=True, hadamard_basis=True)

    diagonal_oracles = {
        "I - 2|0><0|": zero_projection_oracle,
        "Monotonic": monotonic_oracle,
        "Select-K": select_k_oracle,
    }

    runner, results = db_range_runner(
        hamiltonian=H,
        initial_state=None,
        time_step=s,
        diagonal_oracle=diagonal_oracles,
        hadamard_basis=True,
        num_steps_range=[0, 1, 2, 3] if n_qubit > 4 else [0, 1, 2, 3, 4],
        backend="simulator",
        estimate_energy=True,
        shots=1024,
        method="db_sorter",
        output_dir=f"examples/diagonal_zoo/{molecule}"
    )


