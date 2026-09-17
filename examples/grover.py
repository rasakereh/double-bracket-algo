import numpy as np
from qiskit import QuantumCircuit
from db_qite import db_range_runner
from db_qite.utils import create_zero_projection_gate


n_qubit = 4
# marked state is "|0010>" (little-endian)
evolution_oracle = QuantumCircuit(4)
evolution_oracle.x([i for i in range(n_qubit) if i != 1])
evolution_oracle.h(1)
evolution_oracle.mcx([i for i in range(n_qubit) if i != 1], 1)
evolution_oracle.h(1)
evolution_oracle.x([i for i in range(n_qubit) if i != 1])
evolution_oracle.name = "evolution_oracle"
evolution_oracle = evolution_oracle.to_gate()

diagonal_oracle = create_zero_projection_gate(s=np.pi, num_qubits=n_qubit, use_mcp=True, ascending=True, hadamard_basis=True)

# with diagonal oracle
runner, results = db_range_runner(
    hamiltonian=None,
    time_step=1.,
    diagonal_oracle=diagonal_oracle,
    evolution_oracle=evolution_oracle,
    hadamard_basis=True, # whether to work in standard or hadamard basis
    num_steps_range=[0, 1, 2, 3, 4, 5],
    backend="simulator",
    estimate_energy=False,
    shots=1024,
    output_dir="grover_output/reflection",
    method="db_sorter",
)

# without diagonal oracle
runner, results = db_range_runner(
    hamiltonian=None,
    time_step=.5,
    diagonal_oracle=None,
    evolution_oracle=evolution_oracle,
    hadamard_basis=True, # whether to work in standard or hadamard basis
    num_steps_range=[0, 1, 2, 3, 4, 5],
    backend="simulator",
    estimate_energy=False,
    shots=1024,
    output_dir="grover_output/monotone",
    method="db_sorter",
)

