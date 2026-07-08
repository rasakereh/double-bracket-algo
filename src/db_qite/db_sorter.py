from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import numpy as np
import pathlib
import matplotlib.pyplot as plt

from .utils import (
    create_evolution_gate,
    create_zero_projection_gate,
    create_monotonic_diagonal,
    to_sparse_pauli,
)
from .circuit_runner import CircuitRunner

class DB_Sorter:
    multiple_s = True

    def __init__(self, hamiltonian, time_step, trotterization=True):
        self.trotterization = trotterization
        self.hamiltonian = to_sparse_pauli(hamiltonian, convert=trotterization)
        self.time_step = time_step
        if isinstance(time_step, float):
            self.multiple_s = False
            self.e_is, self.e_P0 = self._create_rotation_projection(time_step)
        
    def _create_rotation_projection(self, s):
        e_is = create_evolution_gate(s, self.hamiltonian, use_pauli=self.trotterization)
        # e_P0 = create_zero_projection_gate(s, self.hamiltonian.num_qubits, use_mcp=self.trotterization, add_noise=True)
        e_P0 = create_monotonic_diagonal(s, self.hamiltonian.num_qubits)

        return e_is, e_P0

    def create_U_k(self, k, s=None):
        if k == 0:
            U0 = QuantumCircuit(self.hamiltonian.num_qubits)
            U0.id(range(self.hamiltonian.num_qubits))
            return U0
        
        if self.multiple_s and s is None:
            e_is, e_P0 = self._create_rotation_projection(self.time_step[k-1])
        elif s is None:
            e_is, e_P0 = self.e_is, self.e_P0
        else:
            e_is, e_P0 = self._create_rotation_projection(s)
        
        e_P0_inverse = e_P0.inverse()
        e_P0_inverse.label = '$e^{-i\\sqrt{s}|0><0|}$'

        U_k_1 = self.create_U_k(k - 1, s).to_gate(label=f'$U_{k-1}$')
        U_k_1_inverse = U_k_1.inverse()
        U_k_1_inverse.label = f'$U_{{{k-1}}}^\\dagger$'
        U_k = QuantumCircuit(self.hamiltonian.num_qubits)
        U_k.append(e_P0_inverse, range(self.hamiltonian.num_qubits))
        U_k.append(U_k_1, range(self.hamiltonian.num_qubits))
        U_k.append(e_is, range(self.hamiltonian.num_qubits))
        U_k.append(U_k_1_inverse, range(self.hamiltonian.num_qubits))
        U_k.append(e_P0, range(self.hamiltonian.num_qubits))
        U_k.append(U_k_1, range(self.hamiltonian.num_qubits))
        
        return U_k
    
    def create_circuit(self, num_steps):
        circuit = QuantumCircuit(self.hamiltonian.num_qubits, self.hamiltonian.num_qubits)
        circuit.append(self.create_U_k(num_steps), range(self.hamiltonian.num_qubits))
        circuit.measure(range(self.hamiltonian.num_qubits), range(self.hamiltonian.num_qubits))
        circuit.name = f'DB-Sorter_{num_steps}_steps'
        
        return circuit


def db_sorter_range(
    hamiltonian,
    time_step,
    num_steps_range,
    backend="simulator",
    estimate_energy=True,
    shots=1024,
    output_dir='outputs'
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # heatmap of the hamiltonian
    plt.imshow(np.abs(Operator(hamiltonian).data), cmap='viridis')
    plt.colorbar()
    plt.title("Hamiltonian Matrix")
    plt.savefig(f'{output_dir}/hamiltonian_matrix.png')
    plt.close()

    circuits = []
    for num_steps in num_steps_range:
        dbq = DB_Sorter(hamiltonian, time_step, trotterization=backend!="simulator")
        circuit = dbq.create_circuit(num_steps)
        circuit.decompose().draw('mpl', filename=f'{output_dir}/DB-Sorter_{num_steps}_steps.png')
        plt.close()
        circuits.append(circuit)

    runner = CircuitRunner(circuits, backend, estimate_energy, shots, hamiltonian, output_dir=output_dir)

    runner.draw_transpiled_circuits()
    
    print(f"Running circuits...")
    results = runner.run()
    runner.draw_results()
    
    return runner, results

