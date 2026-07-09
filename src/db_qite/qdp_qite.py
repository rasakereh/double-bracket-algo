from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import numpy as np
import pathlib
import matplotlib.pyplot as plt

from .utils import (
    create_evolution_gate,
    create_zero_projection_gate,
    to_sparse_pauli,
)
from .circuit_runner import CircuitRunner
from .QDP import ExpSwap as QDP_Gate

class QDP_QITE:
    multiple_s = True

    def __init__(self, hamiltonian, initial_state, time_step, trotterization=True):
        self.trotterization = trotterization
        self.hamiltonian = to_sparse_pauli(hamiltonian, convert=trotterization)
        self.initial_state = initial_state
        self.time_step = time_step
        if isinstance(time_step, float):
            self.multiple_s = False
            self.e_is = self._create_rotation(time_step)
    
    def get_curr_s(self, s, k):
        if self.multiple_s and s is None:
            return self.time_step[k-1]
        elif s is None:
            return self.time_step
        else:
            return s

    def _create_rotation(self, s):
        e_is = create_evolution_gate(s, self.hamiltonian, use_pauli=self.trotterization)

        return e_is

    def create_U_k(self, k, s=None, random_u0=False):
        if k == 0:
            U0 = QuantumCircuit(self.hamiltonian.num_qubits)
            if self.initial_state is not None:
                argmax = np.argmax(self.initial_state)+1
                closest_qubit = int(np.ceil(np.log2(argmax)))
                for n in range(self.hamiltonian.num_qubits):
                    if closest_qubit == n:
                        U0.x(n)
                    else:
                        U0.id(n)
            elif random_u0:
                for n in range(self.hamiltonian.num_qubits):
                    U0.h(n)
            else:
                U0.id(range(self.hamiltonian.num_qubits))
            return U0
        
        current_s = self.get_curr_s(s, k)
        
        if s is None and not self.multiple_s:
            e_is = self.e_is
        else:
            e_is = self._create_rotation(current_s)

        num_qubits = self.hamiltonian.num_qubits

        U_k_1 = self.create_U_k(k - 1, s, random_u0).to_gate(label=f'$U_{k-1}$')
        e_is_inverse = e_is.inverse()
        e_is_inverse.label = '$e^{-i\\sqrt{s}H}$'
        qdp_gate = QDP_Gate(num_qubits, current_s**0.5).gate
        qdp_gate.label = 'QDP'
        total_qubits = U_k_1.num_qubits

        U_k = QuantumCircuit(2*total_qubits)
        U_k.append(U_k_1, range(total_qubits))
        U_k.append(U_k_1, range(total_qubits, 2*total_qubits))

        U_k.append(e_is_inverse, range(total_qubits, total_qubits+num_qubits))

        U_k.append(
            qdp_gate,
            list(range(num_qubits)) + list(range(total_qubits, total_qubits + num_qubits)),
        )
        
        U_k.append(e_is, range(num_qubits))
        
        return U_k
    
    def create_circuit(self, num_steps, random_u0):
        U_k = self.create_U_k(num_steps, random_u0=random_u0)
        # print(U_k.draw("text"))
        total_qubits = U_k.num_qubits
        circuit = QuantumCircuit(total_qubits, self.hamiltonian.num_qubits)
        circuit.append(U_k, range(total_qubits))
        circuit.measure(range(self.hamiltonian.num_qubits), range(self.hamiltonian.num_qubits))
        circuit.name = f'QDP-QITE_{num_steps}_steps'
        
        return circuit


def qdp_qite_range(
    hamiltonian,
    initial_state,
    time_step,
    random_u0,
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
        qdp = QDP_QITE(hamiltonian, initial_state, time_step, trotterization=True)
        circuit = qdp.create_circuit(num_steps, random_u0)
        circuit.decompose().draw('mpl', filename=f'{output_dir}/QDP-QITE_{num_steps}_steps.png')
        plt.close()
        circuits.append(circuit)

    runner = CircuitRunner(circuits, backend, estimate_energy, shots, hamiltonian, output_dir=output_dir)

    runner.draw_transpiled_circuits()

    print(f"Running circuits...")
    results = runner.run()
    runner.draw_results()
    
    return runner, results

