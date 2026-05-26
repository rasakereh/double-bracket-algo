from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator

import numpy as np
import pathlib
import matplotlib.pyplot as plt

from .utils import create_evolution_gate, create_zero_projection_gate
from .circuit_runner import CircuitRunner

class DB_QITE:
    multiple_s = True

    def __init__(self, hamiltonian, initial_state, time_step, trotterization=True):
        self.trotterization = trotterization
        if isinstance(hamiltonian, (SparsePauliOp, Pauli)) or not trotterization:
            self.hamiltonian = hamiltonian
        else:
            self.hamiltonian = SparsePauliOp.from_operator(Operator(hamiltonian))
        self.initial_state = initial_state
        self.time_step = time_step
        if isinstance(time_step, float):
            self.multiple_s = False
            self.e_is, self.e_P0 = self._create_rotation_projection(time_step)
        
    def _create_rotation_projection(self, s):
        e_is = create_evolution_gate(s, self.hamiltonian, use_pauli=self.trotterization)
        e_P0 = create_zero_projection_gate(s, self.hamiltonian.num_qubits, use_mcp=self.trotterization)

        return e_is, e_P0

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
                # np.random.seed(42)
                # for n in range(self.hamiltonian.num_qubits):
                #     if np.random.rand() < 0.5:
                #         U0.x(n)
                #     else:
                #         U0.id(n)
                for n in range(self.hamiltonian.num_qubits):
                    U0.h(n)
            else:
                U0.id(range(self.hamiltonian.num_qubits))
            return U0
        
        if self.multiple_s and s is None:
            e_is, e_P0 = self._create_rotation_projection(self.time_step[k-1])
        elif s is None:
            e_is, e_P0 = self.e_is, self.e_P0
        else:
            e_is, e_P0 = self._create_rotation_projection(s)

        U_k_1 = self.create_U_k(k - 1, s, random_u0).to_gate(label=f'$U_{k-1}$')
        U_k_1_inverse = U_k_1.inverse()
        U_k_1_inverse.label = f'$U_{{{k-1}}}^\\dagger$'
        e_is_inverse = e_is.inverse()
        e_is_inverse.label = '$e^{-i\\sqrt{s}H}$'
        U_k = QuantumCircuit(self.hamiltonian.num_qubits)
        U_k.append(U_k_1, range(self.hamiltonian.num_qubits))
        U_k.append(e_is_inverse, range(self.hamiltonian.num_qubits))
        U_k.append(U_k_1_inverse, range(self.hamiltonian.num_qubits))
        U_k.append(e_P0, range(self.hamiltonian.num_qubits))
        U_k.append(U_k_1, range(self.hamiltonian.num_qubits))
        U_k.append(e_is, range(self.hamiltonian.num_qubits))
        
        return U_k
    
    def create_circuit(self, num_steps, random_u0):
        circuit = QuantumCircuit(self.hamiltonian.num_qubits, self.hamiltonian.num_qubits)
        circuit.append(self.create_U_k(num_steps, random_u0=random_u0), range(self.hamiltonian.num_qubits))
        circuit.measure(range(self.hamiltonian.num_qubits), range(self.hamiltonian.num_qubits))
        circuit.name = f'DB-QITE_{num_steps}_steps'
        
        return circuit


def db_qite_range(
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
        dbq = DB_QITE(hamiltonian, initial_state, time_step, trotterization=(backend!="simulator"))
        circuit = dbq.create_circuit(num_steps, random_u0)
        circuit.decompose().draw('mpl', filename=f'{output_dir}/DB-QITE_{num_steps}_steps.png')
        plt.close()
        circuits.append(circuit)

    runner = CircuitRunner(circuits, backend, estimate_energy, shots, hamiltonian, output_dir=output_dir)

    runner.draw_transpiled_circuits()

    print(f"Running circuits...")
    results = runner.run()
    runner.draw_results()
    
    return runner, results

