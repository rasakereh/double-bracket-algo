from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import HamiltonianGate, IGate
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, EstimatorV2 as Estimator, accounts
from qiskit.converters import circuit_to_dag
from qiskit.visualization import plot_histogram

import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

ibm_token = os.getenv("IBM_QUANTUM_TOKEN")

if ibm_token:
    try:
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=ibm_token)
    except accounts.exceptions.AccountAlreadyExistsError:
        print("IBM Quantum account already exists. Using existing account.")

class DB_QITE:
    U_evolution = None
    multiple_s = True

    def __init__(self, hamiltonian, initial_state, time_step):
        self.hamiltonian = hamiltonian
        self.initial_state = initial_state
        self.time_step = time_step
        if isinstance(time_step, float):
            self.multiple_s = False
            self.e_is, self.e_P0 = self._create_rotation_projection(time_step)
    
    def _create_rotation_projection(self, s):
        e_is = HamiltonianGate(self.hamiltonian, time=-s**.5, label='$e^{i\\sqrt{s}H}$')
        P0 = np.zeros(self.hamiltonian.dim)
        P0[0, 0] = 1
        e_P0 = HamiltonianGate(P0, time=-s**.5, label='$e^{i\\sqrt{s}|0><0|}$')

        return e_is, e_P0

    def create_U_k(self, k, s=None, random_u0=False):
        if k == 0:
            if self.initial_state is not None:
                argmax = np.argmax(self.initial_state)+1
                closest_qubit = int(np.ceil(np.log2(argmax)))
                U0 = QuantumCircuit(self.hamiltonian.num_qubits)
                for n in range(self.hamiltonian.num_qubits):
                    if closest_qubit == n:
                        U0.x(n)
                    else:
                        U0.id(n)
            elif random_u0:
                # np.random.seed(42)
                # U0 = QuantumCircuit(self.hamiltonian.num_qubits)
                # for n in range(self.hamiltonian.num_qubits):
                #     if np.random.rand() < 0.5:
                #         U0.x(n)
                #     else:
                #         U0.id(n)
                U0 = QuantumCircuit(self.hamiltonian.num_qubits)
                for n in range(self.hamiltonian.num_qubits):
                    U0.h(n)
            else:
                U0 = QuantumCircuit(self.hamiltonian.num_qubits)
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

class CircuitRunner:
    def __init__(self, circuits, backend, estimate_energy, default_shots, hamiltonian=None, output_dir='outputs'):
        # backend can be "simulator" a string name for qiskit backend or an actual backend object, None (we find the backend automatically)
        self.estimate_energy = estimate_energy
        if self.estimate_energy and hamiltonian is None:
            raise ValueError("Hamiltonian must be provided when estimate_energy is True")
        self.hamiltonian = hamiltonian
        self.output_dir = output_dir
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.default_shots = default_shots
        if backend == "simulator":
            self.backend = AerSimulator()
        elif isinstance(backend, str):
            service = QiskitRuntimeService()
            self.backend = service.backend(backend)
        elif backend is not None:
            self.backend = backend
        else:
            service = QiskitRuntimeService()
            self.backend = service.least_busy(simulator=False, operational=True)
        if backend != "simulator":
            self.simulation = False
            self.sampler = Sampler(self.backend)
        else:
            self.simulation = True
            self.sampler = self.backend
        
        self.estimator = Estimator(mode=self.backend, options={"default_shots": default_shots})
        self.circuits = [transpile(circuit, self.backend) for circuit in circuits]
    
    def result_by_eigensolver(self):
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian must be provided to compute eigenvalues and eigenvectors")
        np.eigenvals, np.eigvecs = np.linalg.eigh(Operator(self.hamiltonian).data)
        self.eigenvalues = np.eigenvals
        self.eigenvectors = np.eigvecs

        return self.eigenvalues, self.eigenvectors
    
    @staticmethod
    def _get_active_qubit_count(circuit):
        dag = circuit_to_dag(circuit)
        # A qubit is active if it is not in the list of idle wires
        active_qubits = [q for q in circuit.qubits if q not in dag.idle_wires()]
        return len(active_qubits)
    
    def draw_transpiled_circuits(self):
        for i, circuit in enumerate(self.circuits):
            print(f"Circuit {circuit.name} transpiled for backend {self.backend.name} with depth {circuit.depth()}, num_qubits {self._get_active_qubit_count(circuit)}, and size {circuit.size()}")
            circuit_to_draw = circuit#.decompose() if self.simulation else circuit
            circuit_to_draw.draw('mpl', filename=f'{self.output_dir}/transpiled_circuit_{i}.png')
            plt.close()
    
    def _run_estimate_energy(self):
        self.jobs = self.estimator.run(
            [(circuit, self.hamiltonian) for circuit in self.circuits],
        )
        self.results = [
            (float(res.data.evs), float(res.data.stds))
            for res in self.jobs.result()
        ]

        return self.results
    
    def _run_Z_measurement(self):
        self.jobs = self.sampler.run(self.circuits, shots=self.default_shots)
        if self.simulation:
            self.results = [result.data.counts for result in self.jobs.result().results]
        else:
            self.results = [result.data.c.get_counts() for result in self.jobs.result()]
        return self.results
    
    def run(self):
        if self.estimate_energy:
            return self._run_estimate_energy()
        else:
            return self._run_Z_measurement()
    
    def _draw_estimate_energy(self):
        self.result_by_eigensolver()

        ground_state_energy = self.eigenvalues[0]

        energies = [res[0] for res in self.results]
        stds = [res[1] for res in self.results]
        plt.errorbar(range(len(energies)), energies, yerr=stds, fmt='o-', ecolor='red', capsize=5)
        plt.axhline(ground_state_energy, color='green', linestyle='--', label='Ground State Energy')
        for energy_level in self.eigenvalues[1:]:
            plt.axhline(energy_level, color='gray', linestyle='dotted', alpha=0.5)
        plt.legend()
        plt.xticks(range(len(energies)), [circuit.name for circuit in self.circuits], rotation=45)
        plt.xlabel('Circuit')
        plt.ylabel('Estimated Energy')
        plt.title('Energy Estimates with Standard Deviation')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/energy_estimates.png')
        plt.close()
    
    def _draw_Z_measurement(self):
        for i, result in enumerate(self.results):
            name = self.circuits[i].name
            plot_histogram(result, title=f"Results for {name}")
            plt.savefig(f'{self.output_dir}/results_{name}.png')
            plt.close()
    
    def draw_results(self):
        if self.estimate_energy:
            self._draw_estimate_energy()
        else:
            self._draw_Z_measurement()


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
        dbq = DB_QITE(hamiltonian, initial_state, time_step)
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

