from qiskit import transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, EstimatorV2 as Estimator, accounts
from qiskit.converters import circuit_to_dag
from qiskit.visualization import plot_histogram

import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

def login_ibm_quantum():
    ibm_token = os.getenv("IBM_QUANTUM_TOKEN")

    if ibm_token:
        try:
            print("Logging in to IBM Quantum...")
            QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=ibm_token)
            print("Logged in to IBM Quantum successfully.")
        except accounts.exceptions.AccountAlreadyExistsError:
            print("IBM Quantum account already exists. Using existing account.")
            

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
        if backend != "simulator":
            login_ibm_quantum()
            self.simulation = False
        else:
            self.simulation = True

        if self.simulation:
            self.backend = AerSimulator()
        elif isinstance(backend, str):
            service = QiskitRuntimeService()
            self.backend = service.backend(backend)
        elif backend is not None:
            self.backend = backend
        else:
            service = QiskitRuntimeService()
            self.backend = service.least_busy(simulator=False, operational=True)
            
        if not self.simulation:
            self.sampler = Sampler(self.backend)
        else:
            self.sampler = self.backend
        
        self.estimator = Estimator(mode=self.backend, options={"default_shots": default_shots})
        self.circuits = [transpile(circuit, self.backend) for circuit in circuits]
    
    def result_by_eigensolver(self):
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian must be provided to compute eigenvalues and eigenvectors")
        eigenvals, eigvecs = np.linalg.eigh(Operator(self.hamiltonian).data)
        self.eigenvalues = eigenvals
        self.eigenvectors = eigvecs

        for eigval, eigvec in zip(self.eigenvalues, self.eigenvectors.T):
            print(f"Eigenvalue: {eigval:.2f}")
            print(f"closest state: |{np.argmax(np.abs(eigvec))}>")
            print(f"~Eigenvector: {np.abs(eigvec)}\n")
            print("-" * 40)

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
            if circuit.size() < 500:
                circuit_to_draw = circuit#.decompose() if self.simulation else circuit
                circuit_to_draw.draw('mpl', filename=f'{self.output_dir}/transpiled_circuit_{i}.png')
                plt.close()
    
    def _run_estimate_energy(self):
        self.jobs = self.estimator.run(
            [(
                circuit,
                self.hamiltonian if self.simulation else self.hamiltonian.apply_layout(circuit.layout)
            ) for circuit in self.circuits],
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

