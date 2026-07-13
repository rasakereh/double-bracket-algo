from qiskit import transpile
from qiskit.quantum_info import Operator, SparsePauliOp
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

def get_ibm_runtime():
    ibm_token = os.getenv("IBM_QUANTUM_TOKEN")

    if ibm_token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)
    else:
        raise ValueError("Set IBM_QUANTUM_TOKEN environment variable to your API key")

class CircuitRunner:
    def __init__(self, circuits, backend, estimate_energy, default_shots, hamiltonian=None, output_dir='outputs'):
        # backend can be "simulator" a string name for qiskit backend or an actual backend object, None (we find the backend automatically)
        self.estimate_energy = estimate_energy
        if self.estimate_energy and hamiltonian is None:
            raise ValueError("Hamiltonian must be provided when estimate_energy is True")
        self.hamiltonian = hamiltonian
        self.output_dir = output_dir
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.initial_circuits = circuits
        self.default_shots = default_shots
        self.prepare_runtime(backend)
        self.transpile_circuits()
    
    def prepare_runtime(self, backend):
        if backend != "simulator":
            service = get_ibm_runtime()
            self.simulation = False
        else:
            self.simulation = True
        
        if self.simulation:
            max_qubit_cnt = max(circuit.num_qubits for circuit in self.initial_circuits)
            sim_method = 'matrix_product_state' if max_qubit_cnt > 20 else 'statevector'
            self.backend = AerSimulator(method=sim_method)
        elif isinstance(backend, str):
            self.backend = service.backend(backend)
        elif backend is not None:
            self.backend = backend
        else:
            self.backend = service.least_busy(simulator=False, operational=True)
            
        self.sampler = Sampler(self.backend)
        
        self.estimator = Estimator(mode=self.backend, options={"default_shots": self.default_shots})

    def transpile_circuits(self):
        self.circuits = [self._transpile_circuit(circuit) for circuit in self.initial_circuits]


    def result_by_eigensolver(self):
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian must be provided to compute eigenvalues and eigenvectors")
        eigenvals, eigvecs = np.linalg.eigh(Operator(self.hamiltonian).data)
        self.eigenvalues = eigenvals
        self.eigenvectors = eigvecs

        for eigval, eigvec in zip(self.eigenvalues[:5], self.eigenvectors.T[:5]):
            print(f"Eigenvalue: {eigval:.2f}")
            if len(eigvec) <= 16:
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
    
    def _prepare_estimator_circuit(self, circuit_idx):
        initial_circuit = self.initial_circuits[circuit_idx]
        circuit = self.circuits[circuit_idx]
        num_qubits = initial_circuit.num_qubits
        num_ancillas = num_qubits - self.hamiltonian.num_qubits
        ancilla_identity = SparsePauliOp("I" * num_ancillas)
        full_hamiltonian = SparsePauliOp.from_operator(Operator(self.hamiltonian)).tensor(ancilla_identity)
        full_hamiltonian = full_hamiltonian.apply_layout(circuit.layout)
        
        return (circuit, full_hamiltonian)
    
    def _transpile_circuit(self, circuit):
        return transpile(
            circuit,
            backend=self.backend,
            layout_method='sabre',
            routing_method='sabre',
            optimization_level=3,
            approximation_degree=1 if self.simulation else .999
        )

    def _run_estimate_energy(self):
        jobs_to_submiut = [self._prepare_estimator_circuit(idx) for idx, _ in enumerate(self.circuits)]
        self.jobs = self.estimator.run(jobs_to_submiut)
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
        plt.savefig(f'{self.output_dir}/energy_estimates_{self.circuits[0].name.split("_")[0]}.png')
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

