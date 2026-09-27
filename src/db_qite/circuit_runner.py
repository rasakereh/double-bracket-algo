from qiskit import transpile
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, EstimatorV2 as Estimator, accounts
from qiskit.converters import circuit_to_dag
from qiskit.visualization import plot_histogram

from functools import reduce
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

from .db_qite import DB_QITE
from .db_sorter import DB_Sorter
from .qdp_qite import QDP_QITE

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
    """You can batch run the circuits with the same H and number of qubits using this class"""

    def __init__(self, circuits, backend, estimate_energy, default_shots, hamiltonian=None, output_dir='outputs'):
        # backend can be "simulator" a string name for qiskit backend or an actual backend object, None (we find the backend automatically)
        self.estimate_energy = estimate_energy
        if self.estimate_energy and hamiltonian is None:
            raise ValueError("Hamiltonian must be provided when estimate_energy is True")
        self.hamiltonian = hamiltonian
        self.output_dir = output_dir
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.circuit_partitions = {circuit_group: len(circuits[circuit_group]) for circuit_group in circuits}
        self.initial_circuits = reduce(
            lambda x, y: x + y,
            circuits.values(),
            []
        )
        self.default_shots = default_shots
        self.prepare_runtime(backend)
        self.transpile_circuits()
    
    def _find_circuit_partition(self, circuit_idx): #TODO: we can just store ranges
        total = 0
        for circuit_group, group_size in self.circuit_partitions.items():
            total += group_size
            if total > circuit_idx:
                return circuit_group
    
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
                circuit_group = self._find_circuit_partition(i)
                circuit_to_draw.draw('mpl', filename=f'{self.output_dir}/transpiled_circuit_{circuit_group}_{i}.png')
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
    
    def _partition_results(self, results):
        self.results = {}
        offset = 0
        for circuit_group in self.circuit_partitions:
            self.results[circuit_group] = results[
                offset:(offset+self.circuit_partitions[circuit_group])
            ]
            offset += self.circuit_partitions[circuit_group]
        
        return self.results

    def _run_estimate_energy(self):
        jobs_to_submiut = [self._prepare_estimator_circuit(idx) for idx, _ in enumerate(self.circuits)]
        self.jobs = self.estimator.run(jobs_to_submiut)
        results = [
            (float(res.data.evs), float(res.data.stds))
            for res in self.jobs.result()
        ]

        return self._partition_results(results)
    
    def _run_Z_measurement(self):
        self.jobs = self.sampler.run(self.circuits, shots=self.default_shots)
        results = [result.data.c.get_counts() for result in self.jobs.result()]
        
        return self._partition_results(results)
    
    def run(self):
        if self.estimate_energy:
            return self._run_estimate_energy()
        else:
            return self._run_Z_measurement()
    
    def _draw_estimate_energy(self):
        self.result_by_eigensolver()

        ground_state_energy = self.eigenvalues[0]

        for circuit_group, result_group in self.results.items():
            energies = [res[0] for res in result_group]
            stds = [res[1] for res in result_group]
            plt.errorbar(range(len(energies)), energies, yerr=stds, fmt='o-', ecolor='red', capsize=5, label=circuit_group)

        plt.axhline(ground_state_energy, color='green', linestyle='--', label='Ground State Energy')
        for energy_level in self.eigenvalues[1:]:
            plt.axhline(energy_level, color='gray', linestyle='dotted', alpha=0.5)
        plt.legend()
        plt.xticks(range(len(energies)), [circuit.name for circuit in self.circuits[:len(energies)]], rotation=45)
        plt.xlabel('Circuit')
        plt.ylabel('Estimated Energy')
        plt.title('Energy Estimates with Standard Deviation')
        plt.tight_layout()
        method = '_'.join(self.circuits[0].name.split("_")[:-2])
        plt.savefig(f'{self.output_dir}/energy_estimates_{method}.png')
        plt.close()
    
    def _draw_Z_measurement(self):
        i = 0
        for circuit_group in self.results:
            for result in self.results[circuit_group]:
                name = self.circuits[i].name
                plot_histogram(result, title=f"Results for {name}")
                plt.savefig(f'{self.output_dir}/results_{circuit_group}::{name}.png')
                plt.close()
                i += 1
    
    def draw_results(self):
        if self.estimate_energy:
            self._draw_estimate_energy()
        else:
            self._draw_Z_measurement()


def db_range_runner(
    hamiltonian=None,
    time_step=.5,
    num_steps_range=[1],
    initial_state=None,
    evolution_oracle=None,
    diagonal_oracle=None,
    hadamard_basis=False,
    backend="simulator",
    estimate_energy=True,
    shots=1024,
    output_dir='outputs',
    method="db_qite"
):
    """
    Run the specific DB algorithm for a range of time steps.

    Args:
        hamiltonian (qiskit.SparsePauliOp | qiskit.Operator | np.ndarray): The Hamiltonian operator.
        time_step (float|list[float]): The time step(s) for the evolution.
        num_steps_range (list[int]): A list of number of steps to run.
        initial_state (qiskit.QuantumCircuit): The circuit to prepare the initial state for the circuit.
        evolution_oracle (qiskit.QuantumCircuit | None): The oracle for the evolution (exp(-is^.5H)).
        diagonal_oracle (qiskit.QuantumCircuit | dict(str, qiskit.QuantumCircuit) | None): The oracle for D evolution (exp(-is^.5D)). List can be provided to compare different oracles
        hadamard_basis (bool): Whether to use the Hadamard basis. Defaults to False.
        backend (str | qiskit.BaseBackend | None): The backend to use for simulation. If None, the least busy backend will be used. If "simulator", the proper simulator will be used.
        estimate_energy (bool): Whether to estimate the energy or measure the final state.
        shots (int): The number of shots for each measurement.
        output_dir (str): The directory to save the output files.

    Returns:
        The runner and the results.
    """
    assert method in ["db_qite", "db_sorter", "qdp_qite"], f"Unknown method: {method}"

    db_class = {
        "db_qite": DB_QITE,
        "db_sorter": DB_Sorter,
        "qdp_qite": QDP_QITE
    }[method]

    db_name = {
        "db_qite": "DB-QITE",
        "db_sorter": "DB-Sorter",
        "qdp_qite": "QDP-QITE"
    }

    trotterization = False if backend == "simulator" else True
    if method == "qdp_qite":
        trotterization = True
    measure = not estimate_energy
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # heatmap of the hamiltonian if the number of qubits is small enough
    if hamiltonian and hamiltonian.num_qubits <= 10:
        plt.imshow(np.abs(Operator(hamiltonian).data), cmap='viridis')
        plt.colorbar()
        plt.title("Hamiltonian Matrix")
        plt.savefig(f'{output_dir}/hamiltonian_matrix.png')
        plt.close()

    circuits = {}
    if not diagonal_oracle or not isinstance(diagonal_oracle, dict):
        circuit_partitions = {db_name[method]: diagonal_oracle}
    else:
        circuit_partitions = {oracle_name: d_oracle for oracle_name, d_oracle in diagonal_oracle.items()}
    for partition_name, d_oracle in circuit_partitions.items():
        circuits[partition_name] = []
        for num_steps in num_steps_range:
            db_circuit = db_class(
                hamiltonian,
                time_step,
                trotterization=trotterization,
                measure=measure,
                initial_state=initial_state,
                evolution_oracle=evolution_oracle,
                diagonal_oracle=d_oracle,
                hadamard_basis=hadamard_basis
            )
            circuit = db_circuit.create_circuit(num_steps)
            circuit.name = f"{method}_{num_steps}_steps"
            circuit.decompose().draw('mpl', filename=f'{output_dir}/{partition_name}_{num_steps}_steps.png')
            plt.close()
            circuits[partition_name].append(circuit)

    runner = CircuitRunner(circuits, backend, estimate_energy, shots, hamiltonian, output_dir=output_dir)

    runner.draw_transpiled_circuits()

    print(f"Running circuits...")
    # results = None
    results = runner.run()
    runner.draw_results()
    
    return runner, results


