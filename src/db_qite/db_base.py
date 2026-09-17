from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .utils import to_sparse_pauli

class DB_Base(ABC):
    """Base class for the Double Bracket algorithms. This class is not meant to be used directly, but rather to be inherited by other classes that implement specific versions of the algorithm.

    Attributes:
        hamiltonian (qiskit.SparsePauliOp | qiskit.Operator | np.ndarray): The Hamiltonian of the system.
        time_step (float | list[float]): The time step(s) for the evolution.
        trotterization (bool): Whether to use Trotterization.
        initial_state (qiskit.QuantumCircuit | None): The circuit that prepares the initial state of the system.
        measure (bool): Whether to measure the final state.
        evolution_oracle (qiskit.QuantumCircuit | None): The oracle for the evolution (exp(-is^.5H)).
        diagonal_oracle (qiskit.QuantumCircuit | None): The oracle for D evolution (exp(-is^.5D)).
        hadamard_basis (bool): Whether to use the Hadamard basis. Defaults to False.
        num_qubits (int): The number of qubits in the system.
    """

    _multiple_s = True

    def __init__(
        self,
        hamiltonian=None,
        time_step=None,
        trotterization=True,
        measure=False,
        initial_state=None,
        evolution_oracle=None,
        diagonal_oracle=None,
        hadamard_basis=False,
    ):
        """Initialize the DB_Base class.

        Args:
            hamiltonian (qiskit.SparsePauliOp | qiskit.Operator | np.ndarray): The Hamiltonian of the system.
            time_step (float | list[float]): The time step(s) for the evolution. If a list is provided, it should have the same length as the number of steps in the evolution.
            trotterization (bool, optional): Whether to use Trotterization. Defaults to True.
            measure (bool, optional): Whether to measure the final state. Defaults to False.
            initial_state (qiskit.QuantumCircuit | None, optional): The circuit that prepares the initial state of the system. If None, the initial state will be |0>^n . Defaults to None.
            evolution_oracle (qiskit.QuantumCircuit | None, optional): The oracle for the evolution (exp(-is^.5H)). Defaults to None.
            diagonal_oracle (qiskit.QuantumCircuit | None, optional): The oracle for D evolution (exp(-is^.5D)). Defaults to None.
            hadamard_basis (bool, optional): Whether to use the Hadamard basis. Defaults to False.
        """

        assert hamiltonian or evolution_oracle, "Either hamiltonian or evolution_oracle must be provided"
        if hamiltonian is not None:
            self.hamiltonian = to_sparse_pauli(hamiltonian, convert=trotterization)
            self.num_qubits = self.num_qubits
        else:
            self.hamiltonian = None
            self.num_qubits = evolution_oracle.num_qubits

        self.trotterization = trotterization
        self.initial_state = initial_state
        self.time_step = time_step
        self.measure = measure
        self.evolution_oracle = evolution_oracle
        self.diagonal_oracle = diagonal_oracle
        self.hadamard_basis = hadamard_basis
        if isinstance(time_step, float):
            self._multiple_s = False
            self.e_is, self.e_P0 = self._create_auxiliary_gates(time_step)
        else:
            assert isinstance(time_step, list), "time_step must be a float or a list of floats"
            assert all(isinstance(t, float) for t in time_step), "All elements of time_step must be floats"
    
    def get_curr_s(self, s, k):
        """Get the current time step for the k-th step of the evolution.
        s can be inherited from the previous step or provided as an argument. If s is None, the time step will be taken from the time_step attribute.

        Args:
            s (float | None): The time step for the current step. If None, the time step will be taken from the time_step attribute.
            k (int): The index of the current step.
        
        Returns:
            float: The time step for the current step.
        """

        if self._multiple_s and s is None:
            return self.time_step[k-1]
        elif s is None:
            return self.time_step
        else:
            return s
    
    def get_auxiliary_gates(self, s, k):
        """Get the auxiliary gates for the k-th step of the evolution.

        Args:
            s (float | None): The time step for the current step. If None, the time step will be deduced.
            k (int): The index of the current step.

        Returns:
            tuple: The auxiliary gates for the current step.
        """

        current_s = self.get_curr_s(s, k)
        
        if s is None and not self._multiple_s:
            e_is, e_P0 = self.e_is, self.e_P0
        else:
            e_is, e_P0 = self._create_auxiliary_gates(current_s)
        
        # oracles override:
        e_is = self.evolution_oracle if self.evolution_oracle is not None else e_is
        e_P0 = self.diagonal_oracle if self.diagonal_oracle is not None else e_P0

        return e_is, e_P0
    
    @abstractmethod
    def _create_auxiliary_gates(self, s):
        pass
    
    def create_U_0(self):
        """Create the initial unitary U_0 for the evolution. If an initial state is provided, it will be used to prepare the initial state of the system. Otherwise, the initial state will be |0>^n.

        Returns:
            qiskit.QuantumCircuit: The initial unitary U_0.
        """

        U0 = QuantumCircuit(self.num_qubits)

        if self.initial_state is not None:
            assert isinstance(self.initial_state, QuantumCircuit), "initial_state must be a quantum circuit"
            U0 = self.initial_state
        else:
            U0.id(range(self.num_qubits))
        return U0

    @abstractmethod
    def create_U_k(self, k, s=None):
        pass

    def create_circuit(self, num_steps):
        """Create the quantum circuit for the evolution.

        Args:
            num_steps (int): The number of steps in the evolution.
        
        Returns:
            qiskit.QuantumCircuit: The quantum circuit for the evolution.
        """
        
        U_k = self.create_U_k(num_steps)
        total_qubits = U_k.num_qubits
        H_qubits = self.num_qubits
        circuit = QuantumCircuit(total_qubits, H_qubits)
        if self.hadamard_basis:
            circuit.h(range(H_qubits))
        circuit.append(U_k, range(total_qubits))
        if self.measure:
            circuit.measure(range(self.num_qubits), range(self.num_qubits))
        
        return circuit

