from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

from .utils import to_sparse_pauli

class DB_Base(ABC):
    """Base class for the Double Bracket algorithms. This class is not meant to be used directly, but rather to be inherited by other classes that implement specific versions of the algorithm.

    Attributes:
        hamiltonian (qiskit.SparsePauliOp | qiskit.Operator | np.ndarray): The Hamiltonian of the system.
        time_step (float | list[float]): The time step(s) for the evolution.
        trotterization (bool): Whether to use Trotterization.
        initial_state (qiskit.QuantumCircuit | None): The circuit that prepares the initial state of the system.
    """

    _multiple_s = True

    def __init__(self, hamiltonian, time_step, trotterization=True, initial_state=None):
        """Initialize the DB_Base class.

        Args:
            hamiltonian (qiskit.SparsePauliOp | qiskit.Operator | np.ndarray): The Hamiltonian of the system.
            time_step (float | list[float]): The time step(s) for the evolution. If a list is provided, it should have the same length as the number of steps in the evolution.
            trotterization (bool, optional): Whether to use Trotterization. Defaults to True.
            initial_state (qiskit.QuantumCircuit | None, optional): The circuit that prepares the initial state of the system. If None, the initial state will be |0>^n . Defaults to None.
        """

        self.trotterization = trotterization
        self.hamiltonian = to_sparse_pauli(hamiltonian, convert=trotterization)
        self.initial_state = initial_state
        self.time_step = time_step
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

        return e_is, e_P0
    
    @abstractmethod
    def _create_auxiliary_gates(self, s):
        pass
    
    def create_U_0(self):
        """Create the initial unitary U_0 for the evolution. If an initial state is provided, it will be used to prepare the initial state of the system. Otherwise, the initial state will be |0>^n.

        Returns:
            qiskit.QuantumCircuit: The initial unitary U_0.
        """

        U0 = QuantumCircuit(self.hamiltonian.num_qubits)
        if self.initial_state is not None:
            assert isinstance(self.initial_state, QuantumCircuit), "initial_state must be a quantum circuit"
            U0 = self.initial_state
        else:
            U0.id(range(self.hamiltonian.num_qubits))
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
        circuit = QuantumCircuit(total_qubits, self.hamiltonian.num_qubits)
        circuit.append(U_k, range(total_qubits))
        circuit.measure(range(self.hamiltonian.num_qubits), range(self.hamiltonian.num_qubits))
        circuit.name = f'QDP-QITE_{num_steps}_steps'
        
        return circuit

