from qiskit import QuantumCircuit

from .utils import (
    create_evolution_gate,
    create_monotonic_diagonal,
)

from .db_base import DB_Base

class DB_Sorter(DB_Base):
    """Class for the Sorter algorithm using the DBSorter framework.
    Inherits from the DB_Base class and implements the Double Bracket-Sorter.

    It diagonalizes a Hamiltonian of a quantum system using the Double Bracket-Sorter algorithm and prepares the eigenstate associated with the smallest eigenvalue.

    Learn more about the method at https://doi.org/10.22331/q-2024-04-09-1316
    """


    def _create_auxiliary_gates(self, s):
        e_is = create_evolution_gate(s, self.hamiltonian, use_pauli=self.trotterization)
        e_P0 = create_monotonic_diagonal(s, self.hamiltonian.num_qubits)

        return e_is, e_P0

    def create_U_k(self, k, s=None):
        if k == 0:
            return self.create_U_0()
        
        e_is, e_P0 = self.get_auxiliary_gates(s, k)
        
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
    
