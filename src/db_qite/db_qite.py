from qiskit import QuantumCircuit

from .utils import (
    create_evolution_gate,
    create_zero_projection_gate,
)

from .db_base import DB_Base

class DB_QITE(DB_Base):
    """Class for the QITE algorithm using the DBQITE framework.
    Inherits from the DB_Base class and implements the Double Bracket-QITE.

    It estimates the imaginary time evolution of a quantum system using the Double Bracket-QITE algorithm.

    Learn more about the method at https://doi.org/10.48550/arXiv.2412.04554
    """

    def _create_auxiliary_gates(self, s):
        e_is = create_evolution_gate(s, self.hamiltonian, use_pauli=self.trotterization)
        e_P0 = create_zero_projection_gate(s, self.num_qubits, use_mcp=self.trotterization)

        return e_is, e_P0

    def create_U_k(self, k, s=None):
        if k == 0:
            return self.create_U_0()
        
        e_is, e_P0 = self.get_auxiliary_gates(s, k)

        U_k_1 = self.create_U_k(k - 1, s).to_gate(label=f'$U_{k-1}$')
        U_k_1_inverse = U_k_1.inverse()
        U_k_1_inverse.label = f'$U_{{{k-1}}}^\\dagger$'
        e_is_inverse = e_is.inverse()
        e_is_inverse.label = '$e^{-i\\sqrt{s}H}$'
        U_k = QuantumCircuit(self.num_qubits)
        U_k.append(U_k_1, range(self.num_qubits))
        U_k.append(e_is_inverse, range(self.num_qubits))
        U_k.append(U_k_1_inverse, range(self.num_qubits))
        U_k.append(e_P0, range(self.num_qubits))
        U_k.append(U_k_1, range(self.num_qubits))
        U_k.append(e_is, range(self.num_qubits))
        
        return U_k
    

