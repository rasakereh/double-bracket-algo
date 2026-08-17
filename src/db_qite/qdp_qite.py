from qiskit import QuantumCircuit

from .utils import (
    create_evolution_gate,
    create_zero_projection_gate,
)

from .db_base import DB_Base
from .QDP import ExpSwap as QDP_Gate

class QDP_QITE(DB_Base):
    """Class for the QITE algorithm using the QDP-QITE framework.
    Inherits from the DB_Base class and implements the QDP-QITE.

    It estimates the imaginary time evolution of a quantum system using the QDP-QITE algorithm. It is a variant of the Double Bracket-QITE algorithm that uses the Quantum Dynamic Programming (QDP) to control the depth of the circuit at the cost of increasing the number of qubits.
    
    Learn more about the method at https://doi.org/10.48550/arXiv.2403.09187
    """

    def _create_auxiliary_gates(self, s):
        e_is = create_evolution_gate(s, self.hamiltonian, use_pauli=self.trotterization)

        return e_is, None

    def create_U_k(self, k, s=None):
        if k == 0:
            return self.create_U_0()
        
        e_is, _ = self.get_auxiliary_gates(s, k)
        current_s = self.get_curr_s(s, k)

        num_qubits = self.hamiltonian.num_qubits

        U_k_1 = self.create_U_k(k - 1, s).to_gate(label=f'$U_{k-1}$')
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
    
