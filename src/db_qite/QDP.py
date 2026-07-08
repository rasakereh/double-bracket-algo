from qiskit.circuit.library import XXPlusYYGate
from qiskit import QuantumCircuit

class ExpSwap:
    qc = None

    def __init__(self, num_qubits, t):
        self.qc = QuantumCircuit(num_qubits * 2, name="exp(-it.S)")
        for i in range(num_qubits):
            self.single_swap(t, i, i + num_qubits)
        
        self.gate = self.qc.to_gate(label='exp(-it.S)')
    
    def single_swap(self, t, inpA, inpB):
        # I can show that exp(-it.SWAP) = CNOT * I.Rz(t) * CNOT * XXPlusYY(2t)
        self.qc.append(XXPlusYYGate(2 * t, 0), [inpA, inpB])
        self.qc.cx(inpA, inpB)
        self.qc.rz(t, inpB)
        self.qc.cx(inpA, inpB)


