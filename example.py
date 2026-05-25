# you need `pip install  qiskit-nature pyscf` for this

from qiskit.quantum_info import SparsePauliOp, Operator
import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from db_qite import db_qite_range

# Define the hamiltonian (H2 molecule)
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
problem = driver.run()
hamiltonian = problem.hamiltonian
mapper = JordanWignerMapper()
H = mapper.map(hamiltonian.second_q_op())

runner, results = db_qite_range(
    hamiltonian=H,
    initial_state=None,
    time_step=.5,
    random_u0=True,
    num_steps_range=[1, 2, 3, 4],
    backend="simulator",
    estimate_energy=True,
    shots=1024
)

