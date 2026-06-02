# you need `pip install  qiskit-nature pyscf` for this

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from db_qite import db_qite_range, db_sorter_range, DB_Insight

# Define the hamiltonian (H2 molecule)
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
problem = driver.run()
hamiltonian = problem.hamiltonian
mapper = JordanWignerMapper()
H = mapper.map(hamiltonian.second_q_op())

# DB-QITE
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

# DB-Sorter
runner, results = db_sorter_range(
    hamiltonian=H,
    time_step=.3,
    num_steps_range=[1, 2, 3],
    backend=None, # Uses real quantum machine
    estimate_energy=True,
    shots=1024
)


# Visualizations
H_mat = H.to_matrix()

insight = DB_Insight(hamiltonian=H_mat, step_size=0.5)
insight.create_evolution_gif(num_steps=100)
insight.create_DBI_evolution_gif(num_steps=200)

