# you need `pip install  qiskit-nature pyscf` for this

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from db_qite import db_range_runner, DB_Insight

# Define the hamiltonian (H2 molecule)
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
problem = driver.run()
hamiltonian = problem.hamiltonian
mapper = JordanWignerMapper()
H = mapper.map(hamiltonian.second_q_op())

for method in ["db_qite", "db_sorter", "qdp_qite"]:
    runner, results = db_range_runner(
        hamiltonian=H,
        initial_state=None,
        time_step=.5,
        num_steps_range=[1, 2, 3],
        backend="simulator",
        estimate_energy=True,
        shots=1024,
        method=method
    )


# Visualizations
H_mat = H.to_matrix()

insight = DB_Insight(hamiltonian=H_mat, step_size=0.5)
insight.create_evolution_gif(num_steps=100)
insight.create_DBI_evolution_gif(num_steps=200)

