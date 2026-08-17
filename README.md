# Double-Bracket Methods for Ground State Preparation

Implementation of [DB-QITE](https://doi.org/10.48550/arXiv.2412.04554) (by Gluza et al), [QDP-QITE](https://doi.org/10.48550/arXiv.2403.09187) and [Ground state by DBI](https://doi.org/10.22331/q-2024-04-09-1316) (by Gluza)

This repository implements three recently emerged algorithms for Hamiltonian ground state preparation:
* **DB-sorter**: Based on DBI algorithm sorts the eigenvalues and eigenvectors and chooses the smallest one
* **DB-QITE**: Mimics imaginary time evolution by approximating Hamiltonian commutation with `|0><0|`
* **QDP-QITE**: Previous algorithm with slower depth growth using Quantum Dynamic Programming method

![Brockett Flow](./images/evolution.gif)
![DBI Flow](./images/dbi_evolution.gif)

## Installation
```bash
pip install db-qite
```

## Usage
### To create a `QuantumCircuit`

#### DB-sorter (or DB-QITE or QDP-QITE)
```python
from db_qite import DB_Sorter  # or DB_QITE or QDP_QITE

dbq = DB_Sorter(
    hamiltonian = H,         # hamiltonian: `SparsePauliOp`
    initial_state = None,    # initial_state array-like, default None
    time_step = s,           # time step. list or a single value
    trotterization = True,   # whether to approx H, default True
)

circuit = dbq.create_circuit(
    num_steps,               # #iterations
)
```


### Run for a range of iterations with plots

You can create and run your favorite circuit for a range of `s` values and create plots
showing how fidelity increases with depth. It can be used for any of `db_qite`, `db_sorter` or `qdp_qite` methods.

```python
from db_qite import db_range_runner 

runner, results = db_range_runner(
    hamiltonian=H,
    initial_state=None, # or initial state
    time_step=.5,       # or a list
    num_steps_range=range(1, 3),
    backend=None, #  or "simulator" or a Qiskit Backend
    estimate_energy=True, # wether to estimate energy or just prepare the state
    shots=1024,
    method="db_qite",
)
```


**Note:** if you provide None, or a qiskit backend name or object, it will run on IBM quantum machines. You have to set `IBM_QUANTUM_TOKEN` environment variable for it
