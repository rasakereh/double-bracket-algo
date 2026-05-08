# Double-Bracket Quantum Imaginary Time Evolution (DB-QITE)

Implementation of [DB-QITE](https://doi.org/10.48550/arXiv.2412.04554) (by Gluza et al)

## Installation
```bash
pip install qiskit-dbqite
```

## Usage
### To create a `QuantumCircuit`
```python
from db_qite import DB_QITE

dbq = DB_QITE(
    hamiltonian = H,         # hamiltonian: `SparsePauliOp`
    initial_state = None,    # initial_state array-like, default None
    time_step = s,           # time step. list or a single value
)
circuit = dbq.create_circuit(
    num_steps,               # #iterations
    random_u0,               # boolean. U0 is random or not
)
```


### Run for a range of iterations with plots
```python
from db_qite import db_qite_range

runner, results = db_qite_range(
    hamiltonian=H,
    initial_state=None, # or initial state
    time_step=.5,       # or a list
    random_u0=True,     # or False
    num_steps_range=range(1, 6),
    backend="simulator", #  or None to detect a qiskit backend
    estimate_energy=True, # wether to estimate energy or just prepare the state
    shots=1024
)
```

**Note:** if you provide None, or a qiskit backend name or object, it will run on IBM quantum machines. You have to set `IBM_QUANTUM_TOKEN` environment variable for it