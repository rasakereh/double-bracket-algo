import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.linalg import expm
import os
from tqdm import tqdm

def _create_monotonic_diagonal(s, num_qubits):
    phases = [np.pi/2/(2**i) for i in range(num_qubits)]
    # phases = phases[::-1]  # Reverse to have the largest phase on the last qubit
    U = [
        [1, 0],
        [0, np.exp(1j*phases[0])]
    ]
    for phase in phases[1:]:
        current_U = [
            [1, 0],
            [0, np.exp(1j*phase)]
        ]
        U = np.kron(U, current_U)
    
    D = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    np.fill_diagonal(D, np.real(-1j*np.log(np.diag(U))))
    print(f"eigenvalues of D: {np.real(np.diag(D))}")

    np.fill_diagonal(U, np.diag(U)**(s**.5))
    
    return D, U

def abs_det(matrix):
    return np.abs(np.linalg.det(matrix))
        

class DB_Insight:
    def __init__(self, hamiltonian, step_size, diagonal_matrix=None, diagonal_unitary=None, inject_noise=False, name=None):
        if inject_noise:
            noise = np.random.normal(
                0,
                np.linalg.norm(hamiltonian)/hamiltonian.shape[0]/hamiltonian.shape[0]/100,
                hamiltonian.shape
            )
            noise = (noise + noise.T)/2
            hamiltonian += noise
        self.hamiltonian = hamiltonian
        self.evolved_hamiltonian = hamiltonian.copy()
        self.dbi_hamiltonian = hamiltonian.copy()
        self.step_size = step_size
        self.num_qubits = int(np.log2(hamiltonian.shape[0]))
        if diagonal_matrix is None:
            self.diagonal_matrix, self.diagonal_unitary = _create_monotonic_diagonal(self.step_size, self.num_qubits)
        else:
            self.diagonal_matrix = diagonal_matrix
            self.diagonal_unitary = diagonal_unitary
        
        self.DP_U = {}
        self.name = name or "Hamiltonian"
    
    def accurate_flow_step(self):
        bracket1 = self.evolved_hamiltonian @ self.diagonal_matrix - self.diagonal_matrix @ self.evolved_hamiltonian
        bracket2 = self.evolved_hamiltonian @ bracket1 - bracket1 @ self.evolved_hamiltonian
        self.evolved_hamiltonian += self.step_size * bracket2/np.linalg.norm(bracket2)/2

        return self.evolved_hamiltonian
    
    def accurate_evolve(self, num_steps):
        for _ in range(num_steps):
            self.accurate_flow_step()
        
        return self.evolved_hamiltonian
    
    def create_evolution_gif(self, num_steps, filename="outputs/insight/evolution.gif"):
        images = []
        for step in tqdm(range(num_steps), desc="Creating evolution GIF"):
            self.accurate_flow_step()
            plt.imshow(np.real(self.evolved_hamiltonian), cmap='viridis')
            plt.colorbar()
            plt.title(f"Brockett Flow Step {step+1} for {self.name}")
            plt.savefig(f"outputs/insight/temp_{step}.png")
            images.append(imageio.imread(f"outputs/insight/temp_{step}.png"))
            plt.close()
            os.remove(f"outputs/insight/temp_{step}.png")

        imageio.mimsave(filename, images, duration=num_steps**.5)
    
    def DBI_flow_step(self):
        exp_D = self.diagonal_unitary
        exp_neg_D = np.conj(exp_D.T)
        exp_H_k = expm(-1j * self.dbi_hamiltonian * (self.step_size**.5))
        U_k = exp_neg_D @ exp_H_k @ exp_D
        U_neg_k = np.conj(U_k.T)
        self.dbi_hamiltonian = U_neg_k @ self.dbi_hamiltonian @ U_k

    def DBI_evolve(self, num_steps):
        for _ in range(num_steps):
            self.DBI_flow_step()
        
        return self.dbi_hamiltonian
    
    def create_DBI_evolution_gif(self, num_steps, filename="outputs/insight/dbi_evolution.gif"):
        images = []
        for step in tqdm(range(num_steps), desc="Creating DBI evolution GIF"):
            self.DBI_flow_step()
            plt.imshow(np.real(self.dbi_hamiltonian), cmap='viridis')
            plt.colorbar()
            plt.title(f"DBI Step {step+1} for {self.name}")
            plt.savefig(f"outputs/insight/dbi_temp_{step}.png")
            images.append(imageio.imread(f"outputs/insight/dbi_temp_{step}.png"))
            plt.close()
            os.remove(f"outputs/insight/dbi_temp_{step}.png")

        imageio.mimsave(filename, images, duration=num_steps**.5)


