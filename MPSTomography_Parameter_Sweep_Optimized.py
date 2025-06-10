# Parameter Sweep for MPS Tomography - Optimized
## 1. Imports (with sparse and parallel libs)
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp, state_fidelity
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
import itertools
import time
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh  # Sparse eigenvalue solver
from concurrent.futures import ProcessPoolExecutor  # Use process-based parallelism
from tqdm import tqdm  # Progress tracking
import warnings

# Ignore overflow warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
## 2. Configuration Parameters
# --- Main Experiment Configuration ---
NUM_QUBITS = 11
PSI_ANSATZ_DEPTH_RANGE = [1, 2, 3, 4, 5]
PROJECTOR_QUBIT_COUNT_RANGE = [2, 3, 4, 5]
NUM_TRIALS = 5

# Adaptive COBYLA settings
SCIPY_COBYLA_BASE_TOL = 1e-4
SCIPY_COBYLA_BASE_MAXITER = 40000
SEED = 42


## 3. Core Functions with Optimizations
# Statevector cache to avoid recomputation
statevector_cache = {}

def get_statevector(circuit):
    """Get statevector with improved caching"""
    from hashlib import sha256
    circuit_repr = repr(circuit)
    circuit_hash = sha256(circuit_repr.encode()).hexdigest()
    
    if circuit_hash not in statevector_cache:
        statevector_cache[circuit_hash] = Statevector(circuit)
    return statevector_cache[circuit_hash]

def run_optimization_with_mps(
    num_qubits, 
    psi_preparation_circuit, 
    proj_qubit_count, 
    proj_u_reps, 
    run_seed,
    optimizer_tol,
    optimizer_maxiter
):
    """Finds optimized projectors P* using EstimatorV2 with MPS simulator"""
    np.random.seed(run_seed)
    
    # Initialize Estimator with MPS backend
    # Initialize Estimator with MPS backend
    estimator = AerEstimator(
        options={
            "backend_options": {"method": "matrix_product_state"},
            "run_options": {"shots": 10000, "seed": run_seed}
        }
    )

    qubit_groups = [tuple((i + j) % num_qubits for j in range(proj_qubit_count)) for i in range(num_qubits)]
    u_ansatz = TwoLocal(proj_qubit_count, ['ry', 'rz'], 'cx', 'linear', reps=proj_u_reps)
    num_u_params = u_ansatz.num_parameters
    
    optimized_projectors = []
    
    # Use tqdm for progress tracking
    for q_group in tqdm(qubit_groups, desc="Optimizing projectors", leave=False):
        q_group_str = "".join(map(str, q_group))
        vqa_circuit = psi_preparation_circuit.compose(u_ansatz.inverse(), qubits=list(q_group))
        vqa_circuit = vqa_circuit.decompose()
        
        s_i_full = ['I'] * num_qubits
        s_z_full_list = ['I'] * num_qubits
        for q_idx in q_group: 
            s_z_full_list[q_idx] = 'Z'
        
        base_projector_spo = SparsePauliOp([
            "".join(s_i_full)[::-1],
            "".join(s_z_full_list)[::-1]
        ], coeffs=[0.5, 0.5])

        def cost_func(params):
            pub = (vqa_circuit, base_projector_spo, params)
            job = estimator.run(pubs=[pub])
            result = job.result()
            cost = result[0].data.evs
            return np.real(cost)

        init_params = np.random.rand(num_u_params) * 2 * np.pi
        
        try:
            opt_res = minimize(
                fun=cost_func, x0=init_params, method='COBYLA',
                tol=optimizer_tol, options={'maxiter': optimizer_maxiter, 'disp': False}
            )
            
            u_circ_opt = u_ansatz.assign_parameters(opt_res.x)
            u_op_opt = Operator(u_circ_opt)
            identity_full = Operator(np.eye(2**num_qubits))
            u_op_full = identity_full.compose(u_op_opt, qargs=list(q_group))
            p_star_op = u_op_full.adjoint().compose(base_projector_spo).compose(u_op_full)
            optimized_projectors.append(p_star_op)
                
        except Exception as e_opt:
            print(f"ERROR: Optimization failed for P_{q_group_str}: {str(e_opt)[:100]}...")
            optimized_projectors.append(None)
            
    return optimized_projectors
## 4. Parallel Sweep Implementation
def run_trial(params):
    """Run a single trial with given parameters"""
    depth, proj_size, trial = params
    trial_seed = SEED + trial
    np.random.seed(trial_seed)
    
    # Adaptive COBYLA settings based on projector size
    # More conservative settings for small parameters
    optimizer_tol = max(1e-3, SCIPY_COBYLA_BASE_TOL * (2 / proj_size))
    optimizer_maxiter = min(1000, int(SCIPY_COBYLA_BASE_MAXITER * (proj_size / 5)))
    
    # State preparation circuit
    psi_ansatz = RealAmplitudes(NUM_QUBITS, reps=depth, entanglement='linear')
    psi_params = np.random.rand(psi_ansatz.num_parameters) * 2 * np.pi
    psi_preparation_circuit = psi_ansatz.assign_parameters(psi_params).decompose()
    
    # Run optimization
    optimized_projectors_P_star = run_optimization_with_mps(
        NUM_QUBITS, psi_preparation_circuit, proj_size, 
        proj_size - 1, trial_seed, 
        optimizer_tol, optimizer_maxiter
    )
    
    # Skip if optimization failed
    if not optimized_projectors_P_star or any(p is None for p in optimized_projectors_P_star):
        return depth, proj_size, trial, np.nan
    
    # Build Hamiltonian using sparse operations
    from scipy.sparse import csr_matrix
    H_op = csr_matrix((2**NUM_QUBITS, 2**NUM_QUBITS), dtype=complex)
    for p_star in optimized_projectors_P_star:
        if p_star is not None:
            H_op += csr_matrix(p_star.data)
    
    # Sparse eigenvalue solver for ground state only
    eigenvalues, eigenvectors = eigsh(H_op, k=1, which='SA')
    psi_gs_vector = eigenvectors[:, 0]
    psi_initial_vector = get_statevector(psi_preparation_circuit)
    fidelity = state_fidelity(psi_initial_vector, psi_gs_vector)
    
    return depth, proj_size, trial, fidelity

def run_parallel_sweep():
    """Run parameter sweep with parallel processing"""
    results = []
    params_list = list(itertools.product(
        PSI_ANSATZ_DEPTH_RANGE, 
        PROJECTOR_QUBIT_COUNT_RANGE, 
        range(NUM_TRIALS))
    )
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(run_trial, params_list),
            desc="Parameter Sweep Progress",
            total=len(params_list),
            ncols=100
        ))
    
    return results
## 5. Results Analysis and Plotting
if __name__ == '__main__':
    # Main experiment with timing
    print(f"--- Optimized Parameter Sweep for MPS Tomography ---")
    print(f"Config: Qubits={NUM_QUBITS}, Trials={NUM_TRIALS}")
    print(f"Sweeping PSI_ANSATZ_DEPTH over {PSI_ANSATZ_DEPTH_RANGE}")
    print(f"Sweeping PROJECTOR_QUBIT_COUNT over {PROJECTOR_QUBIT_COUNT_RANGE}")
    start_time = time.time()
    results = run_parallel_sweep()
    optimized_runtime = time.time() - start_time

    # Process results
    import pandas as pd
    df = pd.DataFrame(results, columns=['depth', 'proj_size', 'trial', 'fidelity'])
    avg_df = df.groupby(['depth', 'proj_size'])['fidelity'].mean().reset_index()

    print("\n=== Final Results ===")
    print(avg_df)

    # Generate plot
    plt.figure(figsize=(10, 6))
    for proj_size in PROJECTOR_QUBIT_COUNT_RANGE:
        subset = avg_df[avg_df['proj_size'] == proj_size]
        plt.plot(subset['depth'], subset['fidelity'], 
                 marker='o', linestyle='-', 
                 label=f'Projector Qubits={proj_size}')

    plt.xlabel('PSI_ANSATZ_DEPTH')
    plt.ylabel('Average Fidelity')
    plt.title('Average Fidelity vs. Ansatz Depth for Different Projector Sizes')
    plt.xticks(PSI_ANSATZ_DEPTH_RANGE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig('optimized_parameter_sweep_results.png', dpi=300)
    plt.show()
