
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp, state_fidelity
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt # For plotting
import time # To time the experiment

try:
    from scipy.optimize import minimize
except ModuleNotFoundError:
    print("ERROR: scipy is not installed. Please install it: pip install scipy")
    def minimize(*args, **kwargs):
        raise RuntimeError("scipy.optimize.minimize not found. Please install scipy.")


def run_single_projection_experiment(num_qubits, psi_depth, run_seed, projectors_u_reps=1):
    """Runs one instance of the state generation, projector optimization, 
    and iterative filtering, returning the final fidelity."""
    np.random.seed(run_seed) # Set seed for this specific run
    print(f"    WORKER (Seed: {run_seed}, Q:{num_qubits}, D:{psi_depth}): np.random.rand(1) after seed: {np.random.rand(1)}")

    # --- Step 1: Create initial quantum state |psi_initial> ---
    psi_ansatz = RealAmplitudes(num_qubits, reps=psi_depth, entanglement='linear')
    num_psi_params = psi_ansatz.num_parameters
    # Ensure parameters are generated even if num_psi_params is 0 (for depth 0)
    if num_psi_params > 0:
        random_psi_params = np.random.rand(num_psi_params) * 2 * np.pi
        print(f"      WORKER (Seed: {run_seed}): First 2 psi_params: {random_psi_params[:2]}")
        psi_circuit = psi_ansatz.assign_parameters(random_psi_params)
    else: # Depth 0 means no parameters, just the |0...0> state implicitly
        psi_circuit = QuantumCircuit(num_qubits) # Effectively |0...0>

    # Visualization (optional and conditional)
    # if VISUALIZE_CIRCUITS and psi_depth > 0: ... 

    try:
        psi_initial_vector = Statevector(psi_circuit)
    except Exception as e:
        # print(f"    ERROR (Run {run_seed}, Depth {psi_depth}, Qubits {num_qubits}): Could not create Statevector: {e}")
        return np.nan # Return NaN on failure

    # --- Step 2: Define base projectors and U(theta) ansatz ---
    proj_00_on_qubits_static_pauli_sum = {}
    if num_qubits > 1: # Projectors only make sense for > 1 qubit
        for q1_idx in range(num_qubits - 1):
            q2_idx = q1_idx + 1
            pauli_list = [
                ("I" * num_qubits, 0.25),
                ("".join(['Z' if i == q1_idx else 'I' for i in range(num_qubits)]), 0.25),
                ("".join(['Z' if i == q2_idx else 'I' for i in range(num_qubits)]), 0.25),
                ("".join(['Z' if i == q1_idx or i == q2_idx else 'I' for i in range(num_qubits)]), 0.25)
            ]
            reversed_pauli_list = [(p_str[::-1], coeff) for p_str, coeff in pauli_list]
            proj_00_on_qubits_static_pauli_sum[(q1_idx, q2_idx)] = SparsePauliOp.from_list(reversed_pauli_list)

    u_ansatz_2q = TwoLocal(2, rotation_blocks=['ry', 'rz'], entanglement_blocks='cx',
                           reps=projectors_u_reps, entanglement='linear')
    num_u_params = u_ansatz_2q.num_parameters

    # --- Step 3: Optimize projectors ---
    optimized_projectors_P_star = []
    if num_qubits > 1:
        qubit_pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        for q1, q2 in qubit_pairs:
            base_proj_spo = proj_00_on_qubits_static_pauli_sum[(q1, q2)]
            def cost_func(params):
                try:
                    u_circ_2q = u_ansatz_2q.assign_parameters(params)
                    u_full_qc_ = QuantumCircuit(num_qubits)
                    u_full_qc_.compose(u_circ_2q, qubits=[q1, q2], inplace=True)
                    u_op = Operator(u_full_qc_)
                    p_op_obj = u_op @ base_proj_spo @ u_op.adjoint()
                    return np.real(psi_initial_vector.expectation_value(p_op_obj))
                except Exception:
                    return np.nan 
            
            init_params = np.random.rand(num_u_params) * 2 * np.pi
            try:
                opt_res = minimize(fun=cost_func, x0=init_params, method='COBYLA',
                                   tol=SCIPY_COBYLA_TOL, 
                                   options={'maxiter': SCIPY_COBYLA_MAXITER, 'disp': VERBOSE_OPTIMIZATION})
                if opt_res.success or (not np.isnan(opt_res.fun) and opt_res.fun < 1.0): # Accept if fun is reasonable
                    u_circ_2q_opt_ = u_ansatz_2q.assign_parameters(opt_res.x)
                    u_full_qc_opt_ = QuantumCircuit(num_qubits)
                    u_full_qc_opt_.compose(u_circ_2q_opt_, qubits=[q1, q2], inplace=True)
                    u_op_opt_ = Operator(u_full_qc_opt_)
                    p_star_obj = u_op_opt_ @ base_proj_spo @ u_op_opt_.adjoint()
                    optimized_projectors_P_star.append({'qubits': (q1,q2), 'P_star_op': p_star_obj})
            except Exception:
                pass 

    # --- Step 4: Iterative projection ---
    if not optimized_projectors_P_star and num_qubits > 1:
        # If optimization failed for all projectors but there were pairs to optimize
        # this run might not be very informative, could return NaN or treat as fidelity 1
        # For now, assume if no projectors, state doesn't change.
        pass 
        
    current_psi = psi_initial_vector.copy()
    if num_qubits > 0 : # Identity op only if there are qubits
        id_op = Operator(np.eye(2**num_qubits, dtype=complex))
    else: # Should not happen with QUBIT_COUNTS_EXP setting
        return 1.0 

    for p_info in optimized_projectors_P_star:
        k_op = id_op - p_info['P_star_op']
        proj_psi_unnorm = current_psi.evolve(k_op)
        norm = np.linalg.norm(proj_psi_unnorm.data)
        if norm < 1e-9:
            current_psi = Statevector(proj_psi_unnorm.data, dims=proj_psi_unnorm.dims())
            break
        current_psi = Statevector(proj_psi_unnorm.data / norm, dims=proj_psi_unnorm.dims())
    
    final_psi = current_psi

    # --- Step 5: Calculate Fidelity ---
    final_norm = np.linalg.norm(final_psi.data)
    if final_norm < 1e-9:
        return 0.0 
    if abs(final_norm - 1.0) > 1e-7:
        final_psi = Statevector(final_psi.data / final_norm, dims=final_psi.dims())
        
    init_norm = np.linalg.norm(psi_initial_vector.data)
    if abs(init_norm - 1.0) > 1e-7:
        # This shouldn't happen if Statevector(circuit) works correctly
        psi_initial_vector = Statevector(psi_initial_vector.data / init_norm, dims=psi_initial_vector.dims())
        
    return state_fidelity(psi_initial_vector, final_psi)



