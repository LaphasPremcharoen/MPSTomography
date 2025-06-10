# This should be at the top level of your script/notebook, not inside another function.
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp, state_fidelity
import itertools 

def cost_function_for_worker(params, u_ansatz, base_projector_spo, num_qubits, q_group, psi_initial_vector):
    """
    A top-level, pickle-able cost function for use with multiprocessing.
    """
    try:
        u_circ = u_ansatz.assign_parameters(params)
        u_full_qc = QuantumCircuit(num_qubits)
        u_full_qc.compose(u_circ, qubits=list(q_group), inplace=True)
        u_op = Operator(u_full_qc)
        p_op = u_op @ base_projector_spo @ u_op.adjoint()
        return np.real(psi_initial_vector.expectation_value(p_op))
    except Exception:
        # Errors in worker processes can be tricky to debug.
        # Returning a high value is safer.
        return 1e6