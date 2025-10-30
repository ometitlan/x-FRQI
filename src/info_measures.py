# src/info_measures.py

import numpy as np
import pennylane as qml

def density_matrix(state):
    return np.outer(state, np.conj(state))

def entropy(state, wires, base=2):
    """
    Von Neumann entropy of the reduced density matrix on 'wires'.
    """
    dm = density_matrix(state)
    return qml.math.vn_entropy(dm, indices=wires, base=base)

def joint_entropy(state, wires_a, wires_b, base=2):
    """ H(A,B) """
    return entropy(state, wires=wires_a + wires_b, base=base)

def joint_entropy_n(state, wires_list, base=2):
    """
    Joint entropy H(A,B,C,...) for disjoint wire groups.
    wires_list: list of lists of wires, e.g. [[0,1], [2], [3]]
    """
    all_wires = [w for group in wires_list for w in group]
    return entropy(state, wires=all_wires, base=base)

def conditional_entropy(state, wires_a, wires_b, base=2):
    """
    H(A|B) = H(A,B) – H(B)
    """
    H_ab = joint_entropy(state, wires_a, wires_b, base=base)
    H_b  = entropy(state, wires=wires_b, base=base)
    return H_ab - H_b

def conditional_entropy_n(state, wires_a, wires_cond, base=2):
    """
    H(A | B,C,...) = H(A,B,C,...) – H(B,C,...)
    wires_cond: list of wires to condition on
    """
    H_all  = joint_entropy_n(state, [wires_cond, wires_a], base=base)
    H_cond = entropy(state, wires=wires_cond, base=base)
    return H_all - H_cond

def mutual_information(state, wires_a, wires_b, base=2):
    """
    I(A:B) = H(A) + H(B) – H(A,B)
    """
    H_a  = entropy(state, wires=wires_a, base=base)
    H_b  = entropy(state, wires=wires_b, base=base)
    H_ab = joint_entropy(state, wires_a, wires_b, base=base)
    return H_a + H_b - H_ab

def total_correlation(state, subsystems, base=2):
    """
    Total correlation (multi-information): 
      TC = Σ_i H(A_i) – H(A_1,A_2,...,A_n).
    subsystems: list of lists of wires, e.g. [[0,1], [2], [3]]
    """
    H_individual = sum(entropy(state, wires=s, base=base) for s in subsystems)
    H_joint_all  = joint_entropy_n(state, subsystems, base=base)
    return H_individual - H_joint_all

def tripartite_mutual_information(state, wires_a, wires_b, wires_c, base=2):
    """
    Co-information (I3) for three subsystems A,B,C:
      I3 = H(A) + H(B) + H(C)
           – H(A,B) – H(A,C) – H(B,C)
           + H(A,B,C).
    """
    H_a   = entropy(state, wires=wires_a, base=base)
    H_b   = entropy(state, wires=wires_b, base=base)
    H_c   = entropy(state, wires=wires_c, base=base)
    H_ab  = joint_entropy(state, wires_a, wires_b, base=base)
    H_ac  = joint_entropy(state, wires_a, wires_c, base=base)
    H_bc  = joint_entropy(state, wires_b, wires_c, base=base)
    H_abc = joint_entropy_n(state, [wires_a, wires_b, wires_c], base=base)
    return H_a + H_b + H_c - H_ab - H_ac - H_bc + H_abc
