# common_utils.py

import sympy as sp
import itertools
import math

def calculate_weight(vec):
    w_base = {0: sp.Rational(2, 3), 1: sp.Rational(1, 6), -1: sp.Rational(1, 6)}
    weight = 1
    for component in vec:
        weight *= w_base[component]
    return weight

def create_vectors(dim):
    all_vectors = list(itertools.product([-1, 0, 1], repeat=dim))

    vectors = []

    if dim == 2: # D2Q9
#     6  2  5
#      \ | /
#     3- 0 -1
#      / | \
#     7  4  8 
        def lbm_2d_sort_key(vec):
            length_sq = sum(c**2 for c in vec)
            if length_sq == 0: return (0, -100)
            angle = math.atan2(vec[1], vec[0])
            if angle < 0: angle += 2 * math.pi
            return (length_sq, angle)

        sorted_pairs = sorted([(v, calculate_weight(v)) for v in all_vectors], key=lambda x: lbm_2d_sort_key(x[0]))
        vectors = [p[0] for p in sorted_pairs]

    elif dim == 3: # D3Q27
#     2- 0 -1 # <-anti-direction pairing->
        groups = {}
        for v in all_vectors:
            w = calculate_weight(v)
            groups.setdefault(w, []).append(v)

        vectors.append((0, 0, 0)) # put center velocity first
        
        for w in sorted(groups.keys(), reverse=True):
            if w == sp.Rational(8, 27): continue # skip center
            unvisited = list(groups[w])
            unvisited.sort(key=lambda x: (tuple(abs(c) for c in x), x), reverse=True)
            
            while unvisited:
                v1 = unvisited.pop(0)
                v2 = tuple(-c for c in v1)
                unvisited.remove(v2)
                vectors.append(v1)
                vectors.append(v2)

    return vectors


def create_feq_list(dim, rho, vel_syms, vectors, weights, trunc=2):
    feq_list = []    
    cs2 = sp.Rational(1, 3)
    u_sq_sum = sum(u_i**2 for u_i in vel_syms)
    for idx, vec in enumerate(vectors):
        w_v = weights[idx]
        c_dot_u = sum(cx * u_i for cx, u_i in zip(vec, vel_syms))
        if dim == 2 or trunc == 2: # standard LBM approximated f_eq
            feq = w_v * rho * (1 + c_dot_u/cs2 + (c_dot_u**2)/(2 * cs2**2) - u_sq_sum/(2 * cs2))
        else: # more accurate Gauss-Hermite to retain u^2 terms in m_eq; otherwise symmetry of velocity (c) vectors elimate u^2 terms in m_eq resulting from M f_eq
            term1 = c_dot_u / cs2
            term2 = (c_dot_u**2) / (2 * cs2**2) - u_sq_sum / (2 * cs2)
            term3 = (c_dot_u**3) / (6 * cs2**3) - (c_dot_u * u_sq_sum) / (2 * cs2**2)
            term4 = (c_dot_u**4) / (24 * cs2**4) - ((c_dot_u**2) * u_sq_sum) / (4 * cs2**3) + (u_sq_sum**2) / (8 * cs2**2)            
            feq = w_v * rho * (1 + term1 + term2 + term3 + term4)
        feq_list.append(feq)
    return feq_list


def create_trans_matrix(moment_orders, vectors):
    M_list = []
    for order in moment_orders:
        row = []
        for vec in vectors:
            term = 1
            for c, alpha in zip(vec, order):
                term *= (c ** alpha)
            row.append(term)
        M_list.append(row)
    M = sp.Matrix(M_list)
    return M, M.inv()

