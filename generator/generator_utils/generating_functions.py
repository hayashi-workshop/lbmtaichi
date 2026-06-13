# generating_functions.py

# the functions defined in this script are used in #
#                      cumulant_moment_exprs.ipynb #

import sympy as sp
from sympy import symbols, Eq
import itertools
import math

import sympy as sp

from generator_utils.common_utils import calculate_weight
from generator_utils.common_utils import create_vectors
from generator_utils.common_utils import create_trans_matrix

class CumulantGenerator():
    def __init__(self, dim=3):
        self.dim = dim
        self.X_list = sp.symbols(['X', 'Y', 'Z'][:dim])
        self.u_list = sp.symbols(['u', 'v', 'w'][:dim])
        self.K = sp.Function('K')(*(self.X_list))
        self.C = sp.Function('C')(*(self.X_list))
        self.X_vec = sp.Matrix(self.X_list)
        self.u_vec = sp.Matrix(self.u_list)
        self.C_sym = (self.X_vec.T @ self.u_vec)[0] + sp.log(self.K)

        self.alpha, self.beta, self.gamma = sp.symbols('alpha beta gamma', integer=True, positive=True)
        self.order_sym = (self.alpha, self.beta, self.gamma)
        diff_vars = []
        for var, n in zip(self.X_list, self.order_sym):
            diff_vars.append((var, n))

        self.gen_operator = sp.Derivative(self.C, *diff_vars)
        self.density_scaling = True

    def definition_of_generating_function(self):
        return Eq(self.C, self.C_sym)

    def definition_of_cumulant(self):
        lhs_def = sp.Symbol(rf'c_{{\alpha \beta \gamma}}', latex_name=rf'c_{{\alpha \beta \gamma}}')
        return Eq(lhs_def, self.gen_operator)

    def __call__(self, order, density_scaling=None):
        if density_scaling == None: density_scaling = self.density_scaling
        c_expr = self._generate_cumulant_from_central_moment(order, density_scaling=density_scaling)
        lhs = sp.Symbol(f'C_{{{"".join(map(str, order))}}}') if density_scaling else sp.Symbol(f'c_{{{"".join(map(str, order))}}}')
        return Eq(lhs, c_expr)

    def _generate_cumulant_from_central_moment(self, order_tuple, density_scaling):
        dim = self.dim
        if self.dim != len(order_tuple): 
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {len(order_tuple)}.")

        rho = sp.Symbol('rho') # density

        expr = self.C_sym # cumulant generating function in terms of K
        for var, order in zip(self.X_list, order_tuple):
            if order > 0:
                expr = expr.diff(var, order)
        
        def get_deriv_symbol(deriv_obj):
            # enforce kappa100=kappa010=kappa001=0 to suppress term divergence at high orders
            args = deriv_obj.args # get derivative order in Derivative expr, like (X, 2) for partial_{X}^{2} 

            orders = {var: 0 for var in self.X_list} # derivative order count; starting from X:0 Y:0 Z:0
            for arg in args[1:]:
                var, count = arg if isinstance(arg, (tuple, sp.Tuple)) else (arg, 1)            
                if var in orders:
                    orders[var] += count

            total = sum(orders.values())
            
            if total == 0: # 000
                return rho
            elif total == 1: # kappa100 -> 0
                return sp.Integer(0)
            else:
                suffix = "".join([str(orders[var]) for var in self.X_list])
                return sp.Symbol(f'kappa{suffix}')

        subs_rules = {d: get_deriv_symbol(d) for d in expr.atoms(sp.Derivative)} # dictionary for replacement of derivatives of K with kappa symbols        
        subs_rules.update({var: 0 for var in self.X_list}) # evaluate at X=0 (treatment for X dot u term)
        subs_rules[self.K] = rho # replacement rule: K at X=0 -> rho
        multiplyer = rho if density_scaling==True else sp.Integer(1) # density scaling C_{200} = rho c_{200} = kappa_{200}

        return ( expr.subs(subs_rules)*multiplyer ).simplify()




class MomentGenerator():
    def __init__(self, vectors, is_central_moment=None):
        self.is_central_moment = is_central_moment # None: raw moment, True: central moment
        self.vectors = vectors
        self.dim=len(vectors[0])

        all_orders = list(itertools.product((0, 1, 2), repeat=self.dim))
        self.moment_orders = sorted(all_orders, key=lambda x: (sum(x), -x[0]))

        self.M, self.M_inv = create_trans_matrix(self.moment_orders, self.vectors)

        self.mode = 'd3q27' if self.dim==3 else 'd2q9'

        self.X_list = sp.symbols(['X', 'Y', 'Z'][:self.dim])
        self.u_list = sp.symbols(['u', 'v', 'w'][:self.dim])

        self.i  = sp.Symbol('i', integer=True)
        self.N  = sp.Symbol('N', integer=True) 
        self.cx = sp.Function('c_x')(self.i)
        self.cy = sp.Function('c_y')(self.i)
        self.cz = sp.Function('c_z')(self.i)
        self.f  = sp.Function('f')(self.i)
        self.f_list = sp.symbols(f'f0:{len(vectors)}')

        self.c_list = [self.cx, self.cy, self.cz][:self.dim]

        self.u_names = ['u', 'v', 'w'][:self.dim] # velocity components
        self.u_list = sp.symbols(self.u_names)

        self.rho = sp.Symbol('rho') # density

        self.moment_sum = None

        self.alpha, self.beta, self.gamma = sp.symbols('alpha beta gamma', integer=True, positive=True)


    def derive_symbolic_moment_sum(self, order=None):
        if order is None:
            current_order = (self.alpha, self.beta, self.gamma)[:self.dim]
        else:
            if self.dim != len(order):
                raise ValueError(f"Dimension mismatch: expected {self.dim}, got {len(order)}.")
            current_order = order
        
        shift = sp.Integer(1) if self.is_central_moment else sp.Integer(0)
        
        term = self.f * sp.prod(
            [ (self.c_list[j] - self.u_list[j] * shift) ** current_order[j] for j in range(self.dim) ]
        ) 
        
        self.moment_sum = sp.Sum(term, (self.i, 0, self.N - 1))
        
        return self.moment_sum


    def __call__(self, order, vel_exprs=None):
        self.derive_symbolic_moment_sum(order)
        mom = self._expand_symbolic_sum_to_lattice(order, vel_exprs)
        if self.is_central_moment:
            lhs = sp.Symbol(f'\\kappa_{{{"".join(map(str, order))}}}')
        else:
            lhs = sp.Symbol(f'm_{{{"".join(map(str, order))}}}')
        return Eq(lhs, mom)

    def _expand_symbolic_sum_to_lattice(self, order, vel_exprs):        
        concrete_expr = 0
        term = self.moment_sum.function
        for i, v in enumerate(self.vectors):
            f_val = self.f_list[i]
            
            subs_dict = {
                sp.Function('c_x')(sp.Symbol('i', integer=True)): v[0],
                sp.Function('c_y')(sp.Symbol('i', integer=True)): v[1],
                sp.Function('c_z')(sp.Symbol('i', integer=True)): v[2] if self.dim == 3 else 0
            }
            subs_dict[sp.Function('f')(sp.Symbol('i', integer=True))] = f_val
            
            if vel_exprs == None:
              pass
            else:
              velocity_subs = dict(zip(self.u_list, vel_exprs))          
              subs_dict.update(velocity_subs)
            
            concrete_expr += sp.expand(term.subs(subs_dict))
            
        return concrete_expr#.simplify()


    def calculate_moment_with_lattice(self, order):
        # moment expression in terms of f
        if self.dim != len(order):
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {len(order)}.")
        
        f_list = sp.symbols(f'f0:{len(vectors)}') # f0, f1, ..., fn
        
        shift = sp.Integer(1) if self.is_central_moment else sp.Integer(0)
        moment_expr = 0 # sum (f_{i} * c_{ix}^alpha * c_{iy}^beta * c_{iz}^gamma)
        for i, v in enumerate(self.vectors):
            term = self.f_list[i] * sp.prod([ ( v[j] - self.u_list[j]*shift )**order[j] for j in range(self.dim)])
            moment_expr += term
            
        return moment_expr


    def replace_with_moments(self, expr, c_list, prefix='m'): # X**2 * Z**2 -> m202
        poly = sp.Poly(expr, c_list)
        terms = poly.terms()
        new_expr = 0
        for exponent_tuple, coeff in terms:
            suffix = "".join(map(str, exponent_tuple))
            sym = sp.Symbol(f"{prefix}{suffix}")
            new_expr += coeff * sym

        return new_expr

    def derive_central_moment_from_moment(self, order, expr_with_velocity=False):
        kappa = self._derive_central_moment_from_moment(order, expr_with_velocity)
        lhs = sp.Symbol(f'\\kappa_{{{"".join(map(str, order))}}}')
        return Eq(lhs, kappa)

    def _derive_central_moment_from_moment(self, order, expr_with_velocity):
        if self.dim != len(order):
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {len(order)}.")

        c_list = sp.symbols( ['cx', 'cy', 'cz'][:self.dim] ) # lattice velocity

        m000 = sp.Symbol('m' + '0'*self.dim)
        kappa_expr = sp.prod([(c_list[i] - self.u_list[i])**order[i] for i in range(self.dim)]) # construct kappa by definition
        kappa_expr = sp.expand(kappa_expr)
        kappa_expr = self.replace_with_moments(kappa_expr, c_list, prefix='m')

        u_subs = {self.u_list[i]: sp.Symbol(f'm{"0"*i}1{"0"*(self.dim-1-i)}') / m000 for i in range(self.dim)}
        kappa_expr = kappa_expr.xreplace(u_subs)

        if expr_with_velocity: 
            back_subs = {m000: self.rho}
            for i in range(self.dim):
                mom_name = f'm{"0"*i}1{"0"*(self.dim-1-i)}'
                back_subs[sp.Symbol(mom_name)] = self.rho * self.u_list[i]
            kappa_expr = kappa_expr.xreplace(back_subs)

        return kappa_expr

    def express_f_in_terms_of_moments(self):
        
        m_symbols = [sp.Symbol(f"m{''.join(map(str, o))}") for o in self.moment_orders]
        m_vec = sp.Matrix(m_symbols)
        
        f_vec = self.M_inv * m_vec
        
        return f_vec, f_vec.tolist()

  
    def _create_trans_matrix(self):
        M_list = []
        for order in self.moment_orders:
            row = []
            for vec in self.vectors:
                term = 1
                for c, alpha in zip(vec, order):
                    term *= (c ** alpha)
                row.append(term)
            M_list.append(row)
        M = sp.Matrix(M_list)
        return M, M.inv()