#lbm_lib.py
"""
Skelton of LB model

This class implements the basic variables.
Inspired by the elegant LBM approach in LBM_Taichi (https://github.com/hietwll/LBM_Taichi/tree/master) by Dr. Zhuo Wang.
"""

# dimension = 2 (D2Q9) or 3 (D3Q27)
    # omega[1]  : shear -> omega
    # omega[2]  : bulk  -> omP
    # omega[3]  : for sum  of combinations of 120, 102, 210, 012, 201, 021 in D3Q27; for 12, 21 in D2Q9 (om3)
    # omega[4]  : for diff of combinations of 120, 102, 210, 012, 201, 021 in D3Q27; for 12, 21 in D2Q9 (om3)
    # omega[5]  : for 111
    # omega[6]  : for 220, 202, 022; for 22 in D2Q9 (om4)
    # omega[7]  : for 220, 202, 022; for 22 in D2Q9 (om4)
    # omega[8]  : for 211, 121, 112
    # omega[9]  : for 221, 212, 122
    # omega[10] : for 222

import taichi as ti
import taichi.math as tm

@ti.data_oriented
class lbm_skelton:
    def __init__(
        self,
        nd,  # domain size (nx, ny) or (nx, ny, nz)
        model_config,
        omega1=None, # relaxation parameter for shear (omega_{1})
        SoA = True # default SoA layout; False->AoS
    ):
        self.nd = nd
        self.dim = len(nd)
        if self.dim == 2:
            self.nx, self.ny = nd
            nx, ny = nd
        else:
            self.nx, self.ny, self.nz = nd
            nx, ny, nz = nd
        self.Q = 3 ** self.dim
        self.omega = ti.field(dtype=float, shape=11) # omega[0] is not used.
        for i in range(1, 11):
            self.omega[i] = 1.0
        self.omega[1] = 1.0 if omega1 == None else omega1
        self.rho = ti.field(float, shape=self.nd)
        if SoA:
            # - - - - - SoA memory layout - - - - - - - - - #
            self.ti_axes = ti.ij if self.dim == 2 else ti.ijk
            self.vel = ti.Vector.field(self.dim, float)
            for d in range(self.dim):
                ti.root.dense(self.ti_axes, self.nd).place(self.vel.get_scalar_field(d))

            self.f_old = ti.Vector.field(self.Q, float)
            self.f_new = ti.Vector.field(self.Q, float)

            block_old = ti.root.dense(self.ti_axes, self.nd)
            for q in range(self.Q): 
                block_old.place(self.f_old.get_scalar_field(q))

            block_new = ti.root.dense(self.ti_axes, self.nd)
            for q in range(self.Q): # allocation must be done with another for loop
                block_new.place(self.f_new.get_scalar_field(q))
        else:
            # - - - - - AoS memory layout - - - - - - - - - #
            self.vel   = ti.Vector.field(self.dim, float, shape=self.nd)
            self.f_old = ti.Vector.field(self.Q, float, shape=self.nd)
            self.f_new = ti.Vector.field(self.Q, float, shape=self.nd)
        # - - - - - - - - - - - - - - - - - - - - - - - #
        self.mask = ti.field(float, shape=self.nd)
        # - - - - - - - - - - - - - - - - - - - - - - - #
        self.init(model_config)


    def init(self, config): # wrapper for Taichi kernel
        self._init_kernel(config)


    @ti.kernel
    def _init_kernel(self, config: ti.template()):
        self.rho.fill(1.0 - config.density_shift)
        self.vel.fill(0)
        self.mask.fill(0)
        for I in ti.grouped(self.rho): # I = (i, j) or (i, j, k)
            self.f_old[I] = self.f_new[I] = config.f_eq(self, I) 


    def swap(self, step):
        f_post = self.f_new if step % 2 == 0 else self.f_old
        f_pre  = self.f_old if step % 2 == 0 else self.f_new
        return f_pre, f_post
 