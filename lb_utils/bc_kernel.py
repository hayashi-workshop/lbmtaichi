# bc_kernel.py
"""
Guo's simple boundary condition

This class apply boundary conditions to macro and distribution functions.
Inspired by the elegant LBM approach in LBM_Taichi (https://github.com/hietwll/LBM_Taichi/tree/master) by Dr. Zhuo Wang.
"""

import taichi as ti
import taichi.math as tm

import numpy as np

from lb_utils.obstacle import ObstacleManager

@ti.data_oriented
class BoundaryManager:
    # [left,right,bottom,top] boundary conditions: 0 -> inflow ; 1 -> outflow ; 2 -> walls ; 3 -> object mask
    # if bc_type = 0, we need to specify the velocity in bc_value
    def __init__(self, nd, init_bc_type, init_bc_value):
        self.nd = nd
        self.dim = len(nd)
        if self.dim == 2:
            self.nx, self.ny = nd 
        else:
            self.nx, self.ny, self.nz = nd
        num_faces = 2 * self.dim
        self.bc_type = ti.field(int, num_faces)
        self.bc_type.from_numpy(np.array(init_bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(self.dim, float, shape=num_faces)
        self.bc_value.from_numpy(np.array(init_bc_value, dtype=np.float32))

    # wrapper function for private kernel _apply_bc
    def apply_bc(self, lbm, config, f_pre, f_post, obstacle=None): 
        if obstacle == None:
            self._apply_bc(lbm, config, f_pre, f_post)
        else:
            self._apply_bc_obs(lbm, config, f_pre, f_post, obstacle)

    @ti.func
    def _apply_bc_core(self, lbm, config, f_pre, f_post, bc_id, vbc, x_bc, x_nb, x_nb2):
        if bc_id == 0: # inflow
            lbm.vel[x_bc] = vbc #self.bc_value[dr]
            lbm.rho[x_bc] = 2.0 * lbm.rho[x_nb] - lbm.rho[x_nb2]
        elif bc_id == 1: # outflow
            lbm.vel[x_bc] = lbm.vel[x_nb]
            lbm.rho[x_bc] = lbm.rho[x_nb]
        elif bc_id == 2: # noslip wall
            lbm.vel[x_bc] = vbc # self.bc_value[dr] # load boundary velocity to adopt moving wall
            lbm.rho[x_bc] = lbm.rho[x_nb]
        elif bc_id == 3: # object mask
            lbm.vel[x_bc] = ti.Vector.zero(float, self.dim) # velocity is zero inside solid object
            lbm.rho[x_bc] = lbm.rho[x_nb]
        f_post[x_bc] = config.f_eq(lbm, x_bc) + f_post[x_nb] - config.f_eq(lbm, x_nb) # Guo: f_{bc} = f_{bc}^{(0)} + f_{nb}^{(1)}


    @ti.func
    def _apply_bc_outer_wall(self, lbm, config, f_pre, f_post, I):
        dr = 0
        di = ti.Vector.zero(ti.i32, self.dim)
        '''
        # the following algorithm faced a problem that wall bc (in left/right) is overwritten by in/outflow (in bottom/top). Whole outflow patch largely deteriorates stability. 
        for d in ti.static(range(self.dim)):
            if I[d] == 0:
                dr = d * 2
                di[d] = 1
            elif I[d] == lbm.nd[d] - 1:
                dr = d * 2 + 1
                di[d] = -1

        self._apply_bc_core(lbm, config, f_pre, f_post, self.bc_type[dr], dr, I, I + di, I + 2 * di)
        '''
        vbc = ti.Vector.zero(ti.f32, self.dim)

        bc_id = 2 # tentative
        is_wall_node = False
        for d in ti.static(range(self.dim)):
            if I[d] == 0 and self.bc_type[d * 2] == 2:
                dr = d * 2
                is_wall_node = True # lock
                vbc = self.bc_value[dr]
            elif I[d] == lbm.nd[d] - 1 and self.bc_type[d * 2 + 1] == 2:
                dr = d * 2 + 1
                is_wall_node = True # lock
                vbc = self.bc_value[dr]

        if not is_wall_node: # wall have priority over in/outlets
            for d in ti.static(range(self.dim)):
                if I[d] == 0:
                    dr = d * 2
                    bc_id = self.bc_type[dr]
                    vbc = self.bc_value[dr]
                elif I[d] == lbm.nd[d] - 1:
                    dr = d * 2 + 1
                    bc_id = self.bc_type[dr]
                    vbc = self.bc_value[dr]

        for d in ti.static(range(self.dim)):
            if I[d] == 0:
                di[d] =  1
            elif I[d] == lbm.nd[d] - 1:
                di[d] = -1

        self._apply_bc_core(lbm, config, f_pre, f_post, bc_id, vbc, I, I + di, I + 2 * di)


    @ti.func
    def _node_ruling(self, lbm: ti.template(), I):
        is_outer_wall = False
        for d in ti.static(range(self.dim)):
            if I[d] == 0 or I[d] == lbm.nd[d] - 1:
                is_outer_wall = True

        is_obj_node = (lbm.mask[I] == 1.0)

        return is_outer_wall, is_obj_node


    @ti.kernel # no obstacle in the field
    def _apply_bc(self, lbm: ti.template(), config: ti.template(), f_pre: ti.template(), f_post: ti.template()):
        for I in ti.grouped(lbm.mask): # I[0] = i, I[1] = j, I[2] = k
            is_outer_wall, is_obj_node = self._node_ruling(lbm, I)
            if not (is_outer_wall or is_obj_node):
                continue

            self._apply_bc_outer_wall(lbm, config, f_pre, f_post, I)

    @ti.kernel # with obstacle
    def _apply_bc_obs(self, lbm: ti.template(), config: ti.template(), f_pre: ti.template(), f_post: ti.template(), obs: ti.template()):
        for I in ti.grouped(lbm.mask): # I[0] = i, I[1] = j, I[2] = k
            is_outer_wall, is_obj_node = self._node_ruling(lbm, I)
            if not (is_outer_wall or is_obj_node):
                continue

            if is_obj_node:
                n = obs.n[I]
                di = self._get_di(n)
                self._apply_bc_core(lbm, config, f_pre, f_post, 3, ti.Vector.zero(ti.f32, self.dim), I, I + di, I + 2 * di)
            elif is_outer_wall:
                self._apply_bc_outer_wall(lbm, config, f_pre, f_post, I)

    @ti.func
    def _get_di(self, n):
        di = ti.Vector.zero(ti.i32, self.dim)
        for i in ti.static(range(self.dim)):
            di[i] = ti.cast(tm.sign(n[i]), ti.i32) # <- sign is better than round. 
        
        if di.norm_sqr() == 0: # fallback for accidental 0 set
            di[0] = 1

        return di

