# obstacle_JCprob1.py
"""
Obstacle manager for Taichi-accelerated LBM

This class apply boundary conditions to macro and distribution functions.
Inspired by the elegant LBM approach in LBM_Taichi (https://github.com/hietwll/LBM_Taichi/tree/master) by Dr. Zhuo Wang.
"""

import taichi as ti
import taichi.math as tm

@ti.data_oriented
class ObstacleManager:
    def __init__(self, center_list, radius, nd):
        self.dim = len(center_list)
        self.radius = radius
        self.center = ti.Vector.field(self.dim, dtype=ti.f32, shape=())
        self.center[None] = ti.Vector(center_list)
        self.theta0 = ti.field(dtype=ti.f32, shape=())
        self.theta0[None] = 0.0
        #self.phi = ti.field(float, shape=nd) # SDF for future extension
        self.ti_axes = ti.ij if self.dim == 2 else ti.ijk
        self.n = ti.Vector.field(self.dim, float) # unit normal
        for d in range(self.dim):
            ti.root.dense(self.ti_axes, nd).place(self.n.get_scalar_field(d))        

    def move_center(self, new_pos):
        self.center[None] = new_pos

    def apply_to_mask(self, lbm): # wrapper function for private kernel _generate_mask_kernel
        self._generate_mask_kernel(lbm.mask)

    #Johansen and Phillip Colella, JCP 1998
    @ti.kernel
    def _generate_mask_kernel(self, mask: ti.template()):
        mask.fill(0)
        for I in ti.grouped(mask):
            distance = ti.cast(I, ti.f32) - self.center[None]
            d2 = distance.norm_sqr()
            eps = 1e-6
            r  = tm.sqrt(d2) + eps
            theta = ti.atan2(distance[1], distance[0]) + self.theta0[None]
            dr = r - self.radius*((1 + 0.5*tm.cos(6.0*theta)))*1.5 
            mask[I] = (dr < 0.0)
            nx = tm.cos(theta) - tm.sin(theta) * 0.9*tm.sin(6.0*theta)/r
            ny = tm.sin(theta) + tm.cos(theta) * 0.9*tm.sin(6.0*theta)/r
            self.n[I] = ti.Vector([nx,ny])
