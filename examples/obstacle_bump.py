# obstacle_bump.py
"""
Obstacle manager for Taichi-accelerated LBM

This class apply boundary conditions to macro and distribution functions.
Inspired by the elegant LBM approach in LBM_Taichi (https://github.com/hietwll/LBM_Taichi/tree/master) by Dr. Zhuo Wang.
"""

import taichi as ti
import taichi.math as tm

@ti.data_oriented
class ObstacleManager:
    def __init__(self, center_list, radius_list, nd):
        self.num_objects = len(center_list)
        self.dim = len(center_list[0])
        
        self.center = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num_objects)
        self.radius = ti.field(dtype=ti.f32, shape=self.num_objects)        
        for i in range(self.num_objects):
            self.center[i] = ti.Vector(center_list[i])
            self.radius[i] = radius_list[i]

        #self.phi = ti.field(float, shape=nd) # SDF for future extension
        self.ti_axes = ti.ij if self.dim == 2 else ti.ijk
        self.n = ti.Vector.field(self.dim, float) # unit normal
        for d in range(self.dim):
            ti.root.dense(self.ti_axes, nd).place(self.n.get_scalar_field(d))        

    def move_center(self, new_pos):
        pass # do not allow this
#        self.center[None] = new_pos

    def apply_to_mask(self, lbm): # wrapper function for private kernel _generate_mask_kernel
        self._generate_mask_kernel(lbm.mask)

    @ti.kernel
    def _generate_mask_kernel(self, mask: ti.template()):
        mask.fill(0)
        self.n.fill(0)
        for I in ti.grouped(mask):
            for i in range(self.num_objects):
                distance = ti.cast(I, ti.f32) - self.center[i]
                d2 = distance.norm_sqr()
                r2 = self.radius[i] ** 2
                if d2 <= r2:
                    mask[I] = 1.0
                    r = tm.sqrt(d2) + 1e-6
                    self.n[I] = distance / r