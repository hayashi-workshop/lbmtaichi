# render.py
"""
Rendering class

This class renders vorticity field. Coloring
Inspired by the elegant LBM approach in LBM_Taichi (https://github.com/hietwll/LBM_Taichi/tree/master) by Dr. Zhuo Wang.
"""

import taichi as ti
import taichi.math as tm

import numpy as np

@ti.data_oriented
class FluidRenderer:
    def __init__(self, lbm, vmin=-0.02, vmax=0.02):
        self.dim = lbm.dim
        self.nx = lbm.nx
        self.ny = lbm.ny
        self.nz = lbm.nz if self.dim == 3 else 1
        self.vmin = vmin
        self.vmax = vmax
        self.mode = "vorticity" # default: "vorticity" # "vorticity" or "velocity"
        self.obstacle_color = 0.2 # <-gray; set 1 for white color
        self.color_map = None
        w = self.nx + self.nz if self.dim == 3 else self.nx
        h = self.ny + self.nz if self.dim == 3 else self.ny
        self.img_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(w, h)) # set (w, h) pix window
        if self.dim == 2: # vorticity field
            self.vor_field = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        else:
            self.ti_axes = ti.ij if self.dim == 2 else ti.ijk
            self.vor_field = ti.Vector.field(self.dim, float)
            for d in range(self.dim):
                ti.root.dense(self.ti_axes, lbm.nd).place(self.vor_field.get_scalar_field(d))
        self.window = ti.ui.Window("Taichi LBM Solver", (w, h), vsync=False)
        self.canvas = self.window.get_canvas()

    def render(self, lbm, mode=None):
        if mode: self.mode = mode
        if self.mode == "vorticity":
            self.color_map = self._apply_colormap_vorticity
            self._compute_vorticity(lbm)
            if self.dim == 2:
                self._render_kernel_vorticity(lbm)
            else:
                self._render_3d_slices_vorticity(lbm)
        else:
            self.color_map = self._apply_colormap_velocity
            if self.dim == 2:
                self._render_kernel_velocity(lbm)
            else:
                self._render_3d_slices_velocity(lbm)
        self.canvas.set_image(self.img_buffer)
        self.window.show()

    @ti.kernel
    def _compute_vorticity(self, lbm: ti.template()):
        if ti.static(self.dim == 2):
            for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
                self.vor_field[i, j] = (lbm.vel[i+1, j].y - lbm.vel[i-1, j].y) - \
                                       (lbm.vel[i, j+1].x - lbm.vel[i, j-1].x)
        else:
            for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
                dvz_dy = (lbm.vel[i, j+1, k].z - lbm.vel[i, j-1, k].z) * 0.5
                dvy_dz = (lbm.vel[i, j, k+1].y - lbm.vel[i, j, k-1].y) * 0.5
                dvx_dz = (lbm.vel[i, j, k+1].x - lbm.vel[i, j, k-1].x) * 0.5
                dvz_dx = (lbm.vel[i+1, j, k].z - lbm.vel[i-1, j, k].z) * 0.5
                dvy_dx = (lbm.vel[i+1, j, k].y - lbm.vel[i-1, j, k].y) * 0.5
                dvx_dy = (lbm.vel[i, j+1, k].x - lbm.vel[i, j-1, k].x) * 0.5
                
                self.vor_field[i, j, k] = ti.Vector([dvz_dy - dvy_dz, 
                                                     dvx_dz - dvz_dx, 
                                                     dvy_dx - dvx_dy])

    @ti.func
    def _render_obstacle(self, lbm: ti.template(), I):
        if lbm.mask[I] > 0:
            self.img_buffer[I] = tm.vec3(self.obstacle_color)


    @ti.kernel
    def _render_kernel_vorticity(self, lbm: ti.template()):
        for I in ti.grouped(self.vor_field):
            self.img_buffer[I] = self.color_map( self.vor_field[I] )
            self._render_obstacle(lbm, I)

    @ti.kernel
    def _render_kernel_velocity(self, lbm: ti.template()):
        for I in ti.grouped(lbm.vel):
            self.img_buffer[I] = self.color_map( lbm.vel[I].norm() )
            self._render_obstacle(lbm, I)

    @ti.func
    def _render_3d_obstacle(self, lbm: ti.template(), I, II):
        if lbm.mask[II] > 0:
            self.img_buffer[I] = tm.vec3(self.obstacle_color)

    @ti.kernel
    def _render_3d_slices_vorticity(self, lbm: ti.template()):
        cx, cy, cz = self.nx // 2, self.ny // 2, self.nz // 2
        for I in ti.grouped(self.img_buffer):
            i, j = I[0], I[1]
            val = 0.0
            if i < self.nx and j < self.ny: # lower - left panel (x, y)
                II = (i, j, cz)
                val = self.vor_field[II].z
                self.img_buffer[I] = self.color_map(val)
                self._render_3d_obstacle(lbm, I, II)                
            elif i >= self.nx and j < self.ny: # lower - right panel (z, y)
                II = (cx, j, i - self.nx)
                val = self.vor_field[II].x
                self.img_buffer[I] = self.color_map(val)
                self._render_3d_obstacle(lbm, I, II)                
            elif i < self.nx and j >= self.ny: # top - left panel (x, z)
                II = (i, cy, j - self.ny)
                val = self.vor_field[II].y
                self.img_buffer[I] = self.color_map(val)
                self._render_3d_obstacle(lbm, I, II)                

    @ti.kernel
    def _render_3d_slices_velocity(self, lbm: ti.template()):
        cx, cy, cz = self.nx // 2, self.ny // 2, self.nz // 2
        for I in ti.grouped(self.img_buffer):
            i, j = I[0], I[1]
            val = 0.0
            if i < self.nx and j < self.ny: # lower - left panel (x, y)
                II = (i, j, cz)
                val = lbm.vel[II].norm()
                self.img_buffer[I] = self.color_map(val)
                self._render_3d_obstacle(lbm, I, II)                
            elif i >= self.nx and j < self.ny: # lower - right panel (z, y)
                II = (cx, j, i - self.nx)
                val = lbm.vel[II].norm()
                self.img_buffer[I] = self.color_map(val)
                self._render_3d_obstacle(lbm, I, II)                
            elif i < self.nx and j >= self.ny: # top - left panel (x, z)
                II = (i, cy, j - self.ny)
                val = lbm.vel[II].norm()
                self.img_buffer[I] = self.color_map(val)
                self._render_3d_obstacle(lbm, I, II)                
            

    @ti.func
    def _apply_colormap_vorticity(self, v):
        t_v = (v - self.vmin) / (self.vmax - self.vmin)
        t_v = tm.clamp(t_v, 0.0, 1.0)
        # (yellow->orange->black->green->cyan)
        color = tm.vec3(0.0)
        if t_v < 0.25: # yellow (1,1,0) -> orange (0.95, 0.49, 0.02)
            color = tm.mix(tm.vec3(1.0, 1.0, 0.0), tm.vec3(0.953, 0.490, 0.016), t_v * 4)
        elif t_v < 0.5: # orange -> black (0,0,0)
            color = tm.mix(tm.vec3(0.953, 0.490, 0.016), tm.vec3(0.0, 0.0, 0.0), (t_v - 0.25) * 4)
        elif t_v < 0.75: # black -> green (0.18, 0.98, 0.53)
            color = tm.mix(tm.vec3(0.0, 0.0, 0.0), tm.vec3(0.176, 0.976, 0.529), (t_v - 0.5) * 4)
        else: # green -> cyan (0,1,1)
            color = tm.mix(tm.vec3(0.176, 0.976, 0.529), tm.vec3(0.0, 1.0, 1.0), (t_v - 0.75) * 4)            

        return color
    
    @ti.func
    def _apply_colormap_velocity(self, v):
        t_v = (v) / (self.vmax) # t_v = (v - self.vmin) / (self.vmax - self.vmin)
        t_v = tm.clamp(t_v, 0.0, 1.0)
        # plasma (dark purple->purple->red-orange->yellow)
        color = tm.vec3(0.0)
        if t_v < 0.33:
            color = tm.mix(tm.vec3(0.05, 0.02, 0.53), tm.vec3(0.49, 0.01, 0.56), t_v * 3)
        elif t_v < 0.66:
            color = tm.mix(tm.vec3(0.49, 0.01, 0.56), tm.vec3(0.88, 0.26, 0.2), (t_v - 0.33) * 3)
        else:
            color = tm.mix(tm.vec3(0.88, 0.26, 0.2), tm.vec3(0.94, 0.9, 0.13), (t_v - 0.66) * 3)

        return color