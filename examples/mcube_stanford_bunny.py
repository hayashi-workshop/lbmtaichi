# stanford_bunny_scene3d.py

# this class uses trimesh to handle mesh data
# install the following if not yet inplemented
#pip install 'trimesh[easy]'
#pip install networkx
import trimesh

import taichi as ti
import taichi.math as tm

from lb_solver.lbm_lib import lbm_skelton
from lb_utils.bc_kernel import BoundaryManager

from lb_utils.lbm_utils import save_vtk
from lb_utils.lbm_utils import PerformanceMonitor
from lb_utils.render import FluidRenderer

from lb_utils.marching_cube import MarchingCube

from examples.obstacle_stanford_bunny import ObstacleManager

ti.init(arch=ti.gpu, default_fp=ti.f32)

@ti.kernel
def compute_Q_criterion(Q: ti.template(), vel: ti.template()):
    nx, ny, nz = vel.shape
    for I in ti.grouped(vel):
        im = tm.clamp(I[0]-1, 0, nx-1)
        jm = tm.clamp(I[1]-1, 0, ny-1)
        km = tm.clamp(I[2]-1, 0, nz-1)
        ip = tm.clamp(I[0]+1, 0, nx-1)
        jp = tm.clamp(I[1]+1, 0, ny-1)
        kp = tm.clamp(I[2]+1, 0, nz-1)
        Ixp = ti.Vector([ip, I[1], I[2]])
        Ixm = ti.Vector([im, I[1], I[2]])
        Iyp = ti.Vector([I[0], jp, I[2]])
        Iym = ti.Vector([I[0], jm, I[2]])
        Izp = ti.Vector([I[0], I[1], kp])
        Izm = ti.Vector([I[0], I[1], km])
        du_dx = (vel[Ixp].x - vel[Ixm].x) * 0.5
        du_dy = (vel[Iyp].x - vel[Iym].x) * 0.5
        du_dz = (vel[Izp].x - vel[Izm].x) * 0.5
        dv_dx = (vel[Ixp].y - vel[Ixm].y) * 0.5
        dv_dy = (vel[Iyp].y - vel[Iym].y) * 0.5
        dv_dz = (vel[Izp].y - vel[Izm].y) * 0.5
        dw_dx = (vel[Ixp].z - vel[Ixm].z) * 0.5
        dw_dy = (vel[Iyp].z - vel[Iym].z) * 0.5
        dw_dz = (vel[Izp].z - vel[Izm].z) * 0.5
        Q[I] = -0.5 * ( du_dx**2 + dv_dy**2 + dw_dz**2 + 2*du_dy*dv_dx + 2*du_dz*dw_dx + 2*dv_dz*dw_dy )

nd = (241, 121, 121)
length_scale = 60    # size of bunny 
offset = (50, 0, 30) # offset (measured for lower-left corner)

bunny = ObstacleManager(nd=nd, offset=offset, length_scale=length_scale)

u, Re = 0.1, 40000.0
nu = u*length_scale/Re; omega = 1/(3*nu + 0.5)

from lb_solver.d3q27_Cumulant_kernel import ModelConfig
lbm = lbm_skelton(nd, config := ModelConfig(), omega)

bc_manager = BoundaryManager(nd, [0, 1, 2, 2, 2, 2], [ [u,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0] ])
bunny.apply_to_mask(lbm)

# isosurface of Q # 
isovalues = [-0.00035, 0.00035]
Q_field = ti.field(dtype=ti.f32, shape=lbm.nd)
mcube = MarchingCube(isovalues, lbm.nd, report=True)
import os
output_dir = "./output/mcube_sb"
os.makedirs(output_dir, exist_ok=True)

mlups_monitor = PerformanceMonitor(nd)

renderer = FluidRenderer(lbm, vmin=-u*0.5, vmax=u*0.5) 

step, step_end = 0, 10000 # |--- run your simulation ---> #
while renderer.window.running and step < step_end:
    for _ in range(10):
        f_pre, f_post = lbm.swap(step)
        config.col_stream_core(lbm, f_pre, f_post)
        bc_manager.apply_bc(lbm, config, f_pre, f_post, bunny)
        step += 1

    renderer.render(lbm)
    mlups_monitor.update(step)

    if step % 100 == 0:
        compute_Q_criterion(Q_field, lbm.vel)
        mcube.generate_mesh_taichi(Q_field) # generate mesh in Taichi scope
        meshes = mcube.mesh_taichi_to_np(taubin_iter=30) # transfer mesh from Taichi to Python scope; mesh smoothing is applied (taubin_iter=20)
        color_name = ["blue", "red"]
        for surface_idx in range(len(meshes)):
            meshes[surface_idx].export(f"{output_dir}/q_surface_Taichi_{color_name[surface_idx]}_{step}.ply") # export mesh 
