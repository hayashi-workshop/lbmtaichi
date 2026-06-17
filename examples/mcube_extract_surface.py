# extract_surface.py
#
# extract isocontour meshes within Taichi scope
#
# this class uses trimesh to handle mesh data
# install the following if not yet inplemented
#pip install 'trimesh[easy]'
#pip install networkx
#

import taichi as ti

import numpy as np
import trimesh

from lb_utils.marching_cube import MarchingCube

ti.init(arch=ti.gpu, default_fp=ti.f32)

on_taichi = True # True: Taichi paralle; False: single-cpu serial

import os
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

N = 101
nd = (N, N, N)

# extract 10 isosurfaces #
isovalues = np.linspace(-25., 25., 10)

mcube = MarchingCube(isovalues, nd, report=True)

x = np.linspace(0, nd[0]-1, nd[0])
y = np.linspace(0, nd[1]-1, nd[1])
z = np.linspace(0, nd[2]-1, nd[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# spherical surface
r = N*0.25

# Python scope
S = np.sqrt((X-(nd[0]-1)/2)**2 + (Y-(nd[1]-1)/2)**2 + (Z-(nd[2]-1)/2)**2) - r

# Taichi scope (sent from Python scope)
S_field = ti.field(dtype=ti.f32, shape=nd)
S_field.from_numpy(S.astype(np.float32))

from pyevtk.hl import gridToVTK

x = np.arange(nd[0])
y = np.arange(nd[1])
z = np.arange(nd[2])

gridToVTK(f"{output_dir}/S", x, y, z, pointData={"S": (S)}, )

#
# generate mesh vertices using Marching Cube
#
if on_taichi: # taichi parallel
    mcube.generate_mesh_taichi(S_field)
    print(f"[on taichi] extracted")

    meshes = mcube.mesh_taichi_to_np()
    print(f"[on taichi] transferred")

    for surface_idx in range(len(meshes)):
        meshes[surface_idx].export(f"{output_dir}/ext_surf_Taichi_{surface_idx}.ply") # export mesh 

    print(f"[taichi] exported")
else: # cpu serial
    all_vertices = mcube.generate_mesh_np(S) # data shape (N*3, 3)
    print(f"[serial cpu] extracted")

    for i, vertices in enumerate(all_vertices):
        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((0, 3))
            faces = np.zeros((0, 3), dtype=int)
        else:
            faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh     = trimesh.Trimesh(vertices=vertices, faces=faces) # instantiate trimesh with vertices and faces
        mesh.merge_vertices() # eliminate duplicating vertices
        mesh.fix_normals() # smoothing face normal

        mesh.export(f"{output_dir}/ext_surf_serial_{i}.ply") # export mesh 
    print(f"[serial cpu] exported")
