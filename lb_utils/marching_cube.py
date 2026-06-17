# marching_cube.py
#
# extract isocontour meshes within Taichi scope
#
# this class uses trimesh to handle mesh data
# install the following if not yet inplemented
#pip install 'trimesh[easy]'
#pip install networkx
#
# Lorensen, W. E.; Cline, Harvey E. (1987). "Marching cubes: A high resolution 3D surface construction algorithm". ACM SIGGRAPH Computer Graphics. 21 (4): 163–169.
#
# z   y
# ^  /
# | /
# - - > x
#
#     v3 - - (e11)- - - v7
#     /|              / |
#   (e5)            (e7)|
#   / (e1)          / (e3)
# v1 - - (e9) - - v5    |
#  |   |          |     |
#  |  v2 - - (e10)| - - v6
# (e0)/           (e2) /
#  |(e4)          | (e6)
#  |/             | /
# v0 - - (e8) - - v4
# 

import taichi as ti

import numpy as np
import trimesh

# look up table in Python scope
from lb_utils import tri_lookup as lut

@ti.data_oriented
class mcSurface: # marching cube surface. separated from MarchingCube to handle multiple surfaces in MarchingCube
    def __init__(self, max_vertices):
        self.max_vertices = max_vertices
        self.mesh_vertices_buf = ti.Vector.field(3, dtype=ti.f32, shape=(self.max_vertices)) # num_of_extract x [ max_vertices x 3 coordinates ]


@ti.data_oriented
class MarchingCube:
    def __init__(self, isovalues, nd, report=False, buffer_list=[1, 2]):
        self.report = report

        if not isinstance(isovalues, (list, np.ndarray)):
            isovalues = [isovalues]

        self.isovalues = isovalues
        self.num_of_extract = len(self.isovalues)
        self.nd = nd # tuple (nx, ny, nz)
        self.nx = self.nd[0]
        self.ny = self.nd[1]
        self.nz = self.nd[2]

        ###############################
        # send tables to Taichi scope #
        ###############################
        # look up table
        self.tri_table = ti.field(dtype=ti.i32, shape=(256, 16))
        self.tri_table.from_numpy(np.array(lut.triangles, dtype=np.int32))
        # edges connection table
        self.edges = ti.field(dtype=ti.i32, shape=(12, 2))
        self.edges.from_numpy(np.array(lut.edges, dtype=np.int32))
        # vertex shifter table
        self.v_table = ti.field(dtype=ti.i32, shape=(8, 3))
        self.v_table.from_numpy(lut.v_table)

        #################################
        # vertex buffer in Taichi scope #
        #################################
        self.max_vertices = self.nx * self.ny * self.nz * 5 * 3 # (5 triangles, 3 vertices for each)
        self.vertices = []
        for surface_idx in range(self.num_of_extract):
            self.vertices.append( mcSurface(self.max_vertices) )

        self.vertices_send_buffer = []
        self.buffer_size = []
        for b in buffer_list:
            self.buffer_size.append( (self.max_vertices // 5 ) * b )
        for i in range(len(self.buffer_size)):
            self.vertices_send_buffer.append( mcSurface( self.buffer_size[i]) )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # count triangles in cells                #self.cell_tri_count = ti.field(ti.int32, shape=self.nd)
        # [!NOTE] array size is 1 cell smaller since this array is "cells" consisting of nodes as vertices
        self.cell_tri_count = ti.field(ti.int32, shape=(self.nd[0]-1, self.nd[1]-1, self.nd[2]-1))
        
        # number of triangles in each surface
        self.total_triangles = ti.field(dtype=ti.int32, shape=self.num_of_extract) # 


    # --- --- --- --- --- --- --- --->
    #
    # for parallel run in taichi scope
    #
    def generate_mesh_taichi(self, S):
        for surface_idx, isovalue in enumerate(self.isovalues):
            self._count_triangles(S, isovalue)
            self._serial_prefix_sum(surface_idx)
            self._generate_mesh(S, isovalue, self.vertices[surface_idx])

        if self.report:
            print(f"[MarchingCube] {len(self.vertices)} surfaces --->")
            for surface_idx, isovalue in enumerate(self.isovalues):
                print(f"[MarchingCube] surface {surface_idx} : {self.total_triangles[surface_idx]} triangles")

    @ti.func
    def _get_vertex_coord(self, I, v_id):
        return I + ti.Vector([self.v_table[v_id, 0], self.v_table[v_id, 1], self.v_table[v_id, 2]])

    @ti.func
    def _interpolate_edge_taichi(self, I, e, isovalue, S):
        v1_id = self.edges[e, 0]
        v2_id = self.edges[e, 1]
        I1 = self._get_vertex_coord(I, v1_id)
        I2 = self._get_vertex_coord(I, v2_id)
        
        val1 = S[I1]
        val2 = S[I2]
        
        t = 0.5 # fallback for no intersection (same values at vertices)
        if abs(val2 - val1) > 1e-2:
            t = (isovalue - val1) / (val2 - val1) # gradient
            
        return I1 + t * (I2 - I1) # intersection 


    @ti.kernel
    def _count_triangles(self, S: ti.template(), isovalue: ti.f32):
        for i, j, k in ti.ndrange(self.nx - 1, self.ny - 1, self.nz - 1): # loop is for CELLs so ijk loop for nd - "1"
            I = ti.Vector([i,j,k])
            cube_index = 0
            for v_id in ti.static(range(8)):
                coord = self._get_vertex_coord(I, v_id)
                if S[coord] < isovalue:
                    cube_index |= (1 << v_id)
            
            count = 0
            for m in ti.static(range(0, 16, 3)): # from 0 to 15 [0 (1 2), 3 (4 5), 6 (7 8), 9 (10 11), 12 (13 14)), 15] 15 ends with -1
                if self.tri_table[cube_index, m] != -1:
                    count += 1

            self.cell_tri_count[I] = count # number of triangles in cell I


    @ti.kernel
    def _serial_prefix_sum(self, surface_idx: ti.i32):
        # this kernel is run in serial
        ti.loop_config(serialize=True)
        
        N_total = (self.nx - 1) * (self.ny - 1) * (self.nz - 1) # expanding indices into 1D

        acc = 0
        for i, j, k in ti.ndrange(self.nx - 1, self.ny - 1, self.nz - 1):
            current_val = self.cell_tri_count[i, j, k]
            self.cell_tri_count[i, j, k] = acc
            acc += current_val
        self.total_triangles[surface_idx] = acc


    @ti.kernel
    def _generate_mesh(self, S: ti.template(), isovalue: ti.f32, vertices: ti.template()):
        for i, j, k in ti.ndrange(self.nx - 1, self.ny - 1, self.nz - 1): # loop is for CELLs so ijk loop for nd - "1"
            I = ti.Vector([i,j,k])
            offset = self.cell_tri_count[I] # find my place for buffer write
            
            cube_index = 0 # count again
            for v_id in ti.static(range(8)):
                coord = self._get_vertex_coord(I, v_id)
                if S[coord] < isovalue:
                    cube_index |= (1 << v_id)
            
            for m in ti.static(range(0, 16, 3)): # from 0 to 15 [0 (1 2), 3 (4 5), 6 (7 8), 9 (10 11), 12 (13 14)), 15] 15 ends with -1
                e0 = self.tri_table[cube_index, m]
                if e0 != -1:
                    p0 = self._interpolate_edge_taichi(I, self.tri_table[cube_index, m  ], isovalue, S)
                    p1 = self._interpolate_edge_taichi(I, self.tri_table[cube_index, m+1], isovalue, S)
                    p2 = self._interpolate_edge_taichi(I, self.tri_table[cube_index, m+2], isovalue, S)
                    
                    idx = offset * 3 + (m // 3) * 3
                    vertices.mesh_vertices_buf[idx]     = p0
                    vertices.mesh_vertices_buf[idx + 1] = p1
                    vertices.mesh_vertices_buf[idx + 2] = p2


    # $$$ $$$ $$$ $$$ $$$ $$$ $$$ $$$
    # 
    # taichi - python communicator
    #
    # $$$ $$$ $$$ $$$ $$$ $$$ $$$ $$$
    def mesh_taichi_to_np(self, taubin_iter=0):
        mesh_np = []

        # Using vertices_send_buffer for small size meshes #
        for surface_idx in range(len(self.isovalues)):
            is_copied = False
            for i in range(len(self.buffer_size)):
                if self.total_triangles[surface_idx] * 3 < self.buffer_size[i] and is_copied == False:
                    self.copy_partial(self.vertices[surface_idx].mesh_vertices_buf, self.vertices_send_buffer[i].mesh_vertices_buf, self.total_triangles[surface_idx] * 3)
                    vertices_np = (self.vertices_send_buffer[i].mesh_vertices_buf).to_numpy() # bottle-neck
                    is_copied = True
            if is_copied == False:
                vertices_np = self.vertices[surface_idx].mesh_vertices_buf.to_numpy() # bottle-neck
            vertices_np = vertices_np[:self.total_triangles[surface_idx] * 3]
            faces_np    = np.arange(len(vertices_np)).reshape(-1, 3) # prepare face indexes [0,1,2], [3,4,5], ...
            mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np) # instantiate trimesh with vertices and faces
            #mesh.merge_vertices() # eliminate duplicating vertices
            if len(mesh.vertices) > 0:
                trimesh.smoothing.filter_taubin(mesh, iterations=taubin_iter)
            mesh.fix_normals() # smoothing face normal
            mesh_np.append(mesh)
        
        return mesh_np

    @ti.kernel
    def copy_partial(self, src: ti.template(), dest: ti.template(), count: ti.i32):
        for i in range(count):
            dest[i] = src[i]

    # --- --- --- --- --- --- --- --->
    #
    # for serial run in python scope
    #
    def _interpolate_edge_np(self, i, j, k, e, isovalue, S):
        v1_id, v2_id = e
        I1 = lut.get_vertex_coord(i, j, k, v1_id)
        I2 = lut.get_vertex_coord(i, j, k, v2_id)
        
        val1, val2 = S[I1], S[I2]
        
        if abs(val2 - val1) < 1e-6: # fallback for no intersection (same values at vertices)
            t = 0.5
        else:
            t = (isovalue - val1) / (val2 - val1) # gradient
            
        return tuple(I1[c] + t * (I2[c] - I1[c]) for c in range(3)) # intersection 


    # for serial run in python scope
    def generate_mesh_np(self, S):
        all_vertices = []
        nx, ny, nz = S.shape

        for isovalue in self.isovalues:
            mesh_vertices = []

            node_bit = (S < isovalue).astype(np.int8)

            for i in range(nx - 1): # [!NOTE] this "range" gives range from 0 to nx-2
                for j in range(ny - 1):
                    for k in range(nz - 1):
                        cube_index = 0
                        for v_id in range(8): # search all vertices
                            coord = lut.get_vertex_coord(i, j, k, v_id)
                            if node_bit[coord]:
                                cube_index |= (1 << v_id) # bit flag for vertices of S < isovalue
                        
                        triangles = lut.triangles[cube_index] # fetch triangles from look up table
                        
                         # from 0 to 15 [0 (1 2), 3 (4 5), 6 (7 8), 9 (10 11), 12 (13 14)), 15] 15 ends with -1
                        for m in range(0, 16, 3): # expand triangles from the fetched list including 15 integers
                            e0, e1, e2 = triangles[m], triangles[m+1], triangles[m+2]
                            if e0 == -1: break # padding ends
                            
                            p0 = self._interpolate_edge_np(i, j, k, lut.edges[e0], isovalue, S) # coordinates of triangle
                            p1 = self._interpolate_edge_np(i, j, k, lut.edges[e1], isovalue, S)
                            p2 = self._interpolate_edge_np(i, j, k, lut.edges[e2], isovalue, S)
                            
                            mesh_vertices.extend([p0, p1, p2]) # push coordinates
                            
            all_vertices.append(np.array(mesh_vertices))

        if self.report:
            print(f"[MarchingCube] {len(all_vertices)} surfaces --->")
            for surface_idx, isovalue in enumerate(self.isovalues):
                print(f"[MarchingCube] surface {surface_idx} : {len(all_vertices[surface_idx]) // 3} triangles")

        return all_vertices


    def save_as_ply(self, vertices, filename="mc_surface.ply"):
        num_verts = len(vertices)
        num_faces = num_verts // 3
        
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_verts}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n") 
            
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
            for i in range(0, num_verts, 3):
                f.write(f"3 {i} {i+1} {i+2}\n")

