# lbm_utils.py
"""
Utility functions for LB simulations

Coloring  
Inspired by the elegant LBM approach in LBM_Taichi (https://github.com/hietwll/LBM_Taichi/tree/master) by Dr. Zhuo Wang.
"""

import taichi as ti
import taichi.math as tm

import numpy as np
import os
import time
import importlib

from pyevtk.hl import gridToVTK


def get_model_config(nd, collision_model, kernel_file=None):
    def import_helper(module_path):
        try:
            module = importlib.import_module(module_path)
            return getattr(module, "ModelConfig")
        except ImportError:
            raise ImportError(f"Module {module_path} is not found.")

    if kernel_file == None:
        dim = len(nd)
        Q = 9 if dim == 2 else 27
        module_path = f"lb_solver.d{dim}q{Q}_{collision_model}_kernel"
    else:
        module_path = f"lb_solver.{kernel_file.removesuffix('.py')}"

    return import_helper(module_path)


class PerformanceMonitor:
    def __init__(self, num_nodes, window_size=10):
        self.num_nodes = np.prod(num_nodes) # nx * ny * nz
        self.window_size = window_size      # monitoring window (forget old record)
        self.start_time = time.time()       # initial time
        self.prev_time = self.start_time    # previous time
        self.last_step = 0                  # last step
        self.mlups_history = []             # store mlups to calculate average

    def update(self, current_step):
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt >= 1.0:
            step_diff = current_step - self.last_step
            # MLUPS = (lattice points * steps) / (eplsed time * 1e6)
            mlups = (self.num_nodes * step_diff) / (dt * 1e6)
            
            self.mlups_history.append(mlups)
            if len(self.mlups_history) > self.window_size:
                self.mlups_history.pop(0)
                
            print(f"step: {current_step} | MLUPS (current): {mlups:.2f} | avg: {sum(self.mlups_history)/len(self.mlups_history):.2f}")
            
            self.prev_time = current_time
            self.last_step = current_step



def save_vtk(lbm: ti.template(), step, output_dir=None):
    if output_dir == None:
        output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    u_np = lbm.vel.get_scalar_field(0).to_numpy()
    v_np = lbm.vel.get_scalar_field(1).to_numpy()
    x, y = np.arange(lbm.nx), np.arange(lbm.ny)
    if lbm.dim == 2:
        z = np.zeros(1)
        w_np = np.zeros_like(u_np)
        gridToVTK(
            output_dir + f"/step_{step:06d}", 
            x, y, z, 
            pointData={"velocity": (u_np, v_np, w_np)}
        )
    else:
        w_np = lbm.vel.get_scalar_field(2).to_numpy()
        z = np.arange(lbm.nz)
        gridToVTK(
            output_dir + f"/step_{step:06d}", 
            x, y, z, 
            pointData={"velocity": (u_np, v_np, w_np)},
        )
