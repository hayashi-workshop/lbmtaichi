# mlups_subproc.py

import taichi as ti
import taichi.math as tm

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='cpu', choices=['cpu', 'gpu'])
args = parser.parse_args()

arch_type = ti.gpu if args.arch == 'gpu' else ti.cpu
ti.init(arch=arch_type, default_fp=ti.f32)

# - - - - - - - - - - - - - - - - - - - - - - #
from lb_solver.lbm_lib import lbm_skelton
from lb_utils.bc_kernel import BoundaryManager

L = 100
u, Re = 0.01, 1000.0
nu = u*2*L/Re; omega = 1/(3*nu + 0.5)

nd = (L, L, L)

from lb_solver.d3q27_Cumulant_kernel import ModelConfig
lbm = lbm_skelton(nd, config := ModelConfig(), omega)

bc_manager = BoundaryManager(nd, [2, 2, 2, 2, 2, 2], [ [0,0,0], [0,0,0], [0,0,0], [u,0,0], [0,0,0], [0,0,0] ])

def run_simulation(stepEnd):
    step, step_end = 0, stepEnd 
    while step < step_end:
        f_pre, f_post = lbm.swap(step) 
        config.col_stream_core(lbm, f_pre, f_post)
        bc_manager.apply_bc(lbm, config, f_pre, f_post)
        step += 1


stepEnd = 1000

run_simulation(stepEnd) # warmup to eliminate slow resutls at early stage
ti.sync() # syncronization [https://docs.taichi-lang.org/docs/master/kernel_sync]

# ---> mlups measurement
start_time = time.time()
run_simulation(stepEnd)
ti.sync()
end_time = time.time()
# <--- measurement ends

elapsed = end_time - start_time
total_updates = L**len(nd) * stepEnd 
mlups = (total_updates / elapsed) / 1e6

print(f"MLUPS:{mlups:.2f}")