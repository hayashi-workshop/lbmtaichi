# mlups_main.py

import subprocess
import os
import sys
import matplotlib.pyplot as plt

def run_simulation(arch):
    print(f"Running cavity flow simulation on {arch.upper()}...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    subproc_path = os.path.join(base_dir, "mlups_subproc.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    result = subprocess.run(
        [sys.executable, subproc_path, "--arch", arch], 
        capture_output=True,
        text=True,
        check=True,
        env=env
    )

    mlups = 0.0
    for line in result.stdout.splitlines():
        if line.startswith("MLUPS:"):
            mlups = float(line.split(":")[1])
            break
            
    return mlups

def main():
    cpu_mlups = run_simulation("cpu")
    gpu_mlups = run_simulation("gpu")
    
    print(f"\n[Results] CPU: {cpu_mlups} MLUPS / GPU: {gpu_mlups} MLUPS")
    
    backends = ['CPU', 'GPU']
    mlups_values = [cpu_mlups, gpu_mlups]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(backends, mlups_values, color=['cyan', 'magenta'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}\nMLUPS', 
                 ha='center', va='bottom', fontweight='bold')
                 
    plt.title('D3Q27 Cumulant LBM 100^3 on Apple M2 (8core cpu/8core gpu)')
    plt.ylabel('Performance (MLUPS)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('img/benchmark_mlups.png', dpi=300)

if __name__ == "__main__":
    main()
