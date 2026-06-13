# kernel_sample.py

from generator.cumulant_generator import run_generator

collision_model = "MRT"
dimension       = 2
omega           = 1.8

run_generator(collision_model=collision_model, dimension=dimension, omega_config_input=omega, output_filename="samples/generated_kernel.py", target_directory="./")
