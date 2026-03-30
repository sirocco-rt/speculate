# %%
####################################################################
# THIS IS TO CREATE A .PF FILE GRID OF UNIQUE PARAMETER COMBINATIONS
####################################################################

import os, sys
import numpy as np
import itertools

# directory of MCRT code (set by $PYTHON)
PYTHON_DIR = os.environ["PYTHON"]

# directory of helper python scripts
sys.path.append("{}/py_progs/".format(PYTHON_DIR))
import py_read_output as r

# Specify the path to the template.pf file
template_file = r.read_pf("template.pf")

# Items to change in the template file
# ------------------------------------
parameters = [
    'Disk.mdot(msol/yr)',
    'Wind.mdot(msol/yr)',
    'KWD.d(in_units_of_rstar)',
    'KWD.mdot_r_exponent',
    'KWD.acceleration_length(cm)',
    'KWD.acceleration_exponent'
    ]

# New parameter values for the grid
# ---------------------------------
grid = np.array([
    [3e-9,10e-9,30e-9], # Msol/yr
    [0.03,0.1,0.3], # Units of disk mdot
    [0.55,5.5,55], # dmin = 5.5, np.sqrt(30Rstar)
    [0,0.25,1], 
    [7.25182e+08,7.25182e+09,7.25182e+10], # 1Rstar 7.25182e+08
    [0.5,1.5,4.5]
    ])

# All the grid combinations
unique_combinations = []
for i in itertools.product(*grid):
    unique_combinations.append(list(i))

# Updating Wmdot to be in units of disk mdot
for i in range(len(unique_combinations)):
    unique_combinations[i][1] = unique_combinations[i][0] * unique_combinations[i][1]
    

# %%
####################################################################
# RUN THIS IF YOU WANT TO USE LATIN HYPERCUBE SAMPLING INSTEAD OF
# A FULL GRID
####################################################################
import pyDOE

# Number of samples to generate
num_samples = 50

# Perform Latin hypercube sampling
lhs_samples = pyDOE.lhs(len(parameters), samples=num_samples)

# Scale the Latin hypercube samples to the grid ranges
scaled_samples = []
for i in range(len(parameters)):
    min_val = np.min(grid[i])
    max_val = np.max(grid[i])
    print(min_val, max_val)
    scaled_samples.append(min_val + lhs_samples[:, i] * (max_val - min_val))

# Convert the scaled samples to the unique combinations format
reduced_combinations = []
for i in range(num_samples):
    combination = []
    for j in range(len(parameters)):
        combination.append(scaled_samples[j][i])
    reduced_combinations.append(combination)

# Update the unique_combinations with the reduced combinations
unique_combinations = reduced_combinations
for i in range(len(unique_combinations)):
    unique_combinations[i][1] = unique_combinations[i][0] * unique_combinations[i][1]

# %% 
####################################################################
# ARE YOU HAPPY WITH THE GRID (UNIQUE_COMBINATIONS)? IF SO, CONTINUE
# IF YOU ARE RERUNNING THE GRID TO RESAMPLE THE WINDSAVE, CHANGE THE
# RERUN VARIABLE TO TRUE
####################################################################
rerun = False # If you are rerunning the grid to resample the windsave

# Write the new pf files for each combination
for i in range(len(unique_combinations)):
    new_file = template_file.copy()
    # Update the template file
    for j in range(len(parameters)):
        if parameters[j] in new_file.keys():    #if parameters key in new file dictionary  exists
            new_file[parameters[j]] = unique_combinations[i][j]
    if rerun:
        new_file['Wind.old_windfile(root_only)'] = f"run{i}" # Adding the correct windsave file name

    # Write the new template file
    if rerun:
        r.write_pf(f"rerun{i}",new_file)
    else:
        r.write_pf(f"run{i}",new_file)
    
    
# write the unique combinations to a file as a table with run number
from prettytable import PrettyTable
table = PrettyTable(['Run Number', 'Disk.mdot(msol/yr)', 'Wind.mdot(msol/yr)', 'KWD.d(in_units_of_rstar)', 'KWD.mdot_r_exponent', 'KWD.acceleration_length(cm)', 'KWD.acceleration_exponent'])
for i in range(len(unique_combinations)):
    table.add_row([i, f"{unique_combinations[i][0]:.3e}", f"{unique_combinations[i][1]:.3e}", f"{unique_combinations[i][2]:.3e}", f"{unique_combinations[i][3]:.3e}", f"{unique_combinations[i][4]:.3e}", f"{unique_combinations[i][5]:.3e}"])
    #table.add_row([i, unique_combinations[i][0], unique_combinations[i][1], unique_combinations[i][2], unique_combinations[i][3], unique_combinations[i][4], unique_combinations[i][5]])

with open('Grid_runs_logfile.txt', 'w') as f:
    f.write(table.get_string())
    
table2 = PrettyTable(['Run Number', 'Disk.mdot(msol/yr)', 'Wind.mdot(Disk.mdot)', 'KWD.d(in_units_of_rstar)', 'KWD.mdot_r_exponent', 'KWD.acceleration_length(cm)', 'KWD.acceleration_exponent'])
for i in range(grid.shape[1]):
    #round numbers to 5dp
    #table2.add_row([i, f'{grid[0,i]:.5e}', f'{grid[1,i]:.5e}', f'{grid[2,i]:.5e}', f'{grid[3,i]:.5e}', f'{grid[4,i]:.5e}', f'{grid[5,i]:.5e}'])
    table2.add_row([i, grid[0,i], grid[1,i], grid[2,i], grid[3,i], f'{grid[4,i]:.5e}', grid[5,i]])

with open('Grid_runs_combinations.txt', 'w') as f:
    f.write(table2.get_string())
    
# %%
#######################################################################
# This is the submission script for Iridis5 at Southampton using SLURM
# AMD Partion, change maximums if you want to use a different partition
#######################################################################

max_num_of_cpus = 1280
max_cpus_per_node = 64
max_walltime = "40:00:00"
max_jobs = 32

# to help priority on a node, set each python simulation to use 64 cpus
optimal_cpus_per_job = max_cpus_per_node
optimal_jobs = max_num_of_cpus // optimal_cpus_per_job
num_sims_per_job = len(unique_combinations) // optimal_jobs + 1 # to not go over limits

print(f"optimal_cpus_per_job change template , #SBATCH --ntasks={optimal_cpus_per_job}")


#############################################################
# Change the template slurm file to suit your needs/your user
#############################################################
# %%
# Create the .slurm file
import shutil
from collections import OrderedDict

# Copy the template slurm file
shutil.copyfile("submit_python_runs_template.slurm", f"submit_python_runs.slurm") # Copy the template slurm file
# with open("submit_python_runs.slurm", 'r') as f:
#     file_contents = f.readlines()


# %%
#Transfer files ending in .spec from the large_optical_grid_runs_completed folder to the optical_hypercube_spectra folder
import glob
import os
import shutil

# Get a list of all the .spec files in the large_optical_grid_runs_completed folder
spec_files = glob.glob("../large_optical_grid_runs_completed/*.spec")

# Move the .spec files to the optical_hypercube_spectra folder
for spec_file in spec_files:
    shutil.copyfile(spec_file, "../optical_hypercube_spectra/" + os.path.basename(spec_file))
    


# # %%
# # Create a directory to store the .slurm files
# import shutil
# from collections import OrderedDict

# # Create the .slurm files
# for i in range(optimal_jobs):
#     # Copy the template slurm file
#     shutil.copyfile("submit_python_runs_template.slurm", f"submit_python_runs_run{i:0>2d}.slurm") # Copy the template slurm file
    
#     slurm_file_path = f"submit_python_runs_run{i:0>2d}.slurm"

#     with open(slurm_file_path, 'r') as f:
#         file_contents = f.readlines()

#     temp_command = file_contents[-1].split(" ") # split the final line into a list of strings
#     file_contents = file_contents[:-1] # remove the file_contents final line
    
#     # work out the new python simulation command, splitting sims into different jobs
#     for sim in range(0,num_sims_per_job+1):
#         if i + sim*optimal_jobs < len(unique_combinations):
#             command = temp_command[:-1] + [f"run{i + sim*optimal_jobs:0>2d}.pf"]
#             # merge list of strings into one string
#             command = " ".join(command)
#             file_contents.append(command + "\n")
    
#     # write the new file contents to the .slurm file
#     with open(slurm_file_path, 'w') as f:
#         f.writelines(file_contents)


# # %%
# # create an .sh file to submit all the .slurm files
# with open("submit_python_runs.sh", 'w') as f:
#     f.write("#!/bin/bash\n")
#     f.write("export PYTHON=/home/agww1g17/python\n")
#     f.write("export PATH=$PATH:$PYTHON/bin\n")
#     for i in range(optimal_jobs):
#         f.write(f"sbatch submit_python_runs_run{i:0>2d}.slurm\n")
# %%
