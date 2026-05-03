# %%
#!/usr/bin/env/python
import os, sys
import numpy as np
import itertools

# %%
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
    'KWD.acceleration_exponent',
    'Boundary_layer.luminosity(ergs/s)',
    'Boundary_layer.temp(K)'
    ]

# New parameter values for the grid
# ---------------------------------
grid = np.array([
    [3e-9,10e-9,30e-9], # Msol/yr
    [0.03,0.1,0.3], # Units of disk mdot
    [0.55,5.5,55], # dmin = 5.5, np.sqrt(30Rstar)
    [0,0.25,1], 
    [7.25182e+08,7.25182e+09,7.25182e+10], # 1Rstar 7.25182e+08
    [0.5,1.5,4.5],
    [0, 0.3, 1], # in units of L_disk = G M_star * M_dotdisk/2R_star
    [0.1,0.3,1] # in units of H/R_wd for area of BL, Stephan boltzmann law to temperature.
    ])

# All the grid combinations
unique_combinations = []
for i in itertools.product(*grid):
    unique_combinations.append(list(i))

# Updating Wmdot to be in units of disk mdot
for i in range(len(unique_combinations)):
    # Wind.mdot in units of disk mdot
    unique_combinations[i][1] = unique_combinations[i][0] * unique_combinations[i][1]
    # Boundary Layer luminosity in units of disk luminosity
    MSOL = 1.989e33
    YR = 3.1556925e7
    L_disk = 6.67e-8 * template_file['Central_object.mass(msol)']*MSOL * (unique_combinations[i][0]*MSOL/YR) / (2 * template_file['Central_object.radius(cm)']) # L_disk = G M_star * M_dotdisk/2R_star
    unique_combinations[i][6] = unique_combinations[i][6] * L_disk
    # Boundary Layer temperature as a function of area covering the white dwarf H/R_wd
    # area of Spherical segment BL = 4•pi•R_wd^2 * H/R_wd 
    area = 4 * np.pi * template_file['Central_object.radius(cm)']**2 * unique_combinations[i][7]
    # Stefan boltzmann law to temperature L=area*sigma*T^4
    STEFAN_BOLTZMANN = 5.6696e-5
    unique_combinations[i][7] = (unique_combinations[i][6] / area / STEFAN_BOLTZMANN)**0.25

# %%
# For the Latin hypercube sampling
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
    
####################################################################
# ARE YOU HAPPY WITH THE GRID (UNIQUE_COMBINATIONS)? IF SO, CONTINUE 
####################################################################

# %% 
# Write the new pf files for each combination
for i in range(len(unique_combinations)):
    new_file = template_file.copy()
    # Update the template file
    for j in range(len(parameters)):
        new_file[parameters[j]] = unique_combinations[i][j]

    # Write the new template file
    r.write_pf(f"run{i:_}",new_file) # i:0>2d
    
import pandas as pd 
# Create a pandas DataFrame for unique_combinations
df_unique_combinations = pd.DataFrame(unique_combinations, columns=parameters)

# Save the DataFrame to a file
df_unique_combinations.to_csv('Grid_runs_logfile.txt', index=False)

# Create a pandas DataFrame for grid
df_grid = pd.DataFrame(grid.T, columns=parameters)

# Save the DataFrame to a file
df_grid.to_csv('Grid_runs_combinations.txt', index=False)
    
# remove the leading zeros from the file names eg run1_034.pf -> run1_34.pf
import os
for filename in os.listdir("."):
    if filename.startswith("run"):
        try:
            new_filename = filename.split("_")
            new_filename[-1] = str(int(new_filename[-1].split(".")[0]))
            new_filename = "_".join(new_filename) + ".pf"
            os.rename(filename, new_filename)
        except:
            pass

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
import pandas as pd

# Get a list of all the .spec files in the large_optical_grid_runs_completed folder
spec_files = glob.glob("../large_optical_grid_runs_completed/*.spec")

# Move the .spec files to the optical_hypercube_spectra folder
for spec_file in spec_files:
    shutil.copyfile(spec_file, "../optical_hypercube_spectra/" + os.path.basename(spec_file))
    

# %%

import glob
import os
import shutil
import pandas as pd

import os
import re

# Regular expression to match filenames in the format 'runX_Y.pf'
pattern = re.compile(r'run(\d+)_(\d+)\.spec')

for filename in os.listdir("."):
    try:
        match = pattern.match(filename)
        if match:
            X = int(match.group(1))
            Y = int(match.group(2))
            N = X * 1000 + Y
            new_filename = f"run{N:04d}.spec"
            os.rename(filename, new_filename)
            #print(new_filename, filename)
    except:
        print(f"Error with {filename}")


#%% 
# check the run numbers

for i in range(6561):
    if not os.path.exists(f"run{i}.spec"):
        print(f"run{i}.spec does not exist")
























#old code

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
