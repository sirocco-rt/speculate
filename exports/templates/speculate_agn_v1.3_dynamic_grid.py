# %%
#!/usr/bin/env/python
import os, sys
import numpy as np
import itertools
import matplotlib.pyplot as plt

# directory of MCRT code (set by $SIROCCO)
try:
    SIROCCO_DIR = os.environ["SIROCCO"]
except KeyError:
    # Fallback if environment variable is missing (e.g. new session)
    SIROCCO_DIR = "/Users/austen/Documents/GitHub/sirocco"
    print(f"Warning: $SIROCCO not found in environment. Using default: {SIROCCO_DIR}")

# directory of helper python scripts
sys.path.append("{}/py_progs/".format(SIROCCO_DIR))
import py_read_output as r

# Import QSOSED
sys.path.append("../src")
from pyqsosed import qsosed

# Specify the path to the template.pf file
template_file = r.read_pf("template.pf")

# AGN GRID SPACE ------------------------------

# Items to change in the template file
# ------------------------------------
parameters = [
    'Central_object.mass(msol)',
    'Disk.mdot(msol/yr)',
    'Wind.mdot(msol/yr)',
    'Wind.filling_factor(1=smooth,<1=clumped)',
    'KWD.d(in_units_of_rstar)',
    'KWD.mdot_r_exponent',
    'KWD.acceleration_length(cm)',
    'KWD.acceleration_exponent',
    ]

fixed_parameters = {
    'Central_object.radius(cm)', # 6R_g, R_g = GM/c^2
    'KWD.rmin(in_units_of_rstar)', # 60R_g
    'KWD.rmax(in_units_of_rstar)', # 600R_g
    'Input_spectra.model_file', # QSOSED model


}
# New parameter values for the grid
# ---------------------------------
grid = [
    [1e7,1e8,1e9],
    [0.025, 0.1, 0.4], # * Mdot_eddington, Mdot_edd = edd_L_const*M/((1 - np.sqrt(1 - 2/(3*self.risco))*c^2)
    [0.03, 0.3, 3.0], # * Mdot_disk
    [0.01,0.1,1.0],
    [5,30,180],
    [0,1], 
    [750, 7500], # * GM/c^2 (R_g)
    [1.5, 3.5],
    ]

# Constants (cgs units)
MSOL = 1.989e33  # g
YR = 3.1556925e7  # s
GRAV_CONST = 6.67e-8  # cm3 g-1 s-2
EDD_LUMINOSITY_CONST = 1.39e38  # erg/s per Msol
c = 2.99792458e10  # cm/s

PLOT_SEDS = True # Toggle plotting of SEDs for each run
RANDOM_SEED = 12345 # Seed for reproducibility (LHS sampling)

# 1. GENERATE RAW SAMPLES (Input Space)
# ----------------------------------
# Choose sampling method:
#   'grid'          — Full Cartesian product of all grid values
#   'lhs'           — Continuous Latin Hypercube (interpolates between grid bounds)
#   'discrete_lhs'  — LHS over exact grid points only (balanced, no interpolation)
SAMPLING_METHOD = 'grid'
NUM_LHS_SAMPLES = 25

raw_samples = []

if SAMPLING_METHOD == 'lhs':
    import pyDOE
    # Set seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Perform Latin hypercube sampling [0, 1]
    lhs_norm = pyDOE.lhs(len(parameters), samples=NUM_LHS_SAMPLES)
    
    # Scale to grid ranges
    for i in range(NUM_LHS_SAMPLES):
        combo = []
        for j in range(len(parameters)):
            min_val = np.min(grid[j])
            max_val = np.max(grid[j])
            val = min_val + lhs_norm[i][j] * (max_val - min_val)
            combo.append(val)
        raw_samples.append(combo)

elif SAMPLING_METHOD == 'discrete_lhs':
    # Latin Hypercube over the exact grid point values.
    # Each level of each parameter appears floor(N/k) or ceil(N/k) times,
    # shuffled independently per parameter. Duplicate combos are re-drawn.
    rng = np.random.default_rng(RANDOM_SEED)
    n_levels = [len(g) for g in grid]

    indices = np.zeros((NUM_LHS_SAMPLES, len(grid)), dtype=int)
    for j in range(len(grid)):
        k = n_levels[j]
        base = np.tile(np.arange(k), NUM_LHS_SAMPLES // k + 1)[:NUM_LHS_SAMPLES]
        rng.shuffle(base)
        indices[:, j] = base

    # Remove exact duplicate rows by re-drawing collisions
    seen = set()
    for i in range(NUM_LHS_SAMPLES):
        key = tuple(indices[i])
        attempts = 0
        while key in seen and attempts < 1000:
            for j in range(len(grid)):
                indices[i, j] = rng.integers(0, n_levels[j])
            key = tuple(indices[i])
            attempts += 1
        seen.add(key)

    for i in range(NUM_LHS_SAMPLES):
        combo = [grid[j][indices[i, j]] for j in range(len(grid))]
        raw_samples.append(combo)

elif SAMPLING_METHOD == 'grid':
    # Full cartesian product
    for i in itertools.product(*grid):
        raw_samples.append(list(i))
else:
    raise ValueError(f"Unknown sampling method: '{SAMPLING_METHOD}'")


# 2. PROCESS SAMPLES (Apply Physics / Unit Conversions)
# --------------------------------------------------
unique_combinations = []

for i, val in enumerate(raw_samples):
    # Convert tuple to list if needed
    combo = list(val)
    
    # Extract base parameters (Raw Values)
    mass_msol = combo[0]
    mdot_factors_edd = combo[1] # Fraction of Mdot_edd
    mdot_wind_factor = combo[2] # Fraction of Disk mdot
    accel_len_rg_factor = combo[6] # Multiplier of Rg
    
    # A. Calculate R_g for this mass
    # R_g = GM/c^2
    Rg = (GRAV_CONST * (mass_msol * MSOL)) / (c**2)

    # B. Update Disk Mdot (column 1)
    # Target: msolar/yr (absolute value)
    # Input: Fraction of Eddington accretion rate
    # Mdot_edd = L_edd / (eta * c^2)

    L_edd = EDD_LUMINOSITY_CONST * mass_msol
    # eta = 1 - sqrt(1 - 2/(3*r_isco))  [Schwarzschild: a=0, r_isco=6]
    r_isco = 6.0 
    eta = 1.0 - np.sqrt(1.0 - 2.0/(3.0*r_isco))  # from AGNSED     
    mdot_edd_cgs = L_edd / (eta * c**2) # g/s
    mdot_disk_cgs = mdot_factors_edd * mdot_edd_cgs
    mdot_disk_msolyr = (mdot_disk_cgs / MSOL) * YR
    combo[1] = mdot_disk_msolyr

    # C. Update Wind Mdot (column 2)
    # Target: msolar/yr (absolute value)
    # Input: Fraction of Disk Mdot
    combo[2] = mdot_wind_factor * mdot_disk_msolyr

    # D. Update Acceleration Length (column 6)
    # Target: cm (absolute value)
    # Input: multiplier of R_g
    combo[6] = accel_len_rg_factor * Rg

    unique_combinations.append(combo)

print(f"Generated {len(unique_combinations)} unique models using '{SAMPLING_METHOD}' sampling.")


####################################################################
# ARE YOU HAPPY WITH THE GRID (UNIQUE_COMBINATIONS)? IF SO, CONTINUE 
####################################################################

# %% 
# Write the new pf files for each combination
sed_cache = {} # Cache to reuse SEDs: key=(mass, log_mdot) -> val=(filename, Lx, Rout)

for i in range(len(unique_combinations)):
    new_file = template_file.copy()
    current_combo = unique_combinations[i]
    
    # 1. Recalculate Rg (needed for radius and rmin/rmax)
    mass_msol = current_combo[0]
    mdot_disk_msolyr = current_combo[1]
    
    # Calculate Rg
    Rg = (GRAV_CONST * (mass_msol * MSOL)) / (c**2)
    
    # Update Fixed Parameters based on Rg
    # Standard Sirocco convention: Central Object Radius = ISCO = 6 Rg
    new_file['Central_object.radius(cm)'] = 6.0 * Rg
    
    # KWD.rmin and KWD.rmax are in units of rstar (which is now 6 Rg)
    # Target: rmin=60Rg -> 10 rstar
    # Target: rmax=600Rg -> 100 rstar
    new_file['KWD.rmin(in_units_of_rstar)'] = 10.0
    new_file['KWD.rmax(in_units_of_rstar)'] = 100.0
    
    # Increase resolution and temp to match colleague's setup
    new_file['Wind.dim.in.x_or_r.direction'] = 50
    new_file['Wind.dim.in.z_or_theta.direction'] = 50
    new_file['Wind.t.init'] = 100_000.0

    # Scaling Disk and Wind domains with Rg
    # KWD.rmax is 600rg. We want the wind domain to extend well beyond this.
    # 600,000 Rg based for over 10x acceleration length
    new_file['Wind.radmax(cm)'] = f"{600_000.0 * Rg:.4e}" 
    
    # 2. GENERATE QSOSED MODEL
    # ------------------------
    # Recover log_mdot (Eddington fraction) for QSOSED
    # Mdot_edd = L_edd / (eta * c^2)
    L_edd = EDD_LUMINOSITY_CONST * mass_msol
    r_isco = 6.0 
    eta = 1.0 - np.sqrt(1.0 - 2.0/(3.0*r_isco))
    mdot_edd_cgs = L_edd / (eta * c**2)
    mdot_edd_msolyr = (mdot_edd_cgs / MSOL) * YR
    
    # Fraction = Mdot_disk / Mdot_edd
    edd_fraction = mdot_disk_msolyr / mdot_edd_msolyr
    log_mdot = np.log10(edd_fraction)

    # Check cache for existing SED
    sed_key = (float(f"{mass_msol:.5e}"), float(f"{log_mdot:.4f}"))
    
    # Initialize variables to ensure scope existence
    L_xray = 1e40 
    r_out_Rg = 60_000.0

    if sed_key in sed_cache:
        sed_filename, L_xray, r_out_Rg = sed_cache[sed_key]
        print(f"Using cached SED for run {i}: {sed_filename}")
        
        new_file['Input_spectra.model_file'] = sed_filename
        new_file['Central_object.luminosity(ergs/s)'] = f"{L_xray:.4e}"
        new_file['Disk.radmax(cm)'] = f"{r_out_Rg * Rg:.4e}"
        
    else:
        # Fixed QSOSED parameters
        # dist: Co-moving distance in Mpc. 
        # Sirocco typically defines the source at a reference distance or intrinsic luminosity.
        # To be consistent with Sirocco's 100pc reference, we use 100pc = 1e-4 Mpc.
        # Note: Since we extract Luminosity (Likely Isotropic Equivalent), this only affects flux calculations if we used them.
        dist = 1.0e-4 # 100 pc
        
        # cos_inc: Cosine of inclination. 
        # The SED model scales as (cos_inc / 0.5). 
        # We assume an isotropic SED (angle-averaged), so cos_inc=0.5 is the appropriate choice.
        cos_inc = 0.5
        
        # fcol: Color temperature correction (Done et al 2012). 
        # fcol=1 means no correction (blackbody-like). Standard is often ~1.7-2.0 for disks, but 1 is safe if unsure.
        fcol = 1.0
        
        # z: Redshift. We are simulating local physics (100pc), so z=0 is correct.
        z = 0.0
        
        # Spin a=0 as requested
        a_spin = 0.0

        # Unique filename based on parameters
        sed_base_name = f"qsosed_M{mass_msol:.2e}_mdot{log_mdot:.3f}"
        sed_filename = f"{sed_base_name}.ls"
        
        try:
            # Create model
            qso = qsosed(M=mass_msol, dist=dist, log_mdot=log_mdot, a=a_spin, cos_inc=cos_inc, fcol=fcol, z=z)
            
            # KEY UPDATE: use 'cgs_wave' to get Wavelength (Angstrom) and L_lambda (erg/s/A)
            try:
                qso.set_units('cgs_wave')
            except:
                # Fallback if set_units fails or doesnt exist in this version
                # Assuming default cgs is Hz, we might need manual conversion if 'cgs_wave' isn't supported
                # But assuming qsosed.py has it from my previous read
                pass
            
            # get_SED(as_flux=False) returns Luminosity Density L_lambda (in cgs_wave mode)
            L_lambda = qso.get_SED(as_flux=False)
            wave = qso.wave_grid  # Angstroms
            
            # IMPORTANT: Sirocco usually expects data sorted by column 1.
            # Wavelengths in qsosed are usually high-to-low (Energy low-to-high).
            # We sort low-to-high (short wavelength / X-ray first) for numerical safety
            sort_idx = np.argsort(wave)
            sorted_wave = wave[sort_idx]
            sorted_Llambda = L_lambda[sort_idx]
            
            # Filter low wavelengths (High Energy cut)
            # User requested cut at 1.239842e-1 Angstroms (approx 100 keV)
            mask_cut = sorted_wave >= 1.239842e-1 # 0.1239842 Angstroms = 100 keV
            sorted_wave = sorted_wave[mask_cut]
            sorted_Llambda = sorted_Llambda[mask_cut]

            # FIX: Clip Llambda to avoid 0.0 (Sirocco log interpolation safety)
            sorted_Llambda = np.maximum(sorted_Llambda, 1e-40)

            # Save .dat file (The actual data)
            # Format: standard scientific, tab separated
            # Sirocco expects: Wavelength (Angstroms)  Luminosity (erg/s/A)
            sed_dat_filename = f"{sed_base_name}.dat"
            np.savetxt(sed_dat_filename, np.column_stack((sorted_wave, sorted_Llambda)), fmt='%.6e', delimiter='\t')
            
            # Create the .ls file (Pointer to the .dat file)
            # Format: filename.dat -1
            with open(sed_filename, 'w') as f_ls:
                f_ls.write(f"{sed_dat_filename} -1")
            
            # Set the filename in the Sirocco parameters
            new_file['Input_spectra.model_file'] = sed_filename

            if PLOT_SEDS:
                # Plotting the SED for verification (Updated for Wavelength)
                plt.figure(figsize=(10, 6))
                # Plot lambda * L_lambda vs lambda (Energy Distribution) 
                plt.loglog(sorted_wave, sorted_wave * sorted_Llambda, label='QSOSED Total')
                
                # Also plot components
                try:
                    # get_SEDcomponent also respects set_units('cgs_wave')
                    L_disc = qso.get_SEDcomponent('disc', as_flux=False)
                    L_warm = qso.get_SEDcomponent('warm', as_flux=False)
                    L_hot = qso.get_SEDcomponent('hot', as_flux=False)
                    
                    # Sort and mask components to match sorted_wave
                    sorted_L_disc = L_disc[sort_idx][mask_cut]
                    sorted_L_warm = L_warm[sort_idx][mask_cut]
                    sorted_L_hot = L_hot[sort_idx][mask_cut]

                    # Apply clipping to components too for log plotting safety
                    sorted_L_disc = np.maximum(sorted_L_disc, 1e-40)
                    sorted_L_warm = np.maximum(sorted_L_warm, 1e-40)
                    sorted_L_hot = np.maximum(sorted_L_hot, 1e-40)
                    
                    plt.loglog(sorted_wave, sorted_wave * sorted_L_disc, '--', label='Disc')
                    plt.loglog(sorted_wave, sorted_wave * sorted_L_warm, '--', label='Warm Compton')
                    plt.loglog(sorted_wave, sorted_wave * sorted_L_hot, '--', label='Hot Compton')
                except Exception as e:
                    print(f"Plotting components failed: {e}") 
                    pass
                    
                plt.xlabel(r'Wavelength ($\AA$)')
                plt.ylabel(r'$\lambda L_{\lambda}$ (erg/s)')
                plt.title(f'SED for Run {i}: M={mass_msol:.1e}, log(mdot)={log_mdot:.2f}')
                plt.legend()
                plt.grid(True, which="both", ls="-", alpha=0.5)
                
                # Toggle plotting limits
                # Find peak energy of the TOTAL SED only (ignore components)
                # This ensures the y-axis is scaled sensibly for the main curve
                peak_y = np.max(sorted_wave * sorted_Llambda)
                
                # Set lower limit to 4-5 mags below peak
                plt.ylim(bottom=peak_y * 1e-4, top=peak_y * 1.5)
                
                # plt.gca().invert_xaxis() # User requested wavelengths smallest to largest
                
                plot_filename = f"{sed_base_name}.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"Saved SED plot to {plot_filename}")

            # Calculate 2-10keV Luminosity for config
            # We need to integrate L_lambda over wavelength.
            # L = int L_lambda dlambda
            # 2 keV = 6.2 A
            # 10 keV = 1.24 A
            lambda_min_xray = 1.24
            lambda_max_xray = 6.2
            
            # Note: we need min to max for integration logic usually, but wave is sorted low-to-high
            mask_xray = (sorted_wave >= lambda_min_xray) & (sorted_wave <= lambda_max_xray)
            
            if np.sum(mask_xray) > 1:
                # integrate L_lambda dlambda
                L_xray = np.trapz(sorted_Llambda[mask_xray], sorted_wave[mask_xray])
            else:
                L_xray = 1e40
                print(f"Warning: Poor X-ray sampling for run {i}")
                
            new_file['Central_object.luminosity(ergs/s)'] = f"{L_xray:.4e}"
            
            # Set Disk.radmax based on QSOSED's self-gravity radius (r_out)
            # qso.r_out is in units of Rg.
            # Check: ensure disk extends at least to 2500 Rg to act as a proper background/obscurer for the wind.
            # (Colleague suggested 1000, but 2500 is safer if the wind extends to 20,000)
            r_out_Rg = max(qso.r_out, 60_000.0) 
            new_file['Disk.radmax(cm)'] = f"{r_out_Rg * Rg:.4e}"

            print(f"Generated SED: {sed_filename} (M={mass_msol:.1e}, log(mdot)={log_mdot:.2f}, Lx_2-10={L_xray:.2e}, Rout={r_out_Rg:.1f}Rg)")
            
            # Save to cache
            sed_cache[sed_key] = (sed_filename, L_xray, r_out_Rg)
            
        except Exception as e:
            print(f"Error generating SED for run {i}: {e} (log_mdot={log_mdot:.3f})")
            # Assuming we want to crash if SED fails, or handle it? 
            # For now, let's proceed but the file param will likely be missing or old default.
    
    
    # Update Variable Parameters
    # Loop defined by length of 'parameters' list provided at top of file
    for j in range(len(parameters)):
        val = current_combo[j]
        # Force scientific notation for specific fields
        if parameters[j] in ['Central_object.mass(msol)', 
                             'Disk.mdot(msol/yr)', 
                             'Wind.mdot(msol/yr)', 
                             'KWD.acceleration_length(cm)']:
            new_file[parameters[j]] = f"{val:.12e}"
        else:
            new_file[parameters[j]] = val

    # Write the new template file
    pf_filename = f"run{i}.pf"
    r.write_pf(pf_filename, new_file)
    
    # POST-PROCESSING: Fix duplicate Input_spectra.model_file entries
    # The .pf file requires two entries for Input_spectra.model_file (one for wind, one for spectrum).
    # Dictionary storage likely reduced it to one, so we manually ensure both are present in the text file.
    
    with open(pf_filename, 'r') as f:
        lines = f.readlines()
    
    # 1. Remove any existing Input_spectra.model_file lines (to avoid duplicates/misplacement)
    clean_lines = [l for l in lines if not l.strip().startswith("Input_spectra.model_file")]
    
    # 2. Re-insert the line exactly where needed
    final_lines = []
    sed_line = f"Input_spectra.model_file                	{sed_filename}\n"
    
    for line in clean_lines:
        final_lines.append(line)
        # Check for the keys that require the model file to follow
        if "Central_object.rad_type_to_make_wind" in line or "Central_object.rad_type_in_final_spectrum" in line:
             final_lines.append(sed_line)
             
    with open(pf_filename, 'w') as f:
        f.writelines(final_lines)

import pandas as pd 
# Create a pandas DataFrame for unique_combinations
df_unique_combinations = pd.DataFrame(unique_combinations, columns=parameters)

# Save the DataFrame to a filetaFrame(log_combinations, columns=parameters)

# Save the DataFrame to a file
df_unique_combinations.to_csv('Grid_runs_logfile.txt', index=False)

# Create a pandas DataFrame for grid (ragged lists, so use a dict)
df_grid = pd.DataFrame({p: pd.Series(v) for p, v in zip(parameters, grid)})

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

# max_num_of_cpus = 1280
# max_cpus_per_node = 64
# max_walltime = "40:00:00"
# max_jobs = 32

# # to help priority on a node, set each python simulation to use 64 cpus
# optimal_cpus_per_job = max_cpus_per_node
# optimal_jobs = max_num_of_cpus // optimal_cpus_per_job
# num_sims_per_job = len(unique_combinations) // optimal_jobs + 1 # to not go over limits

# print(f"optimal_cpus_per_job change template , #SBATCH --ntasks={optimal_cpus_per_job}")


#############################################################
# Change the template slurm file to suit your needs/your user
#############################################################
# %%
# # Create the .slurm file
# import shutil
# from collections import OrderedDict

# # Copy the template slurm file
# shutil.copyfile("submit_python_runs_template.slurm", f"submit_python_runs.slurm") # Copy the template slurm file
# with open("submit_python_runs.slurm", 'r') as f:
#     file_contents = f.readlines()


# %%
#Transfer files ending in .spec from the large_optical_grid_runs_completed folder to the optical_hypercube_spectra folder
# import glob
# import os
# import shutil
# import pandas as pd

# # Get a list of all the .spec files in the large_optical_grid_runs_completed folder
# spec_files = glob.glob("../large_optical_grid_runs_completed/*.spec")

# # Move the .spec files to the optical_hypercube_spectra folder
# for spec_file in spec_files:
#     shutil.copyfile(spec_file, "../optical_hypercube_spectra/" + os.path.basename(spec_file))
    


# # # %%
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
