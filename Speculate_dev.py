# %% Stage 1) Imports and Speculate Grid Classes/funcitons for 'python'.
# 1) ==========================================================================|

#import autopep8
import os
import random
import emcee
import corner
#import arviz as az # something broke in the enviroment
import math as m
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.stats as st
import Speculate_addons.Spec_functions as spec
from pyinstrument import Profiler
#import gpu_tracker as gput
#from alive_progress import alive_it
from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from Starfish.grid_tools import HDF5Creator
from Starfish.grid_tools import GridInterface
from Starfish.emulator import Emulator
from Starfish.emulator.plotting import plot_eigenspectra
from Starfish.spectrum import Spectrum
from Starfish.models import SpectrumModel
from Speculate_addons.Spec_gridinterfaces import Speculate_cv_bl_grid_v87f
from Speculate_addons.Spec_gridinterfaces import Speculate_cv_no_bl_grid_v87f 
from Speculate_addons.Spec_functions import unique_grid_combinations
print('Imported modules')

plt.style.use('Solarize_Light2') # Plot Style (⌐▀͡ ̯ʖ▀) (ran twice as buggy-ish)

# %% Stage 2.1) Flux grid space (HDF5) setup and PCA inputs
# 2.1) ========================================================================|

""" 
-------------------------------------------------------------------------------|
                          - Set up your grid space! -
        Parameters of model            |            Parameters of model
        (kgrid/20years old)            |         (short spec cv grid/v87a)
-------------------------------------------------------------------------------|
1) wind.mdot (msol/yr)                 | 1) wind.mdot (msol/yr)
2) kn.d                                | 2) KWD.d
3) kn.mdot_r_exponent                  | 3) KWD.v_infinity (in_units_of_vescape)
4) kn.v_infinity (in_units_of_vescape) |
5) kn.acceleration_length (cm)         |
6) kn.acceleration_exponent            |
max_wl_range = (876, 1824)             | max_wl_range = (850, 1850)
-------------------------------------------------------------------------------|
        Parameters of model            |            Parameters of model
   (optical_grid_spec_files/v87b)      |     (broad_short_spec_cv_grid/v87a)
-------------------------------------------------------------------------------|
1) disk.mdot (msol/yr)                 | 1) wind.mdot (msol/yr)
2) wind.mdot (disk.mdot)               | 2) KWD.d
3) KWD.d(in_units_of_rstar)            | 3) KWD.v_infinity (in_units_of_vescape)
4) KWD.mdot_r_exponent                 |
5) KWD.acceleration_length(cm)         |
6) KWD.acceleration_exponent           |
max_wl_range = (850, 7950)             | max_wl_range = (850, 7950)
-------------------------------------------------------------------------------|
        Parameters of model            |            Parameters of model
     (Ha_grid_spec_files/v87b)         |     (CV_release_grid/v87f)
-------------------------------------------------------------------------------|
1) disk.mdot (msol/yr)                 | 1) disk.mdot (msol/yr)
2) wind.mdot (disk.mdot)               | 2) wind.mdot (disk.mdot)
3) KWD.d(in_units_of_rstar)            | 3) KWD.d(in_units_of_rstar)
4) KWD.mdot_r_exponent                 | 4) KWD.mdot_r_exponent  
5) KWD.acceleration_length(cm)         | 5) KWD.acceleration_length(cm) 
6) KWD.acceleration_exponent           | 6) KWD.acceleration_exponent 
max_wl_range = (6385, 6735)            | 7) Boundary_layer.luminosity(ergs/s)
                                       | 8) Boundary_layer.temp(K)
                                       | 9) Inclination angle (degrees: 30,55,80)
                                       | max_wl_range = (800,8000)
-------------------------------------------------------------------------------|
"""

# ----- Inputs here -----------------------------------------------------------|
model_parameters = (1,2,9)  # Including parameter 9 for inclination (30°, 55°, 80°)
wl_range = (850, 1850)       # Wavelength range of your emulator grid space.
                              # Later becomes truncated +/-10Angstom
                              
scale = 'linear'              # Transformation scaling for flux data. 'linear'
                              # 'log' or 'scaled'. scale not implemented yet.

grid_file_name = 'test'  # If Builds fast, file save unnessary.
process_grid = False           # Turn off if planning to use existing grid file.
# kgrid = 0                     # Turn on if planning to use kgrid
# shortspec = 0                 # Turn on if planning to use shortspec_cv_grid
# broadshortspec = 0            # Turn on if planning to use broadshortspec_cv_grid
# opticalspec = 0               # Turn on if planning to use optical_grid_spec_files
# h_alpha = 0                   # Turn on if planning to use Ha_grid_spec_files
# cv_release = 1                # Turn on if planning to use CV_release_grid
speculate_cv_no_bl_grid_v87f = 1 # Turn on if planning to use speculate_cv_no_bl_grid_v87f
speculate_cv_bl_grid_v87f = 0    # Turn on if planning to use speculate_cv_bl_grid_v87f

n_components = 10             # Alter the number of PCA components used.
# Integer for no. of components or decimal (0.0-1.0) for 0%-100% accuracy.
block_diagonal = True         # Use block-diagonal optimization for covariance matrix
# -----------------------------------------------------------------------------|

# Sorting parameters by increasing order
model_parameters = sorted(model_parameters)
# Looping through parameters to create a string of numbers for file name
model_parameters_str = ''.join(str(i) for i in model_parameters)

#TODO Noel's Found bug. There is an offset when comparing spectra to emulated 
# models. The Wavelength calculation in velocity space grows exponentially and 
# is non-linear ∆𝝺/𝝺 = ∆v/c.

# # Selecting the specified grid interface 
# if kgrid == 1:
#     # Change inclination with usecols[1]
#     usecols = (1, 8) # Wavelength, Inclination 8-14 --> 40-70 degrees
#     skiprows = 2  # Start of data within file
#     grid = KWDGridInterface(
#         path='kgrid/sscyg_kgrid090311.210901/',
#         usecols=usecols,
#         skiprows=skiprows,
#         wl_range=wl_range,
#         model_parameters=model_parameters)
#     inclination = usecols[1] * 5
#     emu_file_name = f'Kgrid_emu_{scale}_{usecols[1]*5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'

# if shortspec == 1:
#     # Change inclination with usecols[1]
#     usecols = (1, 16) # Wavelength, Inclination 10-21 --> 30-85 degrees
#     skiprows = 81  # Starting point of data within file
#     grid = ShortSpecGridInterface(
#         path='short_spec_cv_grid/',
#         usecols=usecols,
#         skiprows=skiprows,
#         wl_range=wl_range,
#         model_parameters=model_parameters, 
#         scale=scale
#         )
#     inclination = (usecols[1]-4) * 5
#     emu_file_name = f'SSpec_emu_{scale}_{(usecols[1]-4) * 5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'

# if broadshortspec == 1:
#     # Change inclination with usecols[1]
#     usecols = (1, 16) # Wavelength, Inclination 10-21 --> 30-85 degrees
#     skiprows = 81  # Starting point of data within file
#     grid = BroadShortSpecGridInterface(
#         path='broad_short_spec_cv_grid/',
#         usecols=usecols,
#         skiprows=skiprows,
#         wl_range=wl_range,
#         model_parameters=model_parameters, 
#         scale=scale
#         )
#     inclination = (usecols[1]-4) * 5
#     emu_file_name = f'BSSpec_emu_{scale}_{(usecols[1]-4) * 5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'
    
# if opticalspec == 1:
#     # Change inclination with usecols[1]
#     usecols = (1, 16) # Wavelength, Inclination 10-21 --> 30-85 degrees
#     skiprows = 81  # Starting point of data within file
#     grid = OpticalCVGridInterface(
#         path='optical_grid_spec_files/',
#         usecols=usecols,
#         skiprows=skiprows,
#         wl_range=wl_range,
#         model_parameters=model_parameters, 
#         scale=scale
#         )
#     inclination = (usecols[1]-4) * 5
#     emu_file_name = f'OSpec_emu_{scale}_{(usecols[1]-4) * 5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'
    
# if h_alpha == 1:
#     # Change inclination with usecols[1]
#     usecols = (1, 11) # Wavelength, Inclination tuple 10-14 --> 20,45,60,72.5,85 degrees
#     skiprows = 81  # Starting point of data within file
#     grid = HalphaCVGridInterface(
#         path='Ha_grid_spec_files/',
#         usecols=usecols,
#         skiprows=skiprows,
#         wl_range=wl_range,
#         model_parameters=model_parameters, 
#         scale=scale
#         )
#     inclination = (usecols[1]-4) * 5
#     emu_file_name = f'HaSpec_emu_{scale}_{(usecols[1]-4) * 5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'
    
# if cv_release == 1:
#     # Change inclination with usecols[1]
#     usecols = (1, 15) # Wavelength, Inclination 10-21 --> 30-85 degrees
#     skiprows = 86  # Starting point of data within file
#     grid = CVReleaseGridInterface(
#         path='CV_release_grid_spec/',
#         usecols=usecols,
#         skiprows=skiprows,
#         wl_range=wl_range,
#         model_parameters=model_parameters, 
#         scale=scale
#         )
#     inclination = (usecols[1]-4) * 5
#     if 9 in model_parameters or 10 in model_parameters:
#         emu_file_name = f'CVRel_emu_{scale}_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'
#     else:
#         emu_file_name = f'CVRel_emu_{scale}_{(usecols[1]-4) * 5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'
#     print(emu_file_name)

if speculate_cv_no_bl_grid_v87f == 1:
    base_name = 'speculate_cv_no-bl_grid_v87f_'
    # saved grid file name
    grid_file_name = base_name + 'grid_' + model_parameters_str
    # Can change inclination inclination angle
    inclination_angle = 55  # degrees: 30,35,40,45,50,55,60,65,70,75,80,85
    inclination_column = int(2 + (inclination_angle - 30) / 5) #(30°->2, 35°->3, ..., 85°->13)
    usecols = (1, inclination_column) # Wavelength, Inclination 2-13 --> 30-85 degrees
    skiprows = 82  # Starting point of data within file
    grid = Speculate_cv_no_bl_grid_v87f(
        path='sirocco_grids/speculate_cv_no-bl_grid_v87f/',
        usecols=usecols,
        wl_range=wl_range,
        model_parameters=model_parameters, 
        scale=scale
        )
    if 9 in model_parameters or 10 in model_parameters or 11 in model_parameters:
        emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}_{wl_range[0]}-{wl_range[1]}AA_{n_components}PCA'
    else:
        emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}_{inclination_angle}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}PCA'
    print(emu_file_name)

if speculate_cv_bl_grid_v87f == 1:
    base_name = 'speculate_cv_bl_grid_v87f_'
    # saved grid file name
    grid_file_name = base_name + 'grid_' + model_parameters_str
    # Can change inclination inclination angle
    inclination_angle = 55  # degrees: 30,35,40,45,50,55,60,65,70,75,80,85
    inclination_column = int(2 + (inclination_angle - 30) / 5) #(30°->2, 35°->3, ..., 85°->13)
    usecols = (1, inclination_column) # Wavelength, Inclination 2-13 --> 30-85 degrees
    skiprows = 82  # Starting point of data within file
    grid = Speculate_cv_bl_grid_v87f(
        path='sirocco_grids/speculate_cv_bl_grid_v87f/',
        usecols=usecols,
        wl_range=wl_range,
        model_parameters=model_parameters, 
        scale=scale
        )
    inclination = (usecols[1]+4) * 5
    if 9 in model_parameters or 10 in model_parameters or 11 in model_parameters:
        emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}_{wl_range[0]}-{wl_range[1]}AA_{n_components}PCA'
    else:
        emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}_{inclination_angle}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}PCA'
    print(emu_file_name)


# Faster processing with python's .spec files into a hdf5 file
# Auto-generated keyname's required to integrate with Starfish
keyname = ["param{}{{}}".format(i) for i in model_parameters]
keyname = ''.join(keyname)

#if grid_file exists, skip processing
if os.path.isfile(f'Grid-Emulator_Files/{grid_file_name}.npz'):
    print(f'Grid file {grid_file_name} exists, skipping grid processing.')
else:
    # Set up logging for grid processing verification (file only, no console output)
    import logging
    import os
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/grid_processing.log', mode='w')
        ]
    )
    print("🔧 Logging enabled - file/column details will be saved to logs/grid_processing.log (console output suppressed)")
    
    # Is NPZCreator, backwards compatibility with Starfish
    creator = HDF5Creator(
        grid,
        f'Grid-Emulator_Files/{grid_file_name}.npz',
        key_name=keyname,
        wl_range=wl_range)
    creator.process_grid()

# After grid processing, check the grid_points
data = np.load(f'Grid-Emulator_Files/{grid_file_name}.npz', allow_pickle=True)
grid_points = data['grid_points']
print("Grid shape:", grid_points.shape)
print("Unique values per parameter:")
for i, param_name in enumerate(data['param_names']):
    print(f"  {param_name}: {np.unique(grid_points[:, i])}")

# Add custom name if auto generation undesired?
#emu_file_name = 'custom_name.npz' 

# Checking if the emulator file has been created
if os.path.isfile(f'Grid-Emulator_Files/{emu_file_name}.npz'):
    print(f'Emulator {emu_file_name} already exists.')
    emu = Emulator.load(f"Grid-Emulator_Files/{emu_file_name}.npz")
    emu_exists = 1
    print('Existing emulator loaded.')
else:
    print(f'Emulator {emu_file_name} does not exist.')
    emu_exists = 0
    print('Create new emulator in Stage 3.')


# %% Stage 2.2) Speculate's spectral data exploration tool (SDET).
# 2.2) ========================================================================|
# # The Class should open a new window to allow the user to explore the grid.
# %matplotlib qt
# grid_viewer = spec.InspectGrid(grid, emu) # Emu (Emulator) optional




# %%
# %matplotlib inline

# %% Stage 3) Generating and training a new emulator
# 3) ==========================================================================|
# from pyinstrument import Profiler
# import gpu_tracker as gputracker
# tracker = gput.Tracker()
# tracker.start()

# profiler = Profiler()
# profiler.start()
# Asking if user wants to continue training a new emulator
if emu_exists == 1:
    #emu_exists = 0
    #emu_exists = 0 # developing training /remove when complete
    print("Emulator's name:", emu_file_name)
    print('Do you want to overwrite the existing emulator (y/n)?')
    if input('y/n: ') == 'y':
        emu_exists = 0
        print('Existing emulator will be overwritten')
    else:
        print('Existing emulator will be used')

# Generating/training/saving and displaying the new emulator
# TODO Optimse GP for speed GPyTorch
if emu_exists == 0:
    print('Standardising and PCA-ing the dataset')
    emu = Emulator.from_grid(
        f'Grid-Emulator_Files/{grid_file_name}.npz',
        n_components=n_components,
        svd_solver="full",
        block_diagonal=block_diagonal) 
    # scipy.optimise.minimise routine
    print('Training the GP')
    # Using relative tolerance (ftol) so convergence scales with the loss value automatically
    emu.train(method="Nelder-Mead", options=dict(maxiter=25000, disp=True, ftol=1e-3))
    emu.save(f'Grid-Emulator_Files/{emu_file_name}.npz')
    print(emu)  # Displays the trained emulator's parameters

    # Plotting the loss function history
    if hasattr(emu, 'loss_history') and len(emu.loss_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(emu.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('Gaussian Process Training Loss')
        plt.grid(True)
        plt.savefig(f'Grid-Emulator_Files/{emu_file_name}_training_loss.png')


# profiler.stop()
# profiler.print()
# tracker.stop()
# %% Stage 3.5) Plotting the v11 matrix
# 3.5) ========================================================================|
# print("Plotting v11 matrix...")
# %matplotlib qt
# plt.figure(figsize=(10, 8))
# v11_matrix = emu.v11
# if v11_matrix.ndim == 3:
#     print(f"v11 is Block Diagonal with shape {v11_matrix.shape}")
#     n_blocks = v11_matrix.shape[0]
#     cols = int(np.ceil(np.sqrt(n_blocks)))
#     rows = int(np.ceil(n_blocks / cols))
    
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
#     axes = np.atleast_1d(axes).flatten()
    
#     for i in range(n_blocks):
#         im = axes[i].imshow(np.log10(np.abs(v11_matrix[i])), cmap='viridis', interpolation='nearest')
#         axes[i].set_title(f"Block {i}")
#         plt.colorbar(im, ax=axes[i])
    
#     # Hide unused subplots
#     for i in range(n_blocks, len(axes)):
#         axes[i].axis('off')
        
#     plt.suptitle(f"v11 Matrix Blocks - Shape {v11_matrix.shape}")
#     plt.tight_layout()
#     plt.show()
# else:
#     print(f"v11 is Dense with shape {v11_matrix.shape}")
#     plt.imshow(np.abs(v11_matrix), cmap='viridis', interpolation='nearest')
#     plt.title(f"v11 Matrix - Shape {v11_matrix.shape}")
#     plt.colorbar(label='|value|')
#     plt.tight_layout()
#     plt.show()


# print(tracker)
# %% Stage 3 not converged?: Continue training emulator
# 3.) =========================================================================|
# profiler = Profiler()
# profiler.start()

# emu.train_bfgs(options=dict(maxiter=1e5, disp=True))
# emu.save(f'Grid-Emulator_Files/{emu_file_name}.npz')  # Saving the emulator
# print(emu)

# profiler.stop()
# profiler.print()
# %% Stage 4) Plotting the emulator's eigenspectra and weights slice TODO
# 4) ==========================================================================|

#%matplotlib inline
# Inputs: Displayed parameter (1-X), other parameters' fixed index (0-(X-1))
spec.plot_emulator(emu, grid, 1, 1)
# plot_new_eigenspectra(emu, 51)  # <---- Yet to implement

# =============================================================================|

# # %% plot_new_eigenspectra function

# def plot_new_eigenspectra(emulator, params, filename=None):
#     from matplotlib import gridspec
#     """
#     TODO: Correct the deprecated plotting function from Starfish. 
#     Parameters
#     ----------
#     emulator
#     params
#     filename : str or path-like, optional
#         If provided, will save the plot at the given filename

#     Example of a deconstructed set of eigenspectra

#     .. figure:: assets/eigenspectra.png
#         :align: center
#     """
#     weights = emulator.weights[params]
#     X = emulator.eigenspectra * emulator.flux_std
#     reconstructed = weights @ emulator.eigenspectra + emulator.flux_mean
#     reconstructed *= emulator.norm_factor(emulator.grid_points[params])
#     reconstructed = np.squeeze(reconstructed)
#     height = int(emulator.ncomps) * 1.25
#     fig = plt.figure(figsize=(8, height))
#     gs = gridspec.GridSpec(
#         int(emulator.ncomps) + 1,
#         1,
#         height_ratios=[3] + list(np.ones(int(emulator.ncomps))),
#     )
#     ax = plt.subplot(gs[0])
#     ax.plot(emulator.wl, reconstructed, lw=1)
#     ax.set_ylabel("$f_\\lambda$ [erg/cm^2/s/A]")
#     plt.setp(ax.get_xticklabels(), visible=False)
#     for i in range(emulator.ncomps):
#         ax = plt.subplot(gs[i + 1], sharex=ax)
#         ax.plot(emulator.wl, emulator.eigenspectra[i], c="0.4", lw=1)
#         ax.set_ylabel(rf"$\xi_{i}$")
#         if i < emulator.ncomps - 1:
#             plt.setp(ax.get_xticklabels(), visible=False)
#         ax.legend([rf"$w_{i}$ = {weights[i]:.2e}"])
#     plt.xlabel("Wavelength (A)")
#     plt.tight_layout(h_pad=0.2)

#     plt.show()
# # %% # Stage 5) Plotting the emulator's covariance matrix
# # 5) ==========================================================================|

# emu = Emulator.load(f"Grid-Emulator_Files/{emu_file_name}.npz")
random_grid_point = random.choice(emu.grid_points)
print("Random Grid Point Selection")
print(list(emu.param_names))
print(emu.grid_points[0]) # put emu.grid_points[0] 
print(random_grid_point) # or put: random_grid_point in next line
weights, cov = emu(random_grid_point) # here !!!
X = emu.eigenspectra * (emu.flux_std)
flux = (weights @ X) + emu.flux_mean
emu_cov = X.T @ cov @ X
plt.matshow(emu_cov, cmap='Reds')
plt.title("Emulator Covariance Matrix")
plt.colorbar()
plt.show()
# plt.plot(emu.wl, flux)


# %% # Stage 6) Adding observational spectrum as data
# 6) ==========================================================================|

# ---------- Switches here -----------|
# Four methods to select which type of testing spectrum file you want:
# Turn off/on (0/1) to use this method
data_one = 0                # [1] A kgrid grid point
data_two = 0                # [2] A noisy grid point
data_three = 0              # [3] Interpolatation between two grid points
data_four = 1               # [4] A custom test file from python
# -----------------------------------------------------------------------------|

# # ----------- Inputs here ------------|
# if data_one == 1:
#     # File corresponds to grid points in section 2
#     # Parameter's point given by the XX number in the name (6 params = 12 digits)
#     # 040102000000 = 4.5e-10, 16, 1, 1, 1e10, 3
#     file = 'sscyg_k2_040102000001.spec'
    
#     waves, fluxes = np.loadtxt(
#         f'kgrid/sscyg_kgrid090311.210901/{file}', usecols=usecols, unpack=True, skiprows=skiprows)


# if data_two == 1:
#     file = 'sscyg_k2_040102000001.spec'  # File naming same as 1)
#     noise_std = 0.50                    # Percentage noise (0.05 sigma)
    
#     waves, fluxes = np.loadtxt(
#         f'kgrid/sscyg_kgrid090311.210901/{file}', usecols=usecols, unpack=True, skiprows=skiprows)
#     noise = noise_std * np.std(fluxes)
#     for i in range(len(waves)):
#         fluxes[i] = np.random.normal(fluxes[i], noise)       


# if data_three == 1:
#     print('to do') # TODO


# Observational files
if data_four == 1:
    file = 'rwsex_all.csv' # < CHANGEABLE >
    waves, fluxes, errors = np.loadtxt(f'observation_files/{file}', unpack=True, usecols=(0,1,2), delimiter=',', skiprows=1)

# distance corrections 
distance = 223  # pc < CHANGEABLE >
fluxes = fluxes * (distance**2 / 100**2)
errors = errors * (distance**2 / 100**2)
# Data manipulation/truncation into correct format.
wl_range_data = (wl_range[0] + 10, wl_range[1] - 10)  # Truncation
# waves = np.flip(waves)
# fluxes = np.flip(fluxes)
# fluxes = gaussian_filter1d(fluxes, 50)
if scale == 'log':
    fluxes = [np.log10(i) for i in fluxes]  # log
    fluxes = np.array(fluxes)  # log
if scale == 'scaled':
    fluxes /= np.mean(fluxes)
indexes = np.where((waves >= wl_range_data[0]) & (waves <= wl_range_data[1]))  # Truncation + next 2 lines
waves = waves[indexes[0]]
fluxes = fluxes[indexes[0]]
errors = errors[indexes[0]]
raw_flux = list(fluxes)
sigmas = errors
data = Spectrum(waves, fluxes, sigmas=sigmas, masks=None)
data.plot(yscale="linear")
print(file)


# # %% Stage 7) Measuring the autocorrelation of pixels.
# # 7) ==========================================================================|

# # ----- Inputs here ------|
# # Value of sigma (standard deviation) for the size of filter's gaussian
# # kernel in the high pass filter. The larger the value, the smoother the flux
# # data but baseline can warp to y = c
# high_pass_sigma = 200

# # +/- range (aka lags) of the pixels for the autocorrelation plot.
# lags = 500

# # Specify the standard deviation of the quadratric fit boundaries to
# # remove lines.
# percent = 0.5
# # ------------------------|

# def quadratic(x, coefficients):
#     """Returns y value for a given x value and coefficients of a quadratic"""
#     return coefficients[0] * (x**2) + coefficients[1] * (x) + coefficients[2]


# # Alter startfish_flux sigma too (↓) if changing the smoothening term in
# # the class KWDGridInterface(GridInterface):
# # Smoothened data used for the emulator input
# starfish_flux = gaussian_filter1d(raw_flux, 50)
# # Really weird bug if not with list!
# starfish_flux_original = list(starfish_flux)
# # Coefficients of a quadratic fitting routine
# coefficients = np.polyfit(waves, starfish_flux, 2)
# fit = [quadratic(i, coefficients)
#        for i in waves]   # Y values for the quadratic best fit
# std1 = np.std(fit)                     # Percent standard deviation of best fit
# # Plus 1 standard deviation of best fit for detecting lines
# fit_err_plus = fit + (percent * std1)
# # Minus 1 standard deviation of best fit for detecting lines
# fit_err_minus = fit - (percent * std1)

# # Cutting out large emission/absorption lines. Not efficient but too fast to worry about. Reduces large bumps in high pass filter
# # Fluxes greater than 1 std have indexes +/- 10 intervals flux values
# # replaced by the quadratic fit+noise.
# out_of_bounds_indexes = [1 if starfish_flux[i] >= fit_err_plus[i] or starfish_flux[i] <=
#                          fit_err_minus[i] else 0 for i in range(len(waves))]  # Detecting lines >1std from fit
# for i in range(len(waves)):
#     if out_of_bounds_indexes[i] == 1:
#         if i < 10:  # if/elif/else statement checking boundaries to stop errors
#             waves_limit = range(0, i + 11)
#         elif i > (len(waves) - 11):
#             waves_limit = range(i - 10, len(waves))
#         else:
#             waves_limit = range(i - 10, i + 11)
#         for j in waves_limit:  # +/- 10 flux intervals
#             # change line flux with fit flux plus 5% noise.
#             starfish_flux[j] = np.random.normal(fit[j], (0.05 * std1))

# # High-pass filter for the starfish flux trend.
# smooth_flux = gaussian_filter1d(starfish_flux, high_pass_sigma)
# # Starfish data with underlying flux removed.
# adj_flux = starfish_flux - smooth_flux

# plt.plot(
#     waves,
#     starfish_flux_original,
#     color="grey",
#     label='Starfish Flux',
#     linewidth=0.5)
# plt.plot(
#     waves,
#     starfish_flux,
#     color='red',
#     label='Starfish Flux Lines Removed',
#     linewidth=0.5)
# plt.plot(
#     waves,
#     fit_err_plus,
#     color='orange',
#     label=f'Quadratic fit + {percent}$\\sigma$',
#     linewidth=0.5)
# plt.plot(
#     waves,
#     fit_err_minus,
#     color='orange',
#     label=f'Quadratic fit - {percent}$\\sigma$',
#     linewidth=0.5)
# plt.plot(waves, fit, color='purple', label='Quadratic fit', linewidth=0.5)
# plt.plot(waves, smooth_flux, color='green',
#          label='Smoothened/High Pass Filter')
# if scale == 'linear' or scale == 'scaled':
#     plt.plot(
#         waves,
#         adj_flux,
#         color='blue',
#         label='Flux Filter Adjusted',
#         linewidth=0.5)
# plt.xlabel('$\\lambda [\\AA]$')
# plt.ylabel('$f_\\lambda$ [$erg/cm^2/s/cm$]')
# plt.title('Showing Adjusted Starfish Flux From Being Passed Through \n A High-pass Filter And Quadratic Fit Boundaries To Remove Lines')
# plt.legend()
# plt.show()
# if scale == 'log':
#     plt.plot(
#         waves,
#         adj_flux,
#         color='blue',
#         label='Flux Filter Adjusted',
#         linewidth=0.5)
#     plt.xlabel('$\\lambda [\\AA]$')
#     plt.ylabel('$f_\\lambda$ [$erg/cm^2/s/cm$]')
#     plt.legend()
#     plt.show()

# # Numpy's autocorrelation method for comparison checks.
# mean = np.mean(adj_flux)
# var = np.var(adj_flux)
# ndata = adj_flux - mean
# acorr = np.correlate(ndata, ndata, 'same')
# acorr = acorr / var / len(ndata)
# pixels = np.arange(len(acorr))
# if len(acorr) % 2 == 0:
#     pixels = pixels - len(acorr) / 2
# else:
#     pixels = pixels - len(acorr) / 2 + 0.5
# indx = int(np.where(pixels == 0)[0])
# lagpixels = pixels[indx - lags:indx + lags + 1]
# lagacorr = acorr[indx - lags:indx + lags + 1]
# plt.plot(
#     lagpixels,
#     lagacorr,
#     color='green',
#     label='Using Numpy Correlate',
#     linewidth=0.5)
# plt.title(
#     f'Adjusted (Starfish Flux - Smoothened {high_pass_sigma}$\\sigma$) Autocorrelation using Full Spectrum')
# plt.ylabel('ACF')
# plt.xlabel('Pixels')
# plt.legend()
# plt.show()

# %% Stage 8.1) Kernel Calculators"""
# 8.1) ========================================================================|

# TODO: Implement kernel calculators for the emulator's global matrix

# %% Stage 8.2) Search grid indexes helper
# =============================================================================|

spec.search_grid_points(1, emu, grid) # <-- 1/0 switch for on/off

# %% Stage 9) Assigning the model and initial model plot"""
# 9) ==========================================================================|

# ----- Inputs here ------|
# Natural logarithm of the global covariance's Matern 3/2 kernel amplitude
# log=-52 'linear', log=-8 'log'
log_amp = -52
# 5Natural logarithm of the global covariance's Matern 3/2 kernel lengthscale
log_ls = 5
# ------------------------|

model = SpectrumModel(
    f'Grid-Emulator_Files/{emu_file_name}.npz',
    data,
    # [list, of , grid , points]emu.grid_points[119] [-8.95, 10.26, 1.82]
    grid_params=list(emu.grid_points[1900]),
    Av=0,
    global_cov=dict(log_amp=log_amp, log_ls=log_ls)
)
print(model)
model.plot(yscale="linear")

model_flux, model_cov = model()
plt.matshow(model._glob_cov, cmap='Greens')
plt.title("Global Covariance Matrix")
plt.colorbar()
plt.show()
plt.matshow(model_cov, cmap='Blues')
plt.title("Sum Covariance Matrix")
plt.colorbar()
plt.show()
model.freeze("Av")
print("-- Model Labels --")
print(model.labels)

# %% Stage 10) Assigning the mcmc priors
# 10) =========================================================================|

# Default_priors contains a distribution for every possible parameter
# Mostly uniform across grid space bar global_cov being normal
# Change the default distrubtion if you wish something different.
# WARNING! st.uniform(x, y) is range(x, x+y)
# if kgrid == 1:
#     default_priors = {
#         "param1": st.uniform(1.0e-10, 2.9e-9),
#         "param2": st.uniform(4, 28),
#         "param3": st.uniform(0.0, 1.0),
#         "param4": st.uniform(1.0, 2.0),
#         "param5": st.uniform(1e+10, 6e+10),
#         "param6": st.uniform(1.0, 5.0),
#         "global_cov:log_amp": st.norm(log_amp, 10),
#         "global_cov:log_ls": st.uniform(0.1, 10.9),
#         "Av": st.uniform(0.0, 1.0)
#     }
# if shortspec == 1:
#     default_priors = {
#         # log10 values
#         "param1": st.uniform(np.log10(4e-11), (np.log10(3e-9) - np.log10(4e-11))),
#         "param2": st.uniform(2, 14),
#         "param3": st.uniform(1.0, 2.0),
#         "global_cov:log_amp": st.norm(log_amp, 1),
#         "global_cov:log_ls": st.uniform(1, 7),
#         "Av": st.uniform(0.0, 1.0)
#     }

# if broadshortspec == 1:
#     default_priors = {
#         # log10 values
#         "param1": st.uniform(np.log10(4e-11), (np.log10(2.5e-8) - np.log10(4e-11))),
#         "param2": st.uniform(2, 14),
#         "param3": st.uniform(1.0, 2.0),
#         "global_cov:log_amp": st.norm(log_amp, 1),
#         "global_cov:log_ls": st.uniform(1, 7),
#         "Av": st.uniform(0.0, 1.0)
#     }

# ----------------------------------------|
# | 1) disk.mdot (msol/yr)
# | 2) wind.mdot (disk.mdot)
# | 3) KWD.d(in_units_of_rstar)
# | 4) KWD.mdot_r_exponent  
# | 5) KWD.acceleration_length(cm) 
# | 6) KWD.acceleration_exponent 
# | 7) Boundary_layer.luminosity(ergs/s)
# | 8) Boundary_layer.temp(K)
# | 9) Inclination angle (degrees: 30,55,80)
# | max_wl_range = (800,8000)


# WARNING! st.uniform(x, y) is range(x, x+y)
if speculate_cv_no_bl_grid_v87f == 1:
    default_priors = {
        "param1": st.uniform(np.log10(3e-9), 1),  # disk.mdot (log10)
        "param2": st.uniform(0.03, 0.27),           # wind.mdot
        "param3": st.uniform(0.55, 54.45),          # KWD.d(in_units_of_rstar)
        "param4": st.uniform(0.0, 1.0),             # KWD.mdot_r_exponent  
        "param5": st.uniform(np.log10(7.25182e+08), 2.0), # KWD.acceleration_length(cm) (log10)
        "param6": st.uniform(0.5, 4.0),             # KWD.acceleration_exponent 
        "param9": st.uniform(30, 50),               # Inclination (sparse)
        #"param10": st.uniform(30, 50),              # Inclination (10deg)
        #"param11": st.uniform(30, 55),              # Inclination (5deg)
        "global_cov:log_amp": st.norm(log_amp, 10),
        "global_cov:log_ls": st.uniform(0.1, 10.9),
        "Av": st.uniform(0.0, 1.0)
    }
if speculate_cv_bl_grid_v87f == 1:
    default_priors = {
        "param1": st.uniform(np.log10(3e-9), 1.0),  # disk.mdot (log10)
        "param2": st.uniform(0.03, 0.27),           # wind.mdot
        "param3": st.uniform(0.55, 54.45),          # KWD.d(in_units_of_rstar)
        "param4": st.uniform(0.0, 1.0),             # KWD.mdot_r_exponent  
        "param5": st.uniform(np.log10(7.25182e+08), 2.0), # KWD.acceleration_length(cm) (log10)
        "param6": st.uniform(0.5, 4.0),             # KWD.acceleration_exponent 
        "param7": st.uniform(0.0, 1.0),             # Boundary_layer.luminosity(ergs/s)
        "param8": st.uniform(0.1, 0.9),             # Boundary_layer.temp(K)
        "param9": st.uniform(30, 50),               # Inclination (sparse)
        #"param10": st.uniform(30, 50),              # Inclination (10deg)
        #"param11": st.uniform(30, 55),              # Inclination (5deg)
        "global_cov:log_amp": st.norm(log_amp, 10),
        "global_cov:log_ls": st.uniform(0.1, 10.9),
        "Av": st.uniform(0.0, 1.0)
    }

priors = {}  # Selects the priors required from the model parameters used
for label in model.labels:
    priors[label] = default_priors[label]  # if label in default_priors:

# %% Stage 11) Training model with scipy.optimise.minimize(nelder-mead method)
# 11) =========================================================================|

# TODO: SIMPLEX - Need to add global covariance hyperparameters
initial_simplex = spec.simplex(model, priors) 

# Progress bar for training
maxiter = 10000
pbar = tqdm(total=maxiter, desc="Training Progress")

def callback(xk):
    pbar.update(1)

try:
    model.train(
        priors,
        callback=callback,
        options=dict(
            maxiter=maxiter,
            disp=True,
            initial_simplex=initial_simplex,
            return_all=True))
finally:
    pbar.close()

print(model)

# %% Stage 11.continued) Continue training the model
# 12) =========================================================================|

model.train(priors, options=dict(maxiter=1e5, disp=True))
print(model)

# %% Stage 12.1) Saving and plotting the trained model
# 12.1) =======================================================================|

model.plot(yscale="linear")
model.save("Grid-Emulator_Files/Grid_full_MAP.toml")

# %% Stage 12.2) Reloading the trained model
# 12.2) =======================================================================|

model.load("Grid-Emulator_Files/Grid_full_MAP.toml")
model.freeze("global_cov")
print(model.labels)

# %% Stage 13) Set walkers initial positions/dimensionality and mcmc parameters
# 13) =========================================================================|
#TODO : Everything onwards need to be improved
#os.environ["OMP_NUM_THREADS"] = "1"
#mp.set_start_method('fork', force=True)

# ----- Inputs here ------|
ncpu = cpu_count() - 2      # Pool CPU's used.
nwalkers = 1 * ncpu         # Number of walkers in the MCMC.
# Maximum iterations of the MCMC if convergence is not reached.
max_n = 2000
extra_steps = int(max_n / 10)  # Extra MCMC steps
# ------------------------|

ndim = len(model.labels)
print("{0} CPUs".format(ncpu))
# if kgrid == 1:
#     default_scales = {"param1": 1e-11, "param2": 1e-2, "param3": 1e-2,
#                       "param4": 1e-2, "param5": 1e+9, "param6": 1e-2}
# if shortspec == 1:
#     default_scales = {"param1": 1e-1, "param2": 1e-1, "param3": 1e-1}
    
# if broadshortspec == 1:
#     default_scales = {"param1": 1e-1, "param2": 1e-1, "param3": 1e-1}
if speculate_cv_no_bl_grid_v87f == 1:
    default_scales = {
        "param1": 1e-1,   # disk.mdot (log10)
        "param2": 1e-2,   # wind.mdot
        "param3": 1e-1,   # KWD.d(in_units_of_rstar)
        "param4": 1e-1,   # KWD.mdot_r_exponent  
        "param5": 1e+1,   # KWD.acceleration_length(cm) (log10)
        "param6": 1e-1,   # KWD.acceleration_exponent 
        "param9": 5.0,    # Inclination (sparse)
        #"param10": 5.0,   # Inclination (10deg)
        #"param11": 2.5,   # Inclination (5deg)
        # "global_cov:log_amp": 1.0,
        # "global_cov:log_ls": 1.0,
        # "Av": 0.1
    }
if speculate_cv_bl_grid_v87f == 1:
    default_scales = {
        "param1": 1e-1,   # disk.mdot (log10)
        "param2": 1e-2,   # wind.mdot
        "param3": 1e-1,   # KWD.d(in_units_of_rstar)
        "param4": 1e-1,   # KWD.mdot_r_exponent  
        "param5": 1e+1,   # KWD.acceleration_length(cm) (log10)
        "param6": 1e-1,   # KWD.acceleration_exponent 
        "param7": 1.0,    # Boundary_layer.luminosity(ergs/s)
        "param8": 0.1,    # Boundary_layer.temp(K)
        "param9": 5.0,    # Inclination (sparse)
        #"param10": 5.0,   # Inclination (10deg)
        #"param11": 2.5,   # Inclination (5deg)
        # "global_cov:log_amp": 1.0,
        # "global_cov:log_ls": 1.0,
        # "Av": 0.1
    }
scales = {}  # Selects the priors required from the model parameters used
for label in model.labels:
    scales[label] = default_scales[label]

# Initialize gaussian ball for starting point of walkers
# scales = {"c1": 1e-10, "c2": 1e-2, "c3": 1e-2, "c4": 1e-2}
# model = model_ball_initial
ball = np.random.randn(nwalkers, ndim)
for i, key in enumerate(model.labels):
    ball[:, i] *= scales[key]
    ball[:, i] += model[key]

# %% Stage 14) Running MCMC, maximizing and setting up our backend/sampler
# 14) =========================================================================|

def log_prob(P, priors):
    model.set_param_vector(P)
    return model.log_likelihood(priors)


backend = emcee.backends.HDFBackend(
    "Grid-Emulator_Files/Grid_full_MCMC_chain.npz")
backend.reset(nwalkers, ndim)

#with Pool(ncpu) as pool:
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_prob, args=(priors,), backend=backend
) #pool=pool goes here

index = 0  # Tracking how the average autocorrelation time estimate changes
autocorr = np.empty(max_n)

old_tau = np.inf  # This will be useful to testing convergence

# Now we'll sample for up to max_n steps
for sample in sampler.sample(ball, iterations=max_n, progress=True):
    # Only check convergence every 10 steps
    if sampler.iteration % 10:
        continue
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
    # skip math if it's just going to yell at us
    if np.isnan(tau).any() or (tau == 0).any():
        continue
        # Check convergence
    converged = np.all(tau * 10 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        print(f"Converged at sample {sampler.iteration}")
        break
    old_tau = tau

sampler.run_mcmc(backend.get_last_sample(), extra_steps, progress=True)

# %% Stage 15) Plotting raw MCMC chains
# 15) =========================================================================|

reader = emcee.backends.HDFBackend(
    "Grid-Emulator_Files/Grid_full_MCMC_chain.npz")
full_data = az.from_emcee(reader, var_names=model.labels)
flatchain = reader.get_chain(flat=True)
walker_plot = az.plot_trace(full_data)

# %% Stage 16) Discarding MCMC burn-in
# 16) =========================================================================|

tau = reader.get_autocorr_time(tol=0)
if m.isnan(tau.max()):
    burnin = 0
    thin = 1
    print(burnin, thin)
else:
    burnin = int(tau.max())
    thin = int(0.3 * np.min(tau))
burn_samples = reader.get_chain(discard=burnin, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, thin=thin)
dd = dict(zip(model.labels, burn_samples.T))
burn_data = az.from_dict(dd)

# %% Stage 17) Chain trace and summary
# 17) =========================================================================|

# Plotting the mcmc chains without the burn-in section,
# summarise our mcmc run's parameters and analysis,
# plot our posteriors of each paramater,
# produce a corner plot of our parameters.
burnt_walker_plot = az.plot_trace(burn_data)
print(az.summary(burn_data, round_to=None))
burnt_posteriors = az.plot_posterior(burn_data, [i for i in model.labels])

# %% Stage 18) Cornerplot of our parameters. 
# 18) =========================================================================|

# See https://corner.readthedocs.io/en/latest/pages/sigmas/
sigmas = ((1 - np.exp(-0.5)), (1 - np.exp(-2)))
cornerplot = corner.corner(
    burn_samples.reshape((-1, len(model.labels))),
    labels=model.labels,
    show_titles=True,
    truths=[-8.161150909, 10, 1.8] #list(emu.grid_points[32])
)
 #   quantiles=(0.05, 0.16, 0.84, 0.95),levels=sigmas,

# %% Stage 19) Plotting Best Fit MCMC parameters
# 19) =========================================================================|

# We examine our best fit parameters from the mcmc chains, plot and save our
# final best fit model spectrum.
ee = [np.mean(burn_samples.T[i]) for i in range(len(burn_samples.T))]
ee = dict(zip(model.labels, ee))
model.set_param_dict(ee)
print(model)
model.plot(yscale="linear")
model.save("Grid-Emulator_Files/Grid_full_parameters_sampled.toml")


# %% Processing-time data sets

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Plotting the training time of the emulator CIV
# Datasets
number_of_pca_components = [2, 3, 4, 5, 6, 7, 8, 9, 10] # x-axis
first_train_time_train = [0.7, 2.8, 11.1, 19.2, 46.1, 118.4, 186.9, 249, 335.3] # y1-axis
second_train_time_train = [0.5, 1.7, 6.4, 19.7, 58.5, 92.4, 154.5, 203.5, 288.3] # y2-axis

number2_of_pca_components = [11, 12, 13, 14, 15] # x-axis
first_train_time_test = [545.5, 966.8, 1178.9, 991.9, 983.3] # y1-axis
second_train_time_test = [490.2, 650.5, 900.0, 825.6, 1063.6] # y2-axis

model_linear = LinearRegression(fit_intercept=True)
model_linear.fit(np.array(number_of_pca_components).reshape(-1, 1), np.array(first_train_time_train).reshape(-1, 1))
time_prediction_linear = model_linear.predict(np.array([11,12,13,14,15]).reshape(-1, 1))
print(model_linear.coef_, 'coefficients')
print(time_prediction_linear, 'prediction')
print(mean_squared_error(first_train_time_test, time_prediction_linear, squared=False), 'MSE')
print(model_linear.score(np.array(number_of_pca_components).reshape(-1, 1), np.array(first_train_time_train).reshape(-1, 1)), 'R^2')

#Linear Regression model to predict the training time for larger PCA components
model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=True))])
model.fit(np.array(number_of_pca_components).reshape(-1, 1), np.array(first_train_time_train).reshape(-1, 1))
time_prediction = model.predict(np.array([11,12,13,14,15]).reshape(-1, 1))
print(model.named_steps['linear'].coef_)
print(time_prediction)
print(mean_squared_error(first_train_time_test, time_prediction, squared=False), 'MSE')

model2 = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=True))])
model2.fit(np.array(number_of_pca_components).reshape(-1, 1), np.array(second_train_time_train).reshape(-1, 1))
time_prediction2 = model2.predict(np.array([11,12,13,14,15]).reshape(-1, 1))
print(model2.named_steps['linear'].coef_)
print(time_prediction2)
print(mean_squared_error(second_train_time_test, time_prediction2, squared=False), 'MSE')

# Displaying the plots
plt.plot(number_of_pca_components, first_train_time_train, label='First Training', color='red')
plt.plot(number_of_pca_components, second_train_time_train, label='Second Training', color='blue')
plt.plot(number2_of_pca_components, time_prediction_linear, label='Prediction Linear')
plt.plot(number2_of_pca_components, time_prediction, label='Prediction')
plt.plot(number2_of_pca_components, time_prediction2, label='Prediction 2')
plt.plot(number2_of_pca_components, first_train_time_test, label='First Testing', color='red')
plt.plot(number2_of_pca_components, second_train_time_test, label='Second Testing', color='blue')
plt.xlabel('Number of PCA Components')
plt.ylabel('Training Time (s)')
plt.title('Training Time of the Emulator')
plt.legend()
plt.show()
# %%

# %%
