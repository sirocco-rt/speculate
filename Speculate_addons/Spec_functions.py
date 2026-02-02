# Speculate functions

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import itertools
import matplotlib
from typing import Optional
from tqdm import tqdm
import bisect
from Starfish.emulator import Emulator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, CheckButtons, Button
from Starfish.transforms import rescale, _get_renorm_factor

def plot_emulator(emulator, grid, not_fixed, fixed):
    
    """Takes the emulator and given the emulator's inputed model parameters, 
    displays the weights and Gaussian process interpolation from the SVD (PCA decomposition).

    Args:
        emulator (Starfish.emulator.emulator): Trained emulator from your grid space
        
        removed: model_parameters (tuple): The numbers corresponding to the modelling parameters of the grid
        
        not_fixed (int): The varying model parameter number that the weights plot displays (x-axis)
        
        fixed (int): The python list index of the other model parameters. 
            If a model parameter has 5 values with the grid space. 
            Possible int ranges would be 0-4.
        
    Returns:
        Plot of PCA component weights.
    """
    
    # Placing the grid points values within a dictionary, keyed as 'params{}'
    variables = {}
    for loop in grid.model_parameters:
        variables["param{}".format(loop)] = np.unique(emulator.grid_points[:, grid.model_parameters.index(loop)])
        
    # Creating a custom itertools.product routine which can dynamically input the free varying parameter
    # and the length of the number of parameters depending on what is specified. 
    # params = np.array(list(itertools.product(T, logg[:1], Z[:1]))) # <-- starfish original
    not_fixed_index = grid.model_parameters.index(not_fixed) # Converting parameter number to index position
    params = []
    temp = [variables[emulator.param_names[j]] for j in range(len(variables))] # Creating list from dictionary
    temp2 = [np.array(temp[i]) if i==not_fixed_index else temp[i][fixed] for i in range(len(temp))] # New list fixing the other parameters on the given grid point
    for j in range(len(temp2[not_fixed_index])): # Itertools.product calculation into the same original formatting
        params.append(tuple([temp2[i][j] if temp2[i].size>1 else temp2[i] for i in range(len(temp2))]))
    params = np.array(params)
    idxs = np.array([emulator.get_index(p) for p in params])
    weights = emulator.weights[idxs.astype("int")].T
    if emulator.ncomps < 4:
        fix, axes = plt.subplots(emulator.ncomps, 1, sharex=True, figsize=(8,(emulator.ncomps-1)*2))
    else:
        fix, axes = plt.subplots(
            int(np.ceil(emulator.ncomps/2)), 2, sharex=True, figsize=(13,(emulator.ncomps-1)*2),)
    axes = np.ravel(np.array(axes).T)
    [ax.set_ylabel(f"$weights_{i}$") for i, ax in enumerate(axes)]
    
    param_x_axis = np.unique(emulator.grid_points[:,not_fixed_index]) # Picking out all unique not fixed parameter values
    for i, w in enumerate(weights):
        axes[i].plot(param_x_axis, w, "o")
        
    # Again as above, dynamical input for the gaussian process errors to be plotted for the specified parameter
    param_x_axis_test = np.linspace(param_x_axis.min(), param_x_axis.max(), 100)
    temp2[not_fixed_index] = param_x_axis_test
    Xtest = []
    for j in range(len(temp2[not_fixed_index])):
        Xtest.append(tuple([temp2[i][j] if temp2[i].size>1 else temp2[i] for i in range(len(temp2))]))
    Xtest = np.array(Xtest)
    mus = []
    covs = []
    for X in Xtest:
        m, c = emulator(X)
        mus.append(m)
        covs.append(c)
    mus = np.array(mus)
    covs = np.array(covs)
    sigs = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1))
    xlabel = grid.parameters_description()[f"param{not_fixed}"]
    for i, (m, s) in enumerate(zip(mus.T, sigs.T)):
        axes[i].plot(param_x_axis_test, m, "C1")
        axes[i].fill_between(param_x_axis_test, m - (2 * s), m + (2 * s), color="C1", alpha = 0.4)
        axes[i].set_xlabel(f"Parameter $log_{{10}}({xlabel})$")
    plt.suptitle(f"Weights for Parameter {xlabel} with the other parameters fixed to their {fixed} index grid point", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
def grid_spectrum_plot(grid, wl_range):
    pass

def emulator_spectrum_plot(emu, wl_range):
    pass

def simplex(model, priors):
    """An initial simplex algorithm for training the model's Nelder_Mead routine. 
    Suitable for normal or uniform distributions only. If different add method to simplex routine. 
    We divide the each parameter's range into intervals with equal spacing, equal to the 
    number of parameters. Each simplex column point's, (num of parameters+1) in total, contains interval values
    of an individual parameter's range. Each simplex column (parameter values) is then cycled + 1 compared to each 
    other with np.roll. i.e a rolling simplex (see numpy.roll).
     
    Args:
        model (Starfish.models.spectrum_model.SpectrumModel): A starfish model class with emulator and data
        set built in.
        priors (dictionary): Parameter priors for the minimisation training and the MCMC. 

    Returns:
        simplex: An initial simplex to use for the nelder_mead routine when training the model class with scipy.optimise.minimise.
    """
    
    def uniform_prior(distribution, N):
        "Method for generating simplex values for a uniform distribution prior parameter"
        min_value = distribution.args[0]
        max_value = distribution.args[1] + distribution.args[0]
        the_range = max_value - min_value
        truncated_perc = the_range / 20 # reducing by 20% for both ends of the range 
        truncated_min = min_value + truncated_perc
        truncated_max = max_value - truncated_perc
        truncated_range = truncated_max - truncated_min
        interval = truncated_range / N
        data_column = [truncated_min + interval * multiple for multiple in range(N+1)] # even value distribution across range
        return data_column
    
    def normal_prior(distribution, N):
        "Method for generating simplex values for a noramlly prior parameter"
        mean = distribution.args[0]
        std = distribution.args[1]
        min_value = mean - (std*2) # +/- 2 standard deviations for the range
        max_value = mean + (std*2)
        the_range = max_value - min_value
        interval = the_range / N
        data_column = [min_value + interval * multiple for multiple in range(N+1)] # even value distribution across range
        return data_column
    
    N = len(model.get_param_vector()) # Number of training parameters
    simplex = np.zeros((N+1, N)) # Required size of the simplex for given number of parameters
    iteration = 0 # loop count to add to the simplex columns
    for name, distribution in priors.items(): # priors being used
        if distribution.dist.name == 'uniform': # check if distribution is uniform
            data_column = uniform_prior(distribution, N) # returns an even data vector/'column'
        elif distribution.dist.name == 'norm': # check if distribution is normal
            data_column = normal_prior(distribution, N) # returns an even data vector/'column'
        else:
            raise ValueError(f"Error: {name} is {distribution.dist.name} and not a uniform or normal distribution") 
        simplex[:,iteration] = data_column # adding data column to simplex
        print(f"Simplex {name}: min={min(data_column):.4f}, max={max(data_column):.4f}")
        iteration += 1 # next loop

    # rolling the simplex columns +1 each additional column
    for column in range(N):
        new_column = np.roll(simplex[:,column], column)
        simplex[:,column] = new_column

    return simplex

def is_matrix_pos_def(square_matrix):
    """Simple check to see if square matrix is positive definite. 
        If all eigenvalues are greater than 0. Then true is returned.
        For semi-definite, set to greater than or equal to 0

    Returns:
        boolean: True if positive definite matrix, False if not.
    """
    return np.all(np.linalg.eigvals(square_matrix) > 0)


def unique_grid_combinations(grid):
    """Method to return the unique combinations of grid points in a list.
    
    Returns:
        list: A list of the unique combinations of grid points.
    """
    unique_combinations = []
    for i in itertools.product(*grid.points):
        unique_combinations.append(list(i))
    
    return unique_combinations


def search_grid_points(switch, emu, grid):
    if switch == 1:
        print("- Search index range 0 to {} to find the associated grid point values.".format(len(emu.grid_points) - 1))
        print("- Type '-1' to stop searching.")
        print("- Or type [Input, Parameter, Values] in a list for the specific grid point.")
        print("- No square brackets needed, just commas.")
        print("- Increasing the index increases the parameters grid points like an odometer.")
        print("---------------------------------------------------------------")
        print("Names:", emu.param_names)
        print("Description:", [grid.parameters_description()[i] for i in emu.param_names])
        while True:
            user_input = input("Enter Index Value or {} Parameter Values Separated By Commas".format(
                len(emu.param_names)))
            user_input = user_input.split(",") # turning string input into list
            print("-----------------------------------------------------------")
            if len(user_input) == 1:
                index = int(user_input[0]) # variable for integer index
                grid_vals = None # resetting previous variables for other scenarios.
            else:
                grid_vals = [float(i) for i in user_input] # for parameter list
                index = None
            
            # different senarios for user inputs, index or parameter values    
            grid_points = list(emu.grid_points) # fix for enumerating 
            if index == -1: # quit
                break
            
            # integer grid point search
            elif isinstance(index, int) and 0 <= index <= len(emu.grid_points):
                better_display = [grid_points[index][i] for i in range(len(grid_points[index]))]
                print("Emulator grid point index {} is".format(index), 
                      better_display)
            
            # parameter grid point search        
            elif isinstance(grid_vals, list) and len(grid_vals) == len(emu.param_names):
                skipover = True # A variable to print if point not in emulator
                for index, points in enumerate(grid_points): # enumerating to get index
                    # if there is a grid point that matches the input
                    if all(np.round(points,3) == np.round(grid_vals,3)):
                        print("Parameter's {} is at emulator grid point index {}".format(grid_vals, index))
                        skipover = False # to avoid printing the skipover message
                        break # stop searching enumeration
                if skipover:
                    print("Grid point {} is not in the emulator".format(grid_vals))
                    
            # invalid input, can't find grid point
            else:
                print("{} Not valid input! \nType integer between 0 and {}".format(
                    user_input, len(emu.grid_points) - 1) + 
                    ", a parameter list of length {}".format(
                        len(emu.grid_points[0])) + " or '-1' to quit")

                
class InspectGrid:
    """The InspectGrid Class is a spectral data exploration tool. The Class 
    should be opened in a new window which allows the user to explore the grid and 
    emulator space. 
    
    Hence, run %matplotlib qt in the script before running this.
    
    Upon loading, an animation of the grid space will run. The animation can be 
    paused by clicking the animation button or changing the grid point slider
    manually. The animation speed can be changed with the animation speed slider.
    
    There are checkboxes located around the window to activate and deactivate
    the plotting of the spectra for the current grid points, the current 
    emulator points and any fixed spectra. The fixed spectra are added by the 
    user with the add spectrum button. The fixed spectra can be cleared with
    clear spectrum button. 
    
    The param checkbuttons fix the grid and emulator slider to certain values. 
    This means the user can skip over certain points which aren't of interest. 
    Whether this is if the user already knows the param1 parameter value, hence 
    searching different values of param1 would be unnecessary. Or if the user
    if interested to see how a certain parameter affects the spectrum. This can 
    be investigated too. The sliders change which possible grid space parameter
    value is used. 
    
    The description of the parameters (eg, param1) is printed in the terminal
    or interactive window as a dictionary. 
    
    The tool can be used with or without an emulator. This means the user can 
    use the tool to search the grid space before deciding how to train the 
    emulator. Simply remove the emu argument before initialising the class.
    
    # TODO: Can improve the class to be more pythonic as I get better at OOP.
    
    Args: grid (Spec_gridinterfaces): The grid space to be explored.
            The Spec_gridinterfaces class is a child class of 
            Starfish.grid_tools.GridInterface.
    
        emu: Optional[Emulator] = None (Starfish.emulator.emulator): 
            The emulator to be explored. If None, the class ignores the emulator 
            space and removes the buttons associated with the emulator. 
        
    """
    def __init__(self, grid, emu: Optional[Emulator] = None): 
        # Controlling the grid space ------------------------------------------|
        self.grid = grid # adding grid to class
        
        # Creating a list of all unique combinations of parameters
        # Spec_functions.py unique_grid_combinations() function
        self.unique_combinations = []
        for i in itertools.product(*grid.points):
            self.unique_combinations.append(list(i))
            
        self.all_unique_combinations = self.unique_combinations.copy()
        print("Available parameters:")
        print(self.grid.parameters_description())
        
        # Finding the min/max flux values for the entire grid space    
        self.entire_grid_fluxes = []
        self.axis_min_flux = 0
        self.axis_max_flux = 0 # low to ensure first max flux is higher
        for indexes in range(len(self.unique_combinations)):
            flux = grid.load_flux(self.unique_combinations[indexes])
            self.entire_grid_fluxes.append(flux)
            max_flux = max(flux) # assigned for 1 function evaluation, not 3
            if max_flux > self.axis_max_flux:
                self.axis_max_flux = max_flux # finding the highest flux value
        
        # Producing a dimensionless data set for the entire grid space
        self.dimensionless_grid_fluxes = np.array(self.entire_grid_fluxes)
        # Fluxes.mean(1) does the mean across the rows of the flux grid -
        # i.e. the mean flux of each spectrum.
        norm_factors = self.dimensionless_grid_fluxes.mean(1)
        self.dimensionless_grid_fluxes /= norm_factors[:, np.newaxis]
        # fluxes.mean(0) does the mean across the columns - 
        # i.e. the mean flux at each wavelength bin.
        flux_mean = self.dimensionless_grid_fluxes.mean(0)
        self.dimensionless_grid_fluxes -= flux_mean
        # fluxes.std(0) does the standard deviation across the columns -
        # i.e. the standard deviation of the flux at each wavelength bin.
        # flux_std = self.dimensionless_grid_fluxes.std(0) #changed
        # self.dimensionless_grid_fluxes /= flux_std # changed
        # Set min and max values for the y axis of the plot
        self.dimensionless_axis_min_flux = 1e50 # Initialisationa at high values
        self.dimensionless_axis_max_flux = 1e-50
        for indexes in range(len(self.dimensionless_grid_fluxes)): # iterate spectra
            flux = self.dimensionless_grid_fluxes[indexes]
            if min(flux) < self.dimensionless_axis_min_flux:
                self.dimensionless_axis_min_flux = min(flux)
            if max(flux) > self.dimensionless_axis_max_flux:
                self.dimensionless_axis_max_flux = max(flux)
        
        # Initialising plot and adding grid point slider
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.subplots_adjust(left=0.25, bottom=0.35) # adjusting plot size to fit slider
        grid_axis = self.fig.add_axes([0.25, 0.2, 0.55, 0.03]) # Slider shape
        # grid_slider_list is an updating list of grid points for the slider if freezing parameters
        self.grid_slider_list = [i for i in range(len(self.unique_combinations))]
        self.grid_slider = Slider(grid_axis,
                                  'Grid Point',
                                  0,
                                  len(self.unique_combinations)-1,
                                  valinit=0,
                                  valstep=self.grid_slider_list,
                                  initcolor='none',
                                  handle_style={'facecolor':'black'},
                                  color='#268BD2')
        # If slider changes, call function to update plot to the new grid point.
        self.grid_slider.on_changed(self.slider_updating_spectrum)
        
        # Plot Style (⌐▀͡ ̯ʖ▀)
        logo = matplotlib.image.imread('assets/logos/Speculate_logo.png')
        logo_ax = self.fig.add_axes([0.86, 0.85, 0.15, 0.15])
        logo_ax.imshow(logo)
        logo_ax.axis('off') # Stops a graph of the logo
        plt.style.use('Solarize_Light2') # Theme

        # Dimensionless spectra button
        dimensionless_data_box = self.fig.add_axes([0.10, 0.9, 0.10, 0.04]) # Box shape
        self.dimensionless_data = False
        self.dimensionless_data_checkbox = CheckButtons(dimensionless_data_box, [' Dimensionless'], [self.dimensionless_data])
        self.dimensionless_data_checkbox.on_clicked(self.dimensionless_data_function)
        
        # Visibility of grid plots button
        grid_box = self.fig.add_axes([0.85, 0.2, 0.06, 0.04]) # Grid box shape
        self.grid_visible = True
        self.grid_checkbox = CheckButtons(grid_box, [' Visible'], [self.grid_visible])
        self.grid_checkbox.on_clicked(self.grid_check_box)
        
         # Visibility of fixed plots button
        fixed_box = self.fig.add_axes([0.015, 0.025, 0.06, 0.04]) # Grid box shape
        self.fixed_visible = True
        self.fixed_checkbox = CheckButtons(fixed_box, [' Visible'], [self.fixed_visible])
        self.fixed_checkbox.on_clicked(self.fixed_check_box)       
        
        # Grid point arrow button to +/-1 grid point
        grid_larrow_ax = self.fig.add_axes([0.1, 0.2, 0.03, 0.04]) # Arrow shape
        self.gla = Button(grid_larrow_ax, '<') # Grid left arrow (gla)
        self.gla.on_clicked(self.grid_left_arrow)
        grid_rarrow_ax = self.fig.add_axes([0.13, 0.2, 0.03, 0.04]) # Arrow shape
        self.gra = Button(grid_rarrow_ax, '>') # Grid right arrow (gra)
        self.gra.on_clicked(self.grid_right_arrow)
        
        # Add grid point spectrum permanently and clear buttons
        add_g_spectrum = self.fig.add_axes([0.01, 0.2, 0.03, 0.04]) # Button shape
        self.ags = Button(add_g_spectrum, 'Add')
        self.ags.on_clicked(self.add_grid_spectrum)
        clear_g_spectrum = self.fig.add_axes([0.05, 0.2, 0.03, 0.04]) # Button shape
        self.cgs = Button(clear_g_spectrum, 'Clear')
        self.cgs.on_clicked(self.clear_grid_spectrum)
        # Initialising add plot list and counter
        self.add_grid_plots = {}
        self.add_grid_plots_counter = 0 
        
        # Grid echo animation button
        echo_g_axis = self.fig.add_axes([0.92, 0.2, 0.04, 0.04])
        self.echo_g = Button(echo_g_axis, 'Echo')
        self.echo_g.on_clicked(self.echo_grid_animation)
        self.grid_echoing = False
        # Echo ordering is necessary so that when the freeze sliders change 
        # values and the grid/emulator sliders update, the updating plot prompt
        # for the grid doesn't try do the emulator echo plots before the 
        # emulator point list is updated too.
        self.echo_ordering = True
        self.fig.text(0.045, 0.26, 'Plot a\n Spectrum?', fontsize=12, ha='center')
        
        # Freezing parameters of the unique combinations with a checkbox
        freeze_axis = self.fig.add_axes([0.01, 0.35, 0.19, 0.53]) # Checkbox shape
        self.fig.text(0.045, 0.9, 'Fix\n parameters?', fontsize=12, ha='center')
        self.parameter_labels = self.grid.parameters_description().keys()
        self.parameter_names = self.grid.parameters_description().values()
        
        self.parameter_checkboxes = CheckButtons(freeze_axis,
                                                 labels=self.parameter_names)
        self.parameter_checkboxes.on_clicked(self.freeze_parameter_update)
        
        # Initialisation
        self.freeze_sliders = {} # Slider Class for each frozen parameter
        self.freeze_slider_history = {} # Dict recording previous slider values
        # This is to avoid conflicting values on different sliders
        freeze_slider_axis = [] # Adding frozen parameter sliders axes
        self.frozen = {} # True/False dictionary for each parameter if frozen
        self.frozen_store = {} # Dictionary to store frozen parameter indexes
        
        # Sliders axes correctly positioned next to corresponding checkboxs, 
        # enabled for a dynamic number of parameters emulated
        for i in range(len(self.parameter_labels)):
            freeze_slider_axis.append(self.fig.add_axes(
                [0.05, (0.84-(0.53/(len(self.parameter_labels)+1))) - (0.53/(len(self.parameter_labels)+1))*i, 0.09, 0.03]))
        
        # Setting the slider for each possible parameter to be fixed.
        for i in range(len(freeze_slider_axis)):
            # Use actual parameter numbers from model_parameters, not i+1
            param_num = self.grid.model_parameters[i]
            param_label = f"param{param_num}"
            
            self.freeze_sliders[param_label] = Slider(freeze_slider_axis[i],
                                                        '',
                                                        self.grid.points[i][0],
                                                        self.grid.points[i][-1],
                                                        valinit=self.grid.points[i][0],
                                                        valstep=self.grid.points[i],
                                                        initcolor='none',
                                                        handle_style={'facecolor':'black',
                                                                      'size':7})
            self.frozen[param_label] = False # Start unfrozen
            self.freeze_slider_history[param_label] = self.freeze_sliders[param_label].val
            self.freeze_sliders[param_label].on_changed(self.freeze_slider_update)
        
        # Toggle animation button and animation speed slider.
        animation_button_ax = self.fig.add_axes([0.85, 0.025, 0.08, 0.04])
        self.animation_button = Button(animation_button_ax, 'Animate')
        self.animation_button.on_clicked(self.toggle_animation)
        
        animation_axis = self.fig.add_axes([0.75, 0.03, 0.08, 0.03])
        self.animation_slider = Slider(animation_axis,
                                  '< Faster | Slower >  ',
                                  30,
                                  500,
                                  valinit=100,
                                  valstep=1,
                                  initcolor='none',
                                  handle_style={'facecolor':'black'})
        self.animation_slider.valtext.set_visible(False)
        # If slider changes, call function to update plot to the new grid point.
        self.animation_slider.on_changed(self.animation_speed)
        
        # Controlling the emulator space (comments see grid space)-------------|
        if emu == None:
            self.emu = None
        elif emu != None:
            self.emu = emu # adding emulator to class
            print('Optional emulator loading.')
        # Creating a list of all emulator fluxes, normalised to the grid fluxes
            self.entire_emu_fluxes = []
            for indexes in tqdm(range(len(self.emu.grid_points))):
                flux = self.emu.load_flux(self.emu.grid_points[indexes], norm=True)
                self.entire_emu_fluxes.append(flux)
            
            # Producing a dimensionless data set for the entire emulator space
            self.dimensionless_emu_fluxes = np.array(self.entire_emu_fluxes)
            # Fluxes.mean(1) does the mean across the rows of the flux grid -
            # i.e. the mean flux of each spectrum.
            norm_factors = self.dimensionless_emu_fluxes.mean(1)
            self.dimensionless_emu_fluxes /= norm_factors[:, np.newaxis]
            # fluxes.mean(0) does the mean across the columns - 
            # i.e. the mean flux at each wavelength bin.
            flux_mean = self.dimensionless_emu_fluxes.mean(0)
            self.dimensionless_emu_fluxes -= flux_mean
            # fluxes.std(0) does the standard deviation across the columns -
            # i.e. the standard deviation of the flux at each wavelength bin.
            #flux_std = self.dimensionless_emu_fluxes.std(0) # changed
            #self.dimensionless_emu_fluxes /= flux_std # changed
            
            # Adding emulator point slider    
            self.emu_axis = self.fig.add_axes([0.25, 0.1, 0.55, 0.03])
            # emu_slider_list is an updating list of emulator points for the slider if freezing parameters
            self.emu_slider_list = [i for i in range(len(self.emu.grid_points))]
            self.emu_slider = Slider(self.emu_axis,
                                  'Emulator Point',
                                  0,
                                  len(self.emu.grid_points)-1,
                                  valinit=0,
                                  valstep=self.emu_slider_list,
                                  initcolor='none',
                                  handle_style={'facecolor':'black'},
                                  color='#2AA198')
            # If slider changes, call function to update plot to the new emu point.
            self.emu_slider.on_changed(self.slider_updating_spectrum)
            
            # Visibility of emulator plots button
            self.emu_box = self.fig.add_axes([0.85, 0.1, 0.06, 0.04])
            self.emu_visible = True
            self.emu_checkbox = CheckButtons(self.emu_box, [' Visible'], [self.emu_visible])
            self.emu_checkbox.on_clicked(self.emu_check_box)
            
            # Emulator point arrow button to +/-1 emulator point
            emu_arrow_ax = self.fig.add_axes([0.1, 0.1, 0.03, 0.04])
            self.ela = Button(emu_arrow_ax, '<') # Emulator left arrow (ela)
            self.ela.on_clicked(self.emu_left_arrow)
            emu2_arrow_ax = self.fig.add_axes([0.13, 0.1, 0.03, 0.04])
            self.era = Button(emu2_arrow_ax, '>') # Emulator right arrow (era)
            self.era.on_clicked(self.emu_right_arrow)
            
            # Add grid point spectrum permanently and clear buttons
            add_emu_spectrum = self.fig.add_axes([0.01, 0.1, 0.03, 0.04]) # Button shape
            self.aes = Button(add_emu_spectrum, 'Add')
            self.aes.on_clicked(self.add_emu_spectrum)
            clear_emu_spectrum = self.fig.add_axes([0.05, 0.1, 0.03, 0.04]) # Button shape
            self.ces = Button(clear_emu_spectrum, 'Clear')
            self.ces.on_clicked(self.clear_emu_spectrum)
            # Initialising add plot list and counter
            self.add_emu_plots = {}
            self.add_emu_plots_counter = 0
            
            # Emulator echo animation button
            self.echo_e_axis = self.fig.add_axes([0.92, 0.1, 0.04, 0.04])
            self.echo_e = Button(self.echo_e_axis, 'Echo')
            self.echo_e.on_clicked(self.echo_emu_animation)
            self.emulator_echoing = False
            
            # Interpolate emulator button for more finely spaced grid search
            interpolate_axis = self.fig.add_axes([0.25, 0.025, 0.12, 0.04])
            self.interpolate = Button(interpolate_axis, 'Interpolate Emulator')
            self.interpolate.on_clicked(self.interpolate_emulator)
            # Switch to load and activate the interpolate button
            self.interpolate_emu_loaded = False
            self.interpolate_emu_active = False
        
        # ---------------------------------------------------------------------|
        # Button (mouse) press event pauses animation on mouse click.
        self.fig.canvas.mpl_connect('button_press_event', self.slider_pause) 
        # Matplotlib animation iterating through frames like a while loop.
        self.animation = FuncAnimation(
            self.fig,
            self.animation_setting_new_slider_value,
            interval=self.animation_slider.val,
            frames=len(self.unique_combinations),
            )
        
        self.animation.running = True # Start animation
        
        plt.show()


    def slider_updating_spectrum(self, extra):
        """This function plots the spectrum of the grid point selected by the 
        slider when changed. The running animation automatically changes the 
        slide value which prompts this function to run. The slider can be 
        manually changed too."""
        
        self.ax.clear() # clearing the previous frame
        self.ax.set_xlabel(r"Wavelength ($\AA$)")
        self.ax.set_ylabel("Flux")
        self.ax.set_xlim(min(self.grid.wl), max(self.grid.wl))
        self.ax.set_title(f"Spectral Data Exploration Tool")
        # Clean legend labels, 3 sig fig with 3 trailing decimal places
        glabel = ['{:.3e}'.format(i) for i in self.unique_combinations[self.grid_slider.val]]
        # Plotting fluxes if dimensionless data or not
        if self.dimensionless_data:
            plotting_flux = self.dimensionless_grid_fluxes
            self.ax.set_ylim(self.dimensionless_axis_min_flux*1.1,self.dimensionless_axis_max_flux*1.1)
        else:
            plotting_flux = self.entire_grid_fluxes
            self.ax.set_ylim(self.axis_min_flux*0.9, self.axis_max_flux*1.1) 
        self.ax.plot(self.grid.wl,
                     plotting_flux[self.grid_slider.val],
                     label=f'Grid:{", ".join(glabel)}',
                     visible=self.grid_visible,
                     )
        
        if self.emu != None and self.interpolate_emu_active != True:
            # Clean legend labels, 3 sig fig with 3 trailing decimal places
            elabel = ['{:.3e}'.format(np.round(i,3)) for i in self.emu.grid_points[self.emu_slider.val]]
                    # Plotting fluxes if dimensionless data or not
            if self.dimensionless_data:
                plotting_flux = self.dimensionless_emu_fluxes
            else:
                plotting_flux = self.entire_emu_fluxes
            self.ax.plot(self.emu.wl, 
                         plotting_flux[self.emu_slider.val], 
                         label=f'Emulator:{", ".join(elabel)}',
                         visible=self.emu_visible,
                         )
            
        # if we are using the interpolated emulator
        elif self.emu != None and self.interpolate_emu_active == True:
            # Clean legend labels, 3 sig fig with 3 trailing decimal places
            elabel = ['{:.3e}'.format(np.round(i,3)) for i in self.finer_unique_combinations[self.finer_emu_slider.val]]
            # Plotting fluxes if dimensionless data or not
            if self.dimensionless_data:
                plotting_flux = self.dimensionless_finer_emu_fluxes
            else:
                plotting_flux = self.finer_emu_fluxes
            self.ax.plot(self.emu.wl, 
                         plotting_flux[self.finer_emu_slider.val], 
                         label=f'Emulator:{", ".join(elabel)}',
                         visible=self.finer_emu_visible,
                         )

        # Adding the permanent plots data to a plotting axis    
        for i in self.add_grid_plots.values():
            self.ax.plot(i[0], i[1], label=i[2], visible=self.fixed_visible)
        
        if self.emu != None:
            for i in self.add_emu_plots.values():
                self.ax.plot(i[0], i[1], label=i[2], visible=self.fixed_visible)
         
        if self.grid_echoing:
            number_of_echoes = len(self.grid_slider_list)-1 # incase small grid
            if number_of_echoes > 5:
                number_of_echoes = 5 # maximum 5 echoes
            current = self.grid_slider_list.index(self.grid_slider.val)
            for previous in range(1,number_of_echoes+1): # 5 previous indexes
                last = current - previous # previous indexes
                plot_index = self.grid_slider_list[last] # previous slider index
                self.ax.plot(self.grid.wl,
                             self.entire_grid_fluxes[plot_index],
                             visible=self.grid_visible,
                             alpha=1-(previous+1)/10,
                             color='#268BD2'
                             )
        
        if self.emu != None:
            if self.emulator_echoing:
                if self.echo_ordering: # To stop prompt before frozen slider updates
                    number_of_echoes = len(self.emu_slider_list)-1 # incase small grid
                    if number_of_echoes > 5:
                        number_of_echoes = 5 # maximum 5 echoes
                    current = self.emu_slider_list.index(self.emu_slider.val)
                    for previous in range(1,number_of_echoes+1): # 5 previous indexes
                        last = current - previous # previous indexes
                        plot_index = self.emu_slider_list[last] # previous slider index
                        self.ax.plot(self.emu.wl,
                                    self.entire_emu_fluxes[plot_index],
                                    visible=self.emu_visible,
                                    alpha=1-(previous+1)/10,
                                    color='#2AA198'
                                    )
        
        self.ax.legend()
        self.fig.canvas.draw_idle()


    def animation_speed(self, event):
        """Upon animation speed slider change, updating matplotlib's 
        animation interval value. This could break if matplotlib does something
        stupid with the animation class as _interval is a private variable.
        However, not using it is a massive pain to make this function work."""

        self.animation._interval = self.animation_slider.val 


    def animation_setting_new_slider_value(self, i):
        """This function updates the slider value for the next frame of the 
        animation (the animation is treated like a while loop). 
        The iteration increases the slider value by 1, which prompts the slider
        to update the plot to the next grid point.
        
        Args: i (int): The current frame of the animation. WE DO NOT USE THIS
        VALUE!!! but it is required for the FuncAnimation function to run. We 
        explicitly update the next slider value in case the user uses the slider.
        The animation will then continue from the last slider value (not the 
        last frame value). A change in the slider value will prompt an updated 
        plot."""
        
        if self.animation.running:
            # Setting the next grid point slider value
            grid_current_val = self.grid_slider.val # current slider value
            # check if slider value is in list after freezing parameters
            if grid_current_val in self.grid_slider_list: 
                grid_slider_index = self.grid_slider_list.index(grid_current_val)
            else:
                grid_slider_index = 0
            # check if slider value is the last in the list or continue
            if grid_slider_index == len(self.grid_slider_list)-1:
                grid_next_val = self.grid_slider_list[0]
            else:
                grid_next_val = self.grid_slider_list[grid_slider_index + 1]
            self.grid_slider.set_val(grid_next_val) # prompting next slider plot

            if self.emu != None:
                # Setting the next emulator point slider value
                emu_current_val = self.emu_slider.val # Current slider value
                if emu_current_val in self.emu_slider_list:
                    emu_slider_index = self.emu_slider_list.index(emu_current_val)
                else:
                    emu_slider_index = 0
                # Check if slider value is the last in the list or continue
                if emu_slider_index == len(self.emu_slider_list)-1:
                    emu_next_val = self.emu_slider_list[0]
                else:
                    emu_next_val = self.emu_slider_list[emu_slider_index + 1]
                self.emu_slider.set_val(emu_next_val)
                
    
    def dimensionless_data_function(self, event):
        """This function is to transform the spectra in absolute flux to 
        dimensionless data which has been normalised, centred and whitened. 
        The transformation is done in preparation for the PCA process.
        This function also readjusts the axis limits to the new data."""
        
        self.dimensionless_data = not self.dimensionless_data
        self.clear_grid_spectrum(event) # clearing the potential fixed plots
        if self.emu != None:
            self.clear_emu_spectrum(event)  # clearing the potential fixed plots
        self.grid_slider.set_val(self.grid_slider.val) # updating the plot
        return self.dimensionless_data
                
    def grid_check_box(self, event):
        """This function toggles the visibility of the grid 
        spectra. It is called when the user clicks the check boxes."""
        
        self.grid_visible = not self.grid_visible
        self.grid_slider.set_val(self.grid_slider.val) # updating the plot
        return self.grid_visible
    

    def fixed_check_box(self, event):
        """This function toggles the visibility of the fixed plotted 
        spectra. It is called when the user clicks the check boxes."""
        
        self.fixed_visible = not self.fixed_visible
        self.grid_slider.set_val(self.grid_slider.val) # updating the plot
        return self.fixed_visible
    
    
    def emu_check_box(self, event):
        """This function toggles the visibility of the emulator 
        spectra. It is called when the user clicks the check boxes."""
        
        self.emu_visible = not self.emu_visible
        self.finer_emu_visible = not self.finer_emu_visible
        self.emu_slider.set_val(self.emu_slider.val)
        return self.emu_visible
        
    
    def slider_pause(self, event, *args, **kwargs):
        """Pauses/unpauses the animation on mouse click anywhere on the slider."""
        
        # Identifying the slider's location on the figure
        (xm,ym),(xM,yM) = self.grid_slider.label.clipbox.get_points()
        if self.emu != None:
            (xm2,ym2),(xM2,yM2) = self.emu_slider.label.clipbox.get_points()
            
        if xm < event.x < xM and ym < event.y < yM: # if clicking slider, pause
            self.animation.running = False
        elif self.emu != None:
            if xm2 < event.x < xM2 and ym2 < event.y < yM2:
                self.animation.running = False
    
    
    def freeze_parameter_update(self, label):
        """The function freezes the parameter checkbox selected by the user."""
        
        if label[:5] != 'param': # only to covert tickbox names to the dict keys
            key_list = list(self.parameter_labels)
            val_list = list(self.parameter_names)
            position = val_list.index(label) # finding the position of the label
            label = key_list[position] # finding the key of the label
        
        self.frozen[label] = not self.frozen[label] # toggle True/False from checkbox
        # if parameter frozen, we need to remove non-matching grid points.
        if self.frozen[label] == True:
            # Fix for parameter 9: map parameter number to actual index in model_parameters
            param_number = int(label[-1])  # Extract parameter number from paramX
            try:
                index = self.grid.model_parameters.index(param_number)  # Get actual index in model_parameters
            except ValueError:
                print(f"Warning: Parameter {param_number} not found in model_parameters {self.grid.model_parameters}")
                return
            fixed_param_slider_value = self.freeze_sliders[label].val # value of the slider
            # Indexes to be removed from grid point slider list, if the value of
            # the fixed parameter slider doesn't match the grid point value.
            removing_indexes = [] 
            removing_indexes = [count for count, values in 
                                          enumerate(self.all_unique_combinations) 
                                          if values[index] != fixed_param_slider_value]
            
            # Removing grid points
            for grid_point in removing_indexes: 
                present = False # initially assuming not frozen
                for forzen_list in self.frozen_store.values():
                    if grid_point in forzen_list:
                        present = True # if in frozen store, then don't remove again
                        break # if found, no need to continue
                if present == False: # if not frozen, remove from list
                    self.grid_slider_list.remove(grid_point)
            
            # Removing emulator points if used
            if self.emu != None:
                for emu_point in removing_indexes:
                    present = False # initially assuming not frozen
                    for forzen_list in self.frozen_store.values():
                        if emu_point in forzen_list:
                            present = True # if in frozen store, then don't remove again
                            break # if found, no need to continue
                    if present == False: # if not frozen, remove from list
                        self.emu_slider_list.remove(emu_point)
            
            # Storing indexes to not compute again when unfreezing
            self.frozen_store[label] = removing_indexes
            
        # if parameter unfrozen, we need to add non-matching grid points.   
        elif self.frozen[label] == False:
            adding_indexes = self.frozen_store[label] # retrieve stored indexes
            # reset label to None, but using [-1] as it doesn't break iteration 
            # loops in code. -1 will never be a grid point index value.
            self.frozen_store[label] = [-1]
            for grid_point in adding_indexes:
                present = False # initially assuming not frozen
                for forzen_list in self.frozen_store.values():
                    if grid_point in forzen_list:
                        present = True # if in frozen store, then keep frozen
                        break # if found, no need to continue
                if present == False: # if not frozen, add to list
                    bisect.insort(self.grid_slider_list, grid_point)
                    
            if self.emu != None:
                for grid_point in adding_indexes:
                    present = False # initially assuming not frozen
                    for forzen_list in self.frozen_store.values():
                        if grid_point in forzen_list:
                            present = True # if in frozen store, then keep frozen
                            break # if found, no need to continue
                    if present == False: # if not frozen, add to list
                        bisect.insort(self.emu_slider_list, grid_point) # adding to sorted list
        
        # If frozen to a parameter combination not currently displayed
        # The plot updates to the first parameter combination possible. 
        if self.grid_slider.val not in self.grid_slider_list:
            self.echo_ordering = not self.echo_ordering # Need so emu echo doesn't error
            self.grid_slider.set_val(self.grid_slider_list[0])
        if self.emu != None:
            if self.emu_slider.val not in self.emu_slider_list:
                self.echo_ordering = not self.echo_ordering # Let echo emu proceed.
                self.emu_slider.set_val(self.emu_slider_list[0])

 
    def freeze_slider_update(self, event):
        """This function updates the frozen slider parameter value for plotting."""
        
        for label in self.freeze_sliders.keys():
           if self.freeze_sliders[label].val != self.freeze_slider_history[label] and self.frozen[label] == True: 
               # Update history to current (new) slider value
               self.freeze_slider_history[label] = self.freeze_sliders[label].val
               self.freeze_parameter_update(label) # Adding old points back
               self.freeze_parameter_update(label) # Removing new frozen points
               break # when found, no need to continue
    
    
    def grid_left_arrow(self, event):
        """Button press pauses the animation and moves the slider left by -1.
        If fixed parameter list, it moves to the next possible fixed parameter
        value."""
        
        self.animation.running = False
        current = self.grid_slider_list.index(self.grid_slider.val)
        self.grid_slider.set_val(self.grid_slider_list[current-1])
        
        
    def grid_right_arrow(self, event):
        """Button press pauses the animation and moves the slider right by +1.
        If fixed parameter list, it moves to the next possible fixed parameter
        value."""
        
        self.animation.running = False
        current = self.grid_slider_list.index(self.grid_slider.val)
        if current + 1 == len(self.grid_slider_list): # if next outside list
            current = -1 # start at beginning of list again
        self.grid_slider.set_val(self.grid_slider_list[current+1])
        
        
    def emu_left_arrow(self, event):
        """Button press pauses the animation and moves the slider left by -1.
        If fixed parameter list, it moves to the next possible fixed parameter
        value."""
        
        self.animation.running = False
        # for the finer interpolated emulator grid
        if self.interpolate_emu_active == True:
            current = self.finer_emu_slider_list.index(self.finer_emu_slider.val)
            self.finer_emu_slider.set_val(self.finer_emu_slider_list[current-1])
        # for the normal coarser emulator grid
        else:
            current = self.emu_slider_list.index(self.emu_slider.val)
            self.emu_slider.set_val(self.emu_slider_list[current-1])
        
        
    def emu_right_arrow(self, event):
        """Button press pauses the animation and moves the slider right by +1.
        If fixed parameter list, it moves to the next possible fixed parameter
        value."""
        
        self.animation.running = False
        # for the finer interpolated emulator grid
        if self.interpolate_emu_active == True:
            current = self.finer_emu_slider_list.index(self.finer_emu_slider.val)
            if current + 1 == len(self.finer_emu_slider_list):
                current = -1
            self.finer_emu_slider.set_val(self.finer_emu_slider_list[current+1])
        # for the normal coarser emulator grid
        else:
            current = self.emu_slider_list.index(self.emu_slider.val)
            if current + 1 == len(self.emu_slider_list): # if next outside list
                current = -1 # start at beginning of list again
            self.emu_slider.set_val(self.emu_slider_list[current+1])
        
        
    def add_grid_spectrum(self, event):
        """Adding the current grid plot to a list to be permanently plotted 
        each animation frame until cleared."""
        
        self.add_grid_plots_counter += 1 # iterator for dictionary key
        # Clean legend labels, 3 sig fig with 3 trailing decimal places
        glabel = ['{:.3f}'.format(np.round(i,3)) for i in self.unique_combinations[self.grid_slider.val]]
        # Warning! Adding the current plot's data to a list which is placed in a 
        # dictionary called self.add_grid_plots. This is done this way 
        # as for some reason creating a matplotlib self.ax.plot here doesn't 
        # plot when called in slider_updating_spectrum. This can likely be fixed.
        if self.dimensionless_data:
            plotting_list = [self.grid.wl, self.dimensionless_grid_fluxes[self.grid_slider.val], f'Grid{self.add_grid_plots_counter}:{", ".join(glabel)}', self.grid_visible]
        else:
            plotting_list = [self.grid.wl, self.entire_grid_fluxes[self.grid_slider.val], f'Grid{self.add_grid_plots_counter}:{", ".join(glabel)}', self.grid_visible]
        self.add_grid_plots[self.add_grid_plots_counter] = plotting_list
        self.grid_slider.set_val(self.grid_slider.val) # refreshing the plot
        
        
    def clear_grid_spectrum(self, event):
        """Removing the permanent plotting list on pressing the clear button"""
        
        self.add_grid_plots = {}
        self.add_grid_plots_counter = 0 
        self.grid_slider.set_val(self.grid_slider.val) # refreshing the plot
    
    def add_emu_spectrum(self, event):
        """Adding the current emulator plot to a list to be permanently plotted 
        each animation frame until cleared."""
        
        self.add_emu_plots_counter += 1 # iterator for dictionary key
        if self.interpolate_emu_active == False:
            # Clean legend labels, 3 sig fig with 3 trailing decimal places
            elabel = ['{:.3f}'.format(np.round(i,3)) for i in self.emu.grid_points[self.emu_slider.val]]
            # Warning! Adding the current plot's data to a list which is placed in a 
            # dictionary called self.add_grid_plots. This is done this way 
            # as for some reason creating a matplotlib self.ax.plot here doesn't 
            # plot when called in slider_updating_spectrum. This can likely be fixed.
            if self.dimensionless_data:
                plotting_list = [self.emu.wl, self.dimensionless_emu_fluxes[self.emu_slider.val], f'Emu{self.add_emu_plots_counter}:{", ".join(elabel)}', self.emu_visible]
            else:
                plotting_list = [self.emu.wl, self.entire_emu_fluxes[self.emu_slider.val], f'Emu{self.add_emu_plots_counter}:{", ".join(elabel)}', self.emu_visible]
            self.add_emu_plots[self.add_emu_plots_counter] = plotting_list
            
        elif self.interpolate_emu_active == True:
            # Clean legend labels, 3 sig fig with 3 trailing decimal places
            elabel = ['{:.3f}'.format(np.round(i,3)) for i in self.finer_unique_combinations[self.finer_emu_slider.val]]
            # Warning! Adding the current plot's data to a list which is placed in a 
            # dictionary called self.add_grid_plots. This is done this way 
            # as for some reason creating a matplotlib self.ax.plot here doesn't 
            # plot when called in slider_updating_spectrum. This can likely be fixed.
            if self.dimensionless_data:
                plotting_list = [self.emu.wl, self.dimensionless_finer_emu_fluxes[self.finer_emu_slider.val], f'Emu{self.add_emu_plots_counter}:{", ".join(elabel)}', self.emu_visible]
            else:
                plotting_list = [self.emu.wl, self.finer_emu_fluxes[self.finer_emu_slider.val], f'Emu{self.add_emu_plots_counter}:{", ".join(elabel)}', self.emu_visible]
            self.add_emu_plots[self.add_emu_plots_counter] = plotting_list
            
        self.emu_slider.set_val(self.emu_slider.val) # refreshing the plot
    
    
    def clear_emu_spectrum(self, event):
        """Removing the permanent plotting list on pressing the clear button"""
        
        self.add_emu_plots = {}
        self.add_emu_plots_counter = 0
        self.emu_slider.set_val(self.emu_slider.val) # refreshing the plot
    
    def echo_grid_animation(self, event):
        "Pressing echo button toggles echo plotting for the grid"
        
        self.grid_echoing = not self.grid_echoing
        self.grid_slider.set_val(self.grid_slider.val) # refreshing the plot
    
    def echo_emu_animation(self, event):
        "Pressing echo button toggles echo plotting for the emulator"
        
        self.emulator_echoing = not self.emulator_echoing
        self.emu_slider.set_val(self.emu_slider.val) # refreshing the plot
    
    
    def interpolate_emulator(self, event):
        """This function interpolates the emulator points' fluxes to a finer 
        grid. Upon first press, the new finer grid is created by loading the
        flux. After this the interpolator can be switched on and off."""
        
        if self.interpolate_emu_loaded == False: # First press, load new grid
            # Creating new emulator combinations inbetween the grid points
            limiting_number = 3000 # +1 iteration maximum number of combinations
            unique_number = 0 # number of unique combinations initialisation
            intervals = 0 # number for the linspace function
            while unique_number < limiting_number:
                unique_interpolations = [] # new grid of unique combinations
                intervals += 1 # increasing the number of intervals
                for parameter_values in self.grid.points: # for each parameter
                    interpolated_parameter = [] # new parameter list
                    for value_index in range(len(parameter_values)-1):
                        linspaced = np.linspace(parameter_values[value_index],
                                                parameter_values[value_index+1],
                                                intervals, 
                                                endpoint=False
                                                ) # points between two grid values
                        interpolated_parameter.extend(linspaced) # adding to new list
                    # As endpoint=False, the last parameter values is not included,
                    # so we add the final grid value to the list separately.
                    interpolated_parameter.append(parameter_values[-1])
                    # Add new parameter values to grid list
                    unique_interpolations.append(interpolated_parameter)
                # list to array
                unique_interpolations = np.array(unique_interpolations)
            
                # All finer unique combinations in a 1D list
                self.finer_unique_combinations = []
                for combination in itertools.product(*unique_interpolations):
                    self.finer_unique_combinations.append(list(combination))
                # check if we are generating too many combinations
                unique_number = len(self.finer_unique_combinations)
                
                
            # Loading the fluxes for the new finer grid
            self.finer_emu_fluxes = []
            print('Generating more emulator spectra. Please Wait...')
            for spectrum in tqdm(self.finer_unique_combinations): 
                flux = self.emu.load_flux(spectrum, norm=True)
                self.finer_emu_fluxes.append(flux)
            
            # Producing a dimensionless data set for the finer emulator space
            # Note: the spectrum of idential parameters will not be the same 
            # between the fine grid and coarse grid as it is resampling the 
            # random normal multivariate distribution for the weights
            self.dimensionless_finer_emu_fluxes = np.array(self.finer_emu_fluxes)
            for index, params in enumerate(self.finer_unique_combinations):
                self.dimensionless_finer_emu_fluxes[index] /= self.emu.norm_factor(params)
            #flux_mean = self.dimensionless_finer_emu_fluxes.mean(0)
                self.dimensionless_finer_emu_fluxes[index] -= self.emu.flux_mean
            #flux_std = self.dimensionless_finer_emu_fluxes.std(0)
                #self.dimensionless_finer_emu_fluxes[index] /= self.emu.flux_std # changed

            self.finer_emu_slider_list = [i for i in range(len(self.finer_unique_combinations))] # list of indexes for slider
            # New emulator slider for the new finer grid
            self.finer_emu_visible = True # Slider visibility
            self.finer_emu_axis = self.fig.add_axes([0.25, 0.1, 0.55, 0.03])
            self.finer_emu_slider = Slider(self.finer_emu_axis,
                                    'Finer Emulator',
                                    0,
                                    max(self.finer_emu_slider_list),
                                    valinit=0,
                                    valstep=self.finer_emu_slider_list,
                                    initcolor='none',
                                    handle_style={'facecolor':'black'},
                                    color='#2AA198',
                                    track_color='#c42d2a')
            self.finer_emu_slider.on_changed(self.slider_updating_spectrum)
            # Initialisation of slider to invisible as reversed in next few lines
            self.finer_emu_axis.set_visible(not self.finer_emu_axis.get_visible())
            self.interpolate_emu_loaded = True # Switching to loaded state
        
        
        # Change switch when clicked to on/off
        self.interpolate_emu_active = not self.interpolate_emu_active
        # Switching visibility of the coarser emulator grid 
        self.emu_axis.set_visible(not self.emu_axis.get_visible())
        # Switching visibility of finer emulator grid, opposite to course
        self.finer_emu_axis.set_visible(not self.finer_emu_axis.get_visible())
        self.animation.running = False # Pausing the animations
        # Hiding echo button not to be used by the user in this mode.
        self.echo_e_axis.set_visible(not self.echo_e_axis.get_visible())
        self.emu_slider.set_val(self.emu_slider.val) # refreshing the plot
    
    def toggle_animation(self, event):
        """Button press pauses and unpauses the animation."""
        
        if self.animation.running:
            self.animation.running = False
        else:
            self.animation.running = True
