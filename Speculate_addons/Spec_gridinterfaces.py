import os
import fnmatch
import numpy as np
from Starfish.grid_tools import GridInterface
import sigfig        
        
class Speculate_cv_bl_grid_v87f(GridInterface):
    """
    An Interface to the CV BL grid produced by PYTHON v87f Radiative transfer simulations.
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/Å
    
    Parameters of model
    -------------------
    1) Disk.mdot (msol/yr)
    2) wind.mdot (Disk.mdot)
    3) KWD.d (in_units_of_Rstar)
    4) KWD.mdot_r_exponent
    5) KWD.acceleration_length (cm)
    6) KWD.acceleration_exponent
    7) Boundary_layer.luminosity(ergs/s) = L_Disk * param_points_7, L_Disk = G * M_star * M_dotdisk / 2R_star
    8) Boundary_layer.temp(K) = BL_Luminosity / (Stefan boltzmann law* 4•pi•R_wd^2 * H/R_wd)
    9) Inclination angle - sparse (30, 55, 80 degrees)
    10) Inclination angle - mid (30-80 degrees, 10° steps)
    11) Inclination angle - full (30-85 degrees, 5° steps)
    
    Optional parameters
    -------------------
    Angle of Inclination - Amend value in function 'def load_flux' default's value
    
    """
        
    def __init__(self, path, usecols, air=False, wl_range=(800,8000), model_parameters=(1,2,3,4,5,6,7,8,9,10,11), scale='linear'):
        """
        Initialises an empty grid with parameters and wavelengths.
        
        Args:
            path (str or path-like): The path of the base of the grid space library.
            
            air (bool, optional): Whether the wavelengths are measured in air or not.
                Default is False. (Required due to implementation of inherited GridInterface class)
                
            wl_range (tuple, optional): The (min, max) of the wavelengths in AA. 
                Default is (800,8000), wavelength range.
            
            model_parameters (tuple, optional): Specifiy the parameters 
                you wish to fit by adding intergers to the tuple. 
        """
        
        # The grid points in the parameter space are defined, 
        # param_points_1-3 correspond to the model parameters defined at the top in the respective order.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.usecols = usecols # Wavelength and inclination tuple
        # self.skiprows = skiprows # Deprecated: calculated dynamically
        points = []
        if 1 in model_parameters:
            param_points_1 = np.log10(np.array([3e-09, 1e-08, 3e-08]))
            points.append(param_points_1)
        if 2 in model_parameters:
            param_points_2 = np.array([0.03, 0.1, 0.3])
            points.append(param_points_2)
        if 3 in model_parameters:
            param_points_3 = np.array([0.55, 5.5, 55.0])
            points.append(param_points_3)
        if 4 in model_parameters:
            param_points_4 = np.array([0.0, 0.25, 1.0])
            points.append(param_points_4)
        if 5 in model_parameters:
            param_points_5 = np.log10(np.array([7.25182e+08, 7.25182e+09, 7.25182e+10]))
            points.append(param_points_5)
        if 6 in model_parameters:
            param_points_6 = np.array([0.5, 1.5, 4.5])
            points.append(param_points_6)
        if 7 in model_parameters:
            param_points_7 = np.array([0.0, 0.3, 1.0])
            points.append(param_points_7)
        if 8 in model_parameters:
            param_points_8 = np.array([0.1, 0.3, 1.0])
            points.append(param_points_8)
        if 9 in model_parameters:
            param_points_9 = np.array([30, 55, 80])  # Inclination angles in degrees (sparse)
            points.append(param_points_9)
        if 10 in model_parameters:
            param_points_10 = np.array([30, 40, 50, 60, 70, 80])  # Every 10 degrees (30-80°)
            points.append(param_points_10)
        if 11 in model_parameters:
            param_points_11 = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])  # Every 5 degrees (30-85°)
            points.append(param_points_11)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name='CV_BL_grid_v87f',
            param_names=param_names,
            points=points,
            wave_units='AA',
            flux_units=flux_units,
            air=air,
            wl_range=wl_range,
            path=path,
        )        
    
        # The wavelengths for which the fluxes are measured are retrieved.
        try:
            wls_fname = os.path.join(self.path, 'run0.spec')
            skiprows = self._get_skiprows(wls_fname)
            wls = np.loadtxt(wls_fname, usecols=self.usecols[0], skiprows=skiprows, unpack=True)
            wls = np.flip(wls)
        except:
            raise ValueError("Wavelength file improperly specified")
        
        # Truncating to the wavelength range to the provided values.
        self.wl_full = np.array(wls, dtype=np.float64) #wls[::-1]
        self.ind = (self.wl_full >= self.wl_range[0]) & (
            self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]
        
        
    def get_flux(self, params):
        """
        Constructs path to datafile corresponding to model parameters passed.
        
        Args:
            params (ndarray): Contains the modelling parameters of a required grid point.
            
        Returns:
            str: The path of the datafile corresponding to the input model parameters.
        """
        
        # Parameter definitions for both file lookup (1-8) and full space (1-10)
        param1_name = np.log10([3e-9, 1e-08, 3e-08]) # Disk.mdot
        param2_name = [0.03, 0.1, 0.3] # wind.mdot (Disk.mdot)
        param3_name = [0.55, 5.5, 55.0] # KWD.d
        param4_name = [0.0, 0.25, 1.0] # KWD.mdot_r_exponent
        param5_name = np.log10([7.25182e+08, 7.25182e+09, 7.25182e+10]) # KWD.acceleration_length (cm)
        param6_name = [0.5, 1.5, 4.5] # KWD.acceleration_exponent
        param7_name = [0.0, 0.3, 1.0] # Boundary_layer.luminosity(ergs/s)
        param8_name = [0.1, 0.3, 1.0] # Boundary_layer.temp(K)
        param9_name = [30, 55, 80] # Inclination angle (degrees) - sparse
        param10_name = [30, 40, 50, 60, 70, 80] # Inclination angle (degrees) - mid
        param11_name = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85] # Inclination angle (degrees) - full (5° steps)
        
        # For file lookup: only parameters 1-8 (determines filename)
        #file_lookup_names = [param1_name, param2_name, param3_name, param4_name, param5_name, param6_name, param7_name, param8_name]
        fixed_params = [param1_name[1], param2_name[1], param3_name[1], param4_name[1], param5_name[1], param6_name[1], param7_name[1], param8_name[1]]
        
        # Build the 8-parameter array for file lookup (excluding parameters 9 and 10)
        file_params = []
        param_index = 0
        
        for param_num in range(1, 9):  # Parameters 1-8 only
            if param_num in self.model_parameters:  # Include if in model_parameters
                file_params.append(params[param_index])
                param_index += 1
            else:
                # Use fixed middle value for missing parameters 1-8
                file_params.append(fixed_params[param_num - 1])
        
        file_params = np.array(file_params)
        
        # Create file lookup combinations (only parameters 1-8)
        import itertools
        file_combinations = []
        temp_grid = np.array([param1_name, param2_name, param3_name, param4_name, param5_name, param6_name, param7_name, param8_name])
        for i in itertools.product(*temp_grid):
            file_combinations.append(list(i))

        # Find the matching file using 8-parameter combination
        for i in range(len(file_combinations)):
            if np.all(np.isclose(file_params, file_combinations[i], rtol=1e-9, atol=1e-8)):
                file_name = f'run{i}.spec' # if matched, index is run number
                break
        
        return self.path + file_name # returning the correct filename/path

    def parameters_description(self):
        """Provides a description of the model parameters used.

        Returns:
            dictionary: Description of the 'paramX' names
        """
        dictionary = {
            1:"Disk.mdot (msol/yr)",
            2:"Wind.mdot (Disk.mdot)",
            3:"KWD.d (in_units_of_Rstar)",
            4:"KWD.mdot_r_exponent",
            5:"KWD.acceleration_length (cm)",
            6:"KWD.acceleration_exponent",
            7:"Boundary_layer.luminosity(ergs /s)",
            8:"Boundary_layer.temp(K)",
            9:"Inclination angle - sparse (30, 55, 80 degrees)",
            10:"Inclination angle - mid (30, 40, 50, 60, 70, 80 degrees)",
            11:"Inclination angle - full (30-85 degrees, 5° steps)"
            } # Description of the paramters
        parameters_used = {}
        for i in self.model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
    def _get_skiprows(self, filepath):
        """
        Scans the file to find the number of header lines.
        Assumes header lines start with '#' or 'Freq.' (after stripping whitespace).
        """
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                if line.startswith('#'): continue
                if line.startswith('Freq.'): continue
                # Found the first data line
                return i
        return 0

    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Args:
            parameters (ndarray): Contains parameters of a required grid point
            
            header (bool): Whether to attach param values on return, unimplemented!!
            
            norm (bool): Whether to normalise the return flux (left unimplemented)
        
            scale (str): Change this default too if changing the flux scale. 
                Define the scale for your data and emulator grid space. 'linear', 'log' or 'scaled'
                
        Returns:
            ndarray: List of fluxes in the wavelength range specified on initialisation
            
            dict (Optional): Dictionary of parameter names and values
        """
        
        import logging
        from scipy.ndimage import gaussian_filter1d # Instead of normalising, a 1d gaussian smoothing filter is applied 
        
        # Get the filename for logging
        file_path = self.get_flux(parameters)
        file_name = os.path.basename(file_path)
        
        # Determine which column to use for inclination
        # Check for any inclination parameter (9, 10, or 11)
        inclination_params = [p for p in [9, 10, 11] if p in self.model_parameters]
        
        if inclination_params:
            # Get the first inclination parameter found
            inclination_param_num = inclination_params[0]
            inclination_param_index = list(self.model_parameters).index(inclination_param_num)
            inclination_angle = parameters[inclination_param_index]
            
            # Map inclination angle to column index (30°->2, 35°->3, ..., 85°->13)
            inclination_column = int(2 + (inclination_angle - 30) / 5)
            
            # LOG THE DETAILS
            logger = logging.getLogger(__name__)
            logger.info(f"GRID PROCESSING [Param{inclination_param_num}]: File={file_name}, Parameters={parameters}, Inclination={inclination_angle}°, Column={inclination_column}")
        else:
            # Use the default column from usecols if no inclination parameter in model_parameters
            inclination_column = self.usecols[1]
            
            # LOG THE DETAILS
            logger = logging.getLogger(__name__)
            logger.info(f"GRID PROCESSING [Default]: File={file_name}, Parameters={parameters}, Default_Column={inclination_column}")
        
        # Load flux using wavelength (column 0) and calculated inclination column
        # Using comments=['#', 'Freq.'] to handle variable header length
        # Optimized: Pre-scan for header length to avoid line-by-line comment checking in loadtxt
        skiprows = self._get_skiprows(file_path)
        wl, flux = np.loadtxt(file_path, usecols=(0, inclination_column), skiprows=skiprows, unpack=True)
        flux = np.flip(flux)
        
        flux = flux[:len(self.wl_full)] # THIS CUT IS NEEDED, Random parameters appear in the grid space header leading to mismatching file lengths.
        #flux = gaussian_filter1d(flux, 50)
        if self.scale == 'log':
            flux = np.log10(flux) # logged 10 
        if self.scale == 'scaled': # to values near order of magnitude 10^0. 
            flux = flux/np.mean(flux)
        
        # TODO: Implement header if doing to use
        hdr = {'inclination_column': inclination_column} # Header constructed 
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]

class Speculate_cv_no_bl_grid_v87f(GridInterface):
    """
    An Interface to the CV NO-BL grid produced by PYTHON v87f Radiative transfer simulations.
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/AA
    
    Parameters of model
    -------------------
    1) Disk.mdot (msol/yr)
    2) wind.mdot (Disk.mdot)
    3) KWD.d (in_units_of_Rstar)
    4) KWD.mdot_r_exponent
    5) KWD.acceleration_length (cm)
    6) KWD.acceleration_exponent
    9) Inclination angle - sparse (30, 55, 80 degrees)
    10) Inclination angle - mid (30-80 degrees, 10° steps)
    11) Inclination angle - full (30-85 degrees, 5° steps)
    
    Optional parameters
    -------------------
    Angle of Inclination - Amend value in function 'def load_flux' default's value
    
    """
        
    def __init__(self, path, usecols, air=False, wl_range=(800,8000), model_parameters=(1,2,3,4,5,6,9,10,11), scale='linear'):
        """
        Initialises an empty grid with parameters and wavelengths.
        
        Args:
            path (str or path-like): The path of the base of the grid space library.
            
            air (bool, optional): Whether the wavelengths are measured in air or not.
                Default is False. (Required due to implementation of inherited GridInterface class)
                
            wl_range (tuple, optional): The (min, max) of the wavelengths in AA. 
                Default is (800,8000), wavelength range.
            
            model_parameters (tuple, optional): Specifiy the parameters 
                you wish to fit by adding intergers to the tuple. 
        """
        
        # The grid points in the parameter space are defined, 
        # param_points_1-3 correspond to the model parameters defined at the top in the respective order.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.usecols = usecols # Wavelength and inclination tuple
        # self.skiprows = skiprows # Deprecated: calculated dynamically
        points = []
        if 1 in model_parameters:
            param_points_1 = np.log10(np.array([3e-09, 1e-08, 3e-08]))
            points.append(param_points_1)
        if 2 in model_parameters:
            param_points_2 = np.array([0.03, 0.1, 0.3])
            points.append(param_points_2)
        if 3 in model_parameters:
            param_points_3 = np.array([0.55, 5.5, 55.0])
            points.append(param_points_3)
        if 4 in model_parameters:
            param_points_4 = np.array([0.0, 0.25, 1.0])
            points.append(param_points_4)
        if 5 in model_parameters:
            param_points_5 = np.log10(np.array([7.25182e+08, 7.25182e+09, 7.25182e+10]))
            points.append(param_points_5)
        if 6 in model_parameters:
            param_points_6 = np.array([0.5, 1.5, 4.5])
            points.append(param_points_6)
        if 9 in model_parameters:
            param_points_9 = np.array([30, 55, 80])  # Inclination angles in degrees (sparse)
            points.append(param_points_9)
        if 10 in model_parameters:
            param_points_10 = np.array([30, 40, 50, 60, 70, 80])  # Every 10 degrees (30-80°)
            points.append(param_points_10)
        if 11 in model_parameters:
            param_points_11 = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])  # Every 5 degrees (30-85°)
            points.append(param_points_11)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name='CV_NO-BL_grid_v87f',
            param_names=param_names,
            points=points,
            wave_units='AA',
            flux_units=flux_units,
            air=air,
            wl_range=wl_range,
            path=path,
        )        
    
        # The wavelengths for which the fluxes are measured are retrieved.
        try:
            wls_fname = os.path.join(self.path, 'run0.spec')
            skiprows = self._get_skiprows(wls_fname)
            wls = np.loadtxt(wls_fname, usecols=self.usecols[0], skiprows=skiprows, unpack=True)
            wls = np.flip(wls)
        except:
            raise ValueError("Wavelength file improperly specified")
        
        # Truncating to the wavelength range to the provided values.
        self.wl_full = np.array(wls, dtype=np.float64) #wls[::-1]
        self.ind = (self.wl_full >= self.wl_range[0]) & (
            self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]
        
        
    def get_flux(self, params):
        """
        Constructs path to datafile corresponding to model parameters passed.
        
        Args:
            params (ndarray): Contains the modelling parameters of a required grid point.
            
        Returns:
            str: The path of the datafile corresponding to the input model parameters.
        """
        
        # Parameter definitions for both file lookup (1-8) and full space (1-10)
        param1_name = np.log10([3e-9, 1e-08, 3e-08]) # Disk.mdot
        param2_name = [0.03, 0.1, 0.3] # wind.mdot (Disk.mdot)
        param3_name = [0.55, 5.5, 55.0] # KWD.d
        param4_name = [0.0, 0.25, 1.0] # KWD.mdot_r_exponent
        param5_name = np.log10([7.25182e+08, 7.25182e+09, 7.25182e+10]) # KWD.acceleration_length (cm)
        param6_name = [0.5, 1.5, 4.5] # KWD.acceleration_exponent
        param9_name = [30, 55, 80] # Inclination angle (degrees) - sparse
        param10_name = [30, 40, 50, 60, 70, 80] # Inclination angle (degrees) - mid
        param11_name = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85] # Inclination angle (degrees) - full (5° steps)
        
        # For file lookup: only parameters 1-8 (determines filename)
        #file_lookup_names = [param1_name, param2_name, param3_name, param4_name, param5_name, param6_name]
        fixed_params = [param1_name[1], param2_name[1], param3_name[1], param4_name[1], param5_name[1], param6_name[1]]
        
        # Build the 8-parameter array for file lookup (excluding parameters 9 and 10)
        file_params = []
        param_index = 0
        
        for param_num in range(1, 7):  # Parameters 1-6 only
            if param_num in self.model_parameters:  # Include if in model_parameters
                file_params.append(params[param_index])
                param_index += 1
            else:
                # Use fixed middle value for missing parameters 1-6
                file_params.append(fixed_params[param_num - 1])
        
        file_params = np.array(file_params)
        
        # Create file lookup combinations (only parameters 1-8)
        import itertools
        file_combinations = []
        temp_grid = np.array([param1_name, param2_name, param3_name, param4_name, param5_name, param6_name])
        for i in itertools.product(*temp_grid):
            file_combinations.append(list(i))

        # Find the matching file using 8-parameter combination
        for i in range(len(file_combinations)):
            if np.all(np.isclose(file_params, file_combinations[i], rtol=1e-9, atol=1e-8)):
                file_name = f'run{i}.spec' # if matched, index is run number
                break
    
        
        return self.path + file_name # returning the correct filename/path

    def parameters_description(self):
        """Provides a description of the model parameters used.

        Returns:
            dictionary: Description of the 'paramX' names
        """
        dictionary = {
            1:"Disk.mdot (msol/yr)",
            2:"Wind.mdot (Disk.mdot)",
            3:"KWD.d (in_units_of_Rstar)",
            4:"KWD.mdot_r_exponent",
            5:"KWD.acceleration_length (cm)",
            6:"KWD.acceleration_exponent",
            9:"Inclination angle - sparse (30, 55, 80 degrees)",
            10:"Inclination angle - mid (30, 40, 50, 60, 70, 80 degrees)",
            11:"Inclination angle - full (30-85 degrees, 5° steps)"
            } # Description of the paramters
        parameters_used = {}
        for i in self.model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
    def _get_skiprows(self, filepath):
        """
        Scans the file to find the number of header lines.
        Assumes header lines start with '#' or 'Freq.' (after stripping whitespace).
        """
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                if line.startswith('#'): continue
                if line.startswith('Freq.'): continue
                # Found the first data line
                return i
        return 0

    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Args:
            parameters (ndarray): Contains parameters of a required grid point
            
            header (bool): Whether to attach param values on return, unimplemented!!
            
            norm (bool): Whether to normalise the return flux (left unimplemented)
        
            scale (str): Change this default too if changing the flux scale. 
                Define the scale for your data and emulator grid space. 'linear', 'log' or 'scaled'
                
        Returns:
            ndarray: List of fluxes in the wavelength range specified on initialisation
            
            dict (Optional): Dictionary of parameter names and values
        """
        
        import logging
        from scipy.ndimage import gaussian_filter1d # Instead of normalising, a 1d gaussian smoothing filter is applied 
        
        # Get the filename for logging
        file_path = self.get_flux(parameters)
        file_name = os.path.basename(file_path)
        
        # Determine which column to use for inclination
        # Check for any inclination parameter (9, 10, or 11)
        inclination_params = [p for p in [9, 10, 11] if p in self.model_parameters]
        
        if inclination_params:
            # Get the first inclination parameter found
            inclination_param_num = inclination_params[0]
            inclination_param_index = list(self.model_parameters).index(inclination_param_num)
            inclination_angle = parameters[inclination_param_index]
            
            # Map inclination angle to column index (30°->2, 35°->3, ..., 85°->13)
            inclination_column = int(2 + (inclination_angle - 30) / 5)
            
            # LOG THE DETAILS
            logger = logging.getLogger(__name__)
            logger.info(f"GRID PROCESSING [Param{inclination_param_num}]: File={file_name}, Parameters={parameters}, Inclination={inclination_angle}°, Column={inclination_column}")
        else:
            # Use the default column from usecols if no inclination parameter in model_parameters
            inclination_column = self.usecols[1]
            
            # LOG THE DETAILS
            logger = logging.getLogger(__name__)
            logger.info(f"GRID PROCESSING [Default]: File={file_name}, Parameters={parameters}, Default_Column={inclination_column}")
        
        # Load flux using wavelength (column 0) and calculated inclination column
        # Using comments=['#', 'Freq.'] to handle variable header length
        # Optimized: Pre-scan for header length to avoid line-by-line comment checking in loadtxt
        skiprows = self._get_skiprows(file_path)
        wl, flux = np.loadtxt(file_path, usecols=(0, inclination_column), skiprows=skiprows, unpack=True)
        flux = np.flip(flux)
        
        flux = flux[:len(self.wl_full)] # THIS CUT IS NEEDED, Random parameters appear in the grid space header leading to mismatching file lengths.
        #flux = gaussian_filter1d(flux, 50)
        if self.scale == 'log':
            flux = np.log10(flux) # logged 10 
        if self.scale == 'scaled': # to values near order of magnitude 10^0. 
            flux = flux/np.mean(flux)
        
        # TODO: Implement header if doing to use
        hdr = {'inclination_column': inclination_column} # Header constructed 
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]

