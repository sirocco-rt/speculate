import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from Starfish.grid_tools import GridInterface
import sigfig

class KWDGridInterface(GridInterface):
    """
    An Interface to the KWD grid produced by PYTHON simulation.
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/cm
    
    Parameters of model
    -------------------
    1) wind.mdot (msol/yr)
    2) kn.d
    3) kn.mdot_r_exponent
    4) kn.v_infinity (in_units_of_vescape)
    5) kn.acceleration_length (cm)
    6) kn.acceleration_exponent
    
    Optional parameters
    -------------------
    angle of inclination (taken to be 40.0 for this analysis)
    
    """
    
    def __init__(self, path, usecols, skiprows, air=False, wl_range=(1200,1800), model_parameters=(1,2,3), scale='linear'):
        """
        Initialises an empty grid with parameters and wavelengths.
        
        Parameters
        ----------
        path : str or path-like
            The path of the base of the KWD library.
        air : bool, optional
            Whether the wavelengths are measured in air or not. Default is False
            (Required due to implementation of inherited GridInterface class)
        wl_range : tuple, optional
            The (min, max) of the wavelengths in AA. Default is (1200, 1400) for testing.
        model_parameters : tuple, optional
            Specifiy the parameters you wish to fit by adding intergers to the tuple. 
        """
        # The grid points in the parameter space are defined, 
        # param_points_1-6 correspond to the model parameters defined at the top in the respective order.
        
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.usecols = usecols # Wavelength and inclination tuple
        self.skiprows = skiprows # Starting point of fluxes in the datafile.
        points = []
        if 1 in model_parameters:
            param_points_1 = np.array([1e-10, 1.5e-10, 2.1e-10, 3.1e-10, 4.5e-10, 6.6e-10, 9.7e-10, 1.4e-09, 2.1e-09, 3e-09])
            points.append(param_points_1)
        if 2 in model_parameters:
            param_points_2 = np.array([4, 16, 32])
            points.append(param_points_2)
        if 3 in model_parameters:
            param_points_3 = np.array([0, 0.5, 1])
            points.append(param_points_3)
        if 4 in model_parameters:
            param_points_4 = np.array([1, 2, 3])
            points.append(param_points_4)
        if 5 in model_parameters:
            param_points_5 = np.array([1e+10, 3e+10, 7e+10])
            points.append(param_points_5)
        if 6 in model_parameters:
            param_points_6 = np.array([1, 3, 6])
            points.append(param_points_6)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        super().__init__(
            name='KWD',
            param_names=param_names,
            points=points,
            wave_units='AA',
            flux_units='erg/s/cm^2/AA',
            air=air,
            wl_range=wl_range,
            path=path,
        )
        
        # The wavelengths for which the fluxes are measured are retrieved.
        try:
            wls_fname = os.path.join(self.path, 'sscyg_k2_000000000000.spec')
            wls = np.loadtxt(wls_fname, delimiter=' ', usecols=self.usecols[0], skiprows=self.skiprows)
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
        Constructs path of datafile corresponding to parameters passed.
        
        Parameters
        ----------
        params : ndarray
            Contains the parameters of a required grid point.
            
        Returns
        -------
        str
            The path of the datafile corresponding to the input parameters.
            
        """
        #param_names = ["c{}".format(number) for number in parameter_numbers]
        # dict file name format for different param values.
        param1_name = {1e-10: '00', 1.5e-10: '01', 2.1e-10: '02', 3.1e-10: '03', 4.5e-10: '04',
                       6.6e-10: '05', 9.7e-10: '06', 1.4e-09: '07', 2.1e-09: '08', 3e-09: '09'}
        param2_name = {4: '00', 16: '01', 32: '02'}
        param3_name = {0: '00', 0.5: '01', 1: '02'}
        param4_name = {1: '00', 2: '01', 3: '02'}
        param5_name = {1e+10: '00', 3e+10: '01', 7e+10: '02'}
        param6_name = {1: '00', 3: '01', 6: '02'}
        all_names = [param1_name, param2_name, param3_name, param4_name, param5_name, param6_name] # Can be improved with dictionary
        param_numbers = params
        base = self.path + 'sscyg_k2_'
        for loop in range(len(all_names)):
            if "param{}".format(loop+1) in self.param_names:
                base += all_names[loop][param_numbers[loop]]
            else:
                base += '00'
                param_numbers = np.insert(param_numbers, loop, 0)          
        return base + '.spec'

    def parameters_description(self, model_parameters):
        """
        Provides a description of the model parameters used.

        Args:
            model_parameters (tuple): Numbers of the parameters used in the model.

        Returns:
            dictionary: Description of the 'paramX' name
        """
        dictionary = {
            1:"wind.mdot (msol/yr)",
            2:"kn.d",
            3:"kn.mdot_r_exponent",
            4:"kn.v_infinity (in_units_of_vescape)",
            5:"kn.acceleration_length (cm)",
            6:"n.acceleration_exponent"
            } #Description of the paramters
        parameters_used = {}
        for i in model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Parameters
        ----------
        parameters : ndarray
            Contains parameters of a required grid point
            
        header : bool
            Whether to attach param values on return. Unimplemented!!
            
        norm : bool
            Whether to normalise the return flux (left unimplemented)
            
        angle_inc : int
            Angle of inclination, takes values between 0 to 6 corresponding
            to values between 40.0 and 70.0 with 5.0 degree increment.
        
        scale : str
            Change this default too if changing the flux scale. 
            Define the scale for your data and emulator grid space. 'linear', 'log' or 'scaled'
                
        Returns
        -------
        ndarray
            List of fluxes in the wavelength range specified on initialisation
            
        dict (Optional)
            Dictionary of parameter names and values
        
        """
        from scipy.ndimage import gaussian_filter1d # Instead of normalising, a 1d gaussian smoothing filter is applied 
        flux = np.loadtxt(self.get_flux(parameters), usecols=self.usecols, skiprows=self.skiprows)
        flux = np.flip(flux)
        #flux = gaussian_filter1d(flux, 50)
        if self.scale == 'log':
            flux = np.log10(flux) # logged 10 
        if self.scale == 'scaled': # to values near order of magnitude 10^0. 
            flux = flux/np.mean(flux)
        
        # TODO: Implement header if doing to use
        hdr = {'c0' : self.usecols[1]} # Header constructed (channel 0 corresponds to angle of inclination)
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]
        
class ShortSpecGridInterface(GridInterface):
    """
    An Interface to the short spec grid produced by PYTHON v87a Radiative transfer simulations.
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/cm
    
    Parameters of model
    -------------------
    1) wind.mdot (msol/yr)
    2) KWD.d
    3) kn.v_infinity (in_units_of_vescape)
    
    Optional parameters
    -------------------
    Angle of Inclination - Amend value in function 'def load_flux' default's value
    
    """
    
    def __init__(self, path, usecols, skiprows, air=False, wl_range=(850,1850), model_parameters=(1,2,3), scale='linear'):
        """
        Initialises an empty grid with parameters and wavelengths.
        
        Args:
            path (str or path-like): The path of the base of the grid space library.
            
            air (bool, optional): Whether the wavelengths are measured in air or not.
                Default is False. (Required due to implementation of inherited GridInterface class)
                
            wl_range (tuple, optional): The (min, max) of the wavelengths in AA. 
                Default is (850, 1850), wavelength range.
            
            model_parameters (tuple, optional): Specifiy the parameters 
                you wish to fit by adding intergers to the tuple. 
        """
        
        # The grid points in the parameter space are defined, 
        # param_points_1-3 correspond to the model parameters defined at the top in the respective order.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.usecols = usecols # Wavelength and inclination tuple
        self.skiprows = skiprows # Starting point of fluxes in the datafile.
        points = []
        if 1 in model_parameters: # wind.mdot (msol/yr)
            param_points_1 = np.array([4.e-11, 1.e-10, 4.e-10, 1.e-09, 3.e-09])
            param_points_1 = np.log10(param_points_1) # logging the wind mass loss parameter WMdot
            points.append(param_points_1)
        if 2 in model_parameters: # KWD.d
            param_points_2 = np.array([2, 5, 8, 12, 16])
            points.append(param_points_2)
        if 3 in model_parameters: # KWD.v_infinity (in_units_of_vescape)
            param_points_3 = np.array([1, 1.5, 2, 2.5, 3])
            points.append(param_points_3)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name='Short_spec_cv_grid',
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
            wls_fname = os.path.join(self.path, 'run01_WMdot4e-11_d2_vinf1.spec')
            wls = np.loadtxt(wls_fname, usecols=self.usecols[0], skiprows=self.skiprows, unpack=True)
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
        
        # param_names = ["c{}".format(number) for number in parameter_numbers]
        param1_name = np.log10([4.e-11, 1.e-10, 4.e-10, 1.e-09, 3.e-09]) # WMdot
        param2_name = [2, 5, 8, 12, 16] # KWD.d
        param3_name = [1, 1.5, 2, 2.5, 3] # KWD.v_infinity
        all_names = [param1_name, param2_name, param3_name]
        #if "param1" in self.param_names: # reversing log of the wind mass loss parameter WMdot
         #   params[0] = sigfig.round(10**params[0], sigfigs=2) # rounding for correct file_name sigfigs
        
        # Fixed values to select runs if 2D grid made, change index if wishing for different fixed other param
        fixed_params = [param1_name[2], param2_name[2], param3_name[2]]
        
        for loop in range(len(all_names)):
            if "param{}".format(loop+1) not in self.param_names: # Checking for missing model parameter
                params = np.insert(params, loop, fixed_params[loop]) # Adding fixed default for missing parameter
        file_runs = os.listdir('short_spec_cv_grid/') # Finding all possible run file names
        if params[2]%1 == 0.5: # eg param 3 replacing 1.5 to 1p5 as per the file names.
            file_name = f'*_WMdot{sigfig.round(10**params[0], sigfigs=1)}_d{int(params[1])}_vinf{str(params[2]).replace(".","p")}.spec'
        else:
            file_name = f'*_WMdot{sigfig.round(10**params[0], sigfigs=1)}_d{int(params[1])}_vinf{int(params[2])}.spec'
        for file in fnmatch.filter(file_runs, file_name): # Matching parameters to the file name
            file_match = file
        return self.path + file_match # returning the correct filename/path for the parameter fluxes

    def parameters_description(self):
        """Provides a description of the model parameters used.

        Returns:
            dictionary: Description of the 'paramX' names
        """
        dictionary = {
            1:"wind.mdot (msol/yr)",
            2:"KWD.d",
            3:"KWD.v_infinity (in_units_of_vescape)",
            } # Description of the paramters
        parameters_used = {}
        for i in self.model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Args:
            parameters (ndarray): Contains parameters of a required grid point
            
            header (bool): Whether to attach param values on return, unimplemented!!
            
            norm (bool): Whether to normalise the return flux (left unimplemented)
            
            angle_inc (int): Angle of inclination, takes values between 0 to 11 corresponding
                to values between 30.0 and 85.0 with 5.0 degree increment.
        
            scale (str): Change this default too if changing the flux scale. 
                Define the scale for your data and emulator grid space. 'linear', 'log' or 'scaled'
                
        Returns:
            ndarray: List of fluxes in the wavelength range specified on initialisation
            
            dict (Optional): Dictionary of parameter names and values
        """
        
        from scipy.ndimage import gaussian_filter1d # Instead of normalising, a 1d gaussian smoothing filter is applied 
        flux = np.loadtxt(self.get_flux(parameters), usecols=self.usecols[1], skiprows=self.skiprows)
        flux = np.flip(flux)
        #flux = gaussian_filter1d(flux, 50)
        if self.scale == 'log':
            flux = np.log10(flux) # logged 10 
        if self.scale == 'scaled': # to values near order of magnitude 10^0. 
            flux = flux/np.mean(flux)
        
        # TODO: Implement header if doing to use
        hdr = {'c0' : self.usecols[1]} # Header constructed (channel 0 corresponds to angle of inclination)
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]


class BroadShortSpecGridInterface(GridInterface):
    """
    An Interface to the short spec grid produced by PYTHON v87a Radiative transfer simulations.
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/cm
    
    Parameters of model
    -------------------
    1) wind.mdot (msol/yr)
    2) KWD.d
    3) kWD.v_infinity (in_units_of_vescape)
    
    Optional parameters
    -------------------
    Angle of Inclination - Amend value in function 'def load_flux' default's value
    
    """
    
    def __init__(self, path, usecols, skiprows, air=False, wl_range=(850,7950), model_parameters=(1,2,3), scale='linear'):
        """
        Initialises an empty grid with parameters and wavelengths.
        
        Args:
            path (str or path-like): The path of the base of the grid space library.
            
            air (bool, optional): Whether the wavelengths are measured in air or not.
                Default is False. (Required due to implementation of inherited GridInterface class)
                
            wl_range (tuple, optional): The (min, max) of the wavelengths in AA. 
                Default is (850, 1850), wavelength range.
            
            model_parameters (tuple, optional): Specifiy the parameters 
                you wish to fit by adding intergers to the tuple. 
        """
        
        # The grid points in the parameter space are defined, 
        # param_points_1-3 correspond to the model parameters defined at the top in the respective order.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.usecols = usecols # Wavelength and inclination tuple
        self.skiprows = skiprows # Starting point of fluxes in the datafile.
        points = []
        if 1 in model_parameters: # wind.mdot (msol/yr)
            param_points_1 = np.array([4.e-11, 2.e-10, 1.e-9, 5.e-9, 2.5e-8])
            param_points_1 = np.log10(param_points_1) # logging the wind mass loss parameter WMdot
            points.append(param_points_1)
        if 2 in model_parameters: # KWD.d
            param_points_2 = np.array([2, 5, 8, 12, 16])
            points.append(param_points_2)
        if 3 in model_parameters: # KWD.v_infinity (in_units_of_vescape)
            param_points_3 = np.array([1, 1.5, 2, 2.5, 3])
            points.append(param_points_3)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name='Broad_short_spec_cv_grid',
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
            wls_fname = os.path.join(self.path, 'run01_WMdot4e-11_d2_vinf1.spec')
            wls = np.loadtxt(wls_fname, usecols=self.usecols[0], skiprows=self.skiprows, unpack=True)
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
        
        # param_names = ["c{}".format(number) for number in parameter_numbers]
        param1_name = np.log10([4.e-11, 2.e-10, 1.e-9, 5.e-9, 2.5e-8]) # WMdot
        param2_name = [2, 5, 8, 12, 16] # KWD.d
        param3_name = [1, 1.5, 2, 2.5, 3] # KWD.v_infinity
        all_names = [param1_name, param2_name, param3_name]
        #if "param1" in self.param_names: # reversing log of the wind mass loss parameter WMdot
         #   params[0] = sigfig.round(10**params[0], sigfigs=2) # rounding for correct file_name sigfigs
        
        # Fixed values to select runs if 2D grid made, change index if wishing for different fixed other param
        fixed_params = [param1_name[2], param2_name[2], param3_name[2]]
        
        for loop in range(len(all_names)):
            if "param{}".format(loop+1) not in self.param_names: # Checking for missing model parameter
                params = np.insert(params, loop, fixed_params[loop]) # Adding fixed default for missing parameter
        file_runs = os.listdir('broad_short_spec_cv_grid/') # Finding all possible run file names
        
        # TODO: The original file names are not consistent, need to change the file names to be consistent.
        # The 1e-9 is changed to 1e-09 to avoid conflicts. The grid setup file should be corrected
        # to avoid this issue at a later date. Or a fix for this code. Manually remaining the files
        # is a quickly solution at the moment however.
        if (str(sigfig.round(10**np.log10(2.5e-8), sigfigs=2))).count(".") == 1:# eg param 1 replacing 2.5 to 2p5 as per the file names.
            file_name = f'*_WMdot{str(sigfig.round(10**params[0], sigfigs=2)).replace(".","p")}_'
        else:
            file_name = f'*_WMdot{sigfig.round(10**params[0], sigfigs=1)}_'
        if params[1]%1 == 0.5:
            file_name += f'd{str(params[1]).replace(".","p")}_'
        else:
            file_name += f'd{int(params[1])}_'
        if params[2]%1 == 0.5:
            file_name += f'vinf{str(params[2]).replace(".","p")}.spec'
        else:
            file_name += f'vinf{int(params[2])}.spec'
        
        for file in fnmatch.filter(file_runs, file_name): # Matching parameters to the file name
            file_match = file
        return self.path + file_match # returning the correct filename/path for the parameter fluxes

    def parameters_description(self):
        """Provides a description of the model parameters used.

        Returns:
            dictionary: Description of the 'paramX' names
        """
        dictionary = {
            1:"wind.mdot (msol/yr)",
            2:"KWD.d",
            3:"KWD.v_infinity (in_units_of_vescape)",
            } # Description of the paramters
        parameters_used = {}
        for i in self.model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Args:
            parameters (ndarray): Contains parameters of a required grid point
            
            header (bool): Whether to attach param values on return, unimplemented!!
            
            norm (bool): Whether to normalise the return flux (left unimplemented)
            
            angle_inc (int): Angle of inclination, takes values between 0 to 11 corresponding
                to values between 30.0 and 85.0 with 5.0 degree increment.
        
            scale (str): Change this default too if changing the flux scale. 
                Define the scale for your data and emulator grid space. 'linear', 'log' or 'scaled'
                
        Returns:
            ndarray: List of fluxes in the wavelength range specified on initialisation
            
            dict (Optional): Dictionary of parameter names and values
        """
        
        from scipy.ndimage import gaussian_filter1d # Instead of normalising, a 1d gaussian smoothing filter is applied 
        flux = np.loadtxt(self.get_flux(parameters), usecols=self.usecols[1], skiprows=self.skiprows)
        flux = np.flip(flux)
        #flux = gaussian_filter1d(flux, 50)
        if self.scale == 'log':
            flux = np.log10(flux) # logged 10 
        if self.scale == 'scaled': # to values near order of magnitude 10^0. 
            flux = flux/np.mean(flux)
        
        # TODO: Implement header if doing to use
        hdr = {'c0' : self.usecols[1]} # Header constructed (channel 0 corresponds to angle of inclination)
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]

    
class OpticalCVGridInterface(GridInterface):
    """
    An Interface to the short spec grid produced by PYTHON v87b Radiative transfer simulations.
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/cm
    
    Parameters of model
    -------------------
    1) Disk.mdot (msol/yr)
    2) wind.mdot (Disk.mdot)
    3) KWD.d (in_units_of_Rstar)
    4) KWD.mdot_r_exponent
    5) KWD.acceleration_length (cm)
    6) KWD.acceleration_exponent
    
    Optional parameters
    -------------------
    Angle of Inclination - Amend value in function 'def load_flux' default's value
    
    """
    
    def __init__(self, path, usecols, skiprows, air=False, wl_range=(850,7950), model_parameters=(1,2,3,4,5,6), scale='linear'):
        """
        Initialises an empty grid with parameters and wavelengths.
        
        Args:
            path (str or path-like): The path of the base of the grid space library.
            
            air (bool, optional): Whether the wavelengths are measured in air or not.
                Default is False. (Required due to implementation of inherited GridInterface class)
                
            wl_range (tuple, optional): The (min, max) of the wavelengths in AA. 
                Default is (850, 7950), wavelength range.
            
            model_parameters (tuple, optional): Specifiy the parameters 
                you wish to fit by adding intergers to the tuple. 
        """
        
        # The grid points in the parameter space are defined, 
        # param_points_1-3 correspond to the model parameters defined at the top in the respective order.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.usecols = usecols # Wavelength and inclination tuple
        self.skiprows = skiprows # Starting point of fluxes in the datafile.
        points = []
        if 1 in model_parameters:
            param_points_1 = np.array([3e-09, 1e-08, 3e-08])
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
            param_points_5 = np.array([7.25182e+08, 7.25182e+09, 7.25182e+10])
            points.append(param_points_5)
        if 6 in model_parameters:
            param_points_6 = np.array([0.5, 1.5, 4.5])
            points.append(param_points_6)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name='Optical_cv_grid',
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
            wls = np.loadtxt(wls_fname, usecols=self.usecols[0], skiprows=self.skiprows, unpack=True)
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
        
        # param_names = ["c{}".format(number) for number in parameter_numbers]
        param1_name = np.log10([3e-9 , 1e-08, 3e-08]) # Disk.mdot
        param2_name = [0.03, 0.1, 0.3] # wind.mdot (Disk.mdot)
        param3_name = [0.55, 5.5, 55.0] # KWD.d
        param4_name = [0.0, 0.25, 1.0] # KWD.mdot_r_exponent
        param5_name = [7.25182e+08, 7.25182e+09, 7.25182e+10] # KWD.acceleration_length (cm)
        param6_name = [0.5, 1.5, 4.5] # KWD.acceleration_exponent
        all_names = [param1_name, param2_name, param3_name, param4_name, param5_name, param6_name]
        
        # Fixed values to select runs if 2D grid made, change index if wishing for different fixed other param
        fixed_params = [param1_name[1], param2_name[1], param3_name[1], param4_name[1], param5_name[1], param6_name[1]] 
        
        for loop in range(len(all_names)):
            if "param{}".format(loop+1) not in self.param_names: # Checking for missing model parameter
                params = np.insert(params, loop, fixed_params[loop]) # Adding fixed default for missing parameter
        file_runs = os.listdir('optical_grid_spec_files/') # Finding all possible run file names
        
        # All the grid combinations
        import itertools
        unique_combinations = []
        temp_grid = np.array([10**param1_name, param2_name, param3_name, param4_name, param5_name, param6_name])
        for i in itertools.product(*temp_grid):
            unique_combinations.append(list(i))

        # Generating all possible combinations to match the correct file name
        for i in range(len(unique_combinations)):
            if np.all(np.round(params,decimals=10) == np.round(unique_combinations[i], decimals=10)):
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
            6:"KWD.acceleration_exponent"
            } # Description of the paramters
        parameters_used = {}
        for i in self.model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Args:
            parameters (ndarray): Contains parameters of a required grid point
            
            header (bool): Whether to attach param values on return, unimplemented!!
            
            norm (bool): Whether to normalise the return flux (left unimplemented)
            
            angle_inc (int): Angle of inclination, takes values between 0 to 11 corresponding
                to values between 30.0 and 85.0 with 5.0 degree increment.
        
            scale (str): Change this default too if changing the flux scale. 
                Define the scale for your data and emulator grid space. 'linear', 'log' or 'scaled'
                
        Returns:
            ndarray: List of fluxes in the wavelength range specified on initialisation
            
            dict (Optional): Dictionary of parameter names and values
        """
        
        from scipy.ndimage import gaussian_filter1d # Instead of normalising, a 1d gaussian smoothing filter is applied 
        flux = np.loadtxt(self.get_flux(parameters), usecols=self.usecols[1], skiprows=self.skiprows)
        flux = np.flip(flux)
        #flux = gaussian_filter1d(flux, 50)
        if self.scale == 'log':
            flux = np.log10(flux) # logged 10 
        if self.scale == 'scaled': # to values near order of magnitude 10^0. 
            flux = flux/np.mean(flux)
        
        # TODO: Implement header if doing to use
        hdr = {'c0' : self.usecols[1]} # Header constructed (channel 0 corresponds to angle of inclination)
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]
        
class HalphaCVGridInterface(GridInterface):
    """
    An Interface to the high res H_alpha line grid produced by PYTHON v87f Radiative transfer simulations. (Now, Sirocco)
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/Angstrom
    
    Parameters of model
    -------------------
    1) Disk.mdot (msol/yr)
    2) wind.mdot (Disk.mdot)
    3) KWD.d (in_units_of_Rstar)
    4) KWD.mdot_r_exponent
    5) KWD.acceleration_length (cm)
    6) KWD.acceleration_exponent
    
    Optional parameters
    -------------------
    Angle of Inclination - Amend value in function 'def load_flux' default's value
    
    """
        
    def __init__(self, path, usecols, skiprows, air=False, wl_range=(6385,6735), model_parameters=(1,2,3,4,5,6), scale='linear'):
        """
        Initialises an empty grid with parameters and wavelengths.
        
        Args:
            path (str or path-like): The path of the base of the grid space library.
            
            air (bool, optional): Whether the wavelengths are measured in air or not.
                Default is False. (Required due to implementation of inherited GridInterface class)
                
            wl_range (tuple, optional): The (min, max) of the wavelengths in AA. 
                Default is (6385, 6735), wavelength range.
            
            model_parameters (tuple, optional): Specifiy the parameters 
                you wish to fit by adding intergers to the tuple. 
        """
        
        # The grid points in the parameter space are defined, 
        # param_points_1-3 correspond to the model parameters defined at the top in the respective order.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.usecols = usecols # Wavelength and inclination tuple
        self.skiprows = skiprows # Starting point of fluxes in the datafile.
        points = []
        if 1 in model_parameters:
            param_points_1 = np.array([3e-09, 1e-08, 3e-08])
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
            param_points_5 = np.array([7.25182e+08, 7.25182e+09, 7.25182e+10])
            points.append(param_points_5)
        if 6 in model_parameters:
            param_points_6 = np.array([0.5, 1.5, 4.5])
            points.append(param_points_6)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name='H_alpha_cv_grid',
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
            wls_fname = os.path.join(self.path, 'rerun0.spec')
            wls = np.loadtxt(wls_fname, usecols=self.usecols[0], skiprows=self.skiprows, unpack=True)
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
        
        # param_names = ["c{}".format(number) for number in parameter_numbers]
        param1_name = np.log10([3e-9 , 1e-08, 3e-08]) # Disk.mdot
        param2_name = [0.03, 0.1, 0.3] # wind.mdot (Disk.mdot)
        param3_name = [0.55, 5.5, 55.0] # KWD.d
        param4_name = [0.0, 0.25, 1.0] # KWD.mdot_r_exponent
        param5_name = [7.25182e+08, 7.25182e+09, 7.25182e+10] # KWD.acceleration_length (cm)
        param6_name = [0.5, 1.5, 4.5] # KWD.acceleration_exponent
        all_names = [param1_name, param2_name, param3_name, param4_name, param5_name, param6_name]
        
        # Fixed values to select runs if 2D grid made, change index if wishing for different fixed other param
        fixed_params = [param1_name[1], param2_name[1], param3_name[1], param4_name[1], param5_name[1], param6_name[1]] 
        
        for loop in range(len(all_names)):
            if "param{}".format(loop+1) not in self.param_names: # Checking for missing model parameter
                params = np.insert(params, loop, fixed_params[loop]) # Adding fixed default for missing parameter
        file_runs = os.listdir('Ha_grid_spec_files/') # Finding all possible run file names
        
        # All the grid combinations
        import itertools
        unique_combinations = []
        temp_grid = np.array([10**param1_name, param2_name, param3_name, param4_name, param5_name, param6_name])
        for i in itertools.product(*temp_grid):
            unique_combinations.append(list(i))

        # Generating all possible combinations to match the correct file name
        for i in range(len(unique_combinations)):
            if np.all(np.round(params,decimals=10) == np.round(unique_combinations[i], decimals=10)):
                file_name = f'rerun{i}.spec' # if matched, index is run number
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
            6:"KWD.acceleration_exponent"
            } # Description of the paramters
        parameters_used = {}
        for i in self.model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Args:
            parameters (ndarray): Contains parameters of a required grid point
            
            header (bool): Whether to attach param values on return, unimplemented!!
            
            norm (bool): Whether to normalise the return flux (left unimplemented)
            
            angle_inc (int): Angle of inclination, takes values between 0 to 11 corresponding
                to values between 30.0 and 85.0 with 5.0 degree increment.
        
            scale (str): Change this default too if changing the flux scale. 
                Define the scale for your data and emulator grid space. 'linear', 'log' or 'scaled'
                
        Returns:
            ndarray: List of fluxes in the wavelength range specified on initialisation
            
            dict (Optional): Dictionary of parameter names and values
        """
        
        from scipy.ndimage import gaussian_filter1d # Instead of normalising, a 1d gaussian smoothing filter is applied 
        flux = np.loadtxt(self.get_flux(parameters), usecols=self.usecols[1], skiprows=self.skiprows)
        flux = np.flip(flux)
        #flux = gaussian_filter1d(flux, 50)
        if self.scale == 'log':
            flux = np.log10(flux) # logged 10 
        if self.scale == 'scaled': # to values near order of magnitude 10^0. 
            flux = flux/np.mean(flux)
        
        # TODO: Implement header if doing to use
        hdr = {'c0' : self.usecols[1]} # Header constructed (channel 0 corresponds to angle of inclination)
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]
        

        
class CVReleaseGridInterface(GridInterface):
    """
    An Interface to the short spec grid produced by PYTHON v87b Radiative transfer simulations.
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/cm
    
    Parameters of model
    -------------------
    1) Disk.mdot (msol/yr)
    2) wind.mdot (Disk.mdot)
    3) KWD.d (in_units_of_Rstar)
    4) KWD.mdot_r_exponent
    5) KWD.acceleration_length (cm)
    6) KWD.acceleration_exponent
    7) Boundary_layer.luminosity(ergs/s),
    8) Boundary_layer.temp(K)
    
    Optional parameters
    -------------------
    Angle of Inclination - Amend value in function 'def load_flux' default's value
    
    """
        
    def __init__(self, path, usecols, skiprows, air=False, wl_range=(800,8000), model_parameters=(1,2,3,4,5,6,7,8,9), scale='linear'):
        """
        Initialises an empty grid with parameters and wavelengths.
        
        Args:
            path (str or path-like): The path of the base of the grid space library.
            
            air (bool, optional): Whether the wavelengths are measured in air or not.
                Default is False. (Required due to implementation of inherited GridInterface class)
                
            wl_range (tuple, optional): The (min, max) of the wavelengths in AA. 
                Default is (6385, 6735), wavelength range.
            
            model_parameters (tuple, optional): Specifiy the parameters 
                you wish to fit by adding intergers to the tuple. 
        """
        
        # The grid points in the parameter space are defined, 
        # param_points_1-3 correspond to the model parameters defined at the top in the respective order.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.usecols = usecols # Wavelength and inclination tuple
        self.skiprows = skiprows # Starting point of fluxes in the datafile.
        points = []
        if 1 in model_parameters:
            param_points_1 = np.array([3e-09, 1e-08, 3e-08])
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
            param_points_5 = np.array([7.25182e+08, 7.25182e+09, 7.25182e+10])
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
            param_points_10 = np.array([30, 40, 50, 60, 70, 80])  # Every 10 degrees
            points.append(param_points_10)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name='CV_release_grid_spec',
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
            wls = np.loadtxt(wls_fname, usecols=self.usecols[0], skiprows=self.skiprows, unpack=True)
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
        
        # Parameter definitions for both file lookup (1-8) and full space (1-9)
        param1_name = np.log10([3e-9 , 1e-08, 3e-08]) # Disk.mdot
        param2_name = [0.03, 0.1, 0.3] # wind.mdot (Disk.mdot)
        param3_name = [0.55, 5.5, 55.0] # KWD.d
        param4_name = [0.0, 0.25, 1.0] # KWD.mdot_r_exponent
        param5_name = np.log10([7.25182e+08, 7.25182e+09, 7.25182e+10]) # KWD.acceleration_length (cm)
        param6_name = [0.5, 1.5, 4.5] # KWD.acceleration_exponent
        param7_name = [0.0, 0.3, 1.0] # Boundary_layer.luminosity(ergs/s)
        param8_name = [0.1, 0.3, 1.0] # Boundary_layer.temp(K)
        param9_name = [30, 55, 80] # Inclination angle (degrees)
        param10_name = [30, 40, 50, 60, 70, 80] # Inclination angle (degrees) - every 10 degrees
        
        # For file lookup: only parameters 1-8 (determines filename)
        #file_lookup_names = [param1_name, param2_name, param3_name, param4_name, param5_name, param6_name, param7_name, param8_name]
        fixed_params = [10**param1_name[1], param2_name[1], param3_name[1], param4_name[1], 10**param5_name[1], param6_name[1], param7_name[1], param8_name[1]]
        
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
        
        # # Skip parameters 9 and 10 in param_index counting if they exist
        # # They are handled separately in load_flux for column selection
        # if 9 in self.model_parameters or 10 in self.model_parameters:
        #     # We've used all non-inclination parameters, so param_index is correct
        #     pass
        
        file_params = np.array(file_params)
        
        # Create file lookup combinations (only parameters 1-8)
        import itertools
        file_combinations = []
        temp_grid = np.array([10**param1_name, param2_name, param3_name, param4_name, 10**param5_name, param6_name, param7_name, param8_name])
        for i in itertools.product(*temp_grid):
            file_combinations.append(list(i))

        # Find the matching file using 8-parameter combination
        for i in range(len(file_combinations)):
            if np.all(np.round(file_params, decimals=10) == np.round(file_combinations[i], decimals=10)):
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
            10:"Inclination angle - full (30-80 degrees, 5° steps)"
            } # Description of the paramters
        parameters_used = {}
        for i in self.model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
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
        if 9 in self.model_parameters:
            # Parameter 9: Sparse inclinations (30, 55, 80 degrees)
            inclination_param_index = list(self.model_parameters).index(9)
            inclination_angle = parameters[inclination_param_index]
            
            # Map sparse inclination angles to column index
            # 30°->10, 55°->20, 80°->30 (based on the 5° interval structure)
            inclination_column = int(10 + (inclination_angle - 30) / 5)
            
            # LOG THE DETAILS
            logger = logging.getLogger(__name__)
            logger.info(f"GRID PROCESSING [Param9-Sparse]: File={file_name}, Parameters={parameters}, Inclination={inclination_angle}°, Column={inclination_column}")
            
        elif 10 in self.model_parameters:
            # Parameter 10: Full inclinations (30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80 degrees)
            inclination_param_index = list(self.model_parameters).index(10)
            inclination_angle = parameters[inclination_param_index]
            
            # Map all inclination angles to column index
            # 30°->10, 35°->11, 40°->12, ..., 75°->19, 80°->20
            inclination_column = int(10 + (inclination_angle - 30) / 5)
            
            # LOG THE DETAILS
            logger = logging.getLogger(__name__)
            logger.info(f"GRID PROCESSING [Param10-Full]: File={file_name}, Parameters={parameters}, Inclination={inclination_angle}°, Column={inclination_column}")
            
        else:
            # Use the default column from usecols if neither parameter 9 nor 10 in model_parameters
            inclination_column = self.usecols[1]
            
            # LOG THE DETAILS
            logger = logging.getLogger(__name__)
            logger.info(f"GRID PROCESSING [Default]: File={file_name}, Parameters={parameters}, Default_Column={inclination_column}")
        
        # Load flux using wavelength (column 0) and calculated inclination column
        wl, flux = np.loadtxt(self.get_flux(parameters), usecols=(0, inclination_column), skiprows=self.skiprows, unpack=True)
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

