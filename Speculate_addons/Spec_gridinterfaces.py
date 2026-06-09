import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from Starfish.grid_tools import GridInterface
from Speculate_addons.Spec_functions import fit_power_law_continuum        
from Speculate_addons.grid_registry import (
    default_parameter_value,
    inclination_column as registry_inclination_column,
    parameter_description_map,
    parameter_points,
)


def _grid_index(grid_values, value):
    """Return the unique index of ``value`` in one emulator grid axis."""
    grid_values = np.asarray(grid_values)
    matches = np.where(np.isclose(grid_values, value, rtol=1e-9, atol=1e-8))[0]
    if len(matches) != 1:
        raise ValueError(f"Parameter value {value} does not map uniquely to grid values {grid_values.tolist()}")
    return int(matches[0])


def _inclination_column_for_parameters(grid_name, model_parameters, parameters, usecols):
    """Resolve the raw .spec flux column for a fixed or trainable inclination.

    When inclination is part of ``model_parameters`` the actual angle arrives in
    ``parameters`` and must be mapped through the registry.  When inclination is
    fixed outside the emulator, the caller has already placed the correct flux
    column in ``usecols``.  The Training Tool UI prevents multiple trainable
    inclination axes, but direct API callers can still pass legacy defaults that
    include more than one; in that case the lowest registered inclination axis is
    used, matching the historical CV behavior.
    """
    descriptions = parameter_description_map(grid_name)
    inclination_params = sorted(
        param_id
        for param_id, description in descriptions.items()
        if "Inclination angle" in description and param_id in model_parameters
    )

    if inclination_params:
        inclination_param_num = inclination_params[0]
        inclination_param_index = list(model_parameters).index(inclination_param_num)
        inclination_angle = parameters[inclination_param_index]
        return registry_inclination_column(grid_name, inclination_angle)

    return usecols[1]


def _apply_flux_scale(wl_full, ind, flux, scale):
    """Apply the training-time flux transform used by Speculate grids."""
    if scale == 'log':
        # Log training cannot accept zero or negative fluxes; floor only for the
        # transform, leaving linear and continuum-normalised spectra unchanged.
        flux = np.where(flux > 0, flux, 1e-30)
        return np.log10(flux)
    if scale == 'continuum-normalised':
        # Continuum normalisation is computed only over the wavelength window
        # used for training so the fitted continuum matches the output slice.
        wl_sel = wl_full[:len(flux)][ind]
        continuum, _ = fit_power_law_continuum(wl_sel, flux[ind])
        flux[ind] = flux[ind] / np.where(continuum > 0, continuum, 1.0)
    return flux


def _maybe_smooth_flux(flux, smoothing):
    """Apply the optional Gaussian smoothing used by Training and Quick Fit."""
    if not smoothing:
        return flux
    return gaussian_filter1d(np.asarray(flux, dtype=np.float64), 10)


class Speculate_agn_grid_v1_3(GridInterface):
    """
    An Interface to the AGN v1.3 grid produced by Sirocco radiative transfer simulations.

    Parameters 1-8 use the scaled Cartesian coordinates that define the grid;
    parameters 9 and 10 expose sparse and full inclination axes respectively.
    """

    grid_name = 'speculate_agn_grid_v1.3'

    def __init__(self, path, usecols, air=False, wl_range=(800,8000), model_parameters=(1,2,3,4,5,6,7,8,9,10), scale='linear', smoothing=False):
        """Create an AGN grid reader for a selected parameter subset.

        ``model_parameters`` may omit file-defining axes for lower-dimensional
        emulators.  Missing physical axes are filled from registry defaults in
        ``get_flux()``, while omitted inclination axes are handled by the fixed
        flux column stored in ``usecols``.
        """
        self.model_parameters = model_parameters
        self.scale = scale
        self.smoothing = smoothing
        self.usecols = usecols
        self._skiprows_cache = {}

        # Starfish expects the emulator-space points for exactly the exposed
        # parameters.  The registry provides those points for both physical axes
        # and inclination axes.
        points = [parameter_points(self.grid_name, param_num) for param_num in model_parameters]
        param_names = ["param{}".format(number) for number in model_parameters]

        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'

        super().__init__(
            name=self.grid_name,
            param_names=param_names,
            points=points,
            wave_units='AA',
            flux_units=flux_units,
            air=air,
            wl_range=wl_range,
            path=path,
        )

        try:
            # Use run0 only to establish the wavelength array and wavelength
            # mask.  Individual spectra are loaded later by load_flux().
            wls_fname = os.path.join(self.path, 'run0.spec')
            skiprows = self._get_skiprows(wls_fname)
            wls = np.loadtxt(wls_fname, usecols=self.usecols[0], skiprows=skiprows, unpack=True)
            wls = np.flip(wls)
        except Exception as exc:
            raise ValueError("Wavelength file improperly specified") from exc

        self.wl_full = np.array(wls, dtype=np.float64)
        self.ind = (self.wl_full >= self.wl_range[0]) & (
            self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]

    def get_flux(self, params):
        """Return the AGN ``runN.spec`` path for emulator-space parameters.

        The AGN file index is determined only by parameters 1-8.  Inclination is
        a flux column inside the selected file, not part of the run number.
        """
        file_params = []
        param_index = 0

        for param_num in range(1, 9):
            if param_num in self.model_parameters:
                # Consume supplied emulator coordinates in model-parameter order.
                file_params.append(params[param_index])
                param_index += 1
            else:
                # Lower-dimensional emulators fix omitted file axes at the
                # registry default (middle value, or high value for two-point axes).
                file_params.append(default_parameter_value(self.grid_name, param_num))

        # The run files are ordered as a row-major Cartesian product of physical
        # axes 1-8.  Reconstruct the same mixed-radix index from the grid points.
        grids = [parameter_points(self.grid_name, param_num) for param_num in range(1, 9)]
        run_index = 0
        for grid_values, value in zip(grids, file_params):
            value_index = _grid_index(grid_values, value)
            run_index = run_index * len(grid_values) + value_index

        return os.path.join(self.path, f'run{run_index}.spec')

    def parameters_description(self):
        """Return human-readable labels for the selected AGN parameters."""
        dictionary = parameter_description_map(self.grid_name)
        return {
            "param{}".format(i): dictionary[i]
            for i in self.model_parameters
        }

    def _get_skiprows(self, filepath):
        """Find and cache the first numeric row in an AGN .spec file."""
        if filepath in self._skiprows_cache:
            return self._skiprows_cache[filepath]
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    continue
                if line.startswith('Freq.'):
                    continue
                self._skiprows_cache[filepath] = i
                return i
        self._skiprows_cache[filepath] = 0
        return 0

    def load_flux(self, parameters, header=False, norm=False):
        """Load one AGN spectrum for the requested parameters and inclination."""
        file_path = self.get_flux(parameters)
        inclination_column = _inclination_column_for_parameters(
            self.grid_name, self.model_parameters, parameters, self.usecols
        )

        # All AGN .spec files share the same wavelength column convention:
        # column 1 is Lambda, and columns 2+ are inclination fluxes.
        skiprows = self._get_skiprows(file_path)
        _wl, flux = np.loadtxt(file_path, usecols=(self.usecols[0], inclination_column), skiprows=skiprows, unpack=True)
        flux = np.flip(flux)
        flux = flux[:len(self.wl_full)]
        flux = _maybe_smooth_flux(flux, self.smoothing)
        flux = _apply_flux_scale(self.wl_full, self.ind, flux, self.scale)

        hdr = {'inclination_column': inclination_column}
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]
        
class Speculate_cv_bl_grid_v87f(GridInterface):
    """Interface to the CV boundary-layer grid produced by Sirocco/PYTHON v87f.

    The registry entry named by ``grid_name`` owns the parameter descriptions,
    emulator-space grid points, default fixed axes, and inclination column map.
    Parameters 1-8 determine the ``runN.spec`` file; parameters 9-11 select
    sparse/mid/full inclination columns inside that file.
    """

    grid_name = 'speculate_cv_bl_grid_v87f'
        
    def __init__(self, path, usecols, air=False, wl_range=(800,8000), model_parameters=(1,2,3,4,5,6,7,8,9,10,11), scale='linear', smoothing=False):
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
            
            smoothing (bool, optional): Whether to apply Gaussian smoothing (sigma=10) to spectra.
                Default is False.
        """
        
        # The registry defines the emulator-space points for each selected axis.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.smoothing = smoothing # Gaussian smoothing toggle
        self.usecols = usecols # Wavelength and inclination tuple
        self._skiprows_cache = {}
        # self.skiprows = skiprows # Deprecated: calculated dynamically
        points = [parameter_points(self.grid_name, param_num) for param_num in model_parameters]
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name=self.grid_name,
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
        except Exception as exc:
            raise ValueError("Wavelength file improperly specified") from exc
        
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
        
        # Build the 8-parameter array for file lookup (excluding inclination parameters).
        file_params = []
        param_index = 0
        
        for param_num in range(1, 9):  # Parameters 1-8 only
            if param_num in self.model_parameters:  # Include if in model_parameters
                file_params.append(params[param_index])
                param_index += 1
            else:
                # Use the registry default value for omitted file-defining axes.
                file_params.append(default_parameter_value(self.grid_name, param_num))

        grids = [parameter_points(self.grid_name, param_num) for param_num in range(1, 9)]
        run_index = 0
        for grid_values, value in zip(grids, file_params):
            value_index = _grid_index(grid_values, value)
            run_index = run_index * len(grid_values) + value_index
        file_name = f'run{run_index}.spec'
        
        return os.path.join(self.path, file_name) # returning the correct filename/path

    def parameters_description(self):
        """Provides a description of the model parameters used.

        Returns:
            dictionary: Description of the 'paramX' names
        """
        dictionary = parameter_description_map(self.grid_name)
        return {
            "param{}".format(i): dictionary[i]
            for i in self.model_parameters
        }
        
    def _get_skiprows(self, filepath):
        """
        Scans the file to find the number of header lines.
        Assumes header lines start with '#' or 'Freq.' (after stripping whitespace).
        """
        if filepath in self._skiprows_cache:
            return self._skiprows_cache[filepath]
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                if line.startswith('#'): continue
                if line.startswith('Freq.'): continue
                # Found the first data line
                self._skiprows_cache[filepath] = i
                return i
        self._skiprows_cache[filepath] = 0
        return 0

    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Args:
            parameters (ndarray): Contains parameters of a required grid point
            
            header (bool): Whether to attach param values on return, unimplemented!!
            
            norm (bool): Whether to normalise the return flux (left unimplemented)
        
            scale (str): Change this default too if changing the flux scale. 
                Define the scale for your data and emulator grid space. 'linear', 'log' or 'continuum-normalised'
                
        Returns:
            ndarray: List of fluxes in the wavelength range specified on initialisation
            
            dict (Optional): Dictionary of parameter names and values
        """
        
        
        file_path = self.get_flux(parameters)
        
        inclination_column = _inclination_column_for_parameters(
            self.grid_name, self.model_parameters, parameters, self.usecols
        )
        
        # Load flux using the configured wavelength column and calculated
        # inclination column.  The wavelength array is already cached from run0;
        # this read keeps loadtxt's two-column shape consistent across grids.
        # Optimized: Pre-scan for header length to avoid line-by-line comment checking in loadtxt
        skiprows = self._get_skiprows(file_path)
        _wl, flux = np.loadtxt(file_path, usecols=(self.usecols[0], inclination_column), skiprows=skiprows, unpack=True)
        flux = np.flip(flux)

        flux = flux[:len(self.wl_full)] # THIS CUT IS NEEDED, Random parameters appear in the grid space header leading to mismatching file lengths.
        flux = _maybe_smooth_flux(flux, self.smoothing)
        flux = _apply_flux_scale(self.wl_full, self.ind, flux, self.scale)

        # Header mirrors the selected emulator coordinates plus the raw .spec
        # flux column used for this spectrum.
        hdr = {'inclination_column': inclination_column}
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]

class Speculate_cv_no_bl_grid_v87f(GridInterface):
    """Interface to the CV no-boundary-layer grid produced by Sirocco/PYTHON v87f.

    The registry entry named by ``grid_name`` owns the parameter descriptions,
    emulator-space grid points, default fixed axes, and inclination column map.
    Parameters 1-6 determine the ``runN.spec`` file; parameters 9-11 select
    sparse/mid/full inclination columns inside that file.
    """

    grid_name = 'speculate_cv_no-bl_grid_v87f'
        
    def __init__(self, path, usecols, air=False, wl_range=(800,8000), model_parameters=(1,2,3,4,5,6,9,10,11), scale='linear', smoothing=False):
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
            
            smoothing (bool, optional): Whether to apply Gaussian smoothing (sigma=10) to spectra.
                Default is False.
        """
        
        # The registry defines the emulator-space points for each selected axis.
        self.model_parameters = model_parameters
        self.scale = scale # Flux space scale 
        self.smoothing = smoothing # Gaussian smoothing toggle
        self.usecols = usecols # Wavelength and inclination tuple
        self._skiprows_cache = {}
        # self.skiprows = skiprows # Deprecated: calculated dynamically
        points = [parameter_points(self.grid_name, param_num) for param_num in model_parameters]
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        if self.scale == 'log':
            flux_units = 'log(erg/s/cm^2/AA)'
        else:
            flux_units = 'erg/s/cm^2/AA'
            
        super().__init__(
            name=self.grid_name,
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
        except Exception as exc:
            raise ValueError("Wavelength file improperly specified") from exc
        
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
        
        # Build the 6-parameter array for file lookup (excluding inclination parameters).
        file_params = []
        param_index = 0
        
        for param_num in range(1, 7):  # Parameters 1-6 only
            if param_num in self.model_parameters:  # Include if in model_parameters
                file_params.append(params[param_index])
                param_index += 1
            else:
                # Use the registry default value for omitted file-defining axes.
                file_params.append(default_parameter_value(self.grid_name, param_num))

        grids = [parameter_points(self.grid_name, param_num) for param_num in range(1, 7)]
        run_index = 0
        for grid_values, value in zip(grids, file_params):
            value_index = _grid_index(grid_values, value)
            run_index = run_index * len(grid_values) + value_index
        file_name = f'run{run_index}.spec'
    
        
        return os.path.join(self.path, file_name) # returning the correct filename/path

    def parameters_description(self):
        """Provides a description of the model parameters used.

        Returns:
            dictionary: Description of the 'paramX' names
        """
        dictionary = parameter_description_map(self.grid_name)
        return {
            "param{}".format(i): dictionary[i]
            for i in self.model_parameters
        }
        
    def _get_skiprows(self, filepath):
        """
        Scans the file to find the number of header lines.
        Assumes header lines start with '#' or 'Freq.' (after stripping whitespace).
        """
        if filepath in self._skiprows_cache:
            return self._skiprows_cache[filepath]
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                if line.startswith('#'): continue
                if line.startswith('Freq.'): continue
                # Found the first data line
                self._skiprows_cache[filepath] = i
                return i
        self._skiprows_cache[filepath] = 0
        return 0

    def load_flux(self, parameters, header=False, norm=False):
        """
        Returns the Flux of a given set of parameters.
        
        Args:
            parameters (ndarray): Contains parameters of a required grid point
            
            header (bool): Whether to attach param values on return, unimplemented!!
            
            norm (bool): Whether to normalise the return flux (left unimplemented)
        
            scale (str): Change this default too if changing the flux scale. 
                Define the scale for your data and emulator grid space. 'linear', 'log' or 'continuum-normalised'
                
        Returns:
            ndarray: List of fluxes in the wavelength range specified on initialisation
            
            dict (Optional): Dictionary of parameter names and values
        """
        
        
        file_path = self.get_flux(parameters)
        
        inclination_column = _inclination_column_for_parameters(
            self.grid_name, self.model_parameters, parameters, self.usecols
        )
        
        # Load flux using the configured wavelength column and calculated
        # inclination column.  The wavelength array is already cached from run0;
        # this read keeps loadtxt's two-column shape consistent across grids.
        # Optimized: Pre-scan for header length to avoid line-by-line comment checking in loadtxt
        skiprows = self._get_skiprows(file_path)
        _wl, flux = np.loadtxt(file_path, usecols=(self.usecols[0], inclination_column), skiprows=skiprows, unpack=True)
        flux = np.flip(flux)

        flux = flux[:len(self.wl_full)] # THIS CUT IS NEEDED, Random parameters appear in the grid space header leading to mismatching file lengths.
        flux = _maybe_smooth_flux(flux, self.smoothing)
        flux = _apply_flux_scale(self.wl_full, self.ind, flux, self.scale)

        # Header mirrors the selected emulator coordinates plus the raw .spec
        # flux column used for this spectrum.
        hdr = {'inclination_column': inclination_column}
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]

