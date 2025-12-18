import logging
import os
import warnings
from typing import Sequence, Optional, Union, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# from pyinstrument import Profiler


# ============================================================================
# PyTorch GPU Acceleration (Primary Backend)
# ============================================================================

# Try to import PyTorch for GPU-accelerated operations
try:
    import torch
    PYTORCH_AVAILABLE = True
    
    # ============================================================================
    # PRECISION CONTROL: Change in THREE files (keep synchronized!)
    # ============================================================================
    # float32: Uses ~40 GB for 6561-point grid (FITS IN H100)
    # float64: Uses ~80 GB for 6561-point grid (may OOM on H100)
    # 
    # EDIT these 3 lines (same value in all files):
    #   1. Starfish/emulator/emulator.py (this file, line 29) - TORCH_FLOAT64
    #   2. Starfish/emulator/kernels.py (line 11) - DTYPE
    #   3. Starfish/emulator/_utils.py (line 11) - DTYPE
    #
    TORCH_FLOAT64 = torch.float64  # <-- EDIT THIS (and kernels.py, _utils.py!)
    # ============================================================================

    # Import PCA regardless of GPU availability
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        try: 
            from cuml.decomposition import PCA
            GPU_PCA = True
            print('Using GPU for PCA')
        except ImportError: 
            from sklearn.decomposition import PCA, NMF, FastICA
            print('Warning: Using CPU for PCA')
            GPU_PCA = False
    else:
        # CPU fallback - always import sklearn PCA
        from sklearn.decomposition import PCA, NMF, FastICA
        GPU_PCA = False
        if PYTORCH_AVAILABLE:
            print('CUDA not available - using CPU for PCA')
            
    def _pytorch_log_likelihood_computation(v11, w_hat, device='cuda', dtype=None):
        """PyTorch log likelihood computation with GPU acceleration (float64)
        
        With 64-bit precision, this matches SciPy's numerical accuracy
        while providing ~3-4x speedup on GPU. Uses cached operations for MCMC.
        
        Args:
            v11: Covariance matrix (numpy array)
            w_hat: Weight vector (numpy array)
            device: 'cuda' or 'cpu'
            dtype: torch dtype (defaults to float64)
            
        Returns:
            log_likelihood: Scalar float matching SciPy output
        """
        if dtype is None:
            dtype = TORCH_FLOAT64
            
        # Convert to PyTorch tensors with float64 precision
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        v11_torch = torch.from_numpy(v11).to(device_obj, dtype)
        w_hat_torch = torch.from_numpy(w_hat).to(device_obj, dtype)
        
        if v11_torch.dim() == 3:
            # Block diagonal case (Batched)
            # v11_torch: (n_comp, M, M)
            # w_hat_torch: (n_comp * M,) -> reshape to (n_comp, M, 1)
            n_comp = v11_torch.shape[0]
            M = v11_torch.shape[1]
            w_hat_reshaped = w_hat_torch.view(n_comp, M, 1)
            
            L = torch.linalg.cholesky(v11_torch) # (n_comp, M, M)
            
            # logdet = 2 * sum(log(diag(L)))
            # L.diagonal(dim1=-2, dim2=-1) -> (n_comp, M)
            logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)))
            
            # Solve
            z = torch.linalg.solve_triangular(L, w_hat_reshaped, upper=False) # (n_comp, M, 1)
            solved = torch.linalg.solve_triangular(L.transpose(-2, -1), z, upper=True) # (n_comp, M, 1)
            
            # sqmah = w_hat^T @ solved
            # w_hat_reshaped is (n_comp, M, 1)
            # solved is (n_comp, M, 1)
            # We want sum over all components of w_hat_i^T @ solved_i
            # w_hat_reshaped.transpose(-2, -1) @ solved -> (n_comp, 1, 1)
            sqmah = torch.sum(torch.matmul(w_hat_reshaped.transpose(-2, -1), solved))
            
            log_likelihood = -(logdet + sqmah) / 2.0
            return float(log_likelihood.cpu())

        # Cholesky decomposition: v11 = L @ L.T
        L = torch.linalg.cholesky(v11_torch)
        
        # Log determinant: log|v11| = 2 * sum(log(diag(L)))
        logdet = 2.0 * torch.sum(torch.log(L.diagonal()))
        
        # Solve v11 @ x = w_hat via triangular solves
        # L @ L.T @ x = w_hat
        # First: L @ z = w_hat
        z = torch.linalg.solve_triangular(L, w_hat_torch.unsqueeze(1), upper=False).squeeze()
        # Second: L.T @ x = z
        solved = torch.linalg.solve_triangular(L.T, z.unsqueeze(1), upper=True).squeeze()
        
        # Mahalanobis distance: w_hat^T @ v11^{-1} @ w_hat
        sqmah = torch.dot(w_hat_torch, solved)
        
        # Log likelihood
        log_likelihood = -(logdet + sqmah) / 2.0
        
        return float(log_likelihood.cpu())
    
    def _pytorch_log_likelihood_computation_gpu_only(v11_gpu, w_hat_gpu):
        """PyTorch log likelihood computation that accepts and returns GPU tensors (NO TRANSFERS!)
        
        This version is optimized for training loops where v11 is already on GPU
        and we only need the scalar log-likelihood value.
        
        Eliminates GPU→CPU→GPU round-trip transfers during training!
        
        Args:
            v11_gpu: Covariance matrix (torch.Tensor on GPU, float64)
                     Can be 2D (dense) or 3D (stacked block-diagonal)
            w_hat_gpu: Weight vector (torch.Tensor on GPU, float64)
            
        Returns:
            log_likelihood: Scalar float (only transfers single number to CPU)
        """
        # Everything stays on GPU until final result
        
        # Check if we are using the block-diagonal optimization (3D tensor)
        if v11_gpu.dim() == 3:
            # v11_gpu shape: (n_components, M, M)
            # w_hat_gpu shape: (n_components * M) -> needs reshaping
            
            n_comp, M, _ = v11_gpu.shape
            w_hat_reshaped = w_hat_gpu.view(n_comp, M)
            
            logdet = 0.0
            sqmah = 0.0
            
            # Loop over components to save memory (avoid allocating full L matrix)
            # This reduces peak memory from (n_comp * M^2) to (M^2) for the Cholesky factor
            for i in range(n_comp):
                # Cholesky decomposition for single component
                L = torch.linalg.cholesky(v11_gpu[i]) # (M, M)
                
                # Log determinant
                logdet += 2.0 * torch.sum(torch.log(L.diagonal()))
                
                # Solve v11 @ x = w_hat
                # L @ z = w_hat
                z = torch.linalg.solve_triangular(L, w_hat_reshaped[i].unsqueeze(-1), upper=False)
                
                # L.T @ x = z
                solved = torch.linalg.solve_triangular(L.T, z, upper=True).squeeze(-1)
                
                # Mahalanobis distance
                sqmah += torch.dot(w_hat_reshaped[i], solved)
                
                # L is freed here, keeping peak memory low
            
            log_likelihood = -(logdet + sqmah) / 2.0
            
            return float(log_likelihood.cpu())
            
        else:
            # Original dense implementation (2D matrix)
            L = torch.linalg.cholesky(v11_gpu)
            logdet = 2.0 * torch.sum(torch.log(L.diagonal()))
            
            z = torch.linalg.solve_triangular(L, w_hat_gpu.unsqueeze(1), upper=False).squeeze()
            solved = torch.linalg.solve_triangular(L.T, z.unsqueeze(1), upper=True).squeeze()
            
            sqmah = torch.dot(w_hat_gpu, solved)
            log_likelihood = -(logdet + sqmah) / 2.0
            
            # Only transfer single scalar to CPU
            return float(log_likelihood.cpu())
    
except ImportError:
    PYTORCH_AVAILABLE = False
    TORCH_FLOAT64 = None

from Starfish.grid_tools import NPZInterface
from Starfish.grid_tools.utils import determine_chunk_log
from Starfish.utils import calculate_dv
from .kernels import batch_kernel, batch_kernel_cached, batch_kernel_auto, batch_kernel_pytorch_gpu_only, clear_kernel_cache, get_cache_size, rbf_kernel
from ._utils import get_phi_squared_optimized, get_phi_squared_pytorch, get_w_hat, PYTORCH_AVAILABLE as UTILS_PYTORCH_AVAILABLE
log = logging.getLogger(__name__)


def _create_optimal_interpolator(grid_points, factors, logger=None):
    """
    Create the fastest interpolator based on grid structure.
    
    Tries RegularGridInterpolator first (7M+ times faster for regular grids),
    falls back to LinearNDInterpolator for irregular grids.
    
    Parameters
    ----------
    grid_points : numpy.ndarray
        Shape (N, D) - N points in D-dimensional parameter space
    factors : numpy.ndarray
        Shape (N,) - Values to interpolate (norm factors)
    logger : logging.Logger, optional
        Logger for diagnostics
        
    Returns
    -------
    interpolator : callable
        Interpolation function (RegularGrid or LinearND)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check if grid is regular (Cartesian product)
    n_dims = grid_points.shape[1]
    unique_per_dim = [np.unique(grid_points[:, i]) for i in range(n_dims)]
    
    # Check if product of unique values equals total points
    expected_points = np.prod([len(u) for u in unique_per_dim])
    actual_points = grid_points.shape[0]
    
    if expected_points == actual_points:
        # Grid appears regular - try RegularGridInterpolator
        try:
            logger.info(f"Grid is regular ({' × '.join(map(str, [len(u) for u in unique_per_dim]))} = {actual_points} points)")
            logger.info("Using RegularGridInterpolator (FAST - instant creation)")
            
            # Sort data to match grid structure (lexicographic sort)
            sorted_indices = np.lexsort(grid_points.T[::-1])
            sorted_factors = factors[sorted_indices]
            
            # Reshape to grid
            grid_shape = [len(u) for u in unique_per_dim]
            factors_grid = sorted_factors.reshape(grid_shape)
            
            # Create RegularGridInterpolator
            interpolator = RegularGridInterpolator(
                unique_per_dim, 
                factors_grid, 
                method='linear', 
                bounds_error=False, 
                fill_value=None
            )
            
            logger.info("✓ RegularGridInterpolator created successfully")
            return interpolator
            
        except Exception as e:
            logger.warning(f"RegularGridInterpolator failed: {e}")
            logger.warning("Falling back to LinearNDInterpolator...")
    else:
        logger.info(f"Grid is irregular ({actual_points} points, expected {expected_points} for regular grid)")
        logger.info("Using LinearNDInterpolator (slower but supports irregular grids)")
    
    # Fallback: Use LinearNDInterpolator (works for any grid structure)
    logger.info("Creating LinearNDInterpolator (this may take a while for large grids)...")
    interpolator = LinearNDInterpolator(grid_points, factors, rescale=True)
    logger.info("✓ LinearNDInterpolator created successfully")
    
    return interpolator


class Emulator:
    """
    A Bayesian spectral emulator.

    This emulator offers an interface to spectral libraries that offers interpolation
    while providing a variance-covariance matrix that can be forward-propagated in
    likelihood calculations. For more details, see the appendix from Czekala et al.
    (2015).

    Parameters
    ----------
    grid_points : numpy.ndarray
        The parameter space from the library.
    param_names : array-like of str
        The names of each parameter from the grid
    wavelength : numpy.ndarray
        The wavelength of the library models
    weights : numpy.ndarray
        The PCA weights for the original grid points
    eigenspectra : numpy.ndarray
        The PCA components from the decomposition
    w_hat : numpy.ndarray
        The best-fit weights estimator
    flux_mean : numpy.ndarray
        The mean flux spectrum
    flux_std : numpy.ndarray
        The standard deviation flux spectrum
    lambda_xi : float, optional
        The scaling parameter for the augmented covariance calculations, default is 1
    variances : numpy.ndarray, optional
        The variance parameters for each of Gaussian process, default is array of 1s
    lengthscales : numpy.ndarray, optional
        The lengthscales for each Gaussian process, each row should have length equal
        to number of library parameters, default is arrays of 3 * the max grid
        separation for the grid_points
    name : str, optional
        If provided, will give a name to the emulator; useful for keeping track of
        filenames. Default is None.


    Attributes
    ----------
    params : dict
        The underlying hyperparameter dictionary
    """

    def __init__(
        self,
        grid_points: np.ndarray,
        param_names: Sequence[str],
        wavelength: np.ndarray,
        weights: np.ndarray,
        eigenspectra: np.ndarray,
        w_hat: np.ndarray,
        flux_mean: np.ndarray,
        flux_std: np.ndarray,
        factors: np.ndarray,
        lambda_xi: float = 1.0,
        variances: Optional[np.ndarray] = None,
        lengthscales: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        block_diagonal: bool = False,
        ):
        self.log = logging.getLogger(self.__class__.__name__)
        
        self.grid_points = grid_points
        self.param_names = param_names
        self.wl = wavelength
        self.weights = weights
        self.eigenspectra = eigenspectra
        self.flux_mean = flux_mean
        self.flux_std = flux_std
        self.factors = factors
        self.block_diagonal = block_diagonal
        
        # Create optimal interpolator (RegularGrid if possible, LinearND fallback)
        self.factor_interpolator = _create_optimal_interpolator(
            grid_points, factors, logger=self.log
        )

        self.dv = calculate_dv(wavelength)
        self.ncomps = eigenspectra.shape[0]

        self.hyperparams = {}
        self.name = name

        self.lambda_xi = lambda_xi

        self.variances = (
            variances if variances is not None else 1e4 * np.ones(self.ncomps)
        )

        unique = [sorted(np.unique(param_set)) for param_set in self.grid_points.T]
        self._grid_sep = np.array([np.diff(param).max() for param in unique])

        if lengthscales is None:
            lengthscales = np.tile(3 * self._grid_sep, (self.ncomps, 1))

        self.lengthscales = lengthscales

        # Determine the minimum and maximum bounds of the grid
        self.min_params = grid_points.min(axis=0)
        self.max_params = grid_points.max(axis=0)

        # Initialize covariance matrices
        eigenspectra_matrix = np.array(self.eigenspectra)
        M = self.grid_points.shape[0]
        
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.device('cuda')
            eig_torch = torch.from_numpy(eigenspectra_matrix).to(device, TORCH_FLOAT64)
            
            # Block diagonal optimization: (A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹
            dots = torch.matmul(eig_torch, eig_torch.T)
            dots_inv = torch.linalg.inv(dots)
            
            if self.block_diagonal:
                self.log.info("Using Block-Diagonal Optimization (Memory Efficient)")
                # Extract diagonal elements of dots_inv for block scaling
                # Assuming PCA orthogonality, off-diagonals should be negligible
                dots_inv_diag = torch.diagonal(dots_inv) # (n_comp,)
                self._dots_inv_diag_gpu = dots_inv_diag # Store for training updates
                
                # Create stacked iPhiPhi: (n_comp, M, M)
                # Each block is dots_inv_diag[i] * I_M
                eye_M = torch.eye(M, dtype=TORCH_FLOAT64, device=device)
                
                # Initialize v11_gpu directly with iPhiPhi/lambda_xi
                # This avoids allocating iPhiPhi_gpu separately
                v11_gpu = (dots_inv_diag.view(-1, 1, 1) * eye_M.unsqueeze(0)) / self.lambda_xi
                
                # Add kernel to v11_gpu in-place
                # This avoids allocating kernel_gpu separately
                batch_kernel_pytorch_gpu_only(
                    self.grid_points, self.grid_points, self.variances, self.lengthscales, 
                    device='cuda', return_stacked=True, out=v11_gpu, add_to_out=True
                )
            else:
                self.log.info("Using Full Dense Matrix (Standard)")
                eye_M = torch.eye(M, dtype=TORCH_FLOAT64, device=device)
                iPhiPhi_gpu = torch.kron(dots_inv.contiguous(), eye_M.contiguous())
                
                kernel_gpu = batch_kernel_pytorch_gpu_only(
                    self.grid_points, self.grid_points, self.variances, self.lengthscales, device='cuda'
                )
                v11_gpu = iPhiPhi_gpu / self.lambda_xi + kernel_gpu
                
                self._iPhiPhi_gpu = iPhiPhi_gpu
                self._v11_gpu = v11_gpu
                self._iPhiPhi = None
                self._v11 = None
            
            if self.block_diagonal:
                self._iPhiPhi_gpu = None # Not stored in block diagonal mode to save memory
                self._v11_gpu = v11_gpu
                self._iPhiPhi = None
                self._v11 = None
            else:
                self._dots_inv_diag_gpu = None
        else:
            self._dots_inv_diag_gpu = None
            self.log.info("GPU not available, using CPU")
            
            if self.block_diagonal:
                self.log.info("Using Block-Diagonal Optimization (CPU Mode)")
                # CPU Block Diagonal Implementation
                # 1. Compute diagonal of (Phi Phi^T)^-1
                dots = eigenspectra_matrix @ eigenspectra_matrix.T
                dots_inv = np.linalg.inv(dots)
                dots_inv_diag = np.diag(dots_inv) # (n_comp,)
                
                # 2. Create stacked iPhiPhi: (n_comp, M, M)
                # We use a list of matrices or 3D array. 3D array is better for batch operations.
                iPhiPhi_cpu = np.zeros((self.ncomps, M, M))
                eye_M = np.eye(M)
                for i in range(self.ncomps):
                    iPhiPhi_cpu[i] = dots_inv_diag[i] * eye_M
                
                # 3. Compute stacked kernels
                # We need a CPU version of batch_kernel that returns stacked
                # For now, we can just loop over components using the existing single-kernel function
                # or modify batch_kernel_auto. 
                # Let's construct it manually to be safe and explicit.
                kernel_cpu = np.zeros((self.ncomps, M, M))
                from .kernels import rbf_kernel
                for i in range(self.ncomps):
                    kernel_cpu[i] = rbf_kernel(
                        self.grid_points, 
                        self.grid_points, 
                        self.variances[i], 
                        self.lengthscales[i]
                    )
                
                v11_cpu = iPhiPhi_cpu / self.lambda_xi + kernel_cpu
                
                # Store as _v11 but note it is 3D
                self._iPhiPhi = iPhiPhi_cpu
                self._v11 = v11_cpu
                
            else:
                self.log.info("Using Full Dense Matrix (Standard CPU)")
                phi_squared = get_phi_squared_optimized(eigenspectra_matrix, M)
                self._iPhiPhi = np.linalg.inv(phi_squared)
                self._v11 = self._iPhiPhi / self.lambda_xi + batch_kernel_auto(
                    self.grid_points, self.grid_points, self.variances, self.lengthscales
                )
            
            self._iPhiPhi_gpu = None
            self._v11_gpu = None
        
        self.w_hat = w_hat
        self._trained = False
        
        # PyTorch acceleration settings
        self._use_pytorch = True
        self._pytorch_device = 'cuda' if (PYTORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        
        # GPU training mode
        self._gpu_training_mode = False
        self._w_hat_gpu = None
        self._grid_points_gpu = None
        self._variances_gpu = None
        self._lengthscales_gpu = None
        
        if not PYTORCH_AVAILABLE:
            self.log.info("PyTorch not available - using NumPy fallback")
        elif not torch.cuda.is_available():
            self.log.info("CUDA not available - using CPU")

    

    @property
    def lambda_xi(self) -> float:
        """
        float : The tuning hyperparameter

        :setter: Sets the value.
        """
        return np.exp(self.hyperparams["log_lambda_xi"])

    @lambda_xi.setter
    def lambda_xi(self, value: float):
        self.hyperparams["log_lambda_xi"] = np.log(value)

    @property
    def variances(self) -> np.ndarray:
        """
        numpy.ndarray : The variances for each Gaussian process kernel.

        :setter: Sets the variances given an array.
        """
        values = [
            val
            for key, val in self.hyperparams.items()
            if key.startswith("log_variance:")
        ]
        return np.exp(values)

    @variances.setter
    def variances(self, values: np.ndarray):
        for i, value in enumerate(values):
            self.hyperparams[f"log_variance:{i}"] = np.log(value)

    @property
    def lengthscales(self) -> np.ndarray:
        """
        numpy.ndarray : The lengthscales for each Gaussian process kernel.

        :setter: Sets the lengthscales given a 2d array
        """
        values = [
            val
            for key, val in self.hyperparams.items()
            if key.startswith("log_lengthscale:")
        ]
        return np.exp(values).reshape(self.ncomps, -1)

    @lengthscales.setter
    def lengthscales(self, values: np.ndarray):
        for i, value in enumerate(values):
            for j, ls in enumerate(value):
                self.hyperparams[f"log_lengthscale:{i}:{j}"] = np.log(ls)

    @property
    def iPhiPhi(self) -> np.ndarray:
        """
        numpy.ndarray : The inverse phi-squared matrix.
        
        Lazily transfers from GPU to CPU only when accessed.
        """
        if self._iPhiPhi is None and self._iPhiPhi_gpu is not None:
            # Lazy transfer from GPU
            self._iPhiPhi = self._iPhiPhi_gpu.cpu().numpy()
            self.log.debug("iPhiPhi transferred from GPU to CPU (lazy)")
        return self._iPhiPhi
    
    @iPhiPhi.setter
    def iPhiPhi(self, value):
        self._iPhiPhi = value
    
    @property
    def v11(self) -> np.ndarray:
        """
        numpy.ndarray : The v11 covariance matrix.
        
        Lazily transfers from GPU to CPU only when accessed.
        """
        if self._v11 is None and self._v11_gpu is not None:
            # Lazy transfer from GPU
            self._v11 = self._v11_gpu.cpu().numpy()
            self.log.debug("v11 transferred from GPU to CPU (lazy)")
        return self._v11
    
    @v11.setter
    def v11(self, value):
        self._v11 = value

    def __getitem__(self, key):
        return self.hyperparams[key]

    @classmethod
    def load(cls, filename: Union[str, os.PathLike], block_diagonal: Optional[bool] = None):
        """
        Load an emulator from an NPZ file

        Parameters
        ----------
        filename : str or path-like
        block_diagonal : bool, optional
            Override the block_diagonal setting from the file.
        """
        filename = os.path.expandvars(filename)
        data = np.load(filename, allow_pickle=True)
        
        grid_points = data["grid_points"]
        param_names = data["param_names"]
        wavelength = data["wavelength"]
        weights = data["weights"]
        eigenspectra = data["eigenspectra"]
        flux_mean = data["flux_mean"]
        flux_std = data["flux_std"]
        w_hat = data["w_hat"]
        factors = data["factors"]
        lambda_xi = float(data["lambda_xi"])
        variances = data["variances"]
        lengthscales = data["lengthscales"]
        trained = bool(data["trained"])
        
        # Load block_diagonal flag if present (backward compatibility)
        if block_diagonal is None:
            if "block_diagonal" in data:
                block_diagonal = bool(data["block_diagonal"])
            else:
                block_diagonal = False
        
        if "name" in data:
            name = str(data["name"])
        else:
            name = ".".join(filename.split(".")[:-1])

        emulator = cls(
            grid_points=grid_points,
            param_names=param_names,
            wavelength=wavelength,
            weights=weights,
            eigenspectra=eigenspectra,
            w_hat=w_hat,
            flux_mean=flux_mean,
            flux_std=flux_std,
            lambda_xi=lambda_xi,
            variances=variances,
            lengthscales=lengthscales,
            name=name,
            factors=factors,
            block_diagonal=block_diagonal
        )
        emulator._trained = trained
        return emulator

    def save(self, filename: Union[str, os.PathLike]):
        """
        Save the emulator to an NPZ file

        Parameters
        ----------
        filename : str or path-like
        """
        filename = os.path.expandvars(filename)
        
        # Prepare data for saving
        save_data = {
            "grid_points": self.grid_points,
            "param_names": self.param_names,
            "wavelength": self.wl,
            "wave_units": "Angstrom",
            "weights": self.weights,
            "eigenspectra": self.eigenspectra,
            "eigenspectra_units": "erg/cm^2/s/Angstrom",
            "flux_mean": self.flux_mean,
            "flux_std": self.flux_std,
            "w_hat": self.w_hat,
            "trained": self._trained,
            "factors": self.factors,
            "lambda_xi": self.lambda_xi,
            "variances": self.variances,
            "lengthscales": self.lengthscales,
            "block_diagonal": self.block_diagonal
        }
        
        if self.name is not None:
            save_data["name"] = self.name
            
        np.savez_compressed(filename, **save_data)
        self.log.info("Saved file at {}".format(filename))

    @classmethod
    def from_grid(cls, grid, block_diagonal=False, **pca_kwargs):
        """
        Create an Emulator using PCA decomposition from a GridInterface.

        Parameters
        ----------
        grid : :class:`GridInterface` or str
            The grid interface to decompose
        block_diagonal : bool, optional
            Whether to use block-diagonal approximation for covariance matrix. Default is False.
        pca_kwargs : dict, optional
            The keyword arguments to pass to PCA. By default, `n_components=0.99` and
            `svd_solver='full'`.

        See Also
        --------
        sklearn.decomposition.PCA
        """
        # Load grid if a string is given
        if isinstance(grid, str):
            grid = NPZInterface(grid)

        fluxes = np.array(list(grid.fluxes))
        # Normalize to an average of 1 to remove uninteresting correlation
        norm_factors = fluxes.mean(1)
        fluxes /= norm_factors[:, np.newaxis]
        # Center and whiten
        flux_mean = fluxes.mean(0)
        fluxes -= flux_mean
        flux_std = fluxes.std(0) 
        fluxes /= flux_std 

        # Perform PCA using sklearn
        default_pca_kwargs = dict(n_components=0.99, svd_solver="full") # PCA
        # default_pca_kwargs = dict(n_components=10, algorithm='parallel', whiten='unit-variance') # FastICA
        default_pca_kwargs.update(pca_kwargs)
        pca = PCA(**default_pca_kwargs)
        weights = pca.fit_transform(fluxes)
        print(weights.shape, 'weights shape')
        print(pca.components_.shape, 'eigenvector shape')
        eigenspectra = pca.components_

        exp_var = pca.explained_variance_ratio_.sum()

        # This is basically the mean square error of the reconstruction
        log.info(
            f"PCA fit {exp_var:.2f}% of the variance with {pca.n_components_:d} components."
        )
        print(f"PCA fit {exp_var:.2f}% of the variance with {pca.n_components_:d} components.")

        w_hat = get_w_hat(eigenspectra, fluxes)
        print('Completed get_w_hat')

        # from pyinstrument import Profiler
        # profiler = Profiler()
        # profiler.start()

        emulator = cls(
            grid_points=grid.grid_points,
            param_names=grid.param_names,
            wavelength=grid.wl,
            weights=weights,
            eigenspectra=eigenspectra,
            w_hat=w_hat,
            flux_mean=flux_mean,
            flux_std=flux_std,
            factors=norm_factors,
            block_diagonal=block_diagonal,
        )
        # profiler.stop()
        # profiler.print()

        print("Completed 'from grid'")
        return emulator

    def __call__(
        self,
        params: np.ndarray,
        full_cov: bool = True,
        reinterpret_batch: bool = False,
        use_gpu: bool = True,
        return_tensors: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict the weights and covariance matrix at the given parameters.

        Parameters
        ----------
        params : array_like
            The parameters to sample at. Should be consistent with the shapes of the
            original grid points.
        full_cov : bool, optional
            Return the full covariance or just the variance, default is True. This will
            have no effect of reinterpret_batch is true
        reinterpret_batch : bool, optional
            Will try and return a batch of output matrices if the input params are a
            list of params, default is False.
        use_gpu : bool, optional
            Use GPU acceleration if available. Default is True. Falls back to CPU if
            GPU is unavailable or if an error occurs.
        return_tensors : bool, optional
            If True and using GPU, returns PyTorch tensors on the GPU device.
            If False (default), returns numpy arrays on CPU.

        Returns
        -------
        mu : numpy.ndarray or torch.Tensor (len(params),)
        cov : numpy.ndarray or torch.Tensor (len(params), len(params))

        Raises
        ------
        ValueError
            If full_cov and reinterpret_batch are True
        ValueError
            If querying the emulator outside of its trained grid points
        """
        params = np.atleast_2d(params)

        if full_cov and reinterpret_batch:
            raise ValueError(
                "Cannot reshape the full_covariance matrix for many parameters."
            )

        if not self._trained:
            warnings.warn(
                "This emulator has not been trained and therefore is not reliable. call \
                    emulator.train() to train."
            )

        # If the pars is outside of the range of emulator values, raise a ModelError
        if np.any(params < self.min_params) or np.any(params > self.max_params):
            raise ValueError("Querying emulator outside of original parameter range.", params, self.min_params)
        
        # Try GPU path if requested and available
        if use_gpu and PYTORCH_AVAILABLE and torch.cuda.is_available() and self._v11_gpu is not None:
            try:
                return self._call_gpu(params, full_cov, reinterpret_batch, return_tensors=return_tensors)
            except Exception as e:
                self.log.warning(f"GPU __call__ failed: {e}. Falling back to CPU.")
        
        if return_tensors:
            self.log.warning("GPU not available or failed, but return_tensors=True requested. Returning numpy arrays instead.")
        
        # CPU fallback (original implementation)
        return self._call_cpu(params, full_cov, reinterpret_batch)
    
    def _call_cpu(
        self,
        params: np.ndarray,
        full_cov: bool = True,
        reinterpret_batch: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CPU implementation of Gaussian Process prediction (R&W eqn 2.18, 2.19)
        
        This is the original implementation that uses NumPy on CPU.
        """
        if self.block_diagonal:
            # Block-Diagonal CPU Implementation
            # v11 is (n_comp, M, M)
            
            # 1. Compute stacked kernels
            # We loop because we don't have a stacked CPU kernel function yet
            v12_list = []
            v22_list = []
            for i in range(self.ncomps):
                v12_list.append(rbf_kernel(self.grid_points, params, self.variances[i], self.lengthscales[i]))
                v22_list.append(rbf_kernel(params, params, self.variances[i], self.lengthscales[i]))
            
            v12_stacked = np.stack(v12_list) # (n_comp, M, n_query)
            v22_stacked = np.stack(v22_list) # (n_comp, n_query, n_query)
            v21_stacked = v12_stacked.transpose(0, 2, 1) # (n_comp, n_query, M)
            
            # w_hat is (n_comp * M,) -> reshape to (n_comp, M, 1)
            w_hat_stacked = self.w_hat.reshape(self.ncomps, -1)[..., np.newaxis]
            
            # Solve v11 @ alpha = w_hat
            # np.linalg.solve broadcasts over the first dimension (n_comp)
            alpha = np.linalg.solve(self._v11, w_hat_stacked) # (n_comp, M, 1)
            
            # mu = v21 @ alpha
            mu_stacked = v21_stacked @ alpha # (n_comp, n_query, 1)
            
            # Solve v11 @ X = v12
            v11_inv_v12 = np.linalg.solve(self._v11, v12_stacked) # (n_comp, M, n_query)
            
            # cov = v22 - v21 @ v11^{-1} @ v12
            cov_term = v21_stacked @ v11_inv_v12 # (n_comp, n_query, n_query)
            cov_stacked = v22_stacked - cov_term
            
            # Flatten mu
            mu = mu_stacked.flatten()
            
            if not full_cov:
                # Just return variances
                cov_diag = np.diagonal(cov_stacked, axis1=-2, axis2=-1) # (n_comp, n_query)
                cov = cov_diag.flatten()
            else:
                # Construct full block diagonal matrix
                from scipy.linalg import block_diag
                cov = block_diag(*cov_stacked)
                
        else:
            # Recalculate V12, V21, and V22.
            v12 = batch_kernel_auto(self.grid_points, params, self.variances, self.lengthscales)
            v22 = batch_kernel_auto(params, params, self.variances, self.lengthscales)
            v21 = v12.T

            # Recalculate the covariance
            mu = v21 @ np.linalg.solve(self.v11, self.w_hat)
            cov = v22 - v21 @ np.linalg.solve(self.v11, v12)
            
            if not full_cov:
                cov = np.diag(cov)
                
        if reinterpret_batch:
            mu = mu.reshape(-1, self.ncomps, order="F").squeeze()
            cov = cov.reshape(-1, self.ncomps, order="F").squeeze()
        return mu, cov
    
    def _call_gpu(
        self,
        params: np.ndarray,
        full_cov: bool = True,
        reinterpret_batch: bool = False,
        return_tensors: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        GPU-accelerated implementation of Gaussian Process prediction
        
        Performs the same computation as _call_cpu but on GPU for 10-50× speedup.
        Critical for MCMC where this is called hundreds of thousands of times.
        
        Uses cached GPU tensors (grid_points, w_hat, v11) to minimize transfers.
        """
        device = torch.device('cuda')
        
        # Ensure cached GPU tensors exist
        if self._grid_points_gpu is None:
            self._grid_points_gpu = torch.from_numpy(self.grid_points).to(device, TORCH_FLOAT64)
        if self._w_hat_gpu is None:
            self._w_hat_gpu = torch.from_numpy(self.w_hat).to(device, TORCH_FLOAT64)
        if self._variances_gpu is None:
            self._variances_gpu = torch.from_numpy(self.variances).to(device, TORCH_FLOAT64)
        if self._lengthscales_gpu is None:
            self._lengthscales_gpu = torch.from_numpy(self.lengthscales).to(device, TORCH_FLOAT64)
        
        # Transfer query parameters to GPU (tiny transfer: ~56 bytes for 7 params)
        params_gpu = torch.from_numpy(params).to(device, TORCH_FLOAT64)
        
        # Check if we are in block-diagonal mode
        is_block_diagonal = (self._v11_gpu.dim() == 3)
        
        if is_block_diagonal:
            # Block-diagonal / Stacked mode
            n_comp = self.ncomps
            M = self.grid_points.shape[0]
            n_query = params.shape[0]
            
            # Compute kernels in stacked mode
            v12_gpu = batch_kernel_pytorch_gpu_only(
                self._grid_points_gpu, 
                params_gpu, 
                self._variances_gpu, 
                self._lengthscales_gpu,
                device='cuda',
                return_stacked=True
            ) # (n_comp, M, n_query)
            
            v22_gpu = batch_kernel_pytorch_gpu_only(
                params_gpu, 
                params_gpu, 
                self._variances_gpu, 
                self._lengthscales_gpu,
                device='cuda',
                return_stacked=True
            ) # (n_comp, n_query, n_query)
            
            v21_gpu = v12_gpu.transpose(-2, -1) # (n_comp, n_query, M)
            
            # Reshape w_hat for batch solve
            w_hat_reshaped = self._w_hat_gpu.view(n_comp, M).unsqueeze(-1) # (n_comp, M, 1)
            
            # Solve v11 @ x = w_hat
            # v11_gpu is (n_comp, M, M)
            alpha = torch.linalg.solve(self._v11_gpu, w_hat_reshaped) # (n_comp, M, 1)
            
            # mu = v21 @ alpha
            mu_stacked = torch.matmul(v21_gpu, alpha) # (n_comp, n_query, 1)
            
            # Solve v11 @ X = v12
            v11_inv_v12 = torch.linalg.solve(self._v11_gpu, v12_gpu) # (n_comp, M, n_query)
            
            # cov = v22 - v21 @ v11^{-1} @ v12
            cov_term = torch.matmul(v21_gpu, v11_inv_v12) # (n_comp, n_query, n_query)
            cov_stacked = v22_gpu - cov_term # (n_comp, n_query, n_query)
            
            # Reshape results to match expected output format
            mu_gpu = mu_stacked.squeeze(-1).flatten() # (n_comp * n_query)
            
            if not full_cov:
                # Just return variances (diagonal of cov)
                cov_diag = cov_stacked.diagonal(dim1=-2, dim2=-1) # (n_comp, n_query)
                if return_tensors:
                    return mu_gpu, cov_diag.flatten()
                cov = cov_diag.flatten().cpu().numpy()
                mu = mu_gpu.cpu().numpy()
            else:
                # Construct full block diagonal matrix
                blocks = [cov_stacked[i] for i in range(n_comp)]
                cov_gpu = torch.block_diag(*blocks)
                
                if return_tensors:
                    return mu_gpu, cov_gpu
                
                mu = mu_gpu.cpu().numpy()
                cov = cov_gpu.cpu().numpy()
                
        else:
            # Original dense mode
            # Compute kernels on GPU (massively parallel - this is where the speedup comes from)
            v12_gpu = batch_kernel_pytorch_gpu_only(
                self._grid_points_gpu, 
                params_gpu, 
                self._variances_gpu, 
                self._lengthscales_gpu,
                device='cuda'
            )
            v22_gpu = batch_kernel_pytorch_gpu_only(
                params_gpu, 
                params_gpu, 
                self._variances_gpu, 
                self._lengthscales_gpu,
                device='cuda'
            )
            v21_gpu = v12_gpu.T
            
            # GP prediction equations on GPU (R&W 2.18, 2.19)
            # mu = v21 @ v11^{-1} @ w_hat
            mu_gpu = v21_gpu @ torch.linalg.solve(self._v11_gpu, self._w_hat_gpu)
            
            # cov = v22 - v21 @ v11^{-1} @ v12
            cov_gpu = v22_gpu - v21_gpu @ torch.linalg.solve(self._v11_gpu, v12_gpu)
            
            # Transfer results back to CPU (small transfer: ~3 KB for 18 components)
            if not full_cov:
                if return_tensors:
                    return mu_gpu, cov_gpu.diagonal()
                mu = mu_gpu.cpu().numpy()
                cov = cov_gpu.diagonal().cpu().numpy()
            else:
                if return_tensors:
                    return mu_gpu, cov_gpu
                mu = mu_gpu.cpu().numpy()
                cov = cov_gpu.cpu().numpy()
        
        if reinterpret_batch:
            if return_tensors:
                # PyTorch reshape
                mu = mu.reshape(-1, self.ncomps).T.reshape(-1) # Approximate logic, need to check order="F" equivalent
                # order="F" in numpy is column-major. PyTorch is row-major.
                # mu.reshape(-1, self.ncomps, order="F") means fill columns first.
                # In PyTorch: .T.reshape(...) might be needed or just avoid reinterpret_batch with tensors for now.
                # For MCMC we usually don't use reinterpret_batch.
                pass 
            else:
                mu = mu.reshape(-1, self.ncomps, order="F").squeeze()
                cov = cov.reshape(-1, self.ncomps, order="F").squeeze()
        
        return mu, cov

    @property
    def bulk_fluxes(self) -> np.ndarray:
        """
        numpy.ndarray: A vertically concatenated vector of the eigenspectra, flux_mean,
        and flux_std (in that order). Used for bulk processing with the emulator.
        """
        return np.vstack([self.eigenspectra, self.flux_mean, self.flux_std])

    def load_flux(
        self, params: Union[Sequence[float], np.ndarray], norm=False
    ) -> np.ndarray:
        """
        Interpolate a model given any parameters within the grid's parameter range
        using eigenspectrum reconstruction
        by sampling from the weight distributions.

        Parameters
        ----------
        params : array_like
            The parameters to sample at.

        Returns
        -------
        flux : numpy.ndarray
        """
        mu, cov = self(params, reinterpret_batch=False)
        weights = np.random.multivariate_normal(mu, cov).reshape(-1, self.ncomps)
        X = self.eigenspectra * self.flux_std
        flux = weights @ X + self.flux_mean
        if norm:
            flux *= self.norm_factor(params)[:, np.newaxis]
        return np.squeeze(flux)

    def norm_factor(self, params: Union[Sequence[float], np.ndarray]) -> float:
        """
        Return the scaling factor for the absolute flux units in flux-normalized spectra

        Parameters
        ----------
        params : array_like
            The parameters to interpolate at

        Returns
        -------
        factor: float
            The multiplicative factor to normalize a spectrum to the model's absolute flux units
        """
        _params = np.asarray(params)
        return self.factor_interpolator(_params)

    def determine_chunk_log(self, wavelength: Sequence[float], buffer: float = 50):
        """
        Possibly truncate the wavelength and eigenspectra in response to some new
        wavelengths

        Parameters
        ----------
        wavelength : array_like
            The new wavelengths to truncate to
        buffer : float, optional
            The wavelength buffer, in Angstrom. Default is 50

        See Also
        --------
        Starfish.grid_tools.utils.determine_chunk_log
        """
        wavelength = np.asarray(wavelength)

        # determine the indices
        wl_min = wavelength.min()
        wl_max = wavelength.max()

        wl_min -= buffer
        wl_max += buffer

        ind = determine_chunk_log(self.wl, wl_min, wl_max)
        trunc_wavelength = self.wl[ind]

        assert (trunc_wavelength.min() <= wl_min) and (
            trunc_wavelength.max() >= wl_max
        ), (
            f"Emulator chunking ({trunc_wavelength.min():.2f}, {trunc_wavelength.max():.2f}) didn't encapsulate "
            f"full wl range ({wl_min:.2f}, {wl_max:.2f})."
        )

        self.wl = trunc_wavelength
        self.eigenspectra = self.eigenspectra[:, ind]

    def train(self, **opt_kwargs):
        """
        Trains the emulator's hyperparameters using gradient descent. This is a light wrapper around `scipy.optimize.minimize`. If you are experiencing problems optimizing the emulator, consider implementing your own training loop, using this function as a template.

        Parameters
        ----------
        **opt_kwargs
            Any arguments to pass to the optimizer. By default, `method='Nelder-Mead'`
            and `maxiter=10000`. Use train_fast() for L-BFGS-B.

        See Also
        --------
        scipy.optimize.minimize

        """
        import time
        
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            self._gpu_training_mode = True
        
        self._training_start_time = time.time()
        self._iteration_count = 0
        self._best_loss = np.inf
        self._last_progress_time = time.time()
        
        print("Started training (Nelder-Mead)")
        print(f"Optimizing {len(self.get_param_vector())} hyperparameters")
        
        def nll(P):
            if np.any(~np.isfinite(P)):
                return np.inf
            self.set_param_vector(P)
            if np.any(self.lengthscales < 2 * self._grid_sep):
                return np.inf
            loss = -self.log_likelihood()
            self.log.debug(f"loss: {loss}")
            
            # Progress tracking
            self._iteration_count += 1
            current_time = time.time()
            
            # Update best loss
            if loss < self._best_loss:
                self._best_loss = loss
            
            # Show progress every 10 iterations or every 30 seconds
            if (self._iteration_count % 10 == 0) or (current_time - self._last_progress_time > 30):
                elapsed = current_time - self._training_start_time
                print(f"  Iter {self._iteration_count:4d} | Loss: {loss:8.2f} | Best: {self._best_loss:8.2f} | Time: {elapsed:6.1f}s")
                self._last_progress_time = current_time
            
            return loss

        # Do the optimization
        P0 = self.get_param_vector()
        initial_loss = nll(P0.copy())
        print(f"Initial loss: {initial_loss:.2f}")

        default_kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 10000, "disp": True}}
        default_kwargs.update(opt_kwargs)
        
        soln = minimize(nll, P0,  **default_kwargs)
        
        if self._gpu_training_mode:
            self._gpu_training_mode = False
            self.set_param_dict(self.get_param_dict())
        
        final_time = time.time()
        total_elapsed = final_time - self._training_start_time
        print(f"\nOptimization complete")
        print(f"Total time: {total_elapsed:.1f}s | Iterations: {self._iteration_count} | Final loss: {self._best_loss:.2f}")

        if not soln.success:
            self.log.warning("Optimization did not succeed.")
            self.log.info(soln.message)
        else:
            self.set_param_vector(soln.x)
            self._trained = True
            self.log.info("Finished optimizing emulator hyperparameters")
            self.log.info(self)
            
        # Clean up progress tracking variables
        delattr(self, '_training_start_time')
        delattr(self, '_iteration_count')
        delattr(self, '_best_loss')
        delattr(self, '_last_progress_time')
            
    # def train_fast(self, **opt_kwargs):
    #     """
    #     Fast training using L-BFGS-B
        
    #     Parameters
    #     ----------
    #     **opt_kwargs
    #         Any arguments to pass to the optimizer. By default, `method='L-BFGS-B'`
    #         with tighter tolerances.
    #     """
    #     import time
        
    #     if PYTORCH_AVAILABLE and torch.cuda.is_available():
    #         self._gpu_training_mode = True
        
    #     self._training_start_time = time.time()
    #     self._iteration_count = 0
    #     self._best_loss = np.inf
    #     self._last_progress_time = time.time()
        
    #     print("Started training (L-BFGS-B)")
    #     print(f"Optimizing {len(self.get_param_vector())} hyperparameters")
        
    #     def nll(P):
    #         if np.any(~np.isfinite(P)):
    #             return np.inf
    #         self.set_param_vector(P)
    #         if np.any(self.lengthscales < 2 * self._grid_sep):
    #             return np.inf
    #         loss = -self.log_likelihood()
    #         self.log.debug(f"loss: {loss}")
            
    #         # Progress tracking
    #         self._iteration_count += 1
    #         current_time = time.time()
            
    #         # Update best loss
    #         if loss < self._best_loss:
    #             self._best_loss = loss
            
    #         # Show progress every 10 iterations or every 30 seconds
    #         if (self._iteration_count % 10 == 0) or (current_time - self._last_progress_time > 30):
    #             elapsed = current_time - self._training_start_time
    #             print(f"  Iter {self._iteration_count:4d} | Loss: {loss:8.2f} | Best: {self._best_loss:8.2f} | Time: {elapsed:6.1f}s")
    #             self._last_progress_time = current_time
            
    #         return loss

    #     # Do the optimization
    #     P0 = self.get_param_vector()
    #     initial_loss = nll(P0.copy())
    #     print(f"Initial loss: {initial_loss:.2f}")

    #     default_kwargs = {
    #         "method": "L-BFGS-B", 
    #         "options": {
    #             "maxiter": 2000,        # More iterations
    #             "ftol": 1e-12,          # Tighter function tolerance  
    #             "gtol": 1e-10,          # Tighter gradient tolerance
    #             "eps": 1e-10,           # Smaller step for finite differences
    #             "disp": True            # Show progress
    #         }
    #     }
    #     default_kwargs.update(opt_kwargs)
        
    #     print(f"Starting L-BFGS-B optimization with {default_kwargs['options']['maxiter']} max iterations...")
    #     soln = minimize(nll, P0, **default_kwargs)
        
    #     if self._gpu_training_mode:
    #         self._gpu_training_mode = False
    #         self.set_param_dict(self.get_param_dict())
        
    #     final_time = time.time()
    #     total_elapsed = final_time - self._training_start_time
    #     print(f"\nOptimization complete")
    #     print(f"Total time: {total_elapsed:.1f}s | Iterations: {self._iteration_count} | Final loss: {self._best_loss:.2f}")

    #     if not soln.success:
    #         self.log.warning("Fast optimization did not succeed.")
    #         self.log.info(soln.message)
    #     else:
    #         self.set_param_vector(soln.x)
    #         self._trained = True
    #         self.log.info("Finished fast optimization")
    #         self.log.info(self)
            
    #     # Clean up progress tracking variables
    #     delattr(self, '_training_start_time')
    #     delattr(self, '_iteration_count')
    #     delattr(self, '_best_loss')
    #     delattr(self, '_last_progress_time')
            
    # def train_bfgs(self, **opt_kwargs):
    #     """
    #     Alternative training using BFGS
        
    #     Parameters
    #     ----------
    #     **opt_kwargs
    #         Any arguments to pass to the optimizer. By default, `method='BFGS'`.
    #     """
    #     import time
        
    #     self._training_start_time = time.time()
    #     self._iteration_count = 0
    #     self._best_loss = np.inf
    #     self._last_progress_time = time.time()
        
    #     print("Started training (BFGS)")
    #     print(f"Optimizing {len(self.get_param_vector())} hyperparameters")
        
    #     def nll(P):
    #         if np.any(~np.isfinite(P)):
    #             return np.inf
    #         self.set_param_vector(P)
    #         if np.any(self.lengthscales < 2 * self._grid_sep):
    #             return np.inf
    #         loss = -self.log_likelihood()
    #         self.log.debug(f"loss: {loss}")
            
    #         # Progress tracking
    #         self._iteration_count += 1
    #         current_time = time.time()
            
    #         # Update best loss
    #         if loss < self._best_loss:
    #             self._best_loss = loss
            
    #         # Show progress every 10 iterations or every 30 seconds
    #         if (self._iteration_count % 10 == 0) or (current_time - self._last_progress_time > 30):
    #             elapsed = current_time - self._training_start_time
    #             print(f"  Iter {self._iteration_count:4d} | Loss: {loss:8.2f} | Best: {self._best_loss:8.2f} | Time: {elapsed:6.1f}s")
    #             self._last_progress_time = current_time
            
    #         return loss

    #     P0 = self.get_param_vector()
    #     initial_loss = nll(P0.copy())
    #     print(f"Initial loss: {initial_loss:.2f}")

    #     default_kwargs = {
    #         "method": "BFGS", 
    #         "options": {
    #             "maxiter": 2000,
    #             "gtol": 1e-10,
    #             "eps": 1e-10,
    #             "disp": True
    #         }
    #     }
    #     default_kwargs.update(opt_kwargs)
        
    #     soln = minimize(nll, P0, **default_kwargs)
        
    #     final_time = time.time()
    #     total_elapsed = final_time - self._training_start_time
    #     print(f"\nOptimization complete")
    #     print(f"Total time: {total_elapsed:.1f}s | Iterations: {self._iteration_count} | Final loss: {self._best_loss:.2f}")

    #     if not soln.success:
    #         self.log.warning("BFGS optimization did not succeed.")
    #         self.log.info(soln.message)
    #     else:
    #         self.set_param_vector(soln.x)
    #         self._trained = True
    #         self.log.info("Finished BFGS optimization")
    #         self.log.info(self)
            
    #     # Clean up progress tracking variables
    #     delattr(self, '_training_start_time')
    #     delattr(self, '_iteration_count')
    #     delattr(self, '_best_loss')
    #     delattr(self, '_last_progress_time')
            
    def train_original(self, **opt_kwargs):
        """Original training method using Nelder-Mead"""
        print("Started training (Nelder-Mead)")
        def nll(P):
            if np.any(~np.isfinite(P)):
                return np.inf
            self.set_param_vector(P)
            if np.any(self.lengthscales < 2 * self._grid_sep):
                return np.inf
            loss = -self.log_likelihood()
            self.log.debug(f"loss: {loss}")
            return loss

        # Do the optimization
        P0 = self.get_param_vector()

        default_kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 10000}}
        default_kwargs.update(opt_kwargs)
        soln = minimize(nll, P0,  **default_kwargs)

        if not soln.success:
            self.log.warning("Optimization did not succeed.")
            self.log.info(soln.message)
        else:
            self.set_param_vector(soln.x)
            self._trained = True
            self.log.info("Finished optimizing emulator hyperparameters")
            self.log.info(self)

    def get_index(self, params: Sequence[float]) -> int:
        """
        Given a list of stellar parameters (corresponding to a grid point),
        deliver the index that corresponds to the
        entry in the fluxes, grid_points, and weights.

        Parameters
        ----------
        params : array_like
            The stellar parameters

        Returns
        -------
        index : int

        """
        params = np.atleast_2d(params)
        marks = np.abs(self.grid_points - np.expand_dims(params, 1)).sum(axis=-1)
        return marks.argmin(axis=1).squeeze()

    def get_param_dict(self) -> dict:
        """
        Gets the dictionary of parameters. This is the same as `Emulator.params`

        Returns
        -------
        dict
        """
        return self.hyperparams

    def set_param_dict(self, params: dict):
        """
        Sets the parameters with a dictionary

        Parameters
        ----------
        params : dict
            The new parameters.
        """
        for key, val in params.items():
            if key in self.hyperparams:
                self.hyperparams[key] = val

        if self._gpu_training_mode and PYTORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.device('cuda')
            
            if self._grid_points_gpu is None:
                self._grid_points_gpu = torch.from_numpy(self.grid_points).to(device, TORCH_FLOAT64)
            if self._w_hat_gpu is None:
                self._w_hat_gpu = torch.from_numpy(self.w_hat).to(device, TORCH_FLOAT64)
            
            self._variances_gpu = torch.from_numpy(self.variances).to(device, TORCH_FLOAT64)
            self._lengthscales_gpu = torch.from_numpy(self.lengthscales).to(device, TORCH_FLOAT64)
            
            if self.block_diagonal:
                # Optimized Block-Diagonal Update (Memory Efficient)
                # Reconstruct v11 directly without allocating full iPhiPhi
                
                M = self.grid_points.shape[0]
                eye_M = torch.eye(M, dtype=TORCH_FLOAT64, device=device)
                
                # v11 = (dots_inv_diag * I) / lambda + kernel
                # Initialize v11 with the diagonal part
                v11_gpu = (self._dots_inv_diag_gpu.view(-1, 1, 1) * eye_M.unsqueeze(0)) / self.lambda_xi
                
                # Add kernel in-place
                batch_kernel_pytorch_gpu_only(
                    self._grid_points_gpu, 
                    self._grid_points_gpu, 
                    self._variances_gpu, 
                    self._lengthscales_gpu,
                    device='cuda',
                    return_stacked=True,
                    out=v11_gpu,
                    add_to_out=True
                )
                self._v11_gpu = v11_gpu
                
            else:
                # Standard Dense Update
                if self._iPhiPhi_gpu is None:
                    self._iPhiPhi_gpu = torch.from_numpy(self.iPhiPhi).to(device, TORCH_FLOAT64)
                
                kernel_gpu = batch_kernel_pytorch_gpu_only(
                    self._grid_points_gpu, 
                    self._grid_points_gpu, 
                    self._variances_gpu, 
                    self._lengthscales_gpu,
                    device='cuda',
                    return_stacked=False
                )
                self._v11_gpu = self._iPhiPhi_gpu / self.lambda_xi + kernel_gpu
        else:
            if self.block_diagonal:
                # Block-Diagonal CPU Update
                M = self.grid_points.shape[0]
                kernel_cpu = np.zeros((self.ncomps, M, M))
                
                for i in range(self.ncomps):
                    kernel_cpu[i] = rbf_kernel(
                        self.grid_points, 
                        self.grid_points, 
                        self.variances[i], 
                        self.lengthscales[i]
                    )
                
                if self.iPhiPhi is None:
                    # Reconstruct diagonal scaling factors if iPhiPhi is missing (memory optimization)
                    dots = self.eigenspectra @ self.eigenspectra.T
                    dots_inv = np.linalg.inv(dots)
                    dots_inv_diag = np.diag(dots_inv)
                    
                    # Add (dots_inv_diag / lambda) * I to kernel
                    eye_M = np.eye(M)
                    for i in range(self.ncomps):
                        kernel_cpu[i] += (dots_inv_diag[i] / self.lambda_xi) * eye_M
                    self.v11 = kernel_cpu
                else:
                    self.v11 = self.iPhiPhi / self.lambda_xi + kernel_cpu
            else:
                self.v11 = self.iPhiPhi / self.lambda_xi + batch_kernel_auto(
                    self.grid_points, self.grid_points, self.variances, self.lengthscales
                )


    def get_param_vector(self) -> np.ndarray:
        """
        Get a vector of the current trainable parameters of the emulator

        Returns
        -------
        numpy.ndarray
        """
        values = list(self.get_param_dict().values())
        return np.array(values)

    def set_param_vector(self, params: np.ndarray):
        """
        Set the current trainable parameters given a vector. Must have the same form as
        :meth:`get_param_vector`

        Parameters
        ----------
        params : numpy.ndarray
        """
        parameters = self.get_param_dict()
        if len(params) != len(parameters):
            raise ValueError(
                "params must match length of parameters (get_param_vector())"
            )

        param_dict = dict(zip(self.get_param_dict().keys(), params))
        self.set_param_dict(param_dict)

    def log_likelihood(self) -> float:
        """
        Get the log likelihood of the emulator in its current state as calculated in
        the appendix of Czekala et al. (2015)

        Returns
        -------
        float

        Raises
        ------
        scipy.linalg.LinAlgError
            If the Cholesky factorization fails
        """
        if self._gpu_training_mode and PYTORCH_AVAILABLE and self._v11_gpu is not None:
            try:
                result = _pytorch_log_likelihood_computation_gpu_only(
                    self._v11_gpu,
                    self._w_hat_gpu
                )
                return result
            except Exception as e:
                self.log.warning(f"GPU training mode failed: {e}. Falling back.")
        
        if self._use_pytorch and PYTORCH_AVAILABLE:
            try:
                result = _pytorch_log_likelihood_computation(
                    self.v11, 
                    self.w_hat, 
                    device=self._pytorch_device,
                    dtype=TORCH_FLOAT64
                )
                return result
            except Exception as e:
                self.log.warning(f"PyTorch computation failed: {e}. Falling back to SciPy.")
        
        # CPU Fallback (SciPy/NumPy)
        if self.block_diagonal and self._v11 is not None and self._v11.ndim == 3:
             # Block-Diagonal CPU Implementation
            # v11 is (n_comp, M, M)
            # w_hat is (n_comp * M,) -> reshape to (n_comp, M)
            
            w_hat_reshaped = self.w_hat.reshape(self.ncomps, -1)
            
            logdet = 0.0
            sqmah = 0.0
            
            for i in range(self.ncomps):
                L, flag = cho_factor(self._v11[i])
                logdet += 2 * np.sum(np.log(L.diagonal()))
                
                # Solve for this component
                # w_hat_i is (M,)
                solved_i = cho_solve((L, flag), w_hat_reshaped[i])
                sqmah += np.dot(w_hat_reshaped[i], solved_i)
                
            return -(logdet + sqmah) / 2

        L, flag = cho_factor(self.v11)
        logdet = 2 * np.sum(np.log(L.diagonal()))
        solved = cho_solve((L, flag), self.w_hat)
        sqmah = np.dot(self.w_hat, solved)
        return -(logdet + sqmah) / 2

    def __repr__(self):
        output = "Emulator\n"
        output += "-" * 8 + "\n"
        if self.name is not None:
            output += f"Name: {self.name}\n"
        output += f"Trained: {self._trained}\n"
        output += f"lambda_xi: {self.lambda_xi:.3f}\n"
        output += "Variances:\n"
        output += "\n".join([f"\t{v:.2f}" for v in self.variances])
        output += "\nLengthscales:\n"
        output += "\n".join(
            [
                "\t[ " + " ".join([f"{l:.2f} " for l in ls]) + "]"
                for ls in self.lengthscales
            ]
        )
        output += f"\nLog Likelihood: {self.log_likelihood():.2f}\n"
        return output
