import logging
import os
import warnings
from typing import Sequence, Optional, Union, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar

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
            
    def _memory_efficient_log_likelihood_gpu(
        grid_points_gpu, variances_gpu, lengthscales_gpu,
        dots_inv_diag_gpu, lambda_xi, w_hat_gpu, n_comp,
        strict_weight_fit=False, kernel_type="rbf"
    ):
        """Memory-efficient log likelihood: builds and factors V11 one block at a time.
        
        Instead of materializing the full (n_comp, M, M) V11 tensor, this function
        computes each component's V11 block on-the-fly, Cholesky-factors it,
        accumulates the log-likelihood terms, and discards the block.
        
        Peak VRAM: ~2 × M² × 8 bytes (one kernel block + one Cholesky factor)
        instead of n_comp × M² × 8 bytes for the full V11 tensor.
        
        Mathematically identical to the standard path — no approximations.
        
        Args:
            grid_points_gpu: Grid points tensor (M, d) on GPU
            variances_gpu: Kernel variances (n_comp,) on GPU
            lengthscales_gpu: Kernel lengthscales (n_comp, d) on GPU
            dots_inv_diag_gpu: Diagonal of (Phi^T Phi)^{-1} (n_comp,) on GPU
            lambda_xi: Scalar float — truncation noise hyperparameter
            w_hat_gpu: Weight vector (n_comp * M,) on GPU
            n_comp: Number of PCA components
            strict_weight_fit: If True, skip iPhiPhi/lambda_xi term
            
        Returns:
            log_likelihood: Scalar float
        """
        M = grid_points_gpu.shape[0]
        device = grid_points_gpu.device
        w_hat_reshaped = w_hat_gpu.view(n_comp, M)
        
        logdet = 0.0
        sqmah = 0.0
        
        for i in range(n_comp):
            # --- Build V11 block for component i ---
            # Kernel: compute distance then apply kernel function
            X_scaled = grid_points_gpu / lengthscales_gpu[i]
            dist = torch.cdist(X_scaled, X_scaled, p=2.0)
            _apply_kernel_gpu_inplace(dist, variances_gpu[i], kernel_type)
            # dist is now K_i (M, M)
            
            # Add diagonal: iPhiPhi/lambda_xi + jitter
            # Variance-scaled jitter keeps condition number ~1e6
            jitter = max(1e-5, 1e-6 * variances_gpu[i].item()) if strict_weight_fit else 1e-5
            if strict_weight_fit:
                dist.diagonal().add_(jitter)
            else:
                dist.diagonal().add_(dots_inv_diag_gpu[i].item() / lambda_xi + jitter)
            
            # dist is now V11_i (M, M) — the complete block
            
            # --- Cholesky factor and accumulate ---
            L = torch.linalg.cholesky(dist)
            del dist  # Free the V11 block immediately
            
            logdet += 2.0 * torch.sum(torch.log(L.diagonal()))
            
            z = torch.linalg.solve_triangular(L, w_hat_reshaped[i].unsqueeze(-1), upper=False)
            solved = torch.linalg.solve_triangular(L.T, z, upper=True).squeeze(-1)
            del L  # Free Cholesky factor
            
            sqmah += torch.dot(w_hat_reshaped[i], solved)
            del solved, z
        
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
from .kernels import batch_kernel, batch_kernel_cached, batch_kernel_auto, batch_kernel_pytorch_gpu_only, clear_kernel_cache, get_cache_size, rbf_kernel, get_cpu_kernel_func, _apply_kernel_gpu_inplace
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
        strict_weight_fit: bool = False,
        per_component: bool = False,
        kernel: str = "rbf",
        _param_min: Optional[np.ndarray] = None,
        _param_range: Optional[np.ndarray] = None,
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
        self.strict_weight_fit = strict_weight_fit
        self.per_component = per_component
        self.kernel = kernel
        
        # Parameter normalization: scale grid_points to [0,1] for GP kernel operations
        if _param_min is not None and _param_range is not None:
            # Loaded from file — normalization params already known
            self._param_min = _param_min
            self._param_range = _param_range
        else:
            # Fresh construction — compute normalization from grid
            self._param_min = grid_points.min(axis=0)
            self._param_range = grid_points.max(axis=0) - self._param_min
            self._param_range[self._param_range == 0] = 1.0
        self._grid_points_norm = (grid_points - self._param_min) / self._param_range
        
        # Create optimal interpolator (RegularGrid if possible, LinearND fallback)
        # Uses ORIGINAL-space grid_points for physical-space interpolation
        self.factor_interpolator = _create_optimal_interpolator(
            grid_points, factors, logger=self.log
        )

        self.dv = calculate_dv(wavelength)
        self.ncomps = eigenspectra.shape[0]
        self.pca_explained_variance = None  # Set by from_grid(), persisted in save/load

        self.hyperparams = {}
        self.name = name

        self.lambda_xi = lambda_xi

        self.variances = (
            variances if variances is not None else 1e4 * np.ones(self.ncomps)
        )

        unique = [sorted(np.unique(param_set)) for param_set in self._grid_points_norm.T]
        self._grid_sep = np.array([np.diff(param).max() for param in unique])

        if lengthscales is None:
            lengthscales = np.tile(2 * self._grid_sep, (self.ncomps, 1))

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
                
                # Check if full V11 tensor would fit in currently FREE GPU VRAM.
                # Use free memory, not total, so other processes' allocations are respected.
                v11_bytes = self.ncomps * M * M * 8  # float64
                gpu_vram_free, gpu_vram_total = torch.cuda.mem_get_info(0)
                # Need ~3× V11 for construction + kernel intermediates
                if v11_bytes * 3 > gpu_vram_free:
                    # V11 won't fit — defer construction to memory-efficient training
                    self.log.info(f"V11 ({v11_bytes / 1024**3:.1f} GB) exceeds free GPU VRAM "
                                  f"({gpu_vram_free / 1024**3:.1f} GB free of {gpu_vram_total / 1024**3:.1f} GB). "
                                  f"Deferring V11 construction.")
                    v11_gpu = None  # Will be built on-the-fly during training
                else:
                    # V11 fits — build it now
                    eye_M = torch.eye(M, dtype=TORCH_FLOAT64, device=device)
                    
                    if self.strict_weight_fit:
                        # Variance-scaled jitter keeps condition number ~1e6
                        variances_t = torch.from_numpy(self.variances).to(device, TORCH_FLOAT64)
                        jitter_t = torch.clamp(1e-6 * variances_t, min=1e-5)
                        v11_gpu = (jitter_t.view(-1, 1, 1) * eye_M.unsqueeze(0)).clone()
                    else:
                        v11_gpu = (dots_inv_diag.view(-1, 1, 1) * eye_M.unsqueeze(0)) / self.lambda_xi
                        v11_gpu += 1e-5 * eye_M.unsqueeze(0)
                    
                    # Add kernel to v11_gpu in-place
                    batch_kernel_pytorch_gpu_only(
                        self._grid_points_norm, self._grid_points_norm, self.variances, self.lengthscales, 
                        device='cuda', return_stacked=True, out=v11_gpu, add_to_out=True, kernel_type=self.kernel
                    )
            else:
                self.log.info("Using Full Dense Matrix (Standard)")
                # Guard against OOM when other processes are using GPU memory.
                v11_bytes = self.ncomps * M * M * 8  # float64 (dense is ncomps*M × ncomps*M)
                gpu_vram_free, gpu_vram_total = torch.cuda.mem_get_info(0)
                if v11_bytes * 3 > gpu_vram_free:
                    self.log.warning(
                        f"Full V11 ({v11_bytes / 1024**3:.1f} GB est.) exceeds free GPU VRAM "
                        f"({gpu_vram_free / 1024**3:.1f} GB free of {gpu_vram_total / 1024**3:.1f} GB). "
                        f"Falling back to CPU for V11 construction."
                    )
                    raise torch.cuda.OutOfMemoryError(
                        f"Insufficient free GPU VRAM for full V11 "
                        f"({v11_bytes / 1024**3:.1f} GB needed, {gpu_vram_free / 1024**3:.1f} GB free)."
                    )
                eye_M = torch.eye(M, dtype=TORCH_FLOAT64, device=device)
                iPhiPhi_gpu = torch.kron(dots_inv.contiguous(), eye_M.contiguous())
                
                kernel_gpu = batch_kernel_pytorch_gpu_only(
                    self._grid_points_norm, self._grid_points_norm, self.variances, self.lengthscales, device='cuda', kernel_type=self.kernel
                )
                
                if self.strict_weight_fit:
                    v11_gpu = kernel_gpu + (1e-5 * torch.eye(M * self.ncomps, device=device, dtype=TORCH_FLOAT64))
                else:
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
                _kfunc = get_cpu_kernel_func(self.kernel)
                for i in range(self.ncomps):
                    kernel_cpu[i] = _kfunc(
                        self._grid_points_norm, 
                        self._grid_points_norm, 
                        self.variances[i], 
                        self.lengthscales[i]
                    )
                
                if self.strict_weight_fit:
                    # Variance-scaled jitter keeps condition number ~1e6
                    eye_M_arr = np.eye(M)
                    jitter_arr = np.maximum(1e-6 * self.variances, 1e-5)  # (ncomps,)
                    v11_cpu = kernel_cpu + jitter_arr[:, None, None] * eye_M_arr
                else:
                    v11_cpu = iPhiPhi_cpu / self.lambda_xi + kernel_cpu
                
                # Store as _v11 but note it is 3D
                self._iPhiPhi = iPhiPhi_cpu
                self._v11 = v11_cpu
                
            else:
                self.log.info("Using Full Dense Matrix (Standard CPU)")
                phi_squared = get_phi_squared_optimized(eigenspectra_matrix, M)
                self._iPhiPhi = np.linalg.inv(phi_squared)
                kernel_auto = batch_kernel_auto(
                    self._grid_points_norm, self._grid_points_norm, self.variances, self.lengthscales, kernel_type=self.kernel
                )
                if self.strict_weight_fit:
                    self._v11 = kernel_auto + (1e-5 * np.eye(M * self.ncomps))
                else:
                    self._v11 = self._iPhiPhi / self.lambda_xi + kernel_auto
            
            self._iPhiPhi_gpu = None
            self._v11_gpu = None
        
        self.w_hat = w_hat
        self._trained = False
        
        # GPU training mode
        self._gpu_training_mode = False
        self._memory_efficient_training = False  # Stream-and-discard: build V11 blocks on-the-fly
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
        
        # Load strict_weight_fit flag if present (backward compatibility)
        if "strict_weight_fit" in data:
            strict_weight_fit = bool(data["strict_weight_fit"])
        else:
            strict_weight_fit = False
        
        # Load per_component flag if present (backward compatibility)
        per_component = bool(data["per_component"]) if "per_component" in data else False
        
        # Load kernel type if present (backward compatibility: default to "rbf")
        kernel = str(data["kernel"]) if "kernel" in data else "rbf"
        
        if "name" in data:
            name = str(data["name"])
        else:
            name = ".".join(filename.split(".")[:-1])

        # Load normalization params if present (backward compatibility)
        _param_min = data["_param_min"] if "_param_min" in data else None
        _param_range = data["_param_range"] if "_param_range" in data else None

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
            block_diagonal=block_diagonal,
            strict_weight_fit=strict_weight_fit,
            per_component=per_component,
            kernel=kernel,
            _param_min=_param_min,
            _param_range=_param_range,
        )
        emulator._trained = trained
        
        # Load loss history if available
        if "loss_history" in data:
            emulator.loss_history = list(data["loss_history"])

        # Load true PCA explained variance if available
        if "pca_explained_variance" in data:
            emulator.pca_explained_variance = float(data["pca_explained_variance"])
            
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
            "block_diagonal": self.block_diagonal,
            "strict_weight_fit": self.strict_weight_fit,
            "per_component": self.per_component,
            "kernel": self.kernel,
            "_param_min": self._param_min,
            "_param_range": self._param_range,
        }
        
        if self.name is not None:
            save_data["name"] = self.name
            
        # Save loss history if available (for plotting later)
        if hasattr(self, 'loss_history'):
            save_data["loss_history"] = self.loss_history

        # Save true PCA explained variance if available
        if self.pca_explained_variance is not None:
            save_data["pca_explained_variance"] = self.pca_explained_variance
            
        np.savez_compressed(filename, **save_data)
        self.log.info("Saved file at {}".format(filename))


    @classmethod
    def test_pca(cls, grid, **pca_kwargs):
        """
        Perform a test PCA reconstruction to check explained variance.
        
        Parameters
        ----------
        grid : str or GridInterface
            Path to the grid NPZ file or GridInterface object.
        **pca_kwargs : dict
            Arguments passed to PCA (e.g. n_components).
            
        Returns
        -------
        explained_variance : float
            The sum of explained variance ratios.
        n_components : int
            The number of components used.
        """
        # Load grid if a string is given
        if isinstance(grid, str):
            grid = NPZInterface(grid)

        fluxes = np.array(list(grid.fluxes))
        # Normalize to an average of 1
        norm_factors = fluxes.mean(1)
        fluxes /= norm_factors[:, np.newaxis]
        # Center and whiten
        flux_mean = fluxes.mean(0)
        fluxes -= flux_mean
        flux_std = fluxes.std(0) 
        # Avoid division by zero
        flux_std[flux_std == 0] = 1.0
        fluxes /= flux_std 
        fluxes = np.nan_to_num(fluxes)

        # Perform PCA
        default_pca_kwargs = dict(n_components=0.99, svd_solver="full") 
        default_pca_kwargs.update(pca_kwargs)
        pca = PCA(**default_pca_kwargs)
        pca.fit(fluxes)

        exp_var = pca.explained_variance_ratio_.sum()
        
        return exp_var, pca.n_components_

    @classmethod
    def from_grid(cls, grid, block_diagonal=False, strict_weight_fit=False, per_component=False, kernel="rbf", **pca_kwargs):
        """
        Create an Emulator using PCA decomposition from a GridInterface.

        Parameters
        ----------
        grid : :class:`GridInterface` or str
            The grid interface to decompose
        block_diagonal : bool, optional
            Whether to use block-diagonal approximation for covariance matrix. Default is False.
        strict_weight_fit : bool, optional
            When True, bypasses the lambda_xi truncation penalty and forces the GP
            to fit the PCA weights directly. Default is False.
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
        flux_std[flux_std == 0] = 1.0
        fluxes /= flux_std
        fluxes = np.nan_to_num(fluxes)

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
            strict_weight_fit=strict_weight_fit,
            per_component=per_component,
            kernel=kernel,
        )
        emulator.pca_explained_variance = float(exp_var)
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
        
        # Normalize query params to [0,1] to match the normalized kernel space
        params = (params - self._param_min) / self._param_range
        
        # Try GPU path if requested and available
        if use_gpu and PYTORCH_AVAILABLE and torch.cuda.is_available() and self._v11_gpu is not None:
            try:
                return self._call_gpu(params, full_cov, reinterpret_batch, return_tensors=return_tensors)
            except Exception as e:
                self.log.warning(f"GPU __call__ failed: {e}. Falling back to CPU.")
        
        # Memory-efficient GPU inference: V11 not materialized but GPU is available
        if use_gpu and PYTORCH_AVAILABLE and torch.cuda.is_available() and self._v11_gpu is None and self.block_diagonal:
            try:
                return self._call_gpu_memory_efficient(params, full_cov, reinterpret_batch, return_tensors=return_tensors)
            except Exception as e:
                self.log.warning(f"Memory-efficient GPU inference failed: {e}. Falling back to CPU.")
        
        if return_tensors:
            self.log.warning("GPU not available or failed, but return_tensors=True requested. Returning numpy arrays instead.")
        
        # CPU fallback (original implementation)
        # Check that V11 exists on CPU; if not (large deferred grid), build it first
        if self._v11 is None and self._v11_gpu is None and self.block_diagonal:
            M = self.grid_points.shape[0]
            estimated_gb = self.ncomps * M * M * 8 / (1024**3)
            self.log.warning(
                f"CPU fallback: building V11 on CPU ({estimated_gb:.1f} GB for "
                f"{self.ncomps} components × {M}×{M}). This may be very slow or OOM."
            )
            warnings.warn(
                f"Building V11 on CPU ({estimated_gb:.1f} GB). "
                f"This is a fallback — GPU memory-efficient path should be preferred.",
                RuntimeWarning, stacklevel=2
            )
            kernel_cpu = np.zeros((self.ncomps, M, M))
            _kfunc = get_cpu_kernel_func(self.kernel)
            for i in range(self.ncomps):
                kernel_cpu[i] = _kfunc(self._grid_points_norm, self._grid_points_norm, self.variances[i], self.lengthscales[i])
            if self.strict_weight_fit:
                self._v11 = kernel_cpu + 1e-5 * np.eye(M)
            else:
                dots = self.eigenspectra @ self.eigenspectra.T
                dots_inv_diag = np.diag(np.linalg.inv(dots))
                eye_M = np.eye(M)
                for i in range(self.ncomps):
                    kernel_cpu[i] += ((dots_inv_diag[i] / self.lambda_xi) + 1e-5) * eye_M
                self._v11 = kernel_cpu
        
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
            _kfunc = get_cpu_kernel_func(self.kernel)
            v12_list = []
            v22_list = []
            for i in range(self.ncomps):
                v12_list.append(_kfunc(self._grid_points_norm, params, self.variances[i], self.lengthscales[i]))
                v22_list.append(_kfunc(params, params, self.variances[i], self.lengthscales[i]))
            
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
            v12 = batch_kernel_auto(self._grid_points_norm, params, self.variances, self.lengthscales, kernel_type=self.kernel)
            v22 = batch_kernel_auto(params, params, self.variances, self.lengthscales, kernel_type=self.kernel)
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
    
    def _call_gpu_memory_efficient(
        self,
        params: np.ndarray,
        full_cov: bool = True,
        reinterpret_batch: bool = False,
        return_tensors: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Memory-efficient GPU inference for large grids where V11 won't fit in VRAM.
        
        On the FIRST call, builds V11 blocks one component at a time, Cholesky-
        factors each, and caches L blocks + alpha. Peak construction VRAM:
        cached_L_so_far + 2 × M² × 8 bytes (V11 temp + new L block).
        
        On SUBSEQUENT calls, reuses cached L and alpha for fast O(M²) solves.
        This makes MCMC (32K+ calls) feasible: ~20s first call, <1s thereafter.
        
        Mathematically identical to _call_gpu — no approximations.
        """
        device = torch.device('cuda')
        
        # Ensure cached GPU tensors exist
        if self._grid_points_gpu is None:
            self._grid_points_gpu = torch.from_numpy(self._grid_points_norm).to(device, TORCH_FLOAT64)
        if self._w_hat_gpu is None:
            self._w_hat_gpu = torch.from_numpy(self.w_hat).to(device, TORCH_FLOAT64)
        if self._variances_gpu is None:
            self._variances_gpu = torch.from_numpy(self.variances).to(device, TORCH_FLOAT64)
        if self._lengthscales_gpu is None:
            self._lengthscales_gpu = torch.from_numpy(self.lengthscales).to(device, TORCH_FLOAT64)
        if self._dots_inv_diag_gpu is None:
            dots = torch.from_numpy(self.eigenspectra @ self.eigenspectra.T).to(device, TORCH_FLOAT64)
            self._dots_inv_diag_gpu = torch.diag(torch.linalg.inv(dots))
        
        n_comp = self.ncomps
        M = self.grid_points.shape[0]
        
        # --- Build and cache Cholesky factors L on first call ---
        if not hasattr(self, '_mem_eff_L_blocks') or self._mem_eff_L_blocks is None:
            import time as _time
            _t0 = _time.time()
            self.log.info(f"Building Cholesky cache for {n_comp} components (M={M})...")
            print(f"Building Cholesky cache ({n_comp} components, M={M})...")
            
            w_hat_reshaped = self._w_hat_gpu.view(n_comp, M).unsqueeze(-1)  # (n_comp, M, 1)
            L_blocks = []
            alpha_blocks = []
            
            for i in range(n_comp):
                # Build V11 block for component i
                X_scaled = self._grid_points_gpu / self._lengthscales_gpu[i]
                dist = torch.cdist(X_scaled, X_scaled, p=2.0)
                _apply_kernel_gpu_inplace(dist, self._variances_gpu[i], self.kernel)
                
                # Add diagonal (match training: dots_inv_diag/lambda_xi + 1e-5 jitter)
                if self.strict_weight_fit:
                    dist.diagonal().add_(1e-5)
                else:
                    dist.diagonal().add_(self._dots_inv_diag_gpu[i].item() / self.lambda_xi + 1e-5)
                
                # Cholesky factor — keep L, discard V11
                L_i = torch.linalg.cholesky(dist)
                del dist
                
                # Precompute alpha_i = V11^{-1} @ w_hat_i
                alpha_i = torch.cholesky_solve(w_hat_reshaped[i], L_i)  # (M, 1)
                
                L_blocks.append(L_i)
                alpha_blocks.append(alpha_i)
                torch.cuda.empty_cache()
            
            self._mem_eff_L_blocks = L_blocks        # list of (M, M) tensors
            self._mem_eff_alpha_blocks = alpha_blocks  # list of (M, 1) tensors
            _elapsed = _time.time() - _t0
            L_total_gb = n_comp * M * M * 8 / 1024**3
            print(f"Cholesky cache ready ({_elapsed:.1f}s, {L_total_gb:.1f} GB)")
        
        # --- Fast inference using cached L and alpha ---
        params_gpu = torch.from_numpy(params).to(device, TORCH_FLOAT64)
        n_query = params.shape[0]
        
        # Compute v12, v22 kernel blocks (small: M×n_query and n_query×n_query)
        v12_gpu = batch_kernel_pytorch_gpu_only(
            self._grid_points_gpu, params_gpu, 
            self._variances_gpu, self._lengthscales_gpu,
            device='cuda', return_stacked=True, kernel_type=self.kernel
        )  # (n_comp, M, n_query)
        
        v22_gpu = batch_kernel_pytorch_gpu_only(
            params_gpu, params_gpu, 
            self._variances_gpu, self._lengthscales_gpu,
            device='cuda', return_stacked=True, kernel_type=self.kernel
        )  # (n_comp, n_query, n_query)
        
        v21_gpu = v12_gpu.transpose(-2, -1)  # (n_comp, n_query, M)
        
        mu_list = []
        cov_list = []
        
        for i in range(n_comp):
            # mu_i = v21_i @ alpha_i  (cached alpha, O(M) matmul)
            mu_i = torch.matmul(v21_gpu[i], self._mem_eff_alpha_blocks[i])  # (n_query, 1)
            mu_list.append(mu_i.squeeze(-1))
            
            # cov_i = v22_i - v21_i @ L_i^{-T} L_i^{-1} @ v12_i  (Cholesky solve, O(M²))
            v11_inv_v12_i = torch.cholesky_solve(v12_gpu[i], self._mem_eff_L_blocks[i])
            cov_i = v22_gpu[i] - torch.matmul(v21_gpu[i], v11_inv_v12_i)
            cov_list.append(cov_i)
        
        # Stack results
        mu_gpu = torch.cat(mu_list)  # (n_comp * n_query,)
        
        if not full_cov:
            cov_diag = torch.stack([c.diagonal() for c in cov_list])  # (n_comp, n_query)
            if return_tensors:
                return mu_gpu, cov_diag.flatten()
            mu = mu_gpu.cpu().numpy()
            cov = cov_diag.flatten().cpu().numpy()
        else:
            cov_gpu = torch.block_diag(*cov_list)
            if return_tensors:
                return mu_gpu, cov_gpu
            mu = mu_gpu.cpu().numpy()
            cov = cov_gpu.cpu().numpy()
        
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
            self._grid_points_gpu = torch.from_numpy(self._grid_points_norm).to(device, TORCH_FLOAT64)
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
                return_stacked=True,
                kernel_type=self.kernel
            ) # (n_comp, M, n_query)
            
            v22_gpu = batch_kernel_pytorch_gpu_only(
                params_gpu, 
                params_gpu, 
                self._variances_gpu, 
                self._lengthscales_gpu,
                device='cuda',
                return_stacked=True,
                kernel_type=self.kernel
            ) # (n_comp, n_query, n_query)
            
            v21_gpu = v12_gpu.transpose(-2, -1) # (n_comp, n_query, M)
            
            # Reshape w_hat for batch solve
            w_hat_reshaped = self._w_hat_gpu.view(n_comp, M).unsqueeze(-1) # (n_comp, M, 1)
            
            # --- Optimization: Cached Cholesky Decomposition & Precomputed Alpha ---
            # Using Cholesky (O(N^3) once) + Substitution (O(N^2)) is much faster than Solve (O(N^3))
            # avoiding MAGMA batched warnings by looping explicitly during heavy ops.
            
            # Check if we have a valid cached Cholesky factor L
            # We track the 'source' v11 tensor object to invalidate cache if v11 changes (e.g. re-training)
            current_v11_id = id(self._v11_gpu)
            cached_v11_id = getattr(self, '_L_gpu_source_id', None)
            
            if cached_v11_id != current_v11_id:
                # Cache invalid or missing, recompute L and alpha
                L_list = []
                for i in range(n_comp):
                    # Compute Cholesky decomposition: v11 = L @ L.T
                    # Loop avoids "batched routines" warning for large matrices
                    L_list.append(torch.linalg.cholesky(self._v11_gpu[i]))
                self._L_gpu = torch.stack(L_list)
                self._L_gpu_source_id = current_v11_id
                
                # Compute alpha = v11^-1 @ w_hat using Cholesky
                # alpha = cholesky_solve(w_hat, L)
                alpha_list = []
                for i in range(n_comp):
                    alpha_list.append(torch.cholesky_solve(w_hat_reshaped[i], self._L_gpu[i]))
                self._alpha_gpu = torch.stack(alpha_list)
                
            # Retrieve cached alpha
            alpha = self._alpha_gpu

            # mu = v21 @ alpha
            mu_stacked = torch.matmul(v21_gpu, alpha) # (n_comp, n_query, 1)
            
            # Solve v11 @ X = v12
            # v11_inv_v12 = v11^-1 @ v12 => cholesky_solve(v12, L)
            # Manual batching with Cholesky solve (O(N^2)) is very fast even in Python loop
            v11_inv_v12_list = []
            for i in range(n_comp):
                v11_inv_v12_list.append(torch.cholesky_solve(v12_gpu[i], self._L_gpu[i]))
            v11_inv_v12 = torch.stack(v11_inv_v12_list)
            
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
                device='cuda',
                kernel_type=self.kernel
            )
            v22_gpu = batch_kernel_pytorch_gpu_only(
                params_gpu, 
                params_gpu, 
                self._variances_gpu, 
                self._lengthscales_gpu,
                device='cuda',
                kernel_type=self.kernel
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

    # ==================================================================
    # Benchmark / Leave-One-Out Cross-Validation
    # ==================================================================

    def loo_cv(self, grid=None):
        """
        Analytical leave-one-out (LOO) cross-validation for the trained GP.

        For each PCA component *k* and training point *i*, the LOO predictive
        mean and variance are computed from the inverse covariance matrix
        without retraining (Rasmussen & Williams §5.4.2):

            mu_{-i,k}    = w_{i,k} - (K_k^{-1} w_k)_i / (K_k^{-1})_{ii}
            sigma2_{-i,k} = 1 / (K_k^{-1})_{ii}

        If *grid* is provided (path to grid NPZ or a GridInterface) the LOO
        weight predictions are projected back to flux space and compared with
        the original spectra, giving a per-spectrum RMSE.

        Uses GPU acceleration (PyTorch) when available; falls back to NumPy
        otherwise.

        Parameters
        ----------
        grid : str, GridInterface, or None
            If given, also computes flux-space reconstruction metrics.

        Returns
        -------
        results : dict with keys:
            'loo_mu'          : (ncomps, M) LOO predicted weights
            'loo_var'         : (ncomps, M) LOO predictive variances
            'loo_residuals'   : (ncomps, M) true − predicted weights
            'loo_std_resid'   : (ncomps, M) standardised residuals
            'loo_mse_per_comp': (ncomps,) MSE per component
            'loo_rmse_per_comp': (ncomps,) RMSE per component
            'q2_per_comp'     : (ncomps,) per-component Q² (LOO R²)
            'nlpd_per_comp'   : (ncomps,) LOO neg. log predictive density
            'std_resid_mean'  : float, mean of all standardised residuals
            'std_resid_var'   : float, variance of all standardised residuals

            If *grid* is supplied, extra keys:
            'pca_recon_rmse'  : (M,) per-spectrum PCA reconstruction RMSE
            'loo_flux_rmse'   : (M,) per-spectrum LOO flux RMSE
            'pca_recon_rmse_median' : float
            'pca_recon_rmse_95'     : float
            'loo_flux_rmse_median'  : float
            'loo_flux_rmse_95'      : float
            'max_fractional_resid'  : float
            'pca_max_fractional_resid' : float, PCA-only worst-case
            'pca_per_wl_rmse' : (P,) per-wavelength PCA RMSE
            'loo_per_wl_rmse' : (P,) per-wavelength LOO RMSE
        """
        if not self._trained:
            warnings.warn(
                "Emulator is not trained — LOO results will be unreliable."
            )

        M = self.grid_points.shape[0]  # number of training points

        # Reshape w_hat to (ncomps, M)
        w_hat_all = self.w_hat.reshape(self.ncomps, M)

        # -----------------------------------------------------------
        # Decide GPU vs CPU path
        # -----------------------------------------------------------
        use_gpu = (
            PYTORCH_AVAILABLE
            and torch.cuda.is_available()
            and self._v11_gpu is not None
        )
        # Memory-efficient GPU path: V11 not materialized but GPU is available
        use_gpu_mem_eff = (
            not use_gpu
            and PYTORCH_AVAILABLE
            and torch.cuda.is_available()
            and self._v11_gpu is None
            and self._v11 is None
            and self.block_diagonal
        )

        if use_gpu:
            loo_mu, loo_var = self._loo_cv_gpu(w_hat_all, M)
        elif use_gpu_mem_eff:
            loo_mu, loo_var = self._loo_cv_gpu_memory_efficient(w_hat_all, M)
        else:
            loo_mu, loo_var = self._loo_cv_cpu(w_hat_all, M)

        # -----------------------------------------------------------
        # Derived weight-space metrics (always on CPU/numpy)
        # -----------------------------------------------------------
        loo_residuals = w_hat_all - loo_mu
        loo_std_resid = loo_residuals / np.sqrt(np.maximum(loo_var, 1e-30))
        loo_mse = np.mean(loo_residuals ** 2, axis=1)
        loo_rmse = np.sqrt(loo_mse)

        # Per-component Q²_k (LOO R² per PCA component)
        _w_var = np.var(w_hat_all, axis=1)  # (ncomps,)
        q2_per_comp = 1.0 - loo_mse / np.maximum(_w_var, 1e-30)

        # LOO Negative Log Predictive Density — proper scoring rule
        # (Bastos & O'Hagan 2009; Gneiting & Raftery 2007)
        _nlpd_pointwise = 0.5 * (
            np.log(2.0 * np.pi * np.maximum(loo_var, 1e-30))
            + loo_residuals ** 2 / np.maximum(loo_var, 1e-30)
        )  # (ncomps, M)
        nlpd_per_comp = np.mean(_nlpd_pointwise, axis=1)  # (ncomps,)

        # Standardized residual summary statistics
        _std_flat = loo_std_resid.ravel()
        std_resid_mean = float(np.mean(_std_flat))
        std_resid_var = float(np.var(_std_flat))

        results = {
            'loo_mu': loo_mu,
            'loo_var': loo_var,
            'loo_residuals': loo_residuals,
            'loo_std_resid': loo_std_resid,
            'loo_mse_per_comp': loo_mse,
            'loo_rmse_per_comp': loo_rmse,
            'q2_per_comp': q2_per_comp,
            'nlpd_per_comp': nlpd_per_comp,
            'std_resid_mean': std_resid_mean,
            'std_resid_var': std_resid_var,
        }

        # -----------------------------------------------------------
        # Optional flux-space metrics
        # -----------------------------------------------------------
        if grid is not None:
            if isinstance(grid, str):
                grid = NPZInterface(grid)

            # Original normalised spectra (same transform as from_grid:
            # raw → divide by per-spectrum mean → we compare against this)
            fluxes_raw = np.array(list(grid.fluxes))
            norm_factors = fluxes_raw.mean(1)
            original = fluxes_raw / norm_factors[:, np.newaxis]

            # Inverse PCA transform:  flux = weights @ (eigenspectra * flux_std) + flux_mean
            # This matches load_flux() exactly — using the emulator's stored
            # eigenspectra, flux_std and flux_mean (not recomputed from grid).
            X = self.eigenspectra * self.flux_std  # (K, P)

            # a) PCA truncation error (true weights → reconstructed flux)
            pca_recon = self.weights @ X + self.flux_mean  # (M, P)
            pca_recon_err = original - pca_recon
            pca_recon_rmse = np.sqrt(np.mean(pca_recon_err ** 2, axis=1))

            # b) LOO error (LOO predicted weights → reconstructed flux)
            loo_weights = loo_mu.T  # (M, K)
            loo_recon = loo_weights @ X + self.flux_mean  # (M, P)
            loo_flux_err = original - loo_recon
            loo_flux_rmse = np.sqrt(np.mean(loo_flux_err ** 2, axis=1))

            # Max fractional residual across all spectra and wavelengths
            with np.errstate(divide='ignore', invalid='ignore'):
                loo_fractional = np.abs(loo_flux_err) / (np.abs(original) + 1e-30)
                pca_fractional = np.abs(pca_recon_err) / (np.abs(original) + 1e-30)
            max_frac = float(np.nanmax(loo_fractional))
            pca_max_frac = float(np.nanmax(pca_fractional))

            # Per-wavelength RMSE: aggregate over spectra (axis 0)
            pca_per_wl_rmse = np.sqrt(np.mean(pca_recon_err ** 2, axis=0))  # (P,)
            loo_per_wl_rmse = np.sqrt(np.mean(loo_flux_err ** 2, axis=0))    # (P,)

            # c) LOO flux-space variance (propagate per-component GP
            #    uncertainty through the PCA inverse transform).
            #    flux_var[i, p] = sum_k  X[k,p]^2 * loo_var[k, i]
            #    where X = eigenspectra * flux_std.  Each component is
            #    independent so total variance is additive.
            X2 = X ** 2                                     # (K, P)
            loo_recon_var = loo_var.T @ X2                   # (M, P)

            results.update({
                'original_flux': original,          # (M, P) normalised grid spectra
                'pca_recon_flux': pca_recon,         # (M, P) PCA truncation recon
                'loo_recon_flux': loo_recon,          # (M, P) LOO GP recon
                'loo_recon_var': loo_recon_var,       # (M, P) LOO flux-space variance
                'loo_mu': loo_mu,                    # (K, M) LOO predicted weights
                'loo_var': loo_var,                   # (K, M) LOO predictive variances
                'wavelength': self.wl,               # (P,)
                'pca_recon_rmse': pca_recon_rmse,
                'loo_flux_rmse': loo_flux_rmse,
                'pca_recon_rmse_median': float(np.median(pca_recon_rmse)),
                'pca_recon_rmse_95': float(np.percentile(pca_recon_rmse, 95)),
                'loo_flux_rmse_median': float(np.median(loo_flux_rmse)),
                'loo_flux_rmse_95': float(np.percentile(loo_flux_rmse, 95)),
                'max_fractional_resid': max_frac,
                'pca_max_fractional_resid': pca_max_frac,
                'pca_per_wl_rmse': pca_per_wl_rmse,   # (P,)
                'loo_per_wl_rmse': loo_per_wl_rmse,    # (P,)
            })

        return results

    def _loo_cv_gpu(self, w_hat_all: np.ndarray, M: int):
        """
        GPU-accelerated LOO-CV using PyTorch.

        Performs Cholesky decomposition and inverse computation entirely on
        GPU, avoiding the expensive GPU→CPU transfer of v11 and the even
        more expensive numpy Cholesky on large M×M matrices.

        Parameters
        ----------
        w_hat_all : ndarray (ncomps, M)
        M : int

        Returns
        -------
        loo_mu, loo_var : ndarray (ncomps, M) each
        """
        device = self._v11_gpu.device
        dtype = self._v11_gpu.dtype

        w_hat_gpu = torch.from_numpy(w_hat_all).to(device, dtype)
        eye_M = torch.eye(M, device=device, dtype=dtype)

        loo_mu_gpu = torch.zeros(self.ncomps, M, device=device, dtype=dtype)
        loo_var_gpu = torch.zeros(self.ncomps, M, device=device, dtype=dtype)
        is_block_diagonal = (self._v11_gpu.dim() == 3)

        for k in range(self.ncomps):
            # Extract per-component covariance block
            if is_block_diagonal:
                K_k = self._v11_gpu[k]                                   # (M, M)
            else:
                K_k = self._v11_gpu[k * M:(k + 1) * M, k * M:(k + 1) * M]

            # Cholesky → solve for K_inv  (K L = I → K_inv = cholesky_solve(I, L))
            try:
                L = torch.linalg.cholesky(K_k)
                K_inv = torch.cholesky_solve(eye_M, L)                    # (M, M)
            except torch.linalg.LinAlgError:
                # Jitter and retry
                K_k_jit = K_k + 1e-8 * eye_M
                L = torch.linalg.cholesky(K_k_jit)
                K_inv = torch.cholesky_solve(eye_M, L)

            w_k = w_hat_gpu[k]                                           # (M,)
            K_inv_w = K_inv @ w_k                                        # (M,)
            K_inv_diag = K_inv.diagonal()                                # (M,)

            loo_mu_gpu[k] = w_k - K_inv_w / K_inv_diag
            loo_var_gpu[k] = 1.0 / K_inv_diag

        # Transfer results back to CPU (small: 2 × ncomps × M floats)
        return loo_mu_gpu.cpu().numpy(), loo_var_gpu.cpu().numpy()

    def _loo_cv_gpu_memory_efficient(self, w_hat_all: np.ndarray, M: int):
        """
        Memory-efficient GPU LOO-CV for large grids where V11 won't fit in VRAM.

        Builds each component's V11 block on-the-fly, computes K_inv via
        Cholesky, extracts LOO predictions, then discards the block.
        Peak VRAM: ~2 × M² × 8 bytes per component (same as training).

        Parameters
        ----------
        w_hat_all : ndarray (ncomps, M)
        M : int

        Returns
        -------
        loo_mu, loo_var : ndarray (ncomps, M) each
        """
        device = torch.device('cuda')
        dtype = TORCH_FLOAT64

        # Free the inference L cache to make room — LOO needs ~2×M²×8 VRAM per block.
        # The L cache can be rebuilt on the next inference call.
        if hasattr(self, '_mem_eff_L_blocks') and self._mem_eff_L_blocks is not None:
            self.log.info("Freeing inference L cache to make room for LOO-CV...")
            del self._mem_eff_L_blocks
            del self._mem_eff_alpha_blocks
            self._mem_eff_L_blocks = None
            self._mem_eff_alpha_blocks = None
            torch.cuda.empty_cache()

        # Ensure cached GPU tensors exist
        if self._grid_points_gpu is None:
            self._grid_points_gpu = torch.from_numpy(self._grid_points_norm).to(device, dtype)
        if self._variances_gpu is None:
            self._variances_gpu = torch.from_numpy(self.variances).to(device, dtype)
        if self._lengthscales_gpu is None:
            self._lengthscales_gpu = torch.from_numpy(self.lengthscales).to(device, dtype)
        if self._dots_inv_diag_gpu is None:
            dots = torch.from_numpy(self.eigenspectra @ self.eigenspectra.T).to(device, dtype)
            self._dots_inv_diag_gpu = torch.diag(torch.linalg.inv(dots))

        w_hat_gpu = torch.from_numpy(w_hat_all).to(device, dtype)
        eye_M = torch.eye(M, device=device, dtype=dtype)

        loo_mu_gpu = torch.zeros(self.ncomps, M, device=device, dtype=dtype)
        loo_var_gpu = torch.zeros(self.ncomps, M, device=device, dtype=dtype)

        self.log.info(f"Memory-efficient LOO-CV: building V11 blocks on-the-fly ({self.ncomps} components, M={M})")

        for k in range(self.ncomps):
            # Build V11 block for component k on-the-fly
            X_scaled = self._grid_points_gpu / self._lengthscales_gpu[k]
            K_k = torch.cdist(X_scaled, X_scaled, p=2.0)
            del X_scaled
            _apply_kernel_gpu_inplace(K_k, self._variances_gpu[k], self.kernel)

            # Add diagonal (match training: dots_inv_diag/lambda_xi + jitter)
            # Variance-scaled jitter keeps condition number ~1e6
            jitter = max(1e-5, 1e-6 * self._variances_gpu[k].item()) if self.strict_weight_fit else 1e-5
            if self.strict_weight_fit:
                K_k.diagonal().add_(jitter)
            else:
                K_k.diagonal().add_(self._dots_inv_diag_gpu[k].item() / self.lambda_xi + jitter)

            # Cholesky → K_inv via solve
            try:
                L = torch.linalg.cholesky(K_k)
                del K_k
                K_inv = torch.cholesky_solve(eye_M, L)
                del L
            except torch.linalg.LinAlgError:
                # Rebuild K_k from scratch with 10x jitter for a clean retry
                del K_k
                X_scaled = self._grid_points_gpu / self._lengthscales_gpu[k]
                K_k = torch.cdist(X_scaled, X_scaled, p=2.0)
                del X_scaled
                _apply_kernel_gpu_inplace(K_k, self._variances_gpu[k], self.kernel)
                retry_jitter = 10.0 * jitter
                if self.strict_weight_fit:
                    K_k.diagonal().add_(retry_jitter)
                else:
                    K_k.diagonal().add_(self._dots_inv_diag_gpu[k].item() / self.lambda_xi + retry_jitter)
                L = torch.linalg.cholesky(K_k)
                del K_k
                K_inv = torch.cholesky_solve(eye_M, L)
                del L

            w_k = w_hat_gpu[k]
            K_inv_w = K_inv @ w_k
            K_inv_diag = K_inv.diagonal()

            loo_mu_gpu[k] = w_k - K_inv_w / K_inv_diag
            loo_var_gpu[k] = 1.0 / K_inv_diag

            del K_inv
            torch.cuda.empty_cache()

        return loo_mu_gpu.cpu().numpy(), loo_var_gpu.cpu().numpy()

    def _loo_cv_cpu(self, w_hat_all: np.ndarray, M: int):
        """
        CPU fallback for LOO-CV using NumPy/SciPy.

        Parameters
        ----------
        w_hat_all : ndarray (ncomps, M)
        M : int

        Returns
        -------
        loo_mu, loo_var : ndarray (ncomps, M) each
        """
        v11_cpu = self.v11  # property handles lazy GPU→CPU if needed
        if v11_cpu is None:
            raise RuntimeError("Cannot access v11 for LOO computation.")

        loo_mu = np.zeros((self.ncomps, M))
        loo_var = np.zeros((self.ncomps, M))

        for k in range(self.ncomps):
            if self.block_diagonal:
                K_k = v11_cpu[k]
            else:
                K_k = v11_cpu[k * M:(k + 1) * M, k * M:(k + 1) * M]

            try:
                L = np.linalg.cholesky(K_k)
                K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(M)))
            except np.linalg.LinAlgError:
                K_inv = np.linalg.inv(K_k + 1e-8 * np.eye(M))

            w_k = w_hat_all[k]
            K_inv_w = K_inv @ w_k
            K_inv_diag = np.diag(K_inv)

            loo_mu[k] = w_k - K_inv_w / K_inv_diag
            loo_var[k] = 1.0 / K_inv_diag

        return loo_mu, loo_var

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

    def train(self, optimizer="nelder-mead", refine_lambda_xi=False, **opt_kwargs):
        """
        Trains the emulator's hyperparameters.

        Parameters
        ----------
        optimizer : str
            Optimization method. One of ``"nelder-mead"`` (default),
            ``"l-bfgs-b"``, or ``"cma-es"``.
        refine_lambda_xi : bool
            If True and ``strict_weight_fit`` is False, run a 1-D bounded
            refinement of lambda_xi after per-component training.
        **opt_kwargs
            Any arguments to pass to the optimizer. By default,
            ``method='Nelder-Mead'`` and ``maxiter=10000``.

        See Also
        --------
        scipy.optimize.minimize

        """
        import time

        optimizer = optimizer.lower()
        valid_optimizers = {"nelder-mead", "l-bfgs-b", "cma-es"}
        if optimizer not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Must be one of {valid_optimizers}"
            )
        if optimizer == "cma-es":
            try:
                import cma  # noqa: F401
            except ImportError:
                raise ImportError(
                    "CMA-ES optimizer requires the 'cma' package. "
                    "Install it with: pip install cma"
                )

        # Per-component training: optimize each PCA component independently
        if self.per_component and self.block_diagonal:
            self._train_per_component(optimizer=optimizer, refine_lambda_xi=refine_lambda_xi, **opt_kwargs)
            return
        
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            self._gpu_training_mode = True
            
            # Auto-detect if memory-efficient training is needed for block-diagonal mode
            if self.block_diagonal:
                M = self.grid_points.shape[0]
                n_comp = self.ncomps
                elem_bytes = 8  # float64
                # V11 storage + Cholesky workspace + kernel intermediates ≈ 3× V11
                estimated_bytes = n_comp * M * M * elem_bytes * 3
                gpu_vram = torch.cuda.get_device_properties(0).total_memory
                
                if estimated_bytes > gpu_vram:
                    self._memory_efficient_training = True
                    est_gb = estimated_bytes / (1024**3)
                    vram_gb = gpu_vram / (1024**3)
                    # Peak VRAM in memory-efficient mode: ~2 × M² × 8 (kernel block + Cholesky)
                    efficient_gb = (2 * M * M * elem_bytes) / (1024**3)
                    print(f"Memory-efficient training ENABLED")
                    print(f"  Standard mode would need ~{est_gb:.1f} GB, GPU has {vram_gb:.1f} GB")
                    print(f"  Memory-efficient peak: ~{efficient_gb:.1f} GB (builds V11 blocks on-the-fly)")
                else:
                    self._memory_efficient_training = False
        
        self.loss_history = []
        self._training_start_time = time.time()
        self._iteration_count = 0
        self._best_loss = np.inf
        self._last_progress_time = time.time()
        
        training_mode_str = "memory-efficient" if self._memory_efficient_training else "standard"
        print(f"Started training ({optimizer.upper()}, {training_mode_str})")
        print(f"Optimizing {len(self.get_param_vector())} hyperparameters")
        
        # Calculate maximum allowed variance based on data variance
        # We allow the kernel variance to be up to 100x the data variance
        # This prevents "runaway" optimization while allowing flexibility
        weights_variance = np.var(self.weights, axis=0) # (n_comp,)
        # Use valid max_variance bounds (at least 1.0 to avoid too tight constraints on small components)
        max_variances = np.maximum(100.0 * weights_variance, 1.0)
        
        def nll(P, apply_penalty=True):
            if np.any(~np.isfinite(P)):
                return 1e20
            self.set_param_vector(P)
            
            # Check lengthscales — return large finite penalty (not inf)
            # so gradient-based optimizers can still compute useful gradients
            if np.any(self.lengthscales < 0.5 * self._grid_sep):
                violation = np.sum(np.maximum(0, 0.5 * self._grid_sep - self.lengthscales))
                return 1e15 + 1e6 * violation
                
            # Check variances (prevent runaway) — only for unbounded optimizers
            current_variances = self.variances
            if apply_penalty and np.any(current_variances > max_variances):
                 # Soft penalty instead of hard cutoff for better optimization behavior
                 penalty = np.sum(np.maximum(0, current_variances - max_variances))
                 return 1e15 + penalty
            
            loss = -self.log_likelihood()
            self.loss_history.append(loss)
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

        default_kwargs = {"options": {"maxiter": 10000, "disp": True}}
        default_kwargs.update(opt_kwargs)
        total_max_iter = default_kwargs.get("options", {}).get("maxiter", 10000)

        # ── Build log-space bounds for the full param vector ──
        # Layout matches get_param_vector() / hyperparams dict order:
        #   [(log_lambda_xi),]  <-- first, only when not strict_weight_fit
        #   log_var:0, ..., log_var:K-1,
        #   log_ls:0:0, ..., log_ls:K-1:d-1
        d = self._grid_points_norm.shape[1]
        K = self.ncomps
        ls_floor = 0.5 * self._grid_sep      # (d,)
        ls_ceil  = 5.0 * np.ones(d)

        var_lo  = np.full(K, -30.0)
        var_hi  = np.full(K, np.log(1e4))
        ls_lo   = np.tile(np.log(ls_floor), K)
        ls_hi   = np.tile(np.log(ls_ceil),  K)
        if self.strict_weight_fit:
            joint_lo = np.concatenate([var_lo, ls_lo])
            joint_hi = np.concatenate([var_hi, ls_hi])
        else:
            joint_lo = np.concatenate([[-30.0], var_lo, ls_lo])
            joint_hi = np.concatenate([[ 30.0], var_hi, ls_hi])

        if optimizer == "nelder-mead":
            # ── Nelder-Mead path (with burn-in + auto-scaled fatol) ──
            default_kwargs["method"] = "Nelder-Mead"
            options = default_kwargs.get("options", {})
            if "fatol" not in options:
                rel_tol = options.pop("ftol", 1e-6)
                burn_in_iter = 50
                max_iter_total = options.get("maxiter", 1000)

                if max_iter_total > (burn_in_iter * 2):
                    self.log.info(f"Running burn-in phase ({burn_in_iter} iters) to calibrate convergence tolerance...")
                    print(f"Burn-in phase: {burn_in_iter} iterations")

                    burn_in_options = options.copy()
                    burn_in_options["maxiter"] = burn_in_iter
                    burn_in_options.pop("xatol", None)
                    burn_in_options.pop("fatol", None)
                    burn_in_options.pop("ftol", None)

                    burn_in_kwargs = default_kwargs.copy()
                    burn_in_kwargs["options"] = burn_in_options

                    burn_in_soln = minimize(nll, P0, args=(False,), **burn_in_kwargs)
                    P0 = burn_in_soln.x.copy()
                    current_loss = burn_in_soln.fun

                    print(f"Burn-in complete: Loss {initial_loss:.2e} -> {current_loss:.2e}")
                    self._iteration_count = 0
                    self.loss_history = []
                    initial_loss = current_loss
                    print(f"Main run max_iter: {options.get('maxiter')} (full allowance)")

                calculated_fatol = max(1e-8, abs(initial_loss) * rel_tol)
                options["fatol"] = calculated_fatol
                print(f"Auto-scaled fatol: {calculated_fatol:.4f} (relative tolerance: {rel_tol})")
                default_kwargs["options"] = options

            soln = minimize(nll, P0, args=(True,), **default_kwargs)

        elif optimizer == "l-bfgs-b":
            # ── L-BFGS-B with box constraints ──
            # Box bounds enforce constraints directly, so skip the soft penalty
            # (apply_penalty=False) to avoid early-return before loss computation.
            bounds_list = list(zip(joint_lo, joint_hi))
            callback = default_kwargs.pop("callback", None)
            soln = minimize(
                nll, P0, args=(False,),
                method="L-BFGS-B",
                bounds=bounds_list,
                callback=callback,
                options={
                    "maxiter": total_max_iter,
                    "ftol": 1e-15,
                    "gtol": 1e-12,
                    "eps": 1e-5,
                    "disp": False,
                },
            )

        elif optimizer == "cma-es":
            # ── CMA-ES (derivative-free, population-based) ──
            # Box bounds enforce constraints; skip soft penalty (apply_penalty=False).
            # Use maxfevals (not maxiter/generations) so the user's max_iter
            # maps to total function evaluations as expected.
            import cma
            sigma0 = 0.5
            callback = default_kwargs.pop("callback", None)
            cma_bounds = [joint_lo.tolist(), joint_hi.tolist()]
            es = cma.CMAEvolutionStrategy(
                P0.tolist(), sigma0,
                {
                    "bounds": cma_bounds,
                    "maxfevals": total_max_iter,
                    "verbose": -9,
                    "tolfun": 1e-8,
                },
            )
            best_x, best_f = P0.copy(), initial_loss
            while not es.stop():
                solutions = es.ask()
                fits = [nll(np.array(s), False) for s in solutions]
                es.tell(solutions, fits)
                gen_best = min(fits)
                if gen_best < best_f:
                    best_f = gen_best
                    best_x = np.array(solutions[fits.index(gen_best)])
                if callback is not None:
                    callback(best_x)
            # Wrap into a scipy-like result for the downstream code
            from types import SimpleNamespace
            soln = SimpleNamespace(
                x=best_x, fun=best_f, success=True,
                message="CMA-ES terminated", nit=es.result.iterations,
            )
        
        if self._gpu_training_mode:
            was_memory_efficient = self._memory_efficient_training
            self._gpu_training_mode = False
            self._memory_efficient_training = False  # Restore normal mode
            if was_memory_efficient:
                # Large grid: V11 won't fit on GPU or CPU. Don't rebuild — 
                # inference will use _call_gpu_memory_efficient (streaming blocks).
                # Just ensure hyperparams are synced (lightweight, no V11 alloc).
                self._v11_gpu = None
                self._v11 = None
                self._L_gpu = None
                self._alpha_gpu = None
                self._L_gpu_source_id = None
                self._mem_eff_L_blocks = None
                self._mem_eff_alpha_blocks = None
                print("(Skipping V11 rebuild — memory-efficient inference will be used)")
            else:
                self.set_param_dict(self.get_param_dict())  # Rebuild V11 normally
        
        final_time = time.time()
        total_elapsed = final_time - self._training_start_time
        print(f"\nOptimization complete")
        print(f"Total time: {total_elapsed:.1f}s | Iterations: {self._iteration_count} | Final loss: {self._best_loss:.2f}")

        if not soln.success:
            self.log.warning("Optimization did not succeed.")
            print(f"Optimization Status: {soln.message}") # Show user why it stopped
        else:
            self.log.info("Finished optimizing emulator hyperparameters")
            print("Optimization converged successfully.")
        
        # Always apply soln.x — it holds the best point found, whether converged
        # or stopped at maxiter. For memory-efficient mode, do a lightweight update
        # to avoid triggering a ~38 GB V11 rebuild on CPU.
        if self._v11 is None and self._v11_gpu is None and self.block_diagonal:
            # Lightweight update: set hyperparams without triggering V11 rebuild
            parameters = self.get_param_dict()
            keys = [k for k in parameters.keys() if k != "log_lambda_xi"] if self.strict_weight_fit else list(parameters.keys())
            for key, val in zip(keys, soln.x):
                if key in self.hyperparams:
                    self.hyperparams[key] = val
            # Sync GPU hyperparameter tensors if they exist
            if self._variances_gpu is not None:
                device = self._variances_gpu.device
                self._variances_gpu = torch.from_numpy(self.variances).to(device, TORCH_FLOAT64)
                self._lengthscales_gpu = torch.from_numpy(self.lengthscales).to(device, TORCH_FLOAT64)
            # Invalidate the L cache (hyperparams may have changed)
            self._mem_eff_L_blocks = None
            self._mem_eff_alpha_blocks = None
        else:
            self.set_param_vector(soln.x)
        
        self._trained = True
        self.log.info(self)
            
        # Clean up progress tracking variables
        delattr(self, '_training_start_time')
        delattr(self, '_iteration_count')
        delattr(self, '_best_loss')
        delattr(self, '_last_progress_time')

    def _train_per_component(self, optimizer="nelder-mead", refine_lambda_xi=False, **opt_kwargs):
        """Train each PCA component's GP independently.

        In block_diagonal mode the negative log-likelihood decomposes into
        K independent terms, one per component.  Instead of optimizing all
        K*(1+d) parameters jointly (which puts Nelder-Mead in a very
        high-dimensional space), we run K separate low-dimensional
        optimizations (1 variance + d lengthscales each).

        Parameters
        ----------
        optimizer : str
            One of ``"nelder-mead"`` (default), ``"l-bfgs-b"``, or
            ``"cma-es"``.
        refine_lambda_xi : bool
            If True and ``strict_weight_fit`` is False, run a 1-D
            bounded refinement of lambda_xi after all components.
        **opt_kwargs
            Forwarded to the underlying minimizer.  ``maxiter`` is read
            from ``options`` dict and used as the per-component budget.
        """
        import time

        optimizer = optimizer.lower()
        valid_optimizers = {"nelder-mead", "l-bfgs-b", "cma-es"}
        if optimizer not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Must be one of {valid_optimizers}"
            )

        if optimizer == "cma-es":
            try:
                import cma  # noqa: F401
            except ImportError:
                raise ImportError(
                    "CMA-ES optimizer requires the 'cma' package. "
                    "Install it with: pip install cma"
                )

        self.loss_history = []
        self._training_start_time = time.time()
        self._iteration_count = 0
        self._best_loss = np.inf
        self._last_progress_time = time.time()

        d = self._grid_points_norm.shape[1]  # number of grid parameters
        M = self.grid_points.shape[0]

        # Lengthscale bounds (in normalized [0,1] space)
        ls_floor = 0.5 * self._grid_sep      # (d,) — minimum useful resolution
        ls_ceil  = 5.0 * np.ones(d)           # (d,) — beyond this the GP is ~constant

        # Log-space bounds for bounded optimizers (L-BFGS-B, CMA-ES)
        # variance (index 0): unbounded below, capped at log(1e4)
        # lengthscales (indices 1..d): [log(ls_floor), log(ls_ceil)]
        log_bounds_lower = np.concatenate([[-30.0], np.log(ls_floor)])
        log_bounds_upper = np.concatenate([[np.log(1e4)], np.log(ls_ceil)])

        # Parse user options
        default_kwargs = {
            "method": "Nelder-Mead",
            "options": {"maxiter": 10000, "disp": False},
        }
        default_kwargs.update(opt_kwargs)
        total_max_iter = default_kwargs.get("options", {}).get("maxiter", 10000)
        # Each component gets the FULL iteration budget
        per_comp_max_iter = total_max_iter

        print(f"Per-component training ({optimizer}): {self.ncomps} components × {1+d} params each")
        print(f"  Per-component max_iter: {per_comp_max_iter}")
        print(f"  Lengthscale bounds: floor={ls_floor}, ceil={ls_ceil}")

        # Prepare GPU tensors once
        use_gpu = PYTORCH_AVAILABLE and torch.cuda.is_available()
        if use_gpu:
            device = torch.device("cuda")
            grid_gpu = torch.from_numpy(self._grid_points_norm).to(device, TORCH_FLOAT64)
            w_hat_reshaped = torch.from_numpy(
                self.w_hat.reshape(self.ncomps, M)
            ).to(device, TORCH_FLOAT64)
            if not self.strict_weight_fit:
                if hasattr(self, "_dots_inv_diag_gpu") and self._dots_inv_diag_gpu is not None:
                    dots_inv_diag = self._dots_inv_diag_gpu
                else:
                    dots = self.eigenspectra @ self.eigenspectra.T
                    dots_inv_diag = torch.from_numpy(
                        np.diag(np.linalg.inv(dots))
                    ).to(device, TORCH_FLOAT64)
        else:
            w_hat_reshaped_np = self.w_hat.reshape(self.ncomps, M)
            if not self.strict_weight_fit:
                dots = self.eigenspectra @ self.eigenspectra.T
                dots_inv_diag_np = np.diag(np.linalg.inv(dots))

        callback = opt_kwargs.pop("callback", None)

        for k in range(self.ncomps):
            comp_start = time.time()
            comp_best_loss = np.inf
            comp_best_P = None

            # Initial parameter vector for component k: [log_var, log_ls_0, ..., log_ls_{d-1}]
            p0 = np.empty(1 + d)
            p0[0] = np.log(self.variances[k])
            p0[1:] = np.log(self.lengthscales[k])

            def nll_k(P, _k=k):
                if np.any(~np.isfinite(P)):
                    return np.inf

                log_var = P[0]
                log_ls = P[1:]
                var_k = np.exp(log_var)
                ls_k = np.exp(log_ls)

                # Lengthscale bounds
                if np.any(ls_k < ls_floor) or np.any(ls_k > ls_ceil):
                    return np.inf

                # Variance-scaled jitter for numerical stability
                # Keeps condition number bounded at ~1e6 regardless of variance scale
                jitter = max(1e-5, 1e-6 * var_k) if self.strict_weight_fit else 1e-5

                if use_gpu:
                    var_t = torch.tensor(var_k, device=device, dtype=TORCH_FLOAT64)
                    ls_t = torch.from_numpy(ls_k).to(device, TORCH_FLOAT64)

                    X_scaled = grid_gpu / ls_t
                    dist = torch.cdist(X_scaled, X_scaled, p=2.0)
                    _apply_kernel_gpu_inplace(dist, var_t, self.kernel)

                    if self.strict_weight_fit:
                        dist.diagonal().add_(jitter)
                    else:
                        dist.diagonal().add_(
                            dots_inv_diag[_k].item() / self.lambda_xi + jitter
                        )

                    try:
                        L = torch.linalg.cholesky(dist)
                    except torch.linalg.LinAlgError:
                        return np.inf
                    del dist

                    logdet = 2.0 * torch.sum(torch.log(L.diagonal())).item()

                    z = torch.linalg.solve_triangular(
                        L, w_hat_reshaped[_k].unsqueeze(-1), upper=False
                    )
                    solved = torch.linalg.solve_triangular(
                        L.T, z, upper=True
                    ).squeeze(-1)
                    del L

                    sqmah = torch.dot(w_hat_reshaped[_k], solved).item()
                    del solved, z
                else:
                    # CPU path
                    _kfunc = get_cpu_kernel_func(self.kernel)
                    K = _kfunc(
                        self._grid_points_norm, self._grid_points_norm,
                        var_k, ls_k,
                    )
                    if self.strict_weight_fit:
                        K[np.diag_indices_from(K)] += jitter
                    else:
                        K[np.diag_indices_from(K)] += (
                            dots_inv_diag_np[_k] / self.lambda_xi + jitter
                        )
                    try:
                        Lc, flag = cho_factor(K)
                    except np.linalg.LinAlgError:
                        return np.inf
                    logdet = 2 * np.sum(np.log(Lc.diagonal()))
                    solved = cho_solve((Lc, flag), w_hat_reshaped_np[_k])
                    sqmah = np.dot(w_hat_reshaped_np[_k], solved)

                loss = (logdet + sqmah) / 2.0
                self.loss_history.append(loss)
                self._iteration_count += 1

                # Track best-ever params for this component
                nonlocal comp_best_loss, comp_best_P
                if loss < comp_best_loss:
                    comp_best_loss = loss
                    comp_best_P = P.copy()
                if loss < self._best_loss:
                    self._best_loss = loss

                now = time.time()
                if (self._iteration_count % 10 == 0) or (now - self._last_progress_time > 30):
                    elapsed = now - self._training_start_time
                    print(
                        f"  Comp {_k+1:2d}/{self.ncomps} | "
                        f"Iter {self._iteration_count:5d} | "
                        f"Loss: {loss:10.2f} | Best(comp): {comp_best_loss:10.2f} | "
                        f"Time: {elapsed:6.1f}s"
                    )
                    self._last_progress_time = now

                # Invoke Marimo callback for live chart updates
                if callback is not None and self._iteration_count % 10 == 0:
                    callback(None)

                return loss

            # --- Optimizer dispatch per component ---
            if optimizer == "nelder-mead":
                # Burn-in for fatol calibration
                burn_in_iter = 50
                burn_in_opts = {"maxiter": burn_in_iter, "disp": False}
                burn_soln = minimize(
                    nll_k, p0,
                    method="Nelder-Mead", options=burn_in_opts,
                    callback=callback,
                )
                p0 = burn_soln.x.copy()
                baseline_loss = burn_soln.fun if np.isfinite(burn_soln.fun) else comp_best_loss
                fatol = max(1e-8, abs(baseline_loss) * 1e-6)
                print(f"  Comp {k+1:2d}/{self.ncomps} | Burn-in done: loss {baseline_loss:.2f} → fatol={fatol:.2e}")

                comp_opts = {
                    "maxiter": per_comp_max_iter,
                    "fatol": fatol,
                    "disp": False,
                }
                soln = minimize(
                    nll_k, p0,
                    method="Nelder-Mead", options=comp_opts,
                    callback=callback,
                )

            elif optimizer == "l-bfgs-b":
                # L-BFGS-B with box constraints in log-space
                bounds_list = list(zip(log_bounds_lower, log_bounds_upper))
                soln = minimize(
                    nll_k, p0,
                    method="L-BFGS-B",
                    bounds=bounds_list,
                    callback=callback,
                    options={
                        "maxiter": per_comp_max_iter,
                        "ftol": 1e-15,
                        "gtol": 1e-12,
                        "eps": 1e-5,
                        "disp": False,
                    },
                )

            elif optimizer == "cma-es":
                import cma
                # CMA-ES: population-based evolutionary strategy
                sigma0 = 0.5  # initial step size in log-space
                cma_opts = {
                    "bounds": [log_bounds_lower.tolist(), log_bounds_upper.tolist()],
                    "maxfevals": per_comp_max_iter,
                    "verbose": -9,  # suppress CMA-ES internal output
                    "tolfun": 1e-8,
                }
                es = cma.CMAEvolutionStrategy(p0, sigma0, cma_opts)
                while not es.stop():
                    solutions = es.ask()
                    fitnesses = [nll_k(x) for x in solutions]
                    es.tell(solutions, fitnesses)
                    if callback is not None:
                        callback(None)
                # Build a minimal result-like object for uniform handling
                soln = type("CMAResult", (), {
                    "x": es.result.xbest,
                    "fun": es.result.fbest,
                })()
                print(f"  Comp {k+1:2d}/{self.ncomps} | CMA-ES: {es.result.evaluations} fevals, sigma={es.sigma:.4f}")

            # Restore best-ever parameters (soln.x is usually the best, but
            # this guards against Nelder-Mead returning a non-optimal final simplex vertex)
            best_P = comp_best_P if comp_best_P is not None else soln.x
            self.hyperparams[f"log_variance:{k}"] = best_P[0]
            for j in range(d):
                self.hyperparams[f"log_lengthscale:{k}:{j}"] = best_P[1 + j]

            comp_elapsed = time.time() - comp_start
            print(
                f"  Component {k+1}/{self.ncomps} done | "
                f"Best loss: {comp_best_loss:.2f} | Time: {comp_elapsed:.1f}s | "
                f"var={np.exp(best_P[0]):.4f} ls={np.exp(best_P[1:])}"
            )

        # --- lambda_xi refinement (non-strict mode only) ---
        # After per-component training, optimize the shared lambda_xi scalar
        # via 1D bounded minimization over the total NLL across all components.
        if refine_lambda_xi and not self.strict_weight_fit:
            print("\nRefining λ_ξ (1D Brent optimization)...")
            log_lam_init = np.log(self.lambda_xi)

            def _total_nll_lambda(log_lam):
                lam = np.exp(log_lam)
                total = 0.0
                for kk in range(self.ncomps):
                    var_kk = np.exp(self.hyperparams[f"log_variance:{kk}"])
                    ls_kk = np.array([np.exp(self.hyperparams[f"log_lengthscale:{kk}:{j}"]) for j in range(d)])
                    jitter_kk = max(1e-5, 1e-6 * var_kk) if self.strict_weight_fit else 1e-5
                    shift_kk = dots_inv_diag_val[kk] / lam + jitter_kk

                    if use_gpu:
                        var_t = torch.tensor(var_kk, device=device, dtype=TORCH_FLOAT64)
                        ls_t = torch.from_numpy(ls_kk).to(device, TORCH_FLOAT64)
                        X_sc = grid_gpu / ls_t
                        dist_m = torch.cdist(X_sc, X_sc, p=2.0)
                        _apply_kernel_gpu_inplace(dist_m, var_t, self.kernel)
                        dist_m.diagonal().add_(shift_kk)
                        try:
                            Lm = torch.linalg.cholesky(dist_m)
                        except torch.linalg.LinAlgError:
                            return 1e30
                        logdet = 2.0 * torch.sum(torch.log(Lm.diagonal())).item()
                        z = torch.linalg.solve_triangular(Lm, w_hat_reshaped[kk].unsqueeze(-1), upper=False)
                        solved = torch.linalg.solve_triangular(Lm.T, z, upper=True).squeeze(-1)
                        sqmah = torch.dot(w_hat_reshaped[kk], solved).item()
                        del dist_m, Lm, z, solved
                    else:
                        _kfunc = get_cpu_kernel_func(self.kernel)
                        K = _kfunc(self._grid_points_norm, self._grid_points_norm, var_kk, ls_kk)
                        K[np.diag_indices_from(K)] += shift_kk
                        try:
                            Lc, flag = cho_factor(K)
                        except np.linalg.LinAlgError:
                            return 1e30
                        logdet = 2 * np.sum(np.log(Lc.diagonal()))
                        solved = cho_solve((Lc, flag), w_hat_reshaped_np[kk])
                        sqmah = np.dot(w_hat_reshaped_np[kk], solved)

                    total += (logdet + sqmah) / 2.0
                return total

            # Precompute dots_inv_diag values for all components
            if use_gpu:
                if hasattr(self, "_dots_inv_diag_gpu") and self._dots_inv_diag_gpu is not None:
                    dots_inv_diag_val = [self._dots_inv_diag_gpu[kk].item() for kk in range(self.ncomps)]
                else:
                    dots = self.eigenspectra @ self.eigenspectra.T
                    dots_inv_diag_val = list(np.diag(np.linalg.inv(dots)))
            else:
                dots = self.eigenspectra @ self.eigenspectra.T
                dots_inv_diag_val = list(np.diag(np.linalg.inv(dots)))

            nll_before = _total_nll_lambda(log_lam_init)
            lam_result = minimize_scalar(
                _total_nll_lambda,
                bounds=(log_lam_init - 5.0, log_lam_init + 5.0),
                method="bounded",
                options={"xatol": 1e-6},
            )
            nll_after = lam_result.fun
            if nll_after < nll_before:
                self.hyperparams["log_lambda_xi"] = lam_result.x
                print(f"  λ_ξ: {np.exp(log_lam_init):.6f} → {np.exp(lam_result.x):.6f} | NLL: {nll_before:.2f} → {nll_after:.2f}")
            else:
                print(f"  λ_ξ refinement did not improve NLL ({nll_before:.2f} → {nll_after:.2f}), keeping original value.")

        # --- Post-training housekeeping ---
        # Check if V11 fits in memory; if so rebuild, otherwise leave for
        # memory-efficient inference.
        if use_gpu:
            v11_bytes = self.ncomps * M * M * 8 * 3
            gpu_vram = torch.cuda.get_device_properties(0).total_memory
            if v11_bytes > gpu_vram:
                self._v11_gpu = None
                self._v11 = None
                self._L_gpu = None
                self._alpha_gpu = None
                self._L_gpu_source_id = None
                self._mem_eff_L_blocks = None
                self._mem_eff_alpha_blocks = None
                print("(Skipping V11 rebuild — memory-efficient inference will be used)")
            else:
                self._gpu_training_mode = True
                self.set_param_dict(self.get_param_dict())
                self._gpu_training_mode = False
        else:
            self.set_param_dict(self.get_param_dict())

        total_elapsed = time.time() - self._training_start_time
        print(f"\nPer-component training complete")
        print(
            f"Total time: {total_elapsed:.1f}s | "
            f"Total iters: {self._iteration_count} | "
            f"Best total loss: {self._best_loss:.2f}"
        )

        self._trained = True
        self.log.info(self)

        # Clean up progress tracking variables
        delattr(self, "_training_start_time")
        delattr(self, "_iteration_count")
        delattr(self, "_best_loss")
        delattr(self, "_last_progress_time")

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
                self._grid_points_gpu = torch.from_numpy(self._grid_points_norm).to(device, TORCH_FLOAT64)
            if self._w_hat_gpu is None:
                self._w_hat_gpu = torch.from_numpy(self.w_hat).to(device, TORCH_FLOAT64)
            
            self._variances_gpu = torch.from_numpy(self.variances).to(device, TORCH_FLOAT64)
            self._lengthscales_gpu = torch.from_numpy(self.lengthscales).to(device, TORCH_FLOAT64)
            
            if self.block_diagonal:
                if self._memory_efficient_training:
                    # Memory-efficient mode: skip V11 construction entirely.
                    # The log_likelihood method will build each block on-the-fly.
                    # We only need the small hyperparameter tensors (already updated above).
                    pass
                else:
                    # Standard Block-Diagonal Update — materializes full (n_comp, M, M) tensor
                    
                    M = self.grid_points.shape[0]
                    eye_M = torch.eye(M, dtype=TORCH_FLOAT64, device=device)
                    
                    if self.strict_weight_fit:
                        # Variance-scaled jitter keeps condition number ~1e6
                        jitter_t = torch.clamp(1e-6 * self._variances_gpu, min=1e-5)
                        v11_gpu = (jitter_t.view(-1, 1, 1) * eye_M.unsqueeze(0)).clone()
                    else:
                        v11_gpu = (self._dots_inv_diag_gpu.view(-1, 1, 1) * eye_M.unsqueeze(0)) / self.lambda_xi
                        v11_gpu += 1e-5 * eye_M.unsqueeze(0)

                    # Add kernel in-place
                    batch_kernel_pytorch_gpu_only(
                        self._grid_points_gpu, 
                        self._grid_points_gpu, 
                        self._variances_gpu, 
                        self._lengthscales_gpu,
                        device='cuda',
                        return_stacked=True,
                        out=v11_gpu,
                        add_to_out=True,
                        kernel_type=self.kernel
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
                    return_stacked=False,
                    kernel_type=self.kernel
                )
                
                if self.strict_weight_fit:
                    M = kernel_gpu.shape[0]
                    self._v11_gpu = kernel_gpu + (1e-5 * torch.eye(M, device=device, dtype=TORCH_FLOAT64))
                else:
                    self._v11_gpu = self._iPhiPhi_gpu / self.lambda_xi + kernel_gpu
                    M = self._iPhiPhi_gpu.shape[0]
                    self._v11_gpu += 1e-5 * torch.eye(M, device=device, dtype=TORCH_FLOAT64)
        else:
            if self.block_diagonal:
                # Block-Diagonal CPU Update
                M = self.grid_points.shape[0]
                kernel_cpu = np.zeros((self.ncomps, M, M))
                
                _kfunc = get_cpu_kernel_func(self.kernel)
                for i in range(self.ncomps):
                    kernel_cpu[i] = _kfunc(
                        self._grid_points_norm, 
                        self._grid_points_norm, 
                        self.variances[i], 
                        self.lengthscales[i]
                    )
                
                if self.strict_weight_fit:
                    # Variance-scaled jitter keeps condition number ~1e6
                    eye_M = np.eye(M)
                    jitter_arr = np.maximum(1e-6 * self.variances, 1e-5)  # (ncomps,)
                    self.v11 = kernel_cpu + jitter_arr[:, None, None] * eye_M
                else:
                    if self.iPhiPhi is None:
                        # Reconstruct diagonal scaling factors if iPhiPhi is missing (memory optimization)
                        dots = self.eigenspectra @ self.eigenspectra.T
                        dots_inv = np.linalg.inv(dots)
                        dots_inv_diag = np.diag(dots_inv)
                        
                        # Add (dots_inv_diag / lambda) * I to kernel and Jitter
                        eye_M = np.eye(M)
                        for i in range(self.ncomps):
                            kernel_cpu[i] += ((dots_inv_diag[i] / self.lambda_xi) + 1e-5) * eye_M
                        self.v11 = kernel_cpu
                    else:
                        self.v11 = self.iPhiPhi / self.lambda_xi + kernel_cpu
                        # Add Jitter
                        M = self.v11.shape[-1]
                        eye_M = np.eye(M)
                        for i in range(self.ncomps):
                            self.v11[i] += 1e-5 * eye_M
            else:
                kernel_auto = batch_kernel_auto(
                    self._grid_points_norm, self._grid_points_norm, self.variances, self.lengthscales, kernel_type=self.kernel
                )
                if self.strict_weight_fit:
                    self.v11 = kernel_auto + (1e-5 * np.eye(kernel_auto.shape[0]))
                else:
                    self.v11 = self.iPhiPhi / self.lambda_xi + kernel_auto
                    self.v11 += 1e-5 * np.eye(self.v11.shape[0])


    def get_param_vector(self) -> np.ndarray:
        """
        Get a vector of the current trainable parameters of the emulator.
        
        When strict_weight_fit is active, lambda_xi is frozen and excluded
        from the optimization vector.

        Returns
        -------
        numpy.ndarray
        """
        if self.strict_weight_fit:
            values = [v for k, v in self.get_param_dict().items() if k != "log_lambda_xi"]
        else:
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
        if self.strict_weight_fit:
            keys = [k for k in parameters.keys() if k != "log_lambda_xi"]
        else:
            keys = list(parameters.keys())
            
        if len(params) != len(keys):
            raise ValueError(
                "params must match length of parameters (get_param_vector())"
            )

        param_dict = dict(zip(keys, params))
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
        if self._gpu_training_mode and PYTORCH_AVAILABLE:
            # Memory-efficient path: build each V11 block on-the-fly (no full tensor stored)
            if self._memory_efficient_training and self.block_diagonal:
                try:
                    result = _memory_efficient_log_likelihood_gpu(
                        self._grid_points_gpu,
                        self._variances_gpu,
                        self._lengthscales_gpu,
                        self._dots_inv_diag_gpu,
                        self.lambda_xi,
                        self._w_hat_gpu,
                        self.ncomps,
                        strict_weight_fit=self.strict_weight_fit,
                        kernel_type=self.kernel
                    )
                    return result
                except Exception as e:
                    self.log.warning(f"Memory-efficient GPU training failed: {e}. Falling back.")
            
            # Standard path: full V11 tensor on GPU
            if self._v11_gpu is not None:
                try:
                    result = _pytorch_log_likelihood_computation_gpu_only(
                        self._v11_gpu,
                        self._w_hat_gpu
                    )
                    return result
                except Exception as e:
                    self.log.warning(f"GPU training mode failed: {e}. Falling back.")
        
        # CPU Fallback (SciPy/NumPy)
        # Use the v11 property which handles lazy GPU→CPU transfer
        v11_cpu = self.v11
        if v11_cpu is None:
            raise RuntimeError(
                "Cannot compute log likelihood: V11 is not materialized. "
                "Call set_param_dict() or rebuild V11 first."
            )
        
        if self.block_diagonal and v11_cpu.ndim == 3:
            # Block-Diagonal CPU Implementation
            # v11 is (n_comp, M, M)
            # w_hat is (n_comp * M,) -> reshape to (n_comp, M)
            
            w_hat_reshaped = self.w_hat.reshape(self.ncomps, -1)
            
            logdet = 0.0
            sqmah = 0.0
            
            for i in range(self.ncomps):
                L, flag = cho_factor(v11_cpu[i])
                logdet += 2 * np.sum(np.log(L.diagonal()))
                
                # Solve for this component
                # w_hat_i is (M,)
                solved_i = cho_solve((L, flag), w_hat_reshaped[i])
                sqmah += np.dot(w_hat_reshaped[i], solved_i)
                
            return -(logdet + sqmah) / 2

        L, flag = cho_factor(v11_cpu)
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
        output += f"\nLog Likelihood: {self.log_likelihood():.2f}\n" if (self._v11 is not None or self._v11_gpu is not None) else ""
        output += "\n[V11 not materialized — log likelihood unavailable]\n" if (self._v11 is None and self._v11_gpu is None) else ""
        return output
