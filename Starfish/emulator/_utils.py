import logging

import numpy as np
import scipy.linalg as sl
from scipy.special import loggamma

log = logging.getLogger(__name__)

# Try to import PyTorch for GPU acceleration
try:
    import torch
    PYTORCH_AVAILABLE = True
    # Define dtype here to avoid circular import
    # EDIT THIS LINE to change precision: torch.float32 or torch.float64
    DTYPE = torch.float64  # <-- Same as TORCH_FLOAT64 in emulator.py
except ImportError:
    PYTORCH_AVAILABLE = False
    DTYPE = None

# Global flag to control PyTorch usage in phi_squared computation
# Set to True to enable GPU acceleration for get_phi_squared
USE_PYTORCH_PHI_SQUARED = True


def get_w_hat(eigenspectra, fluxes):
    """
    Since we will overflow memory if we actually calculate Phi, we have to
    determine w_hat in a memory-efficient manner.
    
    Optimized version using vectorized operations and optional GPU acceleration.
    """
    m = len(eigenspectra)
    M = len(fluxes)
    
    # Try GPU-accelerated version first (avoids GPU↔CPU transfers)
    if USE_PYTORCH_PHI_SQUARED and PYTORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return get_w_hat_gpu(eigenspectra, fluxes)
        except Exception as e:
            log.warning(f"GPU acceleration failed in get_w_hat: {e}. Falling back to CPU.")
    
    # CPU fallback
    # Convert eigenspectra to matrix form
    eigenspectra_matrix = np.array(eigenspectra)  # Shape: (m, npix)
    
    # OPTIMIZATION: Avoid creating the massive Kronecker product (A ⊗ I)
    # The system is (A ⊗ I) x = b
    # This is equivalent to A X = B where x = vec(X') and b = vec(B')
    # So X = A^{-1} B
    
    # Compute B: dot products of eigenspectra and fluxes
    # Shape: (m, M)
    dot_products = eigenspectra_matrix @ fluxes.T
    
    # Compute A: dot products of eigenspectra with themselves
    # Shape: (m, m)
    dots = eigenspectra_matrix @ eigenspectra_matrix.T
    
    # Solve A X = B
    # Use Cholesky solve for stability since dots is symmetric positive definite
    fac = sl.cho_factor(dots, check_finite=False, overwrite_a=True)
    w_hat_reshaped = sl.cho_solve(fac, dot_products, check_finite=False)
    
    # Flatten result to match expected output format
    # The expected format is flattened row-major (component by component)
    return w_hat_reshaped.ravel()


def get_w_hat_gpu(eigenspectra, fluxes):
    """
    GPU-accelerated version of get_w_hat that keeps everything on GPU.
    
    Avoids GPU↔CPU transfer overhead by doing all operations on GPU:
    - Matrix multiplication (dot products)
    - Kronecker product (phi_squared)
    - Cholesky factorization
    - Triangular solve
    
    Only transfers final result to CPU (single vector vs multiple large matrices).
    
    Parameters
    ----------
    eigenspectra : array-like
        List of eigenspectra arrays, shape (m, npix)
    fluxes : numpy.ndarray
        Flux array, shape (M, npix)
        
    Returns
    -------
    numpy.ndarray
        w_hat vector, shape (M * m,)
    """
    m = len(eigenspectra)
    M = len(fluxes)
    
    device = torch.device('cuda')
    
    # Convert to PyTorch tensors on GPU (single transfer in)
    eigenspectra_matrix = np.array(eigenspectra)
    eig_torch = torch.from_numpy(eigenspectra_matrix).to(device, DTYPE)
    fluxes_torch = torch.from_numpy(fluxes).to(device, DTYPE)
    
    # 1. Compute dot products on GPU: (m, M)
    dot_products = torch.matmul(eig_torch, fluxes_torch.T)
    
    # 2. Reshape to match original format: out[i * M + j] = eigenspectra[i].T @ fluxes[j]
    out = torch.empty((M * m,), dtype=DTYPE, device=device)
    for i in range(m):
        out[i * M:(i + 1) * M] = dot_products[i, :]
    
    # 3. Compute phi_squared on GPU (inline to avoid transfer)
    # This is get_phi_squared_pytorch inlined:
    dots = torch.matmul(eig_torch, eig_torch.T)  # (m, m)
    
    # OPTIMIZATION: Avoid creating the massive Kronecker product (A ⊗ I)
    # The system is (A ⊗ I) x = b
    # This is equivalent to A X = B where x = vec(X') and b = vec(B')
    # So X = A^{-1} B
    
    # Reshape out to (m, M)
    out_reshaped = out.view(m, M)
    
    # Solve A X = B
    # dots @ w_hat_reshaped = out_reshaped
    # w_hat_reshaped = dots^{-1} @ out_reshaped
    
    # Use Cholesky solve for stability since dots is symmetric positive definite
    L_dots = torch.linalg.cholesky(dots)
    
    # Solve dots @ X = out_reshaped
    # L @ L.T @ X = out_reshaped
    z = torch.linalg.solve_triangular(L_dots, out_reshaped, upper=False)
    w_hat_reshaped = torch.linalg.solve_triangular(L_dots.T, z, upper=True)
    
    # Flatten result
    w_hat = w_hat_reshaped.flatten()
    
    # 6. Transfer only final result to CPU (single small vector)
    return w_hat.cpu().numpy()


def get_w_hat_original(eigenspectra, fluxes):
    """
    Original implementation for comparison/fallback.
    """
    m = len(eigenspectra)
    M = len(fluxes)
    out = np.empty((M * m,))
    for i in range(m):
        for j in range(M):
            out[i * M + j] = eigenspectra[i].T @ fluxes[j]

    phi_squared = get_phi_squared(eigenspectra, M)
    fac = sl.cho_factor(phi_squared, check_finite=False, overwrite_a=True)
    return sl.cho_solve(fac, out, check_finite=False)


def get_phi_squared_optimized(eigenspectra_matrix, M):
    """
    Optimized computation of Phi.T.dot(Phi) using vectorized operations.
    
    WARNING: This function returns a dense matrix of size (m*M, m*M).
    For large M, this will consume a massive amount of memory and may cause OOM.
    Use with caution.
    
    Parameters
    ----------
    eigenspectra_matrix : numpy.ndarray
        Shape (m, npix) - the eigenspectra as a matrix
    M : int
        Number of spectra
    """
    m = eigenspectra_matrix.shape[0]
    
    # Compute eigenspectra dot products: (m, m)
    dots = eigenspectra_matrix @ eigenspectra_matrix.T
    
    # Create the block-diagonal structure efficiently using Kronecker product
    out = np.kron(dots, np.eye(M))
    
    return out


def get_phi_squared_pytorch(eigenspectra_matrix, M, device='cuda'):
    """
    PyTorch GPU-accelerated computation of Phi.T.dot(Phi).
    
    Uses GPU for matrix multiplication and Kronecker product, providing
    significant speedup for large matrices (typically 5-10x faster).
    
    WARNING: This function returns a dense matrix of size (m*M, m*M).
    For large M, this will consume a massive amount of memory and may cause OOM.
    Use with caution.
    
    Parameters
    ----------
    eigenspectra_matrix : numpy.ndarray
        Shape (m, npix) - the eigenspectra as a matrix
    M : int
        Number of spectra
    device : str
        'cuda' or 'cpu', defaults to 'cuda'
        
    Returns
    -------
    numpy.ndarray
        The Phi.T @ Phi matrix as a numpy array
    """
    if not PYTORCH_AVAILABLE:
        # Fallback to optimized NumPy version
        return get_phi_squared_optimized(eigenspectra_matrix, M)
    
    try:
        # Use GPU if available, otherwise CPU
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Convert to PyTorch tensor with configurable precision
        eig_torch = torch.from_numpy(eigenspectra_matrix).to(device_obj, DTYPE)
        
        # Compute eigenspectra dot products on GPU: (m, m)
        dots = torch.matmul(eig_torch, eig_torch.T)
        
        # Create identity matrix on GPU
        eye_M = torch.eye(M, dtype=DTYPE, device=device_obj)
        
        # Kronecker product on GPU
        # torch.kron is available in PyTorch 1.9+
        out = torch.kron(dots, eye_M)
        
        # Convert back to numpy
        return out.cpu().numpy()
        
    except Exception as e:
        log.warning(f"PyTorch computation failed: {e}. Falling back to NumPy.")
        return get_phi_squared_optimized(eigenspectra_matrix, M)


def get_phi_squared(eigenspectra, M):
    """
    Original implementation - compute Phi.T.dot(Phi) in a memory efficient manner.

    eigenspectra is a list of 1D numpy arrays.
    """
    m = len(eigenspectra)
    out = np.zeros((m * M, m * M))

    # Compute all of the dot products pairwise, beforehand
    dots = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            dots[i, j] = eigenspectra[i].T @ eigenspectra[j]

    for i in range(M * m):
        for jj in range(m):
            ii = i // M
            j = jj * M + (i % M)
            out[i, j] = dots[ii, jj]
    return out


def get_altered_prior_factors(eigenspectra, fluxes):
    """
    Compute the altered priors for the :math:`\\lambda_\\xi` term as in eqns. A24 and
    A25 of Czekala et al. 2015.

    Parameters
    ----------
    eigenspectra : numpy.ndarray
        The PCA eigenspectra
    fluxes : numpy.ndarray
        The vertically stacked input spectra

    Returns
    -------
    """
    w_hat = get_w_hat(eigenspectra, fluxes)
    M, npix = fluxes.shape
    m = len(eigenspectra)

    Phi_w_hat = np.empty((M * npix, 1))
    for i in range(M):
        loss_per_M = np.zeros(npix)
        for j in range(m):
            loss_per_M += eigenspectra[j] * w_hat[i + j * M]
        indices = slice(i * npix, (i + 1) * npix)
        Phi_w_hat[indices] = loss_per_M.reshape(npix, -1)

    F = fluxes.ravel()
    a_prime = 0.5 * M * (npix - m)
    b_prime = 0.5 * (F.T @ F - F.T @ Phi_w_hat)

    return a_prime, b_prime[0]


class Gamma:
    def __init__(self, alpha, beta=1):
        self.alpha = alpha
        self.beta = beta

    def logpdf(self, x):
        lp = (
            self.alpha * np.log(self.beta)
            - loggamma(self.alpha)
            + (self.alpha - 1) * np.log(x)
            - self.beta * x
        )
        return lp

    def pdf(self, x):
        return np.exp(self.logpdf(x))
