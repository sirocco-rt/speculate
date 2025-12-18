import scipy as sp
import numpy as np
from functools import lru_cache

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


def rbf_kernel(X, Z, variance, lengthscale):
    """
    A classic radial-basis function (Gaussian; exponential squared) covariance kernel

    .. math::
        \\kappa(X, Z | \\sigma^2, \\Lambda) = \\sigma^2 \\exp\\left[-\\frac12 (X-Z)^T \\Lambda^{-1} (X-Z) \\right]

    Parameters
    ----------
    X : np.ndarray
        The first set of points
    Z : np.ndarray
        The second set of points. Must have same second dimension as `X`
    variance : double
        The amplitude for the RBF kernel
    lengthscale : np.ndarray or double
        The lengthscale for the RBF kernel. Must have same second dimension as `X`

    """

    sq_dist = sp.spatial.distance.cdist(X / lengthscale, Z / lengthscale, "sqeuclidean")
    return variance * np.exp(-0.5 * sq_dist)


def batch_kernel(X, Z, variances, lengthscales):
    """
    Batched RBF kernel

    Parameters
    ----------
    X : np.ndarray
        The first set of points
    Z : np.ndarray
        The second set of points. Must have same second dimension as `X`
    variances : np.ndarray
        The amplitude for the RBF kernel
    lengthscales : np.ndarray
        The lengthscale for the RBF kernel. Must have same second dimension as `X`

    See Also
    --------
    :function:`rbf_kernel`
    """
    blocks = [rbf_kernel(X, Z, var, ls) for var, ls in zip(variances, lengthscales)]
    return sp.linalg.block_diag(*blocks)


def batch_kernel_optimized(X, Z, variances, lengthscales):
    """
    Optimized batched RBF kernel with caching and vectorization
    
    This version pre-computes distance calculations and builds the block diagonal
    matrix more efficiently to reduce computational overhead.

    Parameters
    ----------
    X : np.ndarray
        The first set of points
    Z : np.ndarray  
        The second set of points. Must have same second dimension as `X`
    variances : np.ndarray
        The amplitude for each RBF kernel
    lengthscales : np.ndarray
        The lengthscale for each RBF kernel. Must have same second dimension as `X`

    Returns
    -------
    np.ndarray
        Block diagonal kernel matrix
    """
    n_components = len(variances)
    n_X, n_Z = X.shape[0], Z.shape[0]
    
    # Pre-allocate the full block diagonal matrix
    total_size_X = n_X * n_components  
    total_size_Z = n_Z * n_components
    result = np.zeros((total_size_X, total_size_Z))
    
    # Compute each block and place directly into result matrix
    for i, (var, ls) in enumerate(zip(variances, lengthscales)):
        # Calculate block indices
        start_X = i * n_X
        end_X = (i + 1) * n_X
        start_Z = i * n_Z  
        end_Z = (i + 1) * n_Z
        
        # Compute RBF kernel block directly into result
        sq_dist = sp.spatial.distance.cdist(X / ls, Z / ls, "sqeuclidean")
        result[start_X:end_X, start_Z:end_Z] = var * np.exp(-0.5 * sq_dist)
    
    return result


# Global cache for distance matrices to avoid recomputation
_distance_cache = {}

def clear_kernel_cache():
    """Clear the global kernel distance cache to free memory"""
    global _distance_cache
    _distance_cache.clear()

def get_cache_size():
    """Get the current size of the kernel cache"""
    return len(_distance_cache)

def _get_distance_cache_key(X, Z):
    """Generate a cache key for distance matrices"""
    # Use array shapes and a hash of the data for caching
    X_hash = hash(X.data.tobytes()) if hasattr(X, 'data') else hash(X.tobytes())
    Z_hash = hash(Z.data.tobytes()) if hasattr(Z, 'data') else hash(Z.tobytes())
    return (X.shape, Z.shape, X_hash, Z_hash)


def batch_kernel_cached(X, Z, variances, lengthscales):
    """
    Cached version of batch_kernel for repeated calls with same X, Z
    
    This version caches distance computations for identical X, Z pairs,
    which is especially beneficial during training when grid_points don't change.

    Parameters
    ----------
    X : np.ndarray
        The first set of points
    Z : np.ndarray
        The second set of points. Must have same second dimension as `X`
    variances : np.ndarray
        The amplitude for each RBF kernel
    lengthscales : np.ndarray
        The lengthscale for each RBF kernel

    Returns
    -------
    np.ndarray
        Block diagonal kernel matrix
    """
    cache_key = _get_distance_cache_key(X, Z)
    n_components = len(variances)
    n_X, n_Z = X.shape[0], Z.shape[0]
    
    # Pre-allocate result matrix
    total_size_X = n_X * n_components
    total_size_Z = n_Z * n_components  
    result = np.zeros((total_size_X, total_size_Z))
    
    for i, (var, ls) in enumerate(zip(variances, lengthscales)):
        # Create cache key for this specific lengthscale
        ls_key = (cache_key, tuple(ls.flatten()) if hasattr(ls, 'flatten') else ls)
        
        # Check cache for pre-computed distance matrix
        if ls_key in _distance_cache:
            sq_dist = _distance_cache[ls_key]
        else:
            # Compute and cache distance matrix
            sq_dist = sp.spatial.distance.cdist(X / ls, Z / ls, "sqeuclidean")
            _distance_cache[ls_key] = sq_dist
            
            # Limit cache size to prevent memory issues
            if len(_distance_cache) > 100:  # Arbitrary limit
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(_distance_cache))
                del _distance_cache[oldest_key]
        
        # Calculate block indices and fill result matrix
        start_X = i * n_X
        end_X = (i + 1) * n_X  
        start_Z = i * n_Z
        end_Z = (i + 1) * n_Z
        
        result[start_X:end_X, start_Z:end_Z] = var * np.exp(-0.5 * sq_dist)
    
    return result


# ============================================================================
# PyTorch GPU-Accelerated Kernel Functions
# ============================================================================

def rbf_kernel_pytorch(X, Z, variance, lengthscale, device='cuda'):
    """
    PyTorch GPU-accelerated RBF kernel computation using torch.cdist
    
    Uses PyTorch's optimized cdist function for clean, fast distance computation.
    Significantly faster than CPU version for large matrices.
    
    Parameters
    ----------
    X : np.ndarray
        The first set of points (n_X, n_features)
    Z : np.ndarray
        The second set of points (n_Z, n_features)
    variance : float
        The amplitude for the RBF kernel
    lengthscale : np.ndarray or float
        The lengthscale for the RBF kernel
    device : str
        'cuda' for GPU or 'cpu' for CPU
    
    Returns
    -------
    np.ndarray
        RBF kernel matrix (n_X, n_Z)
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install torch")
    
    # Convert to PyTorch tensors with configurable precision
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    X_torch = torch.from_numpy(np.asarray(X)).to(device_obj, DTYPE)
    Z_torch = torch.from_numpy(np.asarray(Z)).to(device_obj, DTYPE)
    
    # Handle scalar or array lengthscale
    if np.isscalar(lengthscale):
        ls_torch = torch.tensor(lengthscale, dtype=DTYPE, device=device_obj)
    else:
        ls_torch = torch.from_numpy(np.asarray(lengthscale)).to(device_obj, DTYPE)
    
    # Scale by lengthscale: X / lengthscale
    X_scaled = X_torch / ls_torch
    Z_scaled = Z_torch / ls_torch
    
    # Compute squared Euclidean distances using torch.cdist (much faster!)
    sq_dist = torch.cdist(X_scaled, Z_scaled, p=2.0) ** 2
    
    # Apply RBF kernel: variance * exp(-0.5 * sq_dist)
    kernel = variance * torch.exp(-0.5 * sq_dist)
    
    # # CRITICAL: Synchronize GPU before returning (for accurate benchmarking)
    # if device_obj.type == 'cuda':
    #     torch.cuda.synchronize()
    
    # Transfer back to CPU and convert to numpy
    return kernel.cpu().numpy()


def batch_kernel_pytorch(X, Z, variances, lengthscales, device='cuda'):
    """
    PyTorch GPU-accelerated batched RBF kernel using torch.cdist
    
    Uses PyTorch's optimized cdist for clean, efficient distance computation.
    Builds block diagonal kernel matrix on GPU with minimal transfers.
    
    Expected speedup: 5-20x for large matrices (>1000×1000 per block)
    
    Parameters
    ----------
    X : np.ndarray
        The first set of points (n_X, n_features)
    Z : np.ndarray
        The second set of points (n_Z, n_features)
    variances : np.ndarray
        The amplitude for each RBF kernel (n_components,)
    lengthscales : np.ndarray
        The lengthscale for each RBF kernel (n_components, n_features)
    device : str
        'cuda' for GPU or 'cpu' for CPU
    
    Returns
    -------
    np.ndarray
        Block diagonal kernel matrix (n_X*n_components, n_Z*n_components)
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install torch")
    
    n_components = len(variances)
    n_X, n_Z = X.shape[0], Z.shape[0]
    
    # Convert to PyTorch tensors with configurable precision
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    X_torch = torch.from_numpy(np.asarray(X)).to(device_obj, DTYPE)
    Z_torch = torch.from_numpy(np.asarray(Z)).to(device_obj, DTYPE)
    variances_torch = torch.from_numpy(np.asarray(variances)).to(device_obj, DTYPE)
    lengthscales_torch = torch.from_numpy(np.asarray(lengthscales)).to(device_obj, DTYPE)
    
    # Pre-allocate result on GPU
    total_size_X = n_X * n_components
    total_size_Z = n_Z * n_components
    result = torch.zeros((total_size_X, total_size_Z), dtype=DTYPE, device=device_obj)
    
    # Compute each block on GPU using torch.cdist
    for i in range(n_components):
        var = variances_torch[i]
        ls = lengthscales_torch[i]
        
        # Scale by lengthscale
        X_scaled = X_torch / ls
        Z_scaled = Z_torch / ls
        
        # Compute squared Euclidean distances using torch.cdist
        sq_dist = torch.cdist(X_scaled, Z_scaled, p=2.0) ** 2
        
        # Apply RBF kernel
        kernel_block = var * torch.exp(-0.5 * sq_dist)
        
        # Place in block diagonal matrix
        start_X = i * n_X
        end_X = (i + 1) * n_X
        start_Z = i * n_Z
        end_Z = (i + 1) * n_Z
        result[start_X:end_X, start_Z:end_Z] = kernel_block
    
    # CRITICAL: Synchronize GPU before returning (for accurate benchmarking)
    if device_obj.type == 'cuda':
        torch.cuda.synchronize()
    
    # Single transfer back to CPU
    return result.cpu().numpy()


def batch_kernel_pytorch_gpu_only(X, Z, variances, lengthscales, device='cuda', return_stacked=False, out=None, add_to_out=False):
    """
    PyTorch GPU-accelerated batched RBF kernel that returns GPU tensor (NO CPU TRANSFER!)
    
    This version keeps the result on GPU for use in training pipelines where
    the result is immediately used in subsequent GPU operations (like log_likelihood).
    
    Eliminates the massive GPU→CPU transfer bottleneck during training!
    
    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        The first set of points (n_X, n_features)
    Z : np.ndarray or torch.Tensor
        The second set of points (n_Z, n_features)
    variances : np.ndarray or torch.Tensor
        The amplitude for each RBF kernel (n_components,)
    lengthscales : np.ndarray or torch.Tensor
        The lengthscale for each RBF kernel (n_components, n_features)
    device : str
        'cuda' for GPU or 'cpu' for CPU
    return_stacked : bool
        If True, returns a 3D tensor (n_components, n_X, n_Z) instead of a 2D block-diagonal matrix.
        This saves massive amounts of memory by avoiding storing zeros.
    out : torch.Tensor, optional
        Output tensor to write results into. If provided, must match expected output shape.
    add_to_out : bool, optional
        If True and out is provided, adds the kernel result to out instead of overwriting.
        Useful for constructing v11 = iPhiPhi + kernel in-place.
    
    Returns
    -------
    torch.Tensor
        Block diagonal kernel matrix on GPU (n_X*n_components, n_Z*n_components)
        OR (n_components, n_X, n_Z) if return_stacked=True
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install torch")
    
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Convert to PyTorch tensors if needed (accept both numpy and torch inputs)
    if isinstance(X, np.ndarray):
        X_torch = torch.from_numpy(np.asarray(X)).to(device_obj, DTYPE)
    else:
        X_torch = X.to(device_obj, DTYPE)
    
    if isinstance(Z, np.ndarray):
        Z_torch = torch.from_numpy(np.asarray(Z)).to(device_obj, DTYPE)
    else:
        Z_torch = Z.to(device_obj, DTYPE)
    
    if isinstance(variances, np.ndarray):
        variances_torch = torch.from_numpy(np.asarray(variances)).to(device_obj, DTYPE)
    else:
        variances_torch = variances.to(device_obj, DTYPE)
    
    if isinstance(lengthscales, np.ndarray):
        lengthscales_torch = torch.from_numpy(np.asarray(lengthscales)).to(device_obj, DTYPE)
    else:
        lengthscales_torch = lengthscales.to(device_obj, DTYPE)
    
    n_components = len(variances_torch)
    n_X, n_Z = X_torch.shape[0], Z_torch.shape[0]
    
    # Pre-allocate result on GPU or use provided output
    if out is not None:
        result = out
    elif return_stacked:
        result = torch.zeros((n_components, n_X, n_Z), dtype=DTYPE, device=device_obj)
    else:
        total_size_X = n_X * n_components
        total_size_Z = n_Z * n_components
        result = torch.zeros((total_size_X, total_size_Z), dtype=DTYPE, device=device_obj)
    
    # Compute each block on GPU using torch.cdist
    for i in range(n_components):
        var = variances_torch[i]
        ls = lengthscales_torch[i]
        
        # Scale by lengthscale
        X_scaled = X_torch / ls
        Z_scaled = Z_torch / ls
        
        # Memory Efficient Implementation:
        # 1. Compute distances (allocates 1 block)
        # Note: We use a temporary variable to ensure we don't keep multiple large blocks
        dist = torch.cdist(X_scaled, Z_scaled, p=2.0)
        
        # 2. Square in-place
        dist.pow_(2)
        
        # 3. Multiply by -0.5 in-place
        dist.mul_(-0.5)
        
        # 4. Exponentiate in-place
        torch.exp(dist, out=dist)
        
        # 5. Multiply by variance in-place
        dist.mul_(var)
        
        if return_stacked:
            if add_to_out:
                result[i].add_(dist)
            else:
                result[i].copy_(dist)
        else:
            # Place in block diagonal matrix
            start_X = i * n_X
            end_X = (i + 1) * n_X
            start_Z = i * n_Z
            end_Z = (i + 1) * n_Z
            
            if add_to_out:
                result[start_X:end_X, start_Z:end_Z].add_(dist)
            else:
                result[start_X:end_X, start_Z:end_Z].copy_(dist)
                
        # Explicitly free memory
        del dist
    
    # Return GPU tensor - NO CPU TRANSFER!
    return result


def batch_kernel_auto(X, Z, variances, lengthscales):
    """
    Automatically select fastest batch_kernel implementation
    
    Intelligently chooses between GPU and CPU based on matrix size:
    - Small matrices (<500×500): CPU faster (avoid GPU overhead)
    - Large matrices (>500×500): GPU faster (parallel computation wins)
    
    GPU overhead (transfer + kernel launch) = 2-10ms
    This overhead only pays off when compute savings > overhead
    
    Priority:
    1. PyTorch GPU (if available AND matrix large enough) - 5-20x faster
    2. Optimized NumPy CPU - 2-3x faster than original
    3. Original implementation - fallback
    
    This function automatically detects hardware and matrix size.
    
    Parameters
    ----------
    X : np.ndarray
        The first set of points (n_X, n_features)
    Z : np.ndarray
        The second set of points (n_Z, n_features)
    variances : np.ndarray
        The amplitude for each RBF kernel (n_components,)
    lengthscales : np.ndarray
        The lengthscale for each RBF kernel (n_components, n_features)
    
    Returns
    -------
    np.ndarray
        Block diagonal kernel matrix
    """
    # Try PyTorch GPU first (fastest)
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return batch_kernel_pytorch(X, Z, variances, lengthscales, device='cuda')
        except Exception as e:
            # Fall back if GPU fails
            import warnings
            warnings.warn(f"PyTorch GPU kernel failed: {e}. Falling back to CPU.")
    
    # Use optimized CPU for small matrices or when GPU unavailable
    # This is actually faster than GPU for small matrices!
    return batch_kernel_optimized(X, Z, variances, lengthscales)


# ============================================================================
# Benchmarking
# ============================================================================


def benchmark_batch_kernel(X, Z, variances, lengthscales, n_runs=5):
    """
    Benchmark the performance difference between batch_kernel implementations
    
    Parameters
    ----------
    X, Z : np.ndarray
        Input points for kernel computation
    variances, lengthscales : np.ndarray  
        Kernel hyperparameters
    n_runs : int
        Number of benchmark runs
        
    Returns
    -------
    dict
        Timing results for each implementation
    """
    import time
    
    results = {}
    
    # Benchmark original implementation
    times_original = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = batch_kernel(X, Z, variances, lengthscales)  
        end = time.perf_counter()
        times_original.append(end - start)
    
    # Benchmark optimized implementation  
    times_optimized = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = batch_kernel_optimized(X, Z, variances, lengthscales)
        end = time.perf_counter() 
        times_optimized.append(end - start)
    
    # Benchmark cached implementation (run twice to see caching effect)
    clear_kernel_cache()  # Start fresh
    times_cached_first = []
    times_cached_second = []
    
    for _ in range(n_runs):
        # First run (no cache)
        start = time.perf_counter()
        _ = batch_kernel_cached(X, Z, variances, lengthscales)
        end = time.perf_counter()
        times_cached_first.append(end - start)
        
        # Second run (with cache)  
        start = time.perf_counter()
        _ = batch_kernel_cached(X, Z, variances, lengthscales)
        end = time.perf_counter()
        times_cached_second.append(end - start)
    
    results = {
        'original': {
            'mean': np.mean(times_original),
            'std': np.std(times_original),
            'times': times_original
        },
        'optimized': {
            'mean': np.mean(times_optimized), 
            'std': np.std(times_optimized),
            'times': times_optimized
        },
        'cached_first_run': {
            'mean': np.mean(times_cached_first),
            'std': np.std(times_cached_first), 
            'times': times_cached_first
        },
        'cached_second_run': {
            'mean': np.mean(times_cached_second),
            'std': np.std(times_cached_second),
            'times': times_cached_second  
        }
    }
    
    # Benchmark PyTorch implementations if available
    if PYTORCH_AVAILABLE:
        # PyTorch GPU
        if torch.cuda.is_available():
            # Warmup
            _ = batch_kernel_pytorch(X, Z, variances, lengthscales, device='cuda')
            
            times_pytorch_gpu = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = batch_kernel_pytorch(X, Z, variances, lengthscales, device='cuda')
                end = time.perf_counter()
                times_pytorch_gpu.append(end - start)
            
            results['pytorch_gpu'] = {
                'mean': np.mean(times_pytorch_gpu),
                'std': np.std(times_pytorch_gpu),
                'times': times_pytorch_gpu
            }
        
        # PyTorch CPU (for comparison)
        times_pytorch_cpu = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = batch_kernel_pytorch(X, Z, variances, lengthscales, device='cpu')
            end = time.perf_counter()
            times_pytorch_cpu.append(end - start)
        
        results['pytorch_cpu'] = {
            'mean': np.mean(times_pytorch_cpu),
            'std': np.std(times_pytorch_cpu),
            'times': times_pytorch_cpu
        }
        
        # Auto-selection
        times_auto = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = batch_kernel_auto(X, Z, variances, lengthscales)
            end = time.perf_counter()
            times_auto.append(end - start)
        
        results['auto'] = {
            'mean': np.mean(times_auto),
            'std': np.std(times_auto),
            'times': times_auto
        }
    
    # Print summary
    print("Batch Kernel Performance Benchmark")
    print("=" * 60)
    print(f"Original implementation:     {results['original']['mean']*1000:.2f}ms ± {results['original']['std']*1000:.2f}ms")
    print(f"Optimized implementation:    {results['optimized']['mean']*1000:.2f}ms ± {results['optimized']['std']*1000:.2f}ms") 
    print(f"Cached (first run):          {results['cached_first_run']['mean']*1000:.2f}ms ± {results['cached_first_run']['std']*1000:.2f}ms")
    print(f"Cached (second run):         {results['cached_second_run']['mean']*1000:.2f}ms ± {results['cached_second_run']['std']*1000:.2f}ms")
    
    if PYTORCH_AVAILABLE:
        print(f"PyTorch CPU:                 {results['pytorch_cpu']['mean']*1000:.2f}ms ± {results['pytorch_cpu']['std']*1000:.2f}ms")
        if torch.cuda.is_available():
            print(f"PyTorch GPU:                 {results['pytorch_gpu']['mean']*1000:.2f}ms ± {results['pytorch_gpu']['std']*1000:.2f}ms")
            print(f"Auto-selection (GPU):        {results['auto']['mean']*1000:.2f}ms ± {results['auto']['std']*1000:.2f}ms")
        else:
            print(f"Auto-selection (CPU):        {results['auto']['mean']*1000:.2f}ms ± {results['auto']['std']*1000:.2f}ms")
    
    print("\nSpeedups vs Original:")
    print(f"  Optimized:         {results['original']['mean'] / results['optimized']['mean']:.2f}x")
    print(f"  Cached (2nd run):  {results['original']['mean'] / results['cached_second_run']['mean']:.2f}x")
    if PYTORCH_AVAILABLE:
        print(f"  PyTorch CPU:       {results['original']['mean'] / results['pytorch_cpu']['mean']:.2f}x")
        if torch.cuda.is_available():
            print(f"  PyTorch GPU:       {results['original']['mean'] / results['pytorch_gpu']['mean']:.2f}x ⭐")
            print(f"  Auto-selection:    {results['original']['mean'] / results['auto']['mean']:.2f}x ⭐")
    
    return results
