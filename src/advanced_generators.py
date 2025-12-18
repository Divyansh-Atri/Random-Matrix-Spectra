"""
Advanced Matrix Generators

Extensions to matrix_generators.py with:
- Heavy-tailed entry distributions
- Correlated random matrices
- Band matrices and structured ensembles

Author: Divyansh Atri
"""

import numpy as np
from scipy import linalg


def generate_heavy_tailed_goe(n, distribution='cauchy', df=3):
    """
    Generate GOE matrix with heavy-tailed entries.
    
    Heavy-tailed distributions violate the standard RMT assumptions!
    This lets me test the limits of universality.
    
    Args:
        n: Matrix size
        distribution: 'cauchy', 'student-t', or 'pareto'
        df: Degrees of freedom (for Student-t)
    
    Returns:
        Symmetric n x n matrix with heavy-tailed entries
    """
    if distribution == 'cauchy':
        # Cauchy has no finite variance - extreme heavy tails!
        A = np.random.standard_cauchy((n, n))
    elif distribution == 'student-t':
        # Student-t with small df has heavy tails
        A = np.random.standard_t(df, (n, n))
    elif distribution == 'pareto':
        # Pareto distribution: power law tails
        alpha = 2.5  # Shape parameter (smaller = heavier tails)
        A = (np.random.pareto(alpha, (n, n)) + 1)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Make symmetric
    H = (A + A.T) / np.sqrt(2)
    
    # Normalization is tricky for heavy-tailed!
    # For Cauchy, variance is infinite, so I'll use a robust scaling
    # Scale by median absolute deviation instead
    mad = np.median(np.abs(H - np.median(H)))
    H = H / (mad * np.sqrt(n))
    
    return H


def generate_band_matrix(n, bandwidth):
    """
    Generate a band matrix - only entries near diagonal are nonzero.
    
    Band matrices have different spectral properties than full random matrices.
    The eigenvalue distribution depends on the bandwidth!
    
    Args:
        n: Matrix size
        bandwidth: Number of diagonals (bandwidth=1 is tridiagonal)
    
    Returns:
        Symmetric band matrix
    """
    # Start with zeros
    A = np.zeros((n, n))
    
    # Fill bands with random entries
    for k in range(-bandwidth, bandwidth + 1):
        # k-th diagonal
        diag_length = n - abs(k)
        if k >= 0:
            # Upper diagonal
            A[np.arange(diag_length), np.arange(diag_length) + k] = np.random.randn(diag_length)
        else:
            # Lower diagonal
            A[np.arange(diag_length) - k, np.arange(diag_length)] = np.random.randn(diag_length)
    
    # Make symmetric
    H = (A + A.T) / 2
    
    # Normalize
    H = H / np.sqrt(n)
    
    return H


def generate_correlated_wishart(n, p, correlation_matrix):
    """
    Generate Wishart matrix from correlated samples.
    
    Instead of i.i.d. samples, the features are correlated.
    This is more realistic for real data!
    
    Args:
        n: Number of samples
        p: Number of features
        correlation_matrix: p x p correlation matrix (must be positive definite)
    
    Returns:
        Sample covariance matrix with correlation structure
    """
    # Generate correlated data using Cholesky decomposition
    # If Σ is the correlation matrix, X = Z * L where L*L^T = Σ
    
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Generate i.i.d. standard normal
    Z = np.random.randn(n, p)
    
    # Apply correlation structure
    X = Z @ L.T
    
    # Sample covariance
    W = (X.T @ X) / n
    
    return W


def generate_toeplitz_matrix(n):
    """
    Generate a Toeplitz random matrix.
    
    Toeplitz matrices have constant diagonals: A[i,j] depends only on |i-j|.
    These appear in time series analysis!
    
    Args:
        n: Matrix size
    
    Returns:
        Symmetric Toeplitz matrix
    """
    # Generate the first row (determines entire matrix)
    first_row = np.random.randn(n)
    
    # Build Toeplitz matrix
    from scipy.linalg import toeplitz
    H = toeplitz(first_row)
    
    # Make it symmetric (average with transpose)
    H = (H + H.T) / 2
    
    # Normalize
    H = H / np.sqrt(n)
    
    return H


def generate_block_matrix(n, num_blocks, block_correlation=0.8):
    """
    Generate a block-structured random matrix.
    
    Entries within blocks are correlated, between blocks are independent.
    This models community structure!
    
    Args:
        n: Matrix size
        num_blocks: Number of blocks
        block_correlation: Correlation within blocks
    
    Returns:
        Block-structured symmetric matrix
    """
    block_size = n // num_blocks
    A = np.zeros((n, n))
    
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size if b < num_blocks - 1 else n
        size = end - start
        
        # Generate correlated block
        # Use correlation matrix with off-diagonal = block_correlation
        corr = np.eye(size) * (1 - block_correlation) + block_correlation
        
        # Generate correlated entries
        L = np.linalg.cholesky(corr)
        Z = np.random.randn(size, size)
        block = Z @ L.T
        
        A[start:end, start:end] = block
    
    # Make symmetric
    H = (A + A.T) / 2
    
    # Normalize
    H = H / np.sqrt(n)
    
    return H


def generate_sparse_random_matrix(n, density=0.1):
    """
    Generate a sparse random matrix.
    
    Only a fraction 'density' of entries are nonzero.
    Sparse matrices have very different eigenvalue distributions!
    
    Args:
        n: Matrix size
        density: Fraction of nonzero entries
    
    Returns:
        Sparse symmetric matrix
    """
    from scipy.sparse import random as sparse_random
    
    # Generate sparse matrix
    A = sparse_random(n, n, density=density, format='csr', 
                      data_rvs=np.random.randn).toarray()
    
    # Make symmetric
    H = (A + A.T) / 2
    
    # Normalize
    H = H / np.sqrt(n * density)  # Account for sparsity
    
    return H


if __name__ == "__main__":
    print("Testing advanced matrix generators...")
    
    # Test heavy-tailed
    print("\n=== Heavy-Tailed GOE ===")
    H_cauchy = generate_heavy_tailed_goe(100, 'cauchy')
    print(f"Cauchy GOE shape: {H_cauchy.shape}")
    print(f"Symmetric: {np.allclose(H_cauchy, H_cauchy.T)}")
    
    H_student = generate_heavy_tailed_goe(100, 'student-t', df=3)
    print(f"Student-t GOE shape: {H_student.shape}")
    
    # Test band matrix
    print("\n=== Band Matrix ===")
    H_band = generate_band_matrix(100, bandwidth=5)
    print(f"Band matrix shape: {H_band.shape}")
    print(f"Symmetric: {np.allclose(H_band, H_band.T)}")
    
    # Test correlated Wishart
    print("\n=== Correlated Wishart ===")
    p = 50
    # Create a correlation matrix (exponential decay)
    corr = np.array([[0.9**abs(i-j) for j in range(p)] for i in range(p)])
    W_corr = generate_correlated_wishart(200, p, corr)
    print(f"Correlated Wishart shape: {W_corr.shape}")
    print(f"Symmetric: {np.allclose(W_corr, W_corr.T)}")
    
    # Test Toeplitz
    print("\n=== Toeplitz Matrix ===")
    H_toep = generate_toeplitz_matrix(100)
    print(f"Toeplitz shape: {H_toep.shape}")
    
    # Test block matrix
    print("\n=== Block Matrix ===")
    H_block = generate_block_matrix(100, num_blocks=4)
    print(f"Block matrix shape: {H_block.shape}")
    
    # Test sparse
    print("\n=== Sparse Matrix ===")
    H_sparse = generate_sparse_random_matrix(100, density=0.2)
    print(f"Sparse matrix shape: {H_sparse.shape}")
    print(f"Sparsity: {np.sum(H_sparse == 0) / H_sparse.size:.2%}")
    
    print("\n✓ All advanced generators working!")
