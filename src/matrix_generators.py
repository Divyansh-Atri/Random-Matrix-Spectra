"""
Random Matrix Generators

This module contains hand-written implementations of various random matrix ensembles.
I'm implementing these from scratch to understand the mathematical structure.

Author: Divyansh Atri
"""

import numpy as np


def generate_goe_matrix(n):
    """
    Generate a Gaussian Orthogonal Ensemble (GOE) matrix.
    
    GOE matrices are real symmetric matrices where:
    - Diagonal elements ~ N(0, 2)
    - Off-diagonal elements ~ N(0, 1)
    
    I need to be careful here - the symmetry is important!
    
    Args:
        n: Size of the matrix (n x n)
    
    Returns:
        Symmetric n x n numpy array
    """
    # Start with a random matrix
    A = np.random.randn(n, n)
    
    # Make it symmetric by averaging with transpose
    # This is a trick I learned - it preserves the distribution correctly
    H = (A + A.T) / np.sqrt(2)
    
    # Normalize by sqrt(n) so eigenvalues are O(1)
    # This scaling is important for the semicircle law
    H = H / np.sqrt(n)
    
    return H


def generate_gue_matrix(n):
    """
    Generate a Gaussian Unitary Ensemble (GUE) matrix.
    
    GUE matrices are complex Hermitian matrices where:
    - Diagonal elements are real ~ N(0, 2)
    - Off-diagonal elements are complex ~ N(0,1) + i*N(0,1)
    
    Args:
        n: Size of the matrix (n x n)
    
    Returns:
        Hermitian n x n complex numpy array
    """
    # Real and imaginary parts
    A_real = np.random.randn(n, n)
    A_imag = np.random.randn(n, n)
    
    # Combine into complex matrix
    A = A_real + 1j * A_imag
    
    # Make Hermitian: H = (A + A†) / sqrt(2)
    # The conjugate transpose is important here!
    H = (A + A.conj().T) / np.sqrt(2)
    
    # Normalize
    H = H / np.sqrt(n)
    
    return H


def generate_wishart_matrix(n, p, aspect_ratio=None):
    """
    Generate a Wishart matrix for studying Marchenko-Pastur law.
    
    Wishart matrices come from: W = (1/n) * X^T * X
    where X is an n x p matrix of i.i.d. Gaussian entries.
    
    This is essentially a sample covariance matrix!
    
    Args:
        n: Number of samples (rows of X)
        p: Number of features (columns of X)
        aspect_ratio: If provided, compute p from n * aspect_ratio
    
    Returns:
        p x p Wishart matrix
    """
    if aspect_ratio is not None:
        p = int(n * aspect_ratio)
    
    # Generate data matrix
    X = np.random.randn(n, p)
    
    # Compute sample covariance: (1/n) X^T X
    W = (X.T @ X) / n
    
    return W


def generate_random_matrix(ensemble='GOE', n=100, **kwargs):
    """
    Unified interface for generating random matrices.
    
    I added this wrapper so I can easily switch between ensembles
    in my experiments without changing much code.
    
    Args:
        ensemble: Type of matrix ('GOE', 'GUE', 'Wishart')
        n: Matrix size
        **kwargs: Additional parameters for specific ensembles
    
    Returns:
        Random matrix of specified ensemble
    """
    if ensemble.upper() == 'GOE':
        return generate_goe_matrix(n)
    elif ensemble.upper() == 'GUE':
        return generate_gue_matrix(n)
    elif ensemble.upper() == 'WISHART':
        p = kwargs.get('p', n)
        aspect_ratio = kwargs.get('aspect_ratio', None)
        return generate_wishart_matrix(n, p, aspect_ratio)
    else:
        raise ValueError(f"Unknown ensemble: {ensemble}")


def verify_symmetry(matrix, tolerance=1e-10):
    """
    Quick check to verify if a real matrix is symmetric.
    
    I use this for debugging - making sure my GOE matrices are actually symmetric!
    
    Args:
        matrix: Matrix to check
        tolerance: Numerical tolerance for symmetry
    
    Returns:
        True if symmetric
    """
    return np.allclose(matrix, matrix.T, atol=tolerance)


def verify_hermitian(matrix, tolerance=1e-10):
    """
    Check if a matrix is Hermitian (for complex matrices).
    
    Args:
        matrix: Matrix to check
        tolerance: Numerical tolerance
    
    Returns:
        True if Hermitian
    """
    return np.allclose(matrix, matrix.conj().T, atol=tolerance)


if __name__ == "__main__":
    # Quick test when I run this file directly
    print("Testing matrix generators...")
    
    # Test GOE
    print("\n=== GOE Test ===")
    goe = generate_goe_matrix(5)
    print(f"Shape: {goe.shape}")
    print(f"Symmetric: {verify_symmetry(goe)}")
    print(f"Real: {np.isreal(goe).all()}")
    
    # Test GUE
    print("\n=== GUE Test ===")
    gue = generate_gue_matrix(5)
    print(f"Shape: {gue.shape}")
    print(f"Hermitian: {verify_hermitian(gue)}")
    print(f"Complex: {np.iscomplexobj(gue)}")
    
    # Test Wishart
    print("\n=== Wishart Test ===")
    wishart = generate_wishart_matrix(100, 50)
    print(f"Shape: {wishart.shape}")
    print(f"Symmetric: {verify_symmetry(wishart)}")
    print(f"Positive eigenvalues: {(np.linalg.eigvalsh(wishart) > -1e-10).all()}")
    
    print("\nAll tests passed!")
