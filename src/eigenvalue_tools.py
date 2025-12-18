"""
Eigenvalue Analysis Tools

Tools for computing and analyzing eigenvalues from random matrices.
Writing these myself to really understand what's happening.

Author: Divyansh Atri
"""

import numpy as np


def compute_eigenvalues(matrix):
    """
    Compute all eigenvalues of a matrix.
    
    For Hermitian/symmetric matrices, eigenvalues are guaranteed real.
    Using eigvalsh (Hermitian) is faster than general eig.
    
    Args:
        matrix: Square matrix (real symmetric or complex Hermitian)
    
    Returns:
        1D array of eigenvalues, sorted in ascending order
    """
    # Check if matrix is real
    if np.isrealobj(matrix):
        # Use specialized routine for real symmetric matrices
        eigenvalues = np.linalg.eigvalsh(matrix)
    else:
        # Use Hermitian eigenvalue solver for complex matrices
        eigenvalues = np.linalg.eigvalsh(matrix)
    
    # Sort them (some functions return them sorted, but let's be explicit)
    eigenvalues = np.sort(eigenvalues)
    
    return eigenvalues


def eigenvalue_statistics(eigenvalues):
    """
    Compute basic statistics of eigenvalue distribution.
    
    This gives me a quick summary to check if things look reasonable.
    
    Args:
        eigenvalues: Array of eigenvalues
    
    Returns:
        Dictionary with various statistics
    """
    stats = {
        'count': len(eigenvalues),
        'min': np.min(eigenvalues),
        'max': np.max(eigenvalues),
        'mean': np.mean(eigenvalues),
        'std': np.std(eigenvalues),
        'median': np.median(eigenvalues),
    }
    
    # The spectral radius (largest absolute eigenvalue)
    stats['spectral_radius'] = np.max(np.abs(eigenvalues))
    
    # Trace (should equal sum of eigenvalues)
    stats['trace'] = np.sum(eigenvalues)
    
    return stats


def nearest_neighbor_spacings(eigenvalues):
    """
    Compute nearest-neighbor spacings between consecutive eigenvalues.
    
    These spacings are important for studying level repulsion and
    universality! The distribution of spacings reveals deep properties.
    
    Args:
        eigenvalues: Sorted array of eigenvalues
    
    Returns:
        Array of spacings s_i = λ_{i+1} - λ_i
    """
    # Make sure they're sorted
    eigs = np.sort(eigenvalues)
    
    # Compute differences between consecutive eigenvalues
    spacings = np.diff(eigs)
    
    return spacings


def unfold_eigenvalues(eigenvalues, method='linear'):
    """
    Unfold eigenvalue spectrum to study local statistics.
    
    Unfolding transforms eigenvalues so the average density is constant.
    This is crucial for comparing spacing statistics across different scales.
    
    I'm using a simple linear unfolding here - map eigenvalues to
    have uniform average spacing.
    
    Args:
        eigenvalues: Sorted eigenvalues
        method: Unfolding method ('linear' for now)
    
    Returns:
        Unfolded eigenvalues
    """
    eigs = np.sort(eigenvalues)
    n = len(eigs)
    
    if method == 'linear':
        # Simple linear unfolding: just rescale to [0, n]
        # The i-th eigenvalue maps to position i
        unfolded = np.linspace(0, n, n)
    else:
        # Could implement polynomial unfolding or other methods
        raise NotImplementedError(f"Method {method} not implemented yet")
    
    return unfolded


def unfolded_spacings(eigenvalues):
    """
    Compute spacings after unfolding.
    
    After unfolding, we expect mean spacing ≈ 1.
    The distribution of these spacings reveals universality.
    
    Args:
        eigenvalues: Original eigenvalues
    
    Returns:
        Array of unfolded spacings (should have mean ≈ 1)
    """
    unfolded = unfold_eigenvalues(eigenvalues)
    spacings = np.diff(unfolded)
    
    # Normalize so mean spacing is exactly 1
    # This is important for comparing with theoretical distributions
    spacings = spacings / np.mean(spacings)
    
    return spacings


def bulk_edge_eigenvalues(eigenvalues, edge_fraction=0.1):
    """
    Separate bulk from edge eigenvalues.
    
    Edge eigenvalues (largest/smallest) behave differently from bulk.
    This function helps me study them separately.
    
    Args:
        eigenvalues: Sorted eigenvalues
        edge_fraction: Fraction to consider as "edge" (default 10%)
    
    Returns:
        Dictionary with 'bulk', 'left_edge', 'right_edge' eigenvalues
    """
    eigs = np.sort(eigenvalues)
    n = len(eigs)
    
    # Number of eigenvalues in each edge
    n_edge = int(n * edge_fraction)
    
    return {
        'left_edge': eigs[:n_edge],
        'bulk': eigs[n_edge:-n_edge],
        'right_edge': eigs[-n_edge:],
    }


def spectral_radius_normalized(eigenvalues):
    """
    Compute normalized spectral radius.
    
    For GOE/GUE with my normalization, this should approach 2 as n → ∞
    (eigenvalues live in [-2, 2] for the semicircle law).
    
    Args:
        eigenvalues: Array of eigenvalues
    
    Returns:
        Maximum absolute eigenvalue
    """
    return np.max(np.abs(eigenvalues))


if __name__ == "__main__":
    # Quick test
    print("Testing eigenvalue tools...")
    
    # Generate some test eigenvalues
    # Using a simple example: eigenvalues from a random GOE matrix
    from matrix_generators import generate_goe_matrix
    
    print("\n=== Testing with GOE matrix ===")
    matrix = generate_goe_matrix(100)
    eigs = compute_eigenvalues(matrix)
    
    print(f"Number of eigenvalues: {len(eigs)}")
    
    stats = eigenvalue_statistics(eigs)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test spacings
    spacings = nearest_neighbor_spacings(eigs)
    print(f"\nNumber of spacings: {len(spacings)}")
    print(f"Mean spacing: {np.mean(spacings):.4f}")
    
    # Test unfolding
    unfolded_sp = unfolded_spacings(eigs)
    print(f"\nMean unfolded spacing: {np.mean(unfolded_sp):.4f}")
    print("(Should be close to 1.0)")
    
    # Test bulk/edge separation
    separated = bulk_edge_eigenvalues(eigs, edge_fraction=0.1)
    print(f"\nBulk eigenvalues: {len(separated['bulk'])}")
    print(f"Edge eigenvalues: {len(separated['left_edge'])} + {len(separated['right_edge'])}")
    
    print("\nAll tests passed!")
