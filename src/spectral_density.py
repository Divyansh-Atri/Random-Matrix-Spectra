"""
Spectral Density Calculations

Computing empirical and theoretical spectral densities.
The spectral density is the distribution of eigenvalues!

Author: Divyansh Atri
"""

import numpy as np
from scipy import stats


def empirical_density(eigenvalues, bins=50, method='histogram'):
    """
    Compute empirical spectral density from eigenvalues.
    
    The spectral density ρ(λ) tells us how eigenvalues are distributed.
    I can estimate it using histograms or kernel density estimation.
    
    Args:
        eigenvalues: Array of eigenvalues
        bins: Number of bins for histogram (if method='histogram')
        method: 'histogram' or 'kde'
    
    Returns:
        (x, density) where x are the points and density are the values
    """
    if method == 'histogram':
        # Use histogram with normalization
        counts, bin_edges = np.histogram(eigenvalues, bins=bins, density=True)
        
        # Use bin centers as x values
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        density = counts
        
    elif method == 'kde':
        # Kernel Density Estimation - smoother but slower
        kde = stats.gaussian_kde(eigenvalues)
        
        # Evaluate on a grid
        x_min, x_max = np.min(eigenvalues), np.max(eigenvalues)
        margin = (x_max - x_min) * 0.1
        x = np.linspace(x_min - margin, x_max + margin, 200)
        density = kde(x)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return x, density


def wigner_semicircle(x, radius=2):
    """
    Theoretical Wigner semicircle distribution.
    
    For GOE/GUE matrices with my normalization:
    ρ(λ) = (1/2π) * sqrt(4 - λ²)  for |λ| ≤ 2
    
    This is one of the most beautiful results in random matrix theory!
    
    Args:
        x: Points to evaluate density at
        radius: Radius of semicircle (default 2 for standard normalization)
    
    Returns:
        Density values at points x
    """
    x = np.asarray(x)
    density = np.zeros_like(x)
    
    # Only nonzero inside [-radius, radius]
    mask = np.abs(x) <= radius
    
    # The famous formula!
    density[mask] = (1 / (2 * np.pi)) * np.sqrt(radius**2 - x[mask]**2)
    
    return density


def marchenko_pastur(x, gamma, sigma=1):
    """
    Theoretical Marchenko-Pastur distribution.
    
    For Wishart matrices W = (1/n) X^T X where X is n × p:
    γ = p/n is the aspect ratio
    
    The density is:
    ρ(λ) = (1/2πγσ²λ) * sqrt((λ+ - λ)(λ - λ-))
    where λ± = σ²(1 ± sqrt(γ))²
    
    Args:
        x: Points to evaluate density at
        gamma: Aspect ratio p/n
        sigma: Variance of entries (default 1)
    
    Returns:
        Density values at points x
    """
    x = np.asarray(x)
    density = np.zeros_like(x)
    
    # Support endpoints
    lambda_minus = sigma**2 * (1 - np.sqrt(gamma))**2
    lambda_plus = sigma**2 * (1 + np.sqrt(gamma))**2
    
    # Only nonzero in [λ-, λ+]
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    
    # The Marchenko-Pastur formula
    # Being careful with numerical stability here
    numerator = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus))
    denominator = 2 * np.pi * gamma * sigma**2 * x[mask]
    
    density[mask] = numerator / denominator
    
    return density


def compare_densities(eigenvalues, theoretical_density, x_range=None, bins=50):
    """
    Compare empirical vs theoretical spectral density.
    
    Returns both densities on the same x-grid for easy plotting.
    
    Args:
        eigenvalues: Empirical eigenvalues
        theoretical_density: Function(x) -> density
        x_range: Tuple (x_min, x_max) for evaluation grid
        bins: Number of bins for empirical density
    
    Returns:
        Dictionary with 'x', 'empirical', 'theoretical' arrays
    """
    # Compute empirical density
    x_emp, rho_emp = empirical_density(eigenvalues, bins=bins, method='histogram')
    
    # Determine x range
    if x_range is None:
        x_min = min(np.min(eigenvalues), np.min(x_emp))
        x_max = max(np.max(eigenvalues), np.max(x_emp))
        margin = (x_max - x_min) * 0.1
        x_range = (x_min - margin, x_max + margin)
    
    # Create common x grid
    x = np.linspace(x_range[0], x_range[1], 200)
    
    # Evaluate theoretical density
    rho_theory = theoretical_density(x)
    
    return {
        'x': x,
        'empirical_x': x_emp,
        'empirical': rho_emp,
        'theoretical': rho_theory,
    }


def kolmogorov_smirnov_distance(eigenvalues, theoretical_cdf):
    """
    Compute Kolmogorov-Smirnov distance between empirical and theoretical distributions.
    
    This gives me a quantitative measure of how well the empirical
    distribution matches theory.
    
    Args:
        eigenvalues: Empirical eigenvalues
        theoretical_cdf: Function(x) -> cumulative probability
    
    Returns:
        KS statistic (supremum of |F_emp - F_theory|)
    """
    # Sort eigenvalues
    eigs_sorted = np.sort(eigenvalues)
    n = len(eigs_sorted)
    
    # Empirical CDF: F(x) = (# eigenvalues ≤ x) / n
    empirical_cdf = np.arange(1, n + 1) / n
    
    # Theoretical CDF evaluated at eigenvalue points
    theoretical_cdf_values = theoretical_cdf(eigs_sorted)
    
    # KS statistic: max absolute difference
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf_values))
    
    return ks_stat


def integrated_squared_error(empirical_x, empirical_density, theoretical_density):
    """
    Compute integrated squared error between densities.
    
    ISE = ∫ (ρ_emp - ρ_theory)² dx
    
    This is another way to quantify deviation from theory.
    
    Args:
        empirical_x: x points for empirical density
        empirical_density: Empirical density values
        theoretical_density: Function(x) -> theoretical density
    
    Returns:
        Integrated squared error
    """
    # Evaluate theoretical density at empirical points
    rho_theory = theoretical_density(empirical_x)
    
    # Compute squared differences
    squared_diff = (empirical_density - rho_theory)**2
    
    # Integrate using trapezoidal rule
    ise = np.trapz(squared_diff, empirical_x)
    
    return ise


if __name__ == "__main__":
    # Test with a GOE matrix
    print("Testing spectral density calculations...")
    
    from matrix_generators import generate_goe_matrix
    from eigenvalue_tools import compute_eigenvalues
    
    print("\n=== Testing Wigner Semicircle ===")
    matrix = generate_goe_matrix(1000)
    eigenvalues = compute_eigenvalues(matrix)
    
    # Compare with Wigner semicircle
    comparison = compare_densities(eigenvalues, wigner_semicircle)
    
    print(f"Eigenvalue range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    print(f"Expected range: [-2, 2]")
    
    # Check normalization
    theory_integral = np.trapz(comparison['theoretical'], comparison['x'])
    emp_integral = np.trapz(comparison['empirical'], comparison['empirical_x'])
    print(f"Theoretical density integrates to: {theory_integral:.4f}")
    print(f"Empirical density integrates to: {emp_integral:.4f}")
    
    print("\n=== Testing Marchenko-Pastur ===")
    from matrix_generators import generate_wishart_matrix
    
    n, p = 1000, 500
    gamma = p / n
    wishart = generate_wishart_matrix(n, p)
    eigenvalues_wishart = compute_eigenvalues(wishart)
    
    mp_density = lambda x: marchenko_pastur(x, gamma)
    comparison_mp = compare_densities(eigenvalues_wishart, mp_density)
    
    print(f"Aspect ratio γ: {gamma:.3f}")
    print(f"Eigenvalue range: [{eigenvalues_wishart.min():.3f}, {eigenvalues_wishart.max():.3f}]")
    
    # Expected support
    lambda_minus = (1 - np.sqrt(gamma))**2
    lambda_plus = (1 + np.sqrt(gamma))**2
    print(f"Expected range: [{lambda_minus:.3f}, {lambda_plus:.3f}]")
    
    print("\nAll tests passed!")
