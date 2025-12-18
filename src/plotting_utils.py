"""
Plotting Utilities

Helper functions for visualizing eigenvalue distributions and spectral densities.
I keep these separate to avoid cluttering the main analysis code.

Author: Divyansh Atri
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_eigenvalue_histogram(eigenvalues, bins=50, theoretical_density=None, 
                               title="Eigenvalue Distribution", figsize=(10, 6)):
    """
    Plot histogram of eigenvalues with optional theoretical overlay.
    
    This is my go-to visualization for checking if empirical results
    match theoretical predictions.
    
    Args:
        eigenvalues: Array of eigenvalues
        bins: Number of bins
        theoretical_density: Optional function(x) for theoretical density
        title: Plot title
        figsize: Figure size
    
    Returns:
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    counts, bin_edges, patches = ax.hist(eigenvalues, bins=bins, density=True, 
                                          alpha=0.6, color='steelblue', 
                                          edgecolor='black', label='Empirical')
    
    # Overlay theoretical density if provided
    if theoretical_density is not None:
        x = np.linspace(eigenvalues.min(), eigenvalues.max(), 500)
        y = theoretical_density(x)
        ax.plot(x, y, 'r-', linewidth=2, label='Theoretical')
    
    ax.set_xlabel('Eigenvalue λ', fontsize=12)
    ax.set_ylabel('Density ρ(λ)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_density_comparison(x_emp, rho_emp, x_theory, rho_theory,
                            title="Spectral Density Comparison", figsize=(10, 6)):
    """
    Plot empirical vs theoretical densities side by side.
    
    Args:
        x_emp, rho_emp: Empirical density
        x_theory, rho_theory: Theoretical density
        title: Plot title
        figsize: Figure size
    
    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot both densities
    ax.plot(x_emp, rho_emp, 'o-', color='steelblue', alpha=0.6, 
            markersize=4, label='Empirical', linewidth=1.5)
    ax.plot(x_theory, rho_theory, '-', color='red', 
            linewidth=2.5, label='Theoretical')
    
    ax.set_xlabel('Eigenvalue λ', fontsize=12)
    ax.set_ylabel('Density ρ(λ)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_spacing_distribution(spacings, bins=30, title="Nearest-Neighbor Spacings",
                              figsize=(10, 6)):
    """
    Plot distribution of eigenvalue spacings.
    
    For GOE, spacings follow the Wigner surmise (approximately).
    This is key for studying universality!
    
    Args:
        spacings: Array of spacings
        bins: Number of bins
        title: Plot title
        figsize: Figure size
    
    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram of spacings
    ax.hist(spacings, bins=bins, density=True, alpha=0.6, 
            color='forestgreen', edgecolor='black', label='Empirical')
    
    # Wigner surmise for GOE: P(s) = (π/2)s exp(-πs²/4)
    s = np.linspace(0, spacings.max(), 500)
    wigner_surmise = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
    ax.plot(s, wigner_surmise, 'r-', linewidth=2.5, label='Wigner Surmise (GOE)')
    
    # Poisson (no repulsion): P(s) = exp(-s)
    poisson = np.exp(-s)
    ax.plot(s, poisson, 'b--', linewidth=2, label='Poisson (random)')
    
    ax.set_xlabel('Spacing s', fontsize=12)
    ax.set_ylabel('Probability P(s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_convergence(sizes, deviations, xlabel="Matrix Size n", 
                    ylabel="Deviation from Theory", 
                    title="Convergence to Theoretical Distribution",
                    figsize=(10, 6), loglog=True):
    """
    Plot how deviation from theory decreases with matrix size.
    
    I use this to study finite-size effects.
    
    Args:
        sizes: Array of matrix sizes
        deviations: Corresponding deviation metrics
        xlabel, ylabel, title: Labels
        figsize: Figure size
        loglog: Use log-log scale
    
    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if loglog:
        ax.loglog(sizes, deviations, 'o-', color='purple', 
                 markersize=8, linewidth=2, label='Measured')
        
        # Add power law reference line
        # Typically expect ~ 1/sqrt(n) convergence
        reference = deviations[0] * (sizes[0] / sizes)**0.5
        ax.loglog(sizes, reference, '--', color='gray', 
                 linewidth=1.5, label='1/√n reference')
    else:
        ax.plot(sizes, deviations, 'o-', color='purple', 
               markersize=8, linewidth=2, label='Measured')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both' if loglog else 'major')
    
    plt.tight_layout()
    return fig, ax


def plot_eigenvalues_2d(eigenvalues_real, eigenvalues_imag=None,
                       title="Eigenvalue Distribution in Complex Plane",
                       figsize=(8, 8)):
    """
    Plot eigenvalues in the complex plane.
    
    For non-Hermitian matrices, eigenvalues can be complex.
    Even for Hermitian, it's nice to verify they're all real!
    
    Args:
        eigenvalues_real: Real parts (or all eigenvalues if real)
        eigenvalues_imag: Imaginary parts (None if all real)
        title: Plot title
        figsize: Figure size
    
    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if eigenvalues_imag is None:
        # All eigenvalues are real - plot on real axis
        ax.scatter(eigenvalues_real, np.zeros_like(eigenvalues_real), 
                  alpha=0.5, s=10, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.5)
    else:
        # Complex eigenvalues
        ax.scatter(eigenvalues_real, eigenvalues_imag, 
                  alpha=0.5, s=10, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
    
    ax.set_xlabel('Re(λ)', fontsize=12)
    ax.set_ylabel('Im(λ)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, ax


def save_figure(fig, filename, dpi=150):
    """
    Save figure to file.
    
    Just a convenience wrapper to ensure consistent settings.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filename}")


if __name__ == "__main__":
    # Quick test
    print("Testing plotting utilities...")
    
    # Generate some test data
    from matrix_generators import generate_goe_matrix
    from eigenvalue_tools import compute_eigenvalues, nearest_neighbor_spacings
    from spectral_density import wigner_semicircle
    
    matrix = generate_goe_matrix(500)
    eigenvalues = compute_eigenvalues(matrix)
    
    # Test histogram plot
    fig1, _ = plot_eigenvalue_histogram(eigenvalues, 
                                       theoretical_density=wigner_semicircle,
                                       title="Test: GOE Eigenvalues")
    plt.close(fig1)
    
    # Test spacing plot
    spacings = nearest_neighbor_spacings(eigenvalues)
    from eigenvalue_tools import unfolded_spacings
    spacings_unfolded = unfolded_spacings(eigenvalues)
    
    fig2, _ = plot_spacing_distribution(spacings_unfolded)
    plt.close(fig2)
    
    print("All plotting tests passed!")
