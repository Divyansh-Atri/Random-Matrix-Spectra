#!/usr/bin/env python3
"""
Generate Example Plots for Random Matrix Spectra Project

This script creates 5 key visualization plots demonstrating:
1. Wigner semicircle convergence
2. Universality in spacing statistics
3. Marchenko-Pastur with varying aspect ratios
4. Finite-size convergence rate
5. Edge fluctuations

Author: Divyansh Atri
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt

from matrix_generators import generate_goe_matrix, generate_gue_matrix, generate_wishart_matrix
from eigenvalue_tools import compute_eigenvalues, unfolded_spacings
from spectral_density import (
    empirical_density, 
    wigner_semicircle, 
    marchenko_pastur,
    integrated_squared_error
)

print("=" * 60)
print("Generating Example Plots for Random Matrix Spectra Project")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Plot 1: Wigner Semicircle Convergence
# ============================================================================
print("\n[1/5] Generating Wigner Semicircle Convergence plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
sizes = [100, 500, 1000, 5000]

for idx, n in enumerate(sizes):
    print(f"  Processing n = {n}...")
    H = generate_goe_matrix(n)
    eigs = compute_eigenvalues(H)
    
    ax = axes[idx]
    ax.hist(eigs, bins=40, density=True, alpha=0.6, 
            color='steelblue', edgecolor='black', label='Empirical')
    
    x = np.linspace(-2.5, 2.5, 500)
    y = wigner_semicircle(x)
    ax.plot(x, y, 'r-', linewidth=2.5, label='Wigner Semicircle')
    
    ax.set_title(f'n = {n}', fontsize=13, fontweight='bold')
    ax.set_xlabel('λ', fontsize=11)
    ax.set_ylabel('ρ(λ)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-3, 3)

plt.suptitle('Wigner Semicircle Law: Convergence with Increasing n', 
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('experiments/wigner_convergence.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: experiments/wigner_convergence.png")
plt.close()

# ============================================================================
# Plot 2: Universality - GOE vs GUE vs Poisson
# ============================================================================
print("\n[2/5] Generating Universality Comparison plot...")

n = 2000  # Reduced from 3000 for faster generation
print(f"  Generating GOE matrix (n={n})...")
H_goe = generate_goe_matrix(n)
eigs_goe = compute_eigenvalues(H_goe)
spacings_goe = unfolded_spacings(eigs_goe)

print(f"  Generating GUE matrix (n={n})...")
H_gue = generate_gue_matrix(n)
eigs_gue = compute_eigenvalues(H_gue)
spacings_gue = unfolded_spacings(eigs_gue)

fig, ax = plt.subplots(figsize=(12, 7))

ax.hist(spacings_goe, bins=40, density=True, alpha=0.4, 
        color='steelblue', edgecolor='black', label='GOE Empirical')
ax.hist(spacings_gue, bins=40, density=True, alpha=0.4, 
        color='forestgreen', edgecolor='black', label='GUE Empirical')

s = np.linspace(0, 4, 500)
P_goe = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
P_gue = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
P_poisson = np.exp(-s)

ax.plot(s, P_goe, '-', color='blue', linewidth=3, label='GOE: P(s) ∝ s·exp(-s²)')
ax.plot(s, P_gue, '-', color='green', linewidth=3, label='GUE: P(s) ∝ s²·exp(-s²)')
ax.plot(s, P_poisson, '--', color='red', linewidth=3, label='Poisson: P(s) = exp(-s)')

ax.set_xlabel('Spacing s', fontsize=13)
ax.set_ylabel('Probability P(s)', fontsize=13)
ax.set_title('Universality: GOE vs GUE vs Poisson Spacing Statistics', 
             fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 4)

plt.tight_layout()
plt.savefig('experiments/universality_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: experiments/universality_comparison.png")
plt.close()

# ============================================================================
# Plot 3: Marchenko-Pastur with Varying Gamma
# ============================================================================
print("\n[3/5] Generating Marchenko-Pastur Varying Gamma plot...")

n = 1500  # Reduced for faster generation
gammas = [0.2, 0.5, 0.8, 1.5]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, gamma in enumerate(gammas):
    p = int(n * gamma)
    print(f"  Processing γ={gamma} (n={n}, p={p})...")
    
    W = generate_wishart_matrix(n, p)
    eigs = compute_eigenvalues(W)
    
    lambda_minus = (1 - np.sqrt(gamma))**2
    lambda_plus = (1 + np.sqrt(gamma))**2
    
    ax = axes[idx]
    ax.hist(eigs, bins=40, density=True, alpha=0.6, 
            color='mediumseagreen', edgecolor='black')
    
    x = np.linspace(max(0, lambda_minus - 0.2), lambda_plus + 0.2, 500)
    y = marchenko_pastur(x, gamma)
    ax.plot(x, y, 'r-', linewidth=2.5)
    
    ax.axvline(lambda_minus, color='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(lambda_plus, color='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax.set_title(f'γ = {gamma} (p/n = {p}/{n})', fontsize=12, fontweight='bold')
    ax.set_xlabel('λ', fontsize=11)
    ax.set_ylabel('ρ(λ)', fontsize=11)
    ax.grid(alpha=0.3)

plt.suptitle('Marchenko-Pastur Law for Different Aspect Ratios', 
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('experiments/marchenko_pastur_varying_gamma.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: experiments/marchenko_pastur_varying_gamma.png")
plt.close()

# ============================================================================
# Plot 4: Finite-Size Convergence
# ============================================================================
print("\n[4/5] Generating Finite-Size Convergence plot...")

test_sizes = np.array([50, 100, 200, 500, 1000, 2000])  # Removed 5000 for speed
ise_values = []

for n in test_sizes:
    print(f"  Processing n={n}...")
    H = generate_goe_matrix(n)
    eigs = compute_eigenvalues(H)
    x_emp, rho_emp = empirical_density(eigs, bins=50)
    ise = integrated_squared_error(x_emp, rho_emp, wigner_semicircle)
    ise_values.append(ise)

ise_values = np.array(ise_values)

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(test_sizes, ise_values, 'o-', color='darkviolet', 
          markersize=10, linewidth=2, label='Measured ISE')

reference = ise_values[3] * np.sqrt(test_sizes[3] / test_sizes)
ax.loglog(test_sizes, reference, '--', color='gray', 
          linewidth=2, label='1/√n reference')

ax.set_xlabel('Matrix Size n', fontsize=13)
ax.set_ylabel('Integrated Squared Error', fontsize=13)
ax.set_title('Convergence Rate to Wigner Semicircle Law', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('experiments/finite_size_convergence.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: experiments/finite_size_convergence.png")
plt.close()

# ============================================================================
# Plot 5: Edge Fluctuations
# ============================================================================
print("\n[5/5] Generating Edge Fluctuations plot...")

n = 800  # Reduced for faster generation
num_trials = 200  # Reduced from 300

max_eigenvalues = []
print(f"  Running {num_trials} trials with n={n}...")

for i in range(num_trials):
    if (i + 1) % 50 == 0:
        print(f"    Trial {i+1}/{num_trials}")
    
    H = generate_goe_matrix(n)
    eigs = compute_eigenvalues(H)
    max_eigenvalues.append(eigs[-1])

max_eigenvalues = np.array(max_eigenvalues)

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(max_eigenvalues, bins=25, density=True, alpha=0.6, 
        color='coral', edgecolor='black')

ax.axvline(2.0, color='red', linestyle='--', linewidth=2, 
           label='Theoretical edge (λ=2)')

mean_max = np.mean(max_eigenvalues)
ax.axvline(mean_max, color='blue', linestyle='-', linewidth=2, 
           label=f'Mean = {mean_max:.4f}')

ax.set_xlabel('Largest Eigenvalue λ_max', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Distribution of Largest Eigenvalue (n={n}, {num_trials} trials)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/edge_fluctuations.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: experiments/edge_fluctuations.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("✓ All plots generated successfully!")
print("=" * 60)
print("\nGenerated files in experiments/:")
print("  1. wigner_convergence.png")
print("  2. universality_comparison.png")
print("  3. marchenko_pastur_varying_gamma.png")
print("  4. finite_size_convergence.png")
print("  5. edge_fluctuations.png")
print("\nSee experiments/PLOTS_DESCRIPTION.md for detailed descriptions.")
print("=" * 60)
