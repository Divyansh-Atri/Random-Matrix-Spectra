# Spectral Theory of Random Matrices: Empirical Study

**Author:** Divyansh Atri  


A hands-on exploration of eigenvalue distributions in high-dimensional random matrices through computational experiments.

---

## What This Project Studies

This repository contains an empirical investigation of **spectral theory of random matrices** - the study of eigenvalue distributions of large random matrices. Through hands-on experiments, I explore three fundamental phenomena:

### 1. Eigenvalue Distributions in High Dimensions

When matrix size grows large (n → ∞), eigenvalues follow universal, deterministic patterns:

- **Wigner Semicircle Law** (GOE/GUE matrices)
- **Marchenko-Pastur Law** (Wishart/sample covariance matrices)

### 2. Universality Phenomena

Local eigenvalue statistics (spacing distributions) are **universal** - they depend only on symmetry class, not on the specific distribution of matrix entries. This robustness makes random matrix theory applicable across diverse fields.

### 3. Finite-Size Effects

Asymptotic laws are beautiful theoretical results, but in practice we always have finite matrices. I study:

- How fast empirical distributions converge to theory
- Where deviations are largest (bulk vs edges)
- What matrix sizes are "large enough" for practical applications

---

## Core Mathematics

### Wigner Semicircle Law

For GOE/GUE matrices H with normalization E[H²ᵢⱼ] = 1/n, the eigenvalue density converges to:

```
ρ(λ) = (1/2π) √(4 - λ²)    for |λ| ≤ 2
```

This describes a **semicircle** centered at 0 with radius 2.

### Marchenko-Pastur Law

For Wishart matrices W = (1/n) X^T X where X is n×p with i.i.d. entries:

```
ρ(λ) = (1/2πγλ) √((λ₊ - λ)(λ - λ₋))
```

where:
- γ = p/n is the aspect ratio
- λ± = (1 ± √γ)²

### Concentration of Measure

Both laws exhibit **concentration** - eigenvalues concentrate in predictable regions as n grows. The support is deterministic, and fluctuations decrease as O(1/√n).

---

## What I Built

### Core Modules (`src/`)

All code is written from scratch in an educational style:

#### `matrix_generators.py`
Hand-written implementations of random matrix ensembles:
- GOE (Gaussian Orthogonal Ensemble): real symmetric
- GUE (Gaussian Unitary Ensemble): complex Hermitian
- Wishart matrices: sample covariance matrices

#### `eigenvalue_tools.py`
Functions for eigenvalue analysis:
- Eigenvalue computation (using optimized symmetric solvers)
- Statistics: min, max, mean, spectral radius
- Nearest-neighbor spacing calculations
- Unfolding transformations for local statistics

#### `spectral_density.py`
Empirical and theoretical spectral densities:
- Empirical density estimation (histograms, KDE)
- Wigner semicircle distribution (analytical)
- Marchenko-Pastur distribution (analytical)
- Comparison utilities and deviation metrics

#### `plotting_utils.py`
Visualization helpers:
- Eigenvalue histograms with theoretical overlays
- Spacing distribution plots
- Convergence studies
- Comparison plots

### Jupyter Notebooks (`notebooks/`)

Five detailed notebooks documenting my exploration:

1. **`01_introduction.ipynb`** - Introduction to random matrices, basic examples
2. **`02_wigner_semicircle.ipynb`** - Deep dive into Wigner law, convergence studies, edge behavior
3. **`03_marchenko_pastur.ipynb`** - Wishart matrices, varying aspect ratios, signal detection
4. **`04_finite_size_effects.ipynb`** - Systematic convergence analysis, error quantification
5. **`05_universality.ipynb`** - Spacing statistics, level repulsion, universality demonstration

---

## Installation and Setup

### Requirements

- Python 3.7+
- NumPy (≥1.21.0)
- SciPy (≥1.7.0)
- Matplotlib (≥3.4.0)
- Jupyter (≥1.0.0)

### Installation

```bash
# Clone repository
cd random-matrix-spectra

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\\Scripts\\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Running Jupyter Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook

# Navigate to notebooks/ directory and open any notebook
```

I recommend running notebooks in order (01 → 05) for a progressive understanding.

### Running Core Modules Directly

Each module has test code that runs when executed directly:

```bash
cd src

# Test matrix generators
python matrix_generators.py

# Test eigenvalue tools
python eigenvalue_tools.py

# Test spectral density calculations
python spectral_density.py

# Test plotting utilities
python plotting_utils.py
```

---

## Key Results and Insights

### 1. Wigner Semicircle (Notebook 02)

- [PASS] Empirical eigenvalue densities converge to semicircle as n → ∞
- [PASS] Convergence rate is O(1/√n) for bulk density
- [PASS] Largest eigenvalue fluctuates near edge λ = 2 (Tracy-Widom regime)
- [PASS] GOE and GUE give identical bulk distributions (universality)

**Practical Insight**: Matrix size n ≥ 1000 provides excellent match with theory.

### 2. Marchenko-Pastur (Notebook 03)

- [PASS] Sample covariance eigenvalues follow Marchenko-Pastur law
- [PASS] Distribution shape changes dramatically with aspect ratio γ = p/n
- [PASS] Eigenvalues beyond [λ₋, λ₊] indicate **signal** (not noise)
- [PASS] Application to PCA: noise threshold is λ₊!

**Practical Insight**: For signal detection in high-dimensional data, eigenvalues above (1 + √(p/n))² are statistically significant.

### 3. Finite-Size Effects (Notebook 04)

- [PASS] Integrated squared error decreases as ~ 1/√n
- [PASS] Deviations are largest at **edges**, not in bulk
- [PASS] Cumulative distribution converges faster than density
- [PASS] Both Wigner and Marchenko-Pastur have similar convergence behavior

**Practical Insight**: 
- n ≥ 500: Good approximation
- n ≥ 1000: Excellent for bulk properties
- n ≥ 5000: Even edges match well

### 4. Universality (Notebook 05)

- [PASS] Nearest-neighbor spacings follow Wigner surmise (GOE: P(s) ∝ s·exp(-s²))
- [PASS] **Level repulsion**: P(s) → 0 as s → 0 (eigenvalues repel!)
- [PASS] GUE has stronger repulsion than GOE (P(s) ∝ s² vs s)
- [PASS] **Spacing statistics are independent of entry distribution!**


**Practical Insight**: Local statistics are robust - modeling assumptions don't matter for large matrices. This explains why random matrix theory has such broad applicability.

---

## Theoretical Background

### Why Study Random Matrices?

Random matrix eigenvalues appear in:

1. **Nuclear Physics**: Energy levels of heavy atomic nuclei (Wigner's original motivation!)
2. **Quantum Chaos**: Distinguishing chaotic vs integrable systems
3. **High-Dimensional Statistics**: Covariance estimation, PCA, signal detection
4. **Number Theory**: Zeros of Riemann zeta function exhibit RMT statistics
5. **Machine Learning**: Generalization, neural network dynamics
6. **Finance**: Portfolio optimization, correlation matrices
7. **Wireless Communications**: Channel capacity, MIMO systems

### Core Concepts

**Empirical Spectral Density**: For matrix H with eigenvalues λ₁, ..., λₙ:
```
ρₙ(λ) = (1/n) Σᵢ δ(λ - λᵢ)
```

As n → ∞, this converges to a deterministic limit ρ(λ).

**Concentration**: Eigenvalues concentrate in predictable regions. For GOE/GUE: [-2, 2]. For Wishart with γ < 1: [λ₋, λ₊] where λ± = (1 ± √γ)².

**Universality**: Local statistics (spacing, correlations) depend only on:
- Symmetry class (real/complex/quaternion)
- NOT on specific entry distributions

This makes results extremely robust!

---

## Deviations from Theory

I don't just show where theory works - I also study **when and where it fails**:

1. **Small matrices** (n < 100): Significant deviations everywhere
2. **Edge behavior**: Finite-size effects persist longer at edges
3. **Tracy-Widom fluctuations**: Largest eigenvalue has non-Gaussian fluctuations
4. **Rare eigenvalues**: For γ > 1 in Marchenko-Pastur, some eigenvalues stuck at 0

These **deviations are also universal** and well-understood theoretically!

---

## Repository Structure

```
random-matrix-spectra/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── src/                        # Core implementation modules
│   ├── matrix_generators.py   # GOE, GUE, Wishart generators
│   ├── eigenvalue_tools.py    # Eigenvalue computation & analysis
│   ├── spectral_density.py    # Empirical & theoretical densities
│   └── plotting_utils.py      # Visualization helpers
├── notebooks/                  # Jupyter notebooks (main content)
│   ├── 01_introduction.ipynb
│   ├── 02_wigner_semicircle.ipynb
│   ├── 03_marchenko_pastur.ipynb
│   ├── 04_finite_size_effects.ipynb
│   └── 05_universality.ipynb
└── experiments/                # Generated plots and data
    └── (figures created by notebooks)
```

---

## References and Further Reading

### Classic Papers

1. **Wigner, E. P.** (1958). "On the distribution of the roots of certain symmetric matrices". *Ann. Math.*
2. **Marchenko, V. A. & Pastur, L. A.** (1967). "Distribution of eigenvalues for some sets of random matrices". *Mat. Sb.*
3. **Mehta, M. L.** (2004). *Random Matrices*. 3rd edition. Academic Press.

### Modern Texts

1. **Anderson, G. W., Guionnet, A., & Zeitouni, O.** (2010). *An Introduction to Random Matrices*. Cambridge.
2. **Tao, T.** (2012). *Topics in Random Matrix Theory*. AMS.
3. **Forrester, P. J.** (2010). *Log-Gases and Random Matrices*. Princeton.

### Online Resources

1. Terence Tao's blog: [terrytao.wordpress.com](https://terrytao.wordpress.com)
2. Greg Anderson's course notes
3. Arxiv: search "random matrix theory" for latest developments

---

## Technical Notes

### Numerical Considerations

1. **Eigenvalue computation**: I use `numpy.linalg.eigvalsh` for symmetric/Hermitian matrices (faster and more stable than general `eig`).

2. **Normalization**: Matrices are scaled as H/√n so eigenvalues are O(1) - this matches standard RMT conventions.

3. **Unfolding**: For spacing statistics, I use linear unfolding. More sophisticated methods exist (polynomial, spline) but linear suffices for large n.

4. **Histograms vs KDE**: Histograms are faster; KDE is smoother. I use both depending on context.

### Performance

- Generating n×n matrix: O(n²) memory, O(n²) time
- Computing eigenvalues: O(n³) time (but optimized in practice)
- Largest matrix tested: 5000×5000 (takes ~30 seconds)

For very large matrices (n > 10000), consider:
- Sparse matrix techniques (if applicable)
- Using only edge eigenvalues (Lanczos, ARPACK)
- GPU acceleration

---
