# Generated Plots Summary

✅ **All 5 example plots have been successfully generated!**

## Files in `experiments/` directory:

1. **wigner_convergence.png** (172 KB)
   - Shows eigenvalue distributions for n = 100, 500, 1000, 5000
   - Demonstrates convergence to Wigner semicircle law
   
2. **universality_comparison.png** (69 KB)
   - Compares GOE vs GUE vs Poisson spacing statistics
   - Shows level repulsion phenomenon
   
3. **marchenko_pastur_varying_gamma.png** (172 KB)
   - Wishart matrices with γ = 0.2, 0.5, 0.8, 1.5
   - Demonstrates how distribution shape changes with aspect ratio
   
4. **finite_size_convergence.png** (89 KB)
   - Log-log plot showing error decay as ~1/√n
   - Quantifies convergence rate to theory
   
5. **edge_fluctuations.png** (55 KB)
   - Distribution of largest eigenvalue from 200 trials
   - Shows Tracy-Widom fluctuations near the edge

## How to View

Simply open the PNG files in any image viewer, or view them in the file explorer.

## Detailed Descriptions

See `PLOTS_DESCRIPTION.md` in this directory for detailed explanations of what each plot shows and how to interpret them.

## Regenerating Plots

To regenerate all plots, run:
```bash
python3 generate_plots.py
```

Or run the Jupyter notebooks for the complete set of ~18 plots.
