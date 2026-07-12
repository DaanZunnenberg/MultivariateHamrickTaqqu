# mht -- Multivariate Hamrick-Taqqu Stationarity Test

A kernel-based test for time-homogeneity of the diffusion matrix of a
bivariate Itô process.

## Statistical background

Given observations of a bivariate diffusion

    dX(t) = b(X(t), t) dt + σ(X(t), t) dW(t)

the test compares two nonparametric estimators of the integrated diffusion
matrix σ(x,t)σ(x,t)ᵀ:

- a **time-domain smoother** (exponentially-weighted running average), and
- a **state-domain smoother** (indicator kernel over the state space).

Under H₀ (time-homogeneous diffusion, i.e. σ(x,t) = σ(x)) the
standardised difference converges weakly to a Gaussian process whose
component-wise running maximum has a Gumbel-type limit distribution.
The null is rejected when the running maximum exceeds a critical bound
derived from that limit.

The package also provides:

- Benjamini-Hochberg and Benjamini-Yekutieli FDR-controlling procedures
  applied to z-score matrices from simulation studies.
- Batch KPSS and Leybourne-McCabe stationarity tests on process paths.

## Installation

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

Requires Python >= 3.10.

## Quick start

```python
import numpy as np
from mht.models.processes import BivariateOUProcess
from mht.testing.kernel_test import KernelTest, Kernel, TestPlotter

# Simulate a bivariate OU process
ou_config = {
    'T': 365, 'dt': 1/20,
    'sigma1': np.sqrt(2), 'sigma2': np.sqrt(2),
    'theta1': 0.2, 'theta2': 0.2,
    'rho': 0.75,
}
process = BivariateOUProcess(**ou_config)
process.simulate(seed=1)
X, T, n = process.config()

# Set up the test configuration
config = {
    'data': X,
    'kernel_params': {
        'bandwidth': np.sqrt(3) * 9 / ((n ** (1/6)) * np.log(n)),
        'n': n, 'T': T,
        'kernel': Kernel.BaseKernel,
    },
    'time_params': {'bandwidth': 200 * T / n, 'n': n, 'T': T},
}

# Estimate and test
test = KernelTest(**config)
test.time_domain_smoother(lamb=0.99)
test.state_domain_smoother(dist=True)   # True = use KDE for joint density
test.gauss()

bound, scalar_gauss = test.transform_1D_gauss()

# Plot
plotter = TestPlotter(test)
plotter.plot_running_maximum()
```

See `notebooks/example.ipynb` for a full worked example including
Monte Carlo simulations and comparison with KPSS / Leybourne-McCabe.

## Repository structure

```
src/mht/
    testing/
        kernel_test.py        # KernelTest, Simulator, TestPlotter
        hypothesis.py         # MultipleHypTest, UnitRootTest, LaTeXTable
        leybourne_mccabe.py   # Leybourne class (single canonical copy)
    models/
        processes.py          # BivariateOUProcess, BivariateCorrelatedBM, ...
    viz/
        plots.py
    utils/
        decorators.py
data/
    reader.py                 # Reader class for simulation CSV files
simulations/                  # Pre-computed CSV simulation results
notebooks/
    example.ipynb
tests/
```

## References

- Hamrick, J., Taqqu, M. (2009). *Testing the diffusion coefficient*.
- Leybourne, S.J., & McCabe, B.P.M. (1994). A consistent test for a unit root.
  *Journal of Business and Economic Statistics*, 12: 157-166.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
  *Journal of the Royal Statistical Society B*, 57(1): 289-300.
