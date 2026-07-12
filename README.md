# mht -- Multivariate Stationarity Test for Diffusion Processes

A nonparametric test for stationarity of a multivariate Itô diffusion process,
based on comparing two estimators of the diffusion matrix whose convergence
rates diverge under the alternative of nonstationarity.

## Statistical background

### Setup

The process X is assumed to be a d-dimensional time-homogeneous Itô diffusion

    dXt = b(Xt) dt + σ(Xt) dWt,    t ≥ 0,

observed discretely on [0, T] at n equidistant points with step Δn = T/n.
The object of interest is the diffusion matrix c(x) = σ(x)σ(x)ᵀ.

**H₀**: X is stationary (equivalently, positive Harris recurrent with α-index 1).
**H₁**: X is nonstationary (α-index < 1, so the occupation measure Ln,T(x) = o(T)).

### Key identification result

Stationarity is identified through the behaviour of the empirical occupation
measure Ln,T(x) = Δn Σ Kh(Xtj − x). By Darling-Kac (1957) and Meyn-Tweedie
(2012), a stationary process satisfies Ln,T(x) ~ T for every x, while a
nonstationary process has Ln,T(x) = op(T). This difference in divergence rate
is the foundation of the test.

### Two estimators of c(x)

The test is a Durbin-Wu-Hausman-style comparison of two consistent estimators
of c(x) whose asymptotic speeds differ under H₁:

**Time-domain smoother** (Jacod-Protter 2011): a local average of outer products
of increments over a window of kn consecutive observations around a fixed time t,

    ĉ_TD,t = (1 / Δn kn) Σ_{m=1}^{kn} ΔXt · ΔXtᵀ,    kn²Δn → 0.

Its central-limit rate is √kn regardless of stationarity, making it a
stationarity-invariant baseline.

**State-domain smoother** (Bandi-Moloche 2018): a Nadaraya-Watson kernel
regression of squared increments on the state space,

    ĉ_SD(x) = [Σ Kh(Xtj − x) ΔXtj ΔXtjᵀ] / [Δn Σ Kh(Xtj − x)],

with bandwidth hn,T ~ C / (n^{1/(d+4)} log n). Its central-limit rate is
√(n hⁿ_n,T) under stationarity, but it diverges almost surely at that same rate
when the process is nonstationary, because T/Ln,T(x) → ∞.

### Test statistic

Under stationarity the two estimators are asymptotically independent (Theorem 2.3
of the framework, proved via β-mixing), so their standardised difference

    Ztj = A Σ^{-1/2}(x) √(2n hⁿ / d(d+1)) vech(ĉ_SD(x) − ĉ_TD,tj)

is asymptotically N(0,1) for every tj under H₀, and diverges almost surely
under H₁. The time index is re-scaled to t'j = tj / (Δn kn) to induce the
vanishing covariance γ(s) = o(1/log s) required by Berman (1964, 1982).

### Critical bound (running maximum)

Because the entire path (Zt'j)_{j=1,...,n} must be assessed jointly, the test
uses the running maximum ϕn = max_{k≤n} |Zt'k|. By Pickands (1969) / Berman
(1964), for a stationary Gaussian sequence with vanishing correlations,

    an (ϕn − bn) →d Gumbel,    an = √(2 log n),    bn = an − (log log n + log π) / 2an.

H₀ is rejected at level α when

    ϕn > bn − log log α⁻¹ / an.

### Multiple-testing supplements

For simulation studies, Benjamini-Hochberg (1995) and Benjamini-Yekutieli (2001)
FDR procedures are applied to the z-score matrix to control false discoveries
across replications and grid points.

The package also provides:

- Batch KPSS and Leybourne-McCabe stationarity tests on process paths for comparison.

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
        leybourne_mccabe.py   # Leybourne-McCabe test (single canonical copy)
    models/
        processes.py          # BivariateOUProcess, BivariateCorrelatedBM, ...
    io/
        reader.py             # Reader class for simulation CSV files
    viz/                      # TestPlotter re-exported here
    utils/
        decorators.py
simulations/                  # Pre-computed CSV simulation results
notebooks/
    example.ipynb
tests/
    test_processes.py
    test_kernel_test.py
```

## References

- Bandi, F.M., & Moloche, G. (2018). On the functional estimation of multivariate
  diffusion processes. *Econometric Theory*, 34(4): 896–946.
- Jacod, J., & Protter, P. (2011). *Discretization of Processes*. Springer.
- Darling, D.A., & Kac, M. (1957). On occupation times for Markoff processes.
  *Transactions of the American Mathematical Society*, 84(2): 444–458.
- Berman, S.M. (1964). Limit theorems for the maximum term in stationary sequences.
  *Annals of Mathematical Statistics*, 35(2): 502–516.
- Pickands, J. (1969). Asymptotic properties of the maximum in a stationary Gaussian
  process. *Transactions of the American Mathematical Society*, 145: 75–86.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
  *Journal of the Royal Statistical Society B*, 57(1): 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in
  multiple testing under dependency. *Annals of Statistics*, 29(4): 1165–1188.
- Leybourne, S.J., & McCabe, B.P.M. (1994). A consistent test for a unit root.
  *Journal of Business and Economic Statistics*, 12: 157–166.
- Meyn, S.P., & Tweedie, R.L. (2012). *Markov Chains and Stochastic Stability* (2nd ed.).
  Cambridge University Press.
