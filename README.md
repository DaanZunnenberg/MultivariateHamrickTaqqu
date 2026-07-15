# mht -- Multivariate Nonparametric Stationarity Test for Diffusion Processes

A nonparametric test for stationarity of a multivariate Itô diffusion process,
based on comparing two estimators of the diffusion matrix whose convergence
rates diverge under the alternative of nonstationarity.

## Statistical background

### Setup

The process $X$ is assumed to be a $d$-dimensional time-homogeneous Itô diffusion

$$dX_t = b(X_t)\,dt + \sigma(X_t)\,dW_t, \qquad t \ge 0,$$

observed discretely on $[0,T]$ at $n$ equidistant points with step $\Delta_n = T/n$.
The object of interest is the diffusion matrix $c(x) = \sigma(x)\sigma(x)^\top$.

- $H_0$: $X$ is stationary (equivalently, positive Harris recurrent with $\alpha$-index $1$).
- $H_1$: $X$ is nonstationary ($\alpha$-index $< 1$, so the occupation measure $L_{n,T}(x) = o_p(T)$).

### Key identification result

Stationarity is identified through the empirical occupation measure

$$L_{n,T}(x) = \Delta_n \sum_{j=1}^n K_{h}(X_{t_j} - x).$$

By Darling–Kac (1957) and Meyn–Tweedie (2012), a stationary process satisfies $L_{n,T}(x) \asymp T$ for every $x$, while a nonstationary process has $L_{n,T}(x) = o_p(T)$. This difference in divergence rate is the foundation of the test.

### Two estimators of $c(x)$

The test is a Durbin–Wu–Hausman-style comparison of two consistent estimators of $c(x)$ whose asymptotic speeds differ under $H_1$:

**Time-domain smoother** (Jacod–Protter 2011): a local average of outer products of increments over a window of $k_n$ consecutive observations around a fixed time $t$,

$$\hat{c}^n_{\mathrm{TD},t} = \frac{1}{\Delta_n k_n} \sum_{m=1}^{k_n} \Delta^m_n X_t \left(\Delta^m_n X_t\right)^\top, \qquad k_n^2 \Delta_n \to 0.$$

Its central-limit rate is $\sqrt{k_n}$ regardless of stationarity, making it a stationarity-invariant baseline.

**State-domain smoother** (Bandi–Moloche 2018): a Nadaraya–Watson kernel regression of squared increments on the state space,

$$\hat{c}^n_{\mathrm{SD}}(x) = \frac{\displaystyle\sum_{j=1}^n K_{h_{n,T}}(X_{t_j} - x)\,(X_{t_j}-X_{t_{j-1}})(X_{t_j}-X_{t_{j-1}})^\top}{\Delta_n \displaystyle\sum_{j=1}^n K_{h_{n,T}}(X_{t_j} - x)},$$

with near-optimal bandwidth $h_{n,T} \sim C / (n^{1/(d+4)} \log n)$. Its central-limit rate is $\sqrt{n h_{n,T}^d}$ under stationarity, but it diverges almost surely at that same rate when the process is nonstationary, because $T / L_{n,T}(x) \to \infty$.

### Test statistic

Under stationarity the two estimators are asymptotically independent (proved via $\beta$-mixing), so their standardised difference

$$Z_{t_j} = A\,\Sigma^{-1/2}(x) \sqrt{\frac{2n h_{n,T}^d}{d(d+1)}}\;\operatorname{vech}\!\left(\hat{c}^n_{\mathrm{SD}}(x) - \hat{c}^n_{\mathrm{TD},t_j}\right) \xrightarrow{d} \mathcal{N}(0,1)$$

for every $t_j$ under $H_0$, and diverges almost surely under $H_1$. The time index is re-scaled to $t'_j = t_j / (\Delta_n k_n)$ to induce vanishing covariance $\gamma(s) = o(1/\log s)$ required by Berman (1964, 1982).

### Critical bound (running maximum)

Because the full path $(Z_{t'_j})_{j=1,\dots,n}$ must be assessed jointly, the test uses the running maximum $\varphi_n = \max_{k \le n} |Z_{t'_k}|$. By Pickands (1969) / Berman (1964), for a stationary Gaussian sequence with vanishing correlations,

$$a_n(\varphi_n - b_n) \xrightarrow{d} \mathrm{Gumbel}, \qquad a_n = \sqrt{2\log n}, \qquad b_n = a_n - \frac{\log\log n + \log\pi}{2a_n}.$$

$H_0$ is rejected at level $\alpha$ when

$$\varphi_n > b_n - \frac{\log\log\alpha^{-1}}{a_n}.$$

### Multiple-testing supplements

For simulation studies, Benjamini–Hochberg (1995) and Benjamini–Yekutieli (2001) FDR procedures are applied to the $z$-score matrix to control false discoveries across replications and grid points.

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
