"""Smoke tests for stochastic process simulation."""
import numpy as np
import pandas as pd
import pytest

from mht.models.processes import (
    BivariateOUProcess,
    BivariateCorrelatedBM,
    BivariateNonHomogeneous,
    BivariateCorrelatedDiffusion,
)


@pytest.fixture
def ou_process():
    return BivariateOUProcess(
        T=50, dt=0.1,
        sigma1=np.sqrt(2), sigma2=np.sqrt(2),
        theta1=0.2, theta2=0.2,
        rho=0.5,
    )


def test_ou_simulate_shape(ou_process):
    ou_process.simulate(seed=0)
    X, T, n = ou_process.config()
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n, 2)
    assert T == pytest.approx(50)


def test_ou_finite_values(ou_process):
    ou_process.simulate(seed=1)
    X, _, _ = ou_process.config()
    assert np.all(np.isfinite(X.values))


@pytest.mark.parametrize("rho", [0.0, 0.5, -0.5])
def test_correlated_bm_shape(rho):
    proc = BivariateCorrelatedBM(T=30, dt=0.1, sigma1=1.0, sigma2=1.0, rho=rho)
    proc.simulate(seed=2)
    X, T, n = proc.config()
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n, 2)
    assert np.all(np.isfinite(X.values))


def test_non_homogeneous_shape():
    proc = BivariateNonHomogeneous(T=30, dt=0.1, rho=0.3)
    proc.simulate(seed=3)
    X, T, n = proc.config()
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n, 2)
    assert np.all(np.isfinite(X.values))


def test_correlated_diffusion_shape():
    proc = BivariateCorrelatedDiffusion(T=5, dt=0.1, sigma1=0.5, sigma2=0.5, rho=0.3)
    proc.simulate(seed=4)
    X, T, n = proc.config()
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n, 2)
