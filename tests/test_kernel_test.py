"""Smoke tests for KernelTest stationarity test."""
import numpy as np
import pandas as pd
import pytest

from mht.models.processes import BivariateOUProcess
from mht.testing.kernel_test import Kernel, KernelTest


@pytest.fixture(scope="module")
def ou_data():
    proc = BivariateOUProcess(
        T=50, dt=0.5,
        sigma1=np.sqrt(2), sigma2=np.sqrt(2),
        theta1=0.2, theta2=0.2,
        rho=0.5,
    )
    proc.simulate(seed=42)
    X, T, n = proc.config()
    return X, T, n


@pytest.fixture(scope="module")
def kernel_test(ou_data):
    X, T, n = ou_data
    bw = np.sqrt(3) * 9 / ((n ** (1 / 6)) * np.log(n))
    return KernelTest(
        data=X,
        kernel_params={"bandwidth": bw, "n": n, "T": T, "kernel": Kernel.BaseKernel},
        time_params={"bandwidth": 10 * T / n, "n": n, "T": T},
        show_object=False,
    )


def test_kerneltest_init(kernel_test):
    assert hasattr(kernel_test, "kernel_estimates")
    assert hasattr(kernel_test, "time_estimates")


def test_time_domain_smoother(kernel_test):
    kernel_test.time_domain_smoother(lamb=0.99)
    assert len(kernel_test.time_estimates) > 0


def test_state_domain_smoother(kernel_test):
    kernel_test.state_domain_smoother(dist=True)
    assert len(kernel_test.kernel_estimates) > 0


def test_gauss_produces_gaussian(kernel_test):
    kernel_test.gauss()
    assert hasattr(kernel_test, "gaussian")
    assert len(kernel_test.gaussian) > 0


def test_transform_1D_gauss(ou_data):
    X, T, n = ou_data
    bw = np.sqrt(3) * 9 / ((n ** (1 / 6)) * np.log(n))
    kt = KernelTest(
        data=X,
        kernel_params={"bandwidth": bw, "n": n, "T": T, "kernel": Kernel.BaseKernel},
        time_params={"bandwidth": 10 * T / n, "n": n, "T": T},
        show_object=False,
    )
    kt.time_domain_smoother(lamb=0.99)
    kt.state_domain_smoother(dist=True)
    kt.gauss()
    bound, scalar_gauss = kt.transform_1D_gauss()
    assert any(np.isfinite(b) for b in bound if b is not np.nan)
    assert len(scalar_gauss) == n
