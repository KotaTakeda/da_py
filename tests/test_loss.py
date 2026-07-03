"""Tests for the loss/norm helpers in da.loss (test-only, issue #10)."""

import numpy as np

from da.loss import loss_rms, loss_sup, norm_rms, norm_sup


def test_norm_rms_scalar_definition():
    # norm_rms(X) = ||X||_2 / sqrt(N) over the last axis.
    x = np.array([3.0, 4.0])
    assert np.isclose(norm_rms(x), 5.0 / np.sqrt(2))


def test_norm_rms_batched_over_leading_axes():
    X = np.array([[1.0, 1.0], [0.0, 2.0]])
    np.testing.assert_allclose(norm_rms(X), [1.0, np.sqrt(2.0)])


def test_loss_rms_is_rmse_of_difference():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.0, 2.0], [3.0, 2.0]])
    np.testing.assert_allclose(loss_rms(a, b), [1.0 / np.sqrt(2), np.sqrt(2.0)])


def test_norm_and_loss_sup():
    x = np.array([[1.0, -3.0], [0.5, 0.25]])
    np.testing.assert_allclose(norm_sup(x), [3.0, 0.5])
    np.testing.assert_allclose(loss_sup(x, np.zeros_like(x)), [3.0, 0.5])


def test_rms_and_sup_vanish_on_identical_inputs():
    a = np.linspace(0.0, 1.0, 7)
    assert loss_rms(a, a) == 0.0
    assert loss_sup(a, a) == 0.0
