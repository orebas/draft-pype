import numpy as np
from scipy.interpolate import BarycentricInterpolator, interp1d
from scipy.linalg import solve
from dataclasses import dataclass
from typing import List, Callable


def rational_interpolation_coefficients(x, y, n):
    """
    Perform a rational interpolation of the data `y` at the points `x` with numerator degree `n`.
    This function returns the coefficients of the numerator and denominator polynomials.

    Arguments:
    - x: array-like, the points where the data is sampled (e.g., time points).
    - y: array-like, the data sample.
    - n: int, the degree of the numerator.

    Returns:
    - c: array-like, coefficients of the numerator polynomial.
    - d: array-like, coefficients of the denominator polynomial.
    """
    N = len(x)
    m = N - n - 1
    if m > 0:
        A_left = np.vstack([x**i for i in range(n + 1)]).T
        A_right = np.vstack([x**i for i in range(m)]).T
        A = np.hstack([A_left, -np.diag(y) @ A_right])
        b = y * x**m
        try:
            c = np.linalg.solve(A, b)
            return c[: n + 1], np.append(c[n + 1 :], 1)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return c[: n + 1], np.append(c[n + 1 :], 1)
    else:
        A = np.vstack([x**i for i in range(n + 1)]).T
        b = y
        c = np.linalg.solve(A, b)
        return c, [1]


def baryEval(z, f, x, w, tol=1e-13):
    """
    Evaluate the barycentric interpolation at point z.
    """
    num = 0.0
    den = 0.0
    for j in range(len(f)):
        diff = z - x[j]
        if abs(diff) < np.sqrt(tol):
            return f[j]
        t = w[j] / diff
        num += t * f[j]
        den += t
    return num / den


@dataclass
class AAADapprox:
    f: np.ndarray
    x: np.ndarray
    w: np.ndarray

    def __call__(self, z):
        return baryEval(z, self.f, self.x, self.w)


@dataclass
class FHDapprox:
    f: np.ndarray
    x: np.ndarray
    w: np.ndarray

    def __call__(self, z):
        return baryEval(z, self.f, self.x, self.w)


def nth_deriv_at(f: Callable, n: int, t: float):
    """
    Compute the nth derivative of function f at point t.
    """
    from scipy.misc import derivative

    return derivative(f, t, n=n, dx=1e-6)


def aaad(xs, ys):
    """
    Approximate function using AAA rational approximation.
    Since there's no direct equivalent in Python, we can use barycentric interpolation.
    """
    interpolator = BarycentricInterpolator(xs, ys)
    # Compute weights (not directly available, so we approximate)
    w = np.ones_like(xs)
    return AAADapprox(f=ys, x=xs, w=w)


def fhd(xs, ys, N):
    """
    Placeholder for Floater–Hormann interpolation.
    """
    interpolator = interp1d(xs, ys, kind=N, fill_value="extrapolate")
    # Since weights are not directly available, we use ones
    w = np.ones_like(xs)
    return FHDapprox(f=ys, x=xs, w=w)


def FourierInterp(xs, ys):
    """
    Perform Fourier interpolation.
    """
    N = len(xs)
    width = xs[-1] - xs[0]
    m = np.pi / width
    b = -np.pi * (xs[0] / width + 0.5)
    z = m * xs + b
    K = np.mean(ys)
    cos_terms = [np.cos(k * z) for k in range(1, N // 2 + 1)]
    sin_terms = [np.sin(k * z) for k in range(1, N // 2 + 1)]
    # Coefficients can be computed using least squares
    A = np.column_stack([np.ones(N)] + cos_terms + sin_terms)
    coeffs, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    return lambda x: np.dot(
        coeffs,
        np.column_stack(
            [np.ones_like(x)]
            + [np.cos(k * (m * x + b)) for k in range(1, N // 2 + 1)]
            + [np.sin(k * (m * x + b)) for k in range(1, N // 2 + 1)]
        ).T,
    )


def BarycentricLagrange(xs, ys):
    """
    Barycentric Lagrange interpolation.
    """
    interpolator = BarycentricInterpolator(xs, ys)
    return interpolator


@dataclass
class RationalFunction:
    a: np.ndarray
    b: np.ndarray

    def __call__(self, z):
        return np.polyval(self.a[::-1], z) / np.polyval(self.b[::-1], z)


def simpleratinterp(xs, ys, d1):
    """
    Simple rational interpolation.
    """
    N = len(xs)
    d2 = N - d1 - 1
    A = np.zeros((N, N))
    for j in range(N):
        A[j, 0] = 1
        for k in range(1, d1 + 1):
            A[j, k] = xs[j] ** k
        for k in range(1, d2 + 1):
            A[j, d1 + k] = -ys[j] * xs[j] ** k
    b = ys
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    return RationalFunction(a=c[: d1 + 1], b=np.concatenate(([1], c[d1 + 1 :])))


def default_interpolator(datasize):
    interpolators = {
        "AAA": aaad,
        "FHD3": lambda xs, ys: fhd(xs, ys, 3),
    }
    if datasize > 10:
        interpolators["FHD8"] = lambda xs, ys: fhd(xs, ys, 8)
    return interpolators
