"""Radial basis functions in PyTorch.

Port of e3nn.math.soft_one_hot_linspace (smooth_finite basis with optional cutoff).
"""

import math

import torch


def soft_unit_step(x: torch.Tensor) -> torch.Tensor:
    """C-infinity smooth unit step function.

    f(x) = exp(-1/x) for x > 0, 0 otherwise.
    """
    safe_x = torch.where(x > 0, x, torch.ones_like(x))
    return torch.where(x > 0, torch.exp(-1.0 / safe_x), torch.zeros_like(x))


def soft_one_hot_linspace(
    x: torch.Tensor,
    start: float,
    end: float,
    number: int,
    basis: str = "smooth_finite",
    cutoff: bool = True,
) -> torch.Tensor:
    """Smooth one-hot encoding along a linspace.

    Projects input x onto a set of basis functions evenly spaced
    between start and end.

    Parameters
    ----------
    x : torch.Tensor
        Input values.
    start : float
        Start of the linspace.
    end : float
        End of the linspace.
    number : int
        Number of basis functions.
    basis : str
        Basis type. Only 'smooth_finite' is supported.
    cutoff : bool
        If True, basis functions are zero outside [start, end].

    Returns
    -------
    torch.Tensor
        Shape (*x.shape, number) tensor of basis function values.
    """
    if basis != "smooth_finite":
        raise ValueError(f"Only 'smooth_finite' basis is supported, got '{basis}'")

    if cutoff:
        values = torch.linspace(start, end, number + 2, device=x.device, dtype=x.dtype)
        step = values[1] - values[0]
        values = values[1:-1]
    else:
        values = torch.linspace(start, end, number, device=x.device, dtype=x.dtype)
        step = values[1] - values[0]

    diff = (x[..., None] - values) / step

    return 1.14136 * math.e**2 * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)
