"""Radial basis functions in JAX.

Port of e3nn.math.soft_one_hot_linspace (smooth_finite basis with optional cutoff).
"""

import jax
import jax.numpy as jnp


def soft_unit_step(x: jax.Array) -> jax.Array:
    """C-infinity smooth unit step function.

    f(x) = exp(-1/x) for x > 0, 0 otherwise.
    """
    safe_x = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, jnp.exp(-1.0 / safe_x), 0.0)


def soft_one_hot_linspace(
    x: jax.Array,
    start: float,
    end: float,
    number: int,
    basis: str = "smooth_finite",
    cutoff: bool = True,
) -> jax.Array:
    """Smooth one-hot encoding along a linspace.

    Projects input x onto a set of basis functions evenly spaced
    between start and end.

    Parameters
    ----------
    x : jax.Array
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
    jax.Array
        Shape (*x.shape, number) tensor of basis function values.
    """
    if basis != "smooth_finite":
        raise ValueError(f"Only 'smooth_finite' basis is supported, got '{basis}'")

    if cutoff:
        # With cutoff: place number centers inside [start, end], excluding boundaries
        values = jnp.linspace(start, end, number + 2)
        step = values[1] - values[0]
        values = values[1:-1]  # Remove boundary centers
    else:
        values = jnp.linspace(start, end, number)
        step = values[1] - values[0]

    # Compute normalized difference: (x - center) / step
    diff = (x[..., None] - values) / step

    # smooth_finite basis: C-infinity bump function via two soft_unit_steps
    # f(t) = C * soft_unit_step(t+1) * soft_unit_step(1-t)
    # where C = 1.14136 * e^2 normalizes to unit integral
    return 1.14136 * jnp.e**2 * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)
