import jax
import jax.numpy as jnp
import functools
import equinox as eqx
from typing import Callable


# @functools.partial(jax.jit, static_argnames=("scalar_func"))
@eqx.filter_jit
def lie_derivative(
    state: jnp.ndarray,
    scalar_func: Callable[[jnp.ndarray], jnp.ndarray],
    tangent: jnp.ndarray,
):
    """
    Evaluates  ∇b(x)ᵀv  b: scalar_func, v: tangent
    Arguments:
        state: The point at which to evaluate the Lie derivative.
        scalar_func: A function that takes in the state and outputs a scalar.
        tangent: The direction in which to compute the Lie derivative.
    Returns:
        The Lie derivative of the scalar function along the tangent at the given state.
    """
    return jax.jvp(scalar_func, (state,), (tangent,))[1]


@eqx.filter_jit
def lie_derivative_multiple(
    state: jnp.ndarray,
    scalar_func: Callable[[jnp.ndarray], jnp.ndarray],
    tangents: jnp.ndarray,
    dim=1,
):
    """
    Evaluates  ∇b(x)ᵀv  b: scalar_func, v=[v1, v2,.., vn]: n # oftangents
    Arguments:
        state: The point at which to evaluate the Lie derivative.
        scalar_func: A function that takes in the state and outputs a scalar.
        tangents: The directions in which to compute the Lie derivative.
        dim: The dimension along which the tangents are stacked.
    Returns:
        The Lie derivatives of the scalar function along each of the tangents at the given state.
    """
    return jax.vmap(lie_derivative, [None, None, dim])(state, scalar_func, tangents)


def lie_derivative_func(scalar_func, directional_func):
    """
    Computes the function f(x) = ∇b(x)ᵀv(x)  b: scalar_func, v: directional_func
    This is used in computing higher order CBFs
    Arguments:
        scalar_func: A function that takes in the state and outputs a scalar.
        directional_func: A function that takes in the state and outputs a vector.
    Returns:
        A function that computes the Lie derivative of the scalar function along the directional function at a given state.
    """
    return lambda state: jax.jvp(scalar_func, (state,), (directional_func(state),))[1]


def lie_derivative_func_n(order, scalar_func, directional_func):
    """
    Computes the Lie derivative of the scalar function along the directional function at a given state.
    Arguments:
        order: The order of the Lie derivative to compute.
        scalar_func: A function that takes in the state and outputs a scalar.
        directional_func: A function that takes in the state and outputs a vector.
    Returns:
        A function that computes the nth order Lie derivative of the scalar function along the directional function at a given state.

    TODO: (kymleung) Make this more efficient. e.g., Optimize with jax.lax.scan or similar to avoid Python loop overhead.
    """
    sf = scalar_func
    for _ in range(order):
        sf = lie_derivative_func(sf, directional_func)
    return sf


# @functools.partial(
#     jax.jit, static_argnames=("scalar_func", "directional_func", "order")
# )
@eqx.filter_jit
def lie_derivative_n(state, order, scalar_func, directional_func):
    """Computes the nth order Lie derivative of the scalar function along the directional function at a given state.
    Arguments:
        state: The point at which to evaluate the Lie derivative.
        order: The order of the Lie derivative to compute.
        scalar_func: A function that takes in the state and outputs a scalar.
        directional_func: A function that takes in the state and outputs a vector.
    Returns:
        The nth order Lie derivative of the scalar function along the directional function at the given state."""
    return lie_derivative_func_n(order, scalar_func, directional_func)(state)


# @functools.partial(jax.jit, static_argnames=["cbf", "alpha", "dynamics"])
@eqx.filter_jit
def get_cbf_constraint_rd1(state, time, cbf, alpha, dynamics):
    """
    Computes the CBF constraint for a relative degree 1 system.
    Arguments:
        state: The current state of the system.
        time: The current time.
        cbf: The control barrier function.
        alpha: The class K function.
        dynamics: The dynamics of the system.
    Returns:
        A tuple (linear, constant) representing the CBF constraint.
    """
    constant = lie_derivative(
        state, cbf, dynamics.open_loop_dynamics(state, time)
    ) + alpha(cbf(state))
    linear = lie_derivative_multiple(
        state, cbf, dynamics.control_jacobian(state, time), dim=1
    )
    return linear, constant


@functools.partial(jax.jit, static_argnames=["cbf", "alpha1", "alpha2", "dynamics"])
def get_cbf_constraint_rd2(state, time, cbf, alpha1, alpha2, dynamics):
    """
    Computes the CBF constraint for a relative degree 2 system.
    Arguments:
        state: The current state of the system.
        time: The current time.
        cbf: The control barrier function.
        alpha1: The class K function for the first Lie derivative.
        alpha2: The class K function for the second Lie derivative.
        dynamics: The dynamics of the system.
    Returns:
        A tuple (linear, constant) representing the CBF constraint.
    """
    Lf2b = lie_derivative_n(state, 2, cbf, dynamics.open_loop_dynamics)
    Lfb_func = lie_derivative_func(cbf, dynamics.open_loop_dynamics)
    LgLfb = jax.vmap(lie_derivative, [None, None, 1])(
        state, Lfb_func, dynamics.control_jacobian(state, time)
    )
    Lfa1b = lie_derivative(
        state, lambda s: alpha1(cbf(s)), dynamics.open_loop_dynamics(state, time)
    )
    a2_term = alpha2(
        lie_derivative(state, cbf, dynamics.open_loop_dynamics(state, time))
        + alpha1(cbf(state))
    )

    constant = Lf2b + Lfa1b + a2_term
    linear = LgLfb
    return linear, constant
