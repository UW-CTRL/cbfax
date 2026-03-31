from typing import Callable, List, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from dynamaxsys import ControlAffineDynamics


class ControlCertificationFunction(eqx.Module):
    certificate_func: Callable[[jnp.ndarray], jnp.ndarray]
    alpha_func: Union[Callable[[float], float], List[Callable[[float], float]]]
    dynamics: ControlAffineDynamics
    state_dim: int
    relative_degree: int

    def __init__(
        self,
        certificate_func: Callable[[jnp.ndarray], jnp.ndarray],
        alpha_func: Union[Callable[[float], float], List[Callable[[float], float]]],
        dynamics: ControlAffineDynamics,
        state_dim: int,
        relative_degree: int,
    ):
        self.state_dim = state_dim
        self.certificate_func = certificate_func
        self.relative_degree = relative_degree
        self.alpha_func = alpha_func
        self.dynamics = dynamics
        if relative_degree not in [1, 2]:
            raise ValueError("Relative degree must be 1 or 2.")
        # if relative_degree == 1 and not isinstance(alpha_func, callable):
        #     raise ValueError("Alpha function must be a single function for relative degree 1.")
        # if relative_degree == 2 and not isinstance(alpha_func, list) and all(isinstance(func, callable) for func in alpha_func):
        #     raise ValueError("Alpha functions must be a list of functions for relative degree 2.")
        
    @eqx.filter_jit
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.certificate_func(state)

    @eqx.filter_jit
    def gradient(self, state: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(self.certificate_func)(state)

    def _control_constraint_rd1(
        self,
        state: jnp.ndarray,
        time: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return get_cbf_constraint_rd1(state, time, self.certificate_func, self.alpha_func, self.dynamics)

    def _control_constraint_rd2(
        self,
        state: jnp.ndarray,
        time: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return get_cbf_constraint_rd2(state, time, self.certificate_func, self.alpha_func, self.dynamics)

    @eqx.filter_jit
    def control_constraint(
        self,
        state: jnp.ndarray,
        time: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.relative_degree == 1:
            return self._control_constraint_rd1(state, time)
        return self._control_constraint_rd2(state, time)


class ControlBarrierFunction(ControlCertificationFunction):
    def __init__(
        self,
        certificate_func: Callable[[jnp.ndarray], jnp.ndarray],
        alpha_func: Union[Callable[[float], float], List[Callable[[float], float]]],
        dynamics: ControlAffineDynamics,
        state_dim: int,
        relative_degree: int,
    ):
        super().__init__(certificate_func, alpha_func, dynamics, state_dim, relative_degree)


class ControlLyapunovFunction(ControlCertificationFunction):
    def __init__(
        self,
        certificate_func: Callable[[jnp.ndarray], jnp.ndarray],
        alpha_func: Union[Callable[[float], float], List[Callable[[float], float]]],
        dynamics: ControlAffineDynamics,
        state_dim: int,
        relative_degree: int,
    ):
        super().__init__(certificate_func, alpha_func, dynamics, state_dim, relative_degree)


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


def lie_derivative_func(
    scalar_func: Callable[[jnp.ndarray], jnp.ndarray],
    directional_func: Callable[[jnp.ndarray], jnp.ndarray],
):
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


def lie_derivative_func_n(
    order: int,
    scalar_func: Callable[[jnp.ndarray], jnp.ndarray],
    directional_func: Callable[[jnp.ndarray], jnp.ndarray],
):
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


@eqx.filter_jit
def lie_derivative_n(
    state: jnp.ndarray,
    order: int,
    scalar_func: Callable[[jnp.ndarray], jnp.ndarray],
    directional_func: Callable[[jnp.ndarray], jnp.ndarray],
):
    """Computes the nth order Lie derivative of the scalar function along the directional function at a given state.
    Arguments:
        state: The point at which to evaluate the Lie derivative.
        order: The order of the Lie derivative to compute.
        scalar_func: A function that takes in the state and outputs a scalar.
        directional_func: A function that takes in the state and outputs a vector.
    Returns:
        The nth order Lie derivative of the scalar function along the directional function at the given state."""
    return lie_derivative_func_n(order, scalar_func, directional_func)(state)


@eqx.filter_jit
def get_cbf_constraint_rd1(
    state: jnp.ndarray,
    time: float,
    cbf: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: Callable[[float], float],
    dynamics: ControlAffineDynamics,
):
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


@eqx.filter_jit
def get_cbf_constraint_rd2(
    state: jnp.ndarray,
    time: float,
    cbf: Callable[[jnp.ndarray], jnp.ndarray],
    alphas: List[Callable[[float], float]],
    dynamics: ControlAffineDynamics,
):
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
    alpha1, alpha2 = alphas
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
