import jax
import functools

@functools.partial(jax.jit, static_argnames=("scalar_func"))
def lie_derivative(state, scalar_func, tangent):
    '''
     Evaluates  ∇b(x)ᵀv  b: scalar_func, v: tangent
    '''
    return jax.jvp(scalar_func, (state,), (tangent,))[1]

def lie_derivative_func(scalar_func, directional_func):
    '''
     Computes the function f(x) = ∇b(x)ᵀv(x)  b: scalar_func, v: directional_func
     This is used in computing higher order CBFs
    '''
    return lambda state: jax.jvp(scalar_func, (state,), (directional_func(state),))[1]

def lie_derivative_func_n(order, scalar_func, directional_func):
    sf = scalar_func
    for _ in range(order):
        sf = lie_derivative_func(sf, directional_func)
    return sf

@functools.partial(jax.jit, static_argnames=("scalar_func", "directional_func", "order"))
def lie_derivative_n(state, order, scalar_func, directional_func):
    return lie_derivative_func_n(order, scalar_func, directional_func)(state)


@functools.partial(jax.jit, static_argnames=["cbf", "alpha", "dynamics"])
def get_cbf_constraint_rd1(state, time, cbf, alpha, dynamics):
    constant = lie_derivative(state, cbf, dynamics.open_loop_dynamics(state, time)) + alpha(cbf(state))
    linear = jax.vmap(lie_derivative, [None, None, 1])(state, cbf, dynamics.control_jacobian(state, time))
    return linear, constant


@functools.partial(jax.jit, static_argnames=["cbf", "alpha1", "alpha2", "dynamics"])
def get_cbf_constraint_rd2(state, time, cbf, alpha1, alpha2, dynamics):
    Lf2b = lie_derivative_n(state, 2, cbf, dynamics.open_loop_dynamics)
    Lfb_func = lie_derivative_func(cbf, dynamics.open_loop_dynamics)
    LgLfb = jax.vmap(lie_derivative, [None, None, 1])(state, Lfb_func, dynamics.control_jacobian(state, time))
    Lfa1b = lie_derivative(state, lambda s: alpha1(cbf(s)), dynamics.open_loop_dynamics(state, time))
    a2_term = alpha2(lie_derivative(state, cbf, dynamics.open_loop_dynamics(state, time)) + alpha1(cbf(state)))

    constant = Lf2b + Lfa1b + a2_term
    linear = LgLfb
    return linear, constant

