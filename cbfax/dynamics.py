import jax
import jax.numpy as jnp
import equinox as eqx
import functools
import abc
from typing import Callable

def runge_kutta_integrator(dynamics, dt=0.1):
    # zero-order hold
    def integrator(x, u, t):
        dt2 = dt / 2.0
        k1 = dynamics(x, u, t)
        k2 = dynamics(x + dt2 * k1, u, t + dt2)
        k3 = dynamics(x + dt2 * k2, u, t + dt2)
        k4 = dynamics(x + dt * k3, u, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return integrator

@functools.partial(jax.jit, static_argnames=["dynamics"])
def linearize(dynamics, state, control, t):
    A, B = jax.jacobian(dynamics, [0, 1])(state, control, t)
    C = dynamics(state, control, t) - A @ state - B @ control
    return A, B, C

class Dynamics(metaclass=abc.ABCMeta):
    state_dim: int
    control_dim: int

    @abc.abstractmethod
    def ode_dynamics(self, state, control, time=0):
        """Implements the continuous-time dynamics ODE."""

    def discrete_step(self, state, control, time=0, dt=0.1):
        return runge_kutta_integrator(self.ode_dynamics, dt)(state, control, time)

    def __call__(self, state, control, time=0):
        return self.ode_dynamics(state, control, time)



class ControlAffineDynamics(Dynamics):

    def ode_dynamics(self, state, control, time=0):
        return self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control

    @abc.abstractmethod
    def open_loop_dynamics(self, state, time):
        """Implements the open loop dynamics `f(x, t)`."""

    @abc.abstractmethod
    def control_jacobian(self, state, time):
        """Implements the control Jacobian `G_u(x, t)`."""


# class NAgentIntegrator(ControlAffineDynamics):
#     integrator_dim: int
#     N_dim: int
#     n_agents: int


class IntegratorND(ControlAffineDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        self.state_dim = self.integrator_dim * self.N_dim
        self.control_dim = self.N_dim

        self.A = jnp.eye(self.state_dim, k=self.N_dim)
        self.B = jnp.zeros([self.state_dim, self.control_dim])
        self.B = self.B.at[-self.N_dim:].set(jnp.eye(self.N_dim))

    def open_loop_dynamics(self, state, time=0):
        return self.A @ state

    def control_jacobian(self, state, time=0):
        return self.B
        
    # def ode_dynamics(self, state, control, t):
    #     return jnp.concatenate([state[self.N_dim:], control])
    
def DoubleIntegrator2D():
    return IntegratorND(2, 2) 

def DoubleIntegrator1D():
    return IntegratorND(2, 1) 

def SingleIntegrator2D():
    return IntegratorND(1, 2) 

def SingleIntegrator1D():
    return IntegratorND(1, 1) 

class Unicycle(ControlAffineDynamics):
    state_dim: int = 3
    control_dim: int = 2

    def open_loop_dynamics(self, state, time=0):
        return jnp.zeros(self.state_dim)
    
    def control_jacobian(self, state, time=0):
        # v, om = control
        x, y, th = state
        return jnp.array(
            [
                [jnp.cos(th), 0.],
                [jnp.sin(th), 0.],
                [0., 1.]
            ]
        )

class DynamicallyExtendedUnicycle(ControlAffineDynamics):
    state_dim: int = 4
    control_dim: int = 2

    # def ode_dynamics(self, state, control, time=0):
    #     x, y, th, v = state
    #     a, om = control
    #     return jnp.array([v * jnp.cos(th),
    #                       v * jnp.sin(th),
    #                       om,
    #                       a])


    def open_loop_dynamics(self, state, time=0):
        x, y, th, v = state
        # om, a = control
        return jnp.array(
            [
                v * jnp.cos(th),
                v * jnp.sin(th),
                0.,
                0.,
            ]
        )
    
    def control_jacobian(self, state, time=0):
        # om, a = control
        return jnp.array(
            [
                [0., 0.],
                [0., 0.],
                [1., 0.],
                [0., 1.]
            ]
        )


class SimpleCar(ControlAffineDynamics):
    state_dim: int = 4
    control_dim: int = 2
    wheelbase: int

    def __init__(self, wheelbase):
        self.wheelbase = wheelbase

    def open_loop_dynamics(self, state, time=0):
        x, y, th, v = state
        # tandelta, a = control
        return jnp.array(
            [
                v * jnp.cos(th),
                v * jnp.sin(th),
                0.,
                0.,
            ]
        )
    
    def control_jacobian(self, state, time=0):
        # tandelta, a = control, tandelta = tan(delta)
        x, y, th, v = state
        return jnp.array(
            [
                [0., 0.],
                [0., 0.],
                [v / self.wheelbase, 0.],
                [0., 1.]
            ]
        )
    

class TwoPlayerRelativeIntegratorND(ControlAffineDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        self.state_dim = self.integrator_dim * self.N_dim
        self.control_dim = self.N_dim * 2

        self.A = jnp.eye(self.state_dim, k=self.N_dim)
        B = jnp.zeros([self.state_dim, self.N_dim])
        B = B.at[-self.N_dim:].set(jnp.eye(self.N_dim))
        self.B2 = jnp.concatenate([-B, B], axis=-1)


    def open_loop_dynamics(self, state, time=0):
        return self.A @ state

    def control_jacobian(self, state, time=0):
        return self.B2


class RelativeUnicycle(ControlAffineDynamics):
    state_dim: int = 3
    control_dim: int = 4

    # def ode_dynamics(self, state, control, time=0):
    #     xrel, yrel, threl = state
    #     v1, om1, v2, om2 = control
    #     return jnp.array([v2 * jnp.cos(threl) + om1 * yrel - v1,
    #                       v2 * jnp.sin(threl) - om1 * xrel,
    #                       om2 - om1])


    def open_loop_dynamics(self, state, time=0):
        xrel, yrel, threl = state
        # v1, om1, v2, om2 = control
        return jnp.zeros(self.state_dim)

    def control_jacobian(self, state, time=0):
        xrel, yrel, threl = state
        # v1, om1, v2, om2 = control
        return jnp.array(
            [
                [-1., yrel, jnp.cos(threl), 0.],
                [0., -xrel, jnp.sin(threl), 0.],
                [0., -1., 0., 1.]
            ]
        )


class RelativeDynamicallyExtendedUnicycle(ControlAffineDynamics):
    state_dim: int = 5
    control_dim: int = 4

    # def ode_dynamics(self, state, control, time=0):
    #     xrel, yrel, threl, v1, v2 = state
    #     om1, a1, om2, a2 = control
    #     return jnp.array([v2 * jnp.cos(threl) + om1 * yrel - v1,
    #                       v2 * jnp.sin(threl) - om1 * xrel,
    #                       om2 - om1,
    #                       a1,
    #                       a2])

    def open_loop_dynamics(self, state, time=0):
        xrel, yrel, threl, v1, v2 = state
        # om1, a1, om2, a1 = control
        return jnp.array(
            [
                -v1 + v2 * jnp.cos(threl),
                v2 * jnp.sin(threl),
                0.,
                0.,
                0.
            ]
        )

    def control_jacobian(self, state, time=0):
        xrel, yrel, threl, v1, v2 = state
        # om1, a1, om2, a1 = control
        return jnp.array(
            [
                [yrel, 0., 0., 0.],
                [xrel, 0., 0., 0.],
                [-1., 0., 1., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.]
            ]
        )
