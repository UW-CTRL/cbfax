import jax
import jax.numpy as jnp
from typing import Callable
import matplotlib.pyplot as plt
from cbfax.cbf import ControlBarrierFunction, ControlLyapunovFunction


def _plot_halfspace_lessthan(
    normal_vector: jnp.ndarray, constant: float, xlim=(-10, 10), ylim=(-10, 10), linestyle="-", alpha=0.5
):
    """Plots the halfspace defined by the inequality a*x + b*y <= c, where (a, b) is the normal vector and c is the constant.
    Arguments:
        normal_vector: A 2D vector (a, b) that defines the normal vector of the halfspace.
        constant: The constant c in the inequality a*x + b*y <= c.
        xlim: The limits for the x-axis.
        ylim: The limits for the y-axis.
        linestyle: The style of the boundary line (default is solid).
        alpha: The transparency of the halfspace (default is 0.5)."""
    # Define the normal vector and constant
    a, b = normal_vector
    c = constant

    # Create a grid of points
    x = jnp.linspace(xlim[0], xlim[1], 400)
    y = jnp.linspace(ylim[0], ylim[1], 400)
    X, Y = jnp.meshgrid(x, y)

    # Calculate the values of the halfspace
    Z = a * X + b * Y + c

    # Plot the halfspace
    # plt.contourf(X, Y, Z <= c, alpha=0.5, colors=['#ff9999', '#9999ff'])
    plt.contourf(X, Y, Z <= 0, alpha=alpha, colors=["#ffb09c", "#E0FFD2"])
    plt.contour(X, Y, Z, levels=[0], colors="black", linestyles=linestyle)

    # Set the limits and labels
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r"$u_1$")
    plt.ylabel(r"$u_2$")
    # plt.title(f'Halfspace: {a}x + {b}y ≤ {c}')

    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    # plt.show()


def plot_halfspace(normal_vector: jnp.ndarray, constant: float, relation: str, xlim=(-10, 10), ylim=(-10, 10), alpha=0.5):
    """Plots the halfspace defined by the inequality a*x + b*y <= c, a*x + b*y < c, a*x + b*y >= c, or a*x + b*y > c, where (a, b) is the normal vector and c is the constant.
    Arguments:
        normal_vector: A 2D vector (a, b) that defines the normal vector of the halfspace.
        constant: The constant c in the inequality a*x + b*y <= c, a*x + b*y < c, a*x + b*y >= c, or a*x + b*y > c.
        relation: A string that specifies the type of inequality ("<=", "<", ">=", ">").
        xlim: The limits for the x-axis.
        ylim: The limits for the y-axis.
        alpha: The transparency of the halfspace (default is 0.5).
    """

    if relation == "<=":
        _plot_halfspace_lessthan(
            normal_vector, constant, xlim=xlim, ylim=ylim, linestyle="-", alpha=alpha
        )
    elif relation == "<":
        _plot_halfspace_lessthan(
            normal_vector, constant, xlim=xlim, ylim=ylim, linestyle="--", alpha=alpha
        )
    elif relation == ">=":
        _plot_halfspace_lessthan(
            [-normal_vector[0], -normal_vector[1]],
            -constant,
            xlim=xlim,
            ylim=ylim,
            linestyle="-",
            alpha=alpha
        )
    elif relation == ">":
        _plot_halfspace_lessthan(
            [-normal_vector[0], -normal_vector[1]],
            -constant,
            xlim=xlim,
            ylim=ylim,
            linestyle="--",
            alpha=alpha
        )


def interactive_halfspace(a, b, c, relation):
    plot_halfspace([a, b], c, relation)

def plot_cbf(cbf: ControlBarrierFunction, rest_values=None, xlim=(-10, 10), ylim=(-10, 10), N=101, fill=True, zorder=0):
    """Plots the control barrier function defined by the given barrier function.
    Arguments:
        cbf: A function that takes in a state and outputs a scalar value representing the control barrier function.
        rest_values: A list of values for the remaining state dimensions (if any). For 2D and higher-dimensional states,
            the first two dimensions are assumed to be the x and y coordinates for plotting. For 1D states, only the x-axis is varied.
        xlim: The limits for the x-axis.
        ylim: The limits for the y-axis.
        N: The number of points to use in each dimension for the grid."""
    if rest_values is None:
        rest_values = []

    plot_dimension = cbf.state_dim

    if plot_dimension == 1:
        x = jnp.linspace(xlim[0], xlim[1], N)
        Xs = x.reshape(-1, 1)
        Z = jax.vmap(cbf)(Xs)

        plt.fill_between(x, Z, 0, where=Z >= 0, alpha=0.6, color="#99ff99")
        plt.fill_between(x, Z, 0, where=Z < 0, alpha=0.6, color="#ff9999")
        plt.plot(x, Z, color="black")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlim(xlim)
        plt.xlabel("x")
        plt.ylabel("b(x)")
        plt.grid(True)
        return

    # Create a grid of points
    x = jnp.linspace(xlim[0], xlim[1], N)
    y = jnp.linspace(ylim[0], ylim[1], N)
    X, Y = jnp.meshgrid(x, y)
    rest_state = [jnp.ones_like(X) * v for v in rest_values]
    XYs = jnp.stack([X, Y] + rest_state, axis=-1).reshape(-1, 2 + len(rest_values))


    # Evaluate the barrier function
    Z = jax.vmap(cbf)(XYs).reshape(N, N)

    # Plot the CBF
    if fill:
        plt.contourf(X, Y, Z, alpha=0.6, levels=10, cmap="jet", zorder=zorder)
        plt.colorbar()
    else:
        plt.contourf(X, Y, Z >= 0, alpha=0.6, colors=["#ff9999", "#99ff99"], zorder=zorder)
    plt.contour(X, Y, Z, alpha=0.7, levels=10, colors="lightgray", zorder=zorder)
    plt.contour(X, Y, Z, levels=[0], colors="black", zorder=zorder)

    # Set the limits and labels
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title(f'Control Barrier Function: {a}x^2 + {b}y^2 + {c}')

    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.axis("equal")
    
def plot_clf(cbf: ControlLyapunovFunction, rest_values=None, xlim=(-10, 10), ylim=(-10, 10), N=101, fill=True, zorder=0):
    """Plots the control lyapunov function defined by the given lyapunov function.
    Arguments:
        cbf: A function that takes in a state and outputs a scalar value representing the control barrier function.
        rest_values: A list of values for the remaining state dimensions (if any). For 2D and higher-dimensional states,
            the first two dimensions are assumed to be the x and y coordinates for plotting. For 1D states, only the x-axis is varied.
        xlim: The limits for the x-axis.
        ylim: The limits for the y-axis.
        N: The number of points to use in each dimension for the grid."""
    if rest_values is None:
        rest_values = []

    plot_dimension = cbf.state_dim

    if plot_dimension == 1:
        x = jnp.linspace(xlim[0], xlim[1], N)
        Xs = x.reshape(-1, 1)
        Z = jax.vmap(cbf)(Xs)

        plt.fill_between(x, Z, 0, where=Z >= 0, alpha=0.6, color="#99ff99")
        plt.fill_between(x, Z, 0, where=Z < 0, alpha=0.6, color="#ff9999")
        plt.plot(x, Z, color="black")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlim(xlim)
        plt.xlabel("x")
        plt.ylabel("b(x)")
        plt.grid(True)
        return

    # Create a grid of points
    x = jnp.linspace(xlim[0], xlim[1], N)
    y = jnp.linspace(ylim[0], ylim[1], N)
    X, Y = jnp.meshgrid(x, y)
    rest_state = [jnp.ones_like(X) * v for v in rest_values]
    XYs = jnp.stack([X, Y] + rest_state, axis=-1).reshape(-1, 2 + len(rest_values))


    # Evaluate the barrier function
    Z = jax.vmap(cbf)(XYs).reshape(N, N)

    # Plot the CBF
    # plt.contourf(X, Y, Z >= 0, alpha=0.6, colors=["#ff9999", "#99ff99"])
    if fill:
        plt.contourf(X, Y, Z, alpha=0.6, levels=10, cmap="jet", zorder=zorder)
    else:
        plt.contourf(X, Y, Z >= 0, alpha=0.6, colors=["#ff9999", "#99ff99"], zorder=zorder)
    plt.colorbar()
    plt.contour(X, Y, Z, alpha=0.7, levels=10, colors="lightgray", zorder=zorder)
    plt.contour(X, Y, Z, levels=[0], colors="black", zorder=zorder)

    # Set the limits and labels
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title(f'Control Barrier Function: {a}x^2 + {b}y^2 + {c}')

    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.axis("equal")
