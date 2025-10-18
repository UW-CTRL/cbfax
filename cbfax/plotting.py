import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets


def _plot_halfspace_lessthan(
    normal_vector, constant, xlim=(-10, 10), ylim=(-10, 10), linestyle="-", alpha=0.5
):
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


def plot_halfspace(normal_vector, constant, relation, xlim=(-10, 10), ylim=(-10, 10), alpha=0.5):
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


def plot_cbf(barrier_func, rest_values=[], xlim=(-10, 10), ylim=(-10, 10), N=101):
    # Create a grid of points
    x = jnp.linspace(xlim[0], xlim[1], N)
    y = jnp.linspace(ylim[0], ylim[1], N)
    X, Y = jnp.meshgrid(x, y)
    if rest_values is not None:
        rest_state = [jnp.ones_like(X) * v for v in rest_values]
    else:
        rest_state = jnp.zeros([X.shape[0], 0])
    XYs = jnp.stack([X, Y] + rest_state, axis=-1).reshape(-1, 2 + len(rest_values))


    # Evaluate the barrier function
    Z = jax.vmap(barrier_func)(XYs).reshape(N, N)

    # Plot the CBF
    plt.contourf(X, Y, Z >= 0, alpha=0.6, colors=["#ff9999", "#99ff99"])
    plt.contour(X, Y, Z, alpha=0.7, levels=10, colors="lightgray")
    plt.contour(X, Y, Z, levels=[0], colors="black")

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


# # Create interactive widgets
# a_slider = widgets.FloatSlider(value=1, min=-10, max=10, step=0.1, description='a')
# b_slider = widgets.FloatSlider(value=1, min=-10, max=10, step=0.1, description='b')
# c_slider = widgets.FloatSlider(value=1, min=-10, max=10, step=0.1, description='c')
# relation_slider = widgets.SelectionSlider(options=["<=", ">=", "<", ">"], description="relation")
# # Use interact to create the sliders
# interact(interactive_halfspace, a=a_slider, b=b_slider, c=c_slider, relation=relation_slider)


# # Example usage
# normal_vector = [1, 1]
# constant = 0
# plot_halfspace(normal_vector, constant)
# # normal_vector = [-1, -1]
# # plot_halfspace(normal_vector, constant)
