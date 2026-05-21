import numpy as np
from py_pol.stokes import Stokes
from py_pol.jones_vector import Jones_vector
from py_pol.utils import degrees
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import (array, asarray, cos, exp, linspace, matrix, meshgrid,
                   ndarray, ones, outer, real, remainder, sin, size, sqrt,
                   zeros_like)
from scipy import datasets
from scipy.signal import fftconvolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#Will delete this data later, just for testing purposes. 
#J0x = np.array([
    #[0.26567846-0.9640617j,  0.0+0.0j],
    #[0.25248779-0.96760008j, 0.0+0.0j],
    #[0.23933232-0.97093771j, 0.0+0.0j],
    #[0.22621421-0.97407758j, 0.0+0.0j],
    #[0.21313557-0.97702263j, 0.0+0.0j],
    #[0.20009842-0.9797758j,  0.0+0.0j],
    #[0.18710469-0.98233998j, 0.0+0.0j],
    #[0.17415626-0.98471803j, 0.0+0.0j],
    #[0.16125493-0.98691279j, 0.0+0.0j],
    #[0.14840241-0.98892706j, 0.0+0.0j],
    #[0.13560039-0.99076361j, 0.0+0.0j]
#])

#J0y = np.array([
    #[0.0+0.0j, 0.26567846+0.9640617j],
    #[0.0+0.0j, 0.25248779+0.96760008j],
    #[0.0+0.0j, 0.23933232+0.97093771j],
    #[0.0+0.0j, 0.22621421+0.97407758j],
    #[0.0+0.0j, 0.21313557+0.97702263j],
    #[0.0+0.0j, 0.20009842+0.9797758j],
    #[0.0+0.0j, 0.18710469+0.98233998j],
    #[0.0+0.0j, 0.17415626+0.98471803j],
    #[0.0+0.0j, 0.16125493+0.98691279j],
    #[0.0+0.0j, 0.14840241+0.98892706j],
    #[0.0+0.0j, 0.13560039+0.99076361j]
#])

#J=Jones_vector("J0"); J.from_matrix(np.stack([J0x, J0y], axis=1))

#print(J)
####################################################################################

colors = matplotlib.colors.TABLEAU_COLORS
name_colors = list(colors)
linestyles = [
    'dotted', 'dashed', 'dashdot', 'loosely dotted', 'loosely dashed',
    'loosely dashdot'
]

def plot_ellipse(E, figsize=(6, 6), N_angles=91, draw_arrow=True, draw_titles=True, draw_labels=True,
                  limit='', *args, **kwargs):

    dif_index_arrow = 4
    # Calculate the electric field amplitudes and the delays
    if E._type == 'Jones_vector':
        E0x, E0y = E.parameters.amplitudes(shape=False)
        E0u = np.zeros(1)
    else:
        print("Error: Unsupported field type")
    delay = E.parameters.delay(shape=False)
    phase = E.parameters.global_phase(shape=False)
    if phase is None:
        phase = np.zeros_like(E0x)
    if np.isnan(phase).any():
        phase[np.isnan(phase)] = 0

    # Create the angle variables
    angles = linspace(0, 360 * degrees, N_angles)
    Angles, E0X = np.meshgrid(angles, E0x)
    _, E0Y = np.meshgrid(angles, E0y)
    _, Delay = np.meshgrid(angles, delay)
    _, Phase = np.meshgrid(angles, phase)
    if E._type == 'Jones_vector':
        is_linear = E.checks.is_linear(shape=False, out_number=False)
    else:
        is_linear = E.checks.is_linear(shape=False,
                                       out_number=False,
                                       use_nan=False)

    Ex = E0X * np.cos(Phase - Angles)
    Ey = E0Y * np.cos(Phase + Delay - Angles)
    if draw_arrow:
        dx = np.diff(Ex)
        dy = np.diff(Ey)
        dif = dx**2 + dy**2
        index_arrow = np.argmax(dif, axis=-1)
        cond = index_arrow >= N_angles -  + dif_index_arrow
        if np.any(cond):
            index_arrow[cond] = index_arrow[cond] - N_angles

    # Safety arrays
    if E.size < 2:
            is_pol = np.array([True])
    else:
            is_pol = np.ones_like(E0x).flatten()

    if limit in [0, '', [], None]:
        limit = np.array([E0x.max(), E0y.max(), E0u.max()]).max() * 1.2

    # Prepare the figure and the subplots
    fig = plt.figure(figsize=figsize)
    Nx, Ny, Nsubplots, Ncurves = (1, 1, 1, E.size)

    # Main loop
    ax = []
    for ind in range(E.size):  # Loop in curves
        # Initial considerations for the subplot
        indS = int(np.floor(ind / Ncurves))
        indC = int(ind % Ncurves)
        if indC == 0:
            axis = fig.add_subplot(Nx, Ny, indS + 1)
            ax.append(axis)
            if draw_titles:
                plt.title(E.name, fontsize=26)

        color = colors[name_colors[ind % 10]]
        if Ncurves == 1:
                string = 'Polarized'
        else:
                string = str(ind)

        if is_pol[ind]:
            axis.plot(Ex[ind, :], Ey[ind, :], label=string, color=color,  *args, **kwargs)
            if draw_arrow and ~is_linear[ind]:
                axis.arrow(Ex[ind, index_arrow[ind]],
                           Ey[ind, index_arrow[ind]],
                           Ex[ind, index_arrow[ind] + dif_index_arrow] - Ex[ind, index_arrow[ind]],
                           Ey[ind, index_arrow[ind] + dif_index_arrow] - Ey[ind, index_arrow[ind]],
                           width=0,
                           head_width=0.05 * limit,
                           linewidth=0,
                           color=color,
                           length_includes_head=True)
        if indC == Ncurves - 1:
            plt.axis('equal')
            plt.axis('square')
            plt.grid(True)
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-limit, limit)
            if draw_labels:
                axis.set_xlabel('$E_x$', fontsize=14)
                axis.set_ylabel('$E_y$', fontsize=14)
            else:
                axis.set_xticklabels([])
                axis.set_yticklabels([])

            plt.tight_layout()
            if Ncurves > 1:
                plt.legend()

    plt.show()
    return fig, ax
    

