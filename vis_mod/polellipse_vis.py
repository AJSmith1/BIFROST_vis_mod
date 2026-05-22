import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from numpy import linspace
 
degrees = np.pi / 180
 
colors = matplotlib.colors.TABLEAU_COLORS
name_colors = list(colors)

def plot_ellipse(*datasets,
                 figsize=(6, 6),
                 N_angles=91,
                 draw_arrow=True,
                 draw_titles=True,
                 draw_labels=True,
                 limit=None,
                 param=None,
                 param_name=None,
                 colormap=None,
                 n_subplots=1,
                 datasets_per_subplot=None,
                 **kwargs):
    
    n = len(datasets)
    if n == 0:
        raise ValueError("At least one dataset must be provided.")
 
    # Normalise list arguments to length n
    if not isinstance(param, list):
        param = [param] * n
    if not isinstance(param_name, list):
        param_name = [param_name] * n
    if not isinstance(colormap, list):
        colormap = [colormap] * n
 
    # Replace None colormaps with defaults
    for i in range(n):
        if colormap[i] is None:
            colormap[i] = 'viridis' if param[i] is not None else None
 
    # Distribute datasets across subplots
    if datasets_per_subplot is None:
        base = n // n_subplots
        rem  = n % n_subplots
        datasets_per_subplot = [base + (1 if i < rem else 0) for i in range(n_subplots)]
 
    subplot_assignment = []
    for sp_idx, count in enumerate(datasets_per_subplot):
        subplot_assignment.extend([sp_idx] * count)  # 0-indexed
 
    # Compute per-subplot x and y limits independently
    use_fixed_limit = limit not in [0, '', [], None]

    # Computing ellipse data
    all_Ex = []
    all_Ey = []
    for E in datasets:
        if E._type == 'Jones_vector':
            E0x, E0y = E.parameters.amplitudes(shape=False)
        else:
            raise ValueError("Only Jones_vector is supported.")

        delay = E.parameters.delay(shape=False)
        phase = E.parameters.global_phase(shape=False)
        if phase is None:
            phase = np.zeros_like(E0x)
        if np.isnan(phase).any():
            phase[np.isnan(phase)] = 0

        angles = linspace(0, 360 * degrees, N_angles)
        Angles, E0X = np.meshgrid(angles, E0x)
        _,      E0Y = np.meshgrid(angles, E0y)
        _,    Delay = np.meshgrid(angles, delay)
        _,    Phase = np.meshgrid(angles, phase)

        Ex = E0X * np.cos(Phase - Angles)
        Ey = E0Y * np.cos(Phase + Delay - Angles)
        all_Ex.append(Ex)
        all_Ey.append(Ey)

    # Per-subplot limits
    subplot_xlim = {}
    subplot_ylim = {}
    for sp_idx in range(n_subplots):
        ds_in_sp = [i for i in range(n) if subplot_assignment[i] == sp_idx]
        if use_fixed_limit:
            subplot_xlim[sp_idx] = (-limit, limit)
            subplot_ylim[sp_idx] = (-limit, limit)
        else:
            x_vals = np.concatenate([all_Ex[i].ravel() for i in ds_in_sp])
            y_vals = np.concatenate([all_Ey[i].ravel() for i in ds_in_sp])
            x_max = np.abs(x_vals).max() * 1.2 if x_vals.size else 1.0
            y_max = np.abs(y_vals).max() * 1.2 if y_vals.size else 1.0
            subplot_xlim[sp_idx] = (-x_max, x_max)
            subplot_ylim[sp_idx] = (-y_max, y_max)

    # Making arrow heads, size relative to subplot limits
    def arrow_head(sp_idx):
        xrange = subplot_xlim[sp_idx][1] - subplot_xlim[sp_idx][0]
        yrange = subplot_ylim[sp_idx][1] - subplot_ylim[sp_idx][0]
        return 0.05 * max(xrange, yrange) / 2

    # n_subplots columns, 1 row
    fig, axes_grid = plt.subplots(1, n_subplots, figsize=(figsize[0] * n_subplots, figsize[1]),
                                  squeeze=False)
    axes_grid = axes_grid[0]  
 
    dif_index_arrow = 4
 
    for ds_idx, E in enumerate(datasets):
        sp_idx   = subplot_assignment[ds_idx]
        axis     = axes_grid[sp_idx]
        ds_param = param[ds_idx]
        ds_pname = param_name[ds_idx] if param_name[ds_idx] is not None else E.name
        ds_cmap  = colormap[ds_idx]
 
        # Field stuff
        if E._type == 'Jones_vector':
            E0x, E0y = E.parameters.amplitudes(shape=False)
        else:
            raise ValueError(f"Dataset {ds_idx}: only Jones_vector is supported.")
 
        delay = E.parameters.delay(shape=False)
        phase = E.parameters.global_phase(shape=False)
        if phase is None:
            phase = np.zeros_like(E0x)
        if np.isnan(phase).any():
            phase[np.isnan(phase)] = 0
 
        is_linear = E.checks.is_linear(shape=False, out_number=False)
        is_pol    = np.ones(E.size, dtype=bool) if E.size >= 2 else np.array([True])
 
        Ex = all_Ex[ds_idx]
        Ey = all_Ey[ds_idx]
 
        # Arrows
        if draw_arrow:
            dx = np.diff(Ex)
            dy = np.diff(Ey)
            dif_sq = dx**2 + dy**2
            index_arrow = np.argmax(dif_sq, axis=-1)
            cond = index_arrow >= N_angles - dif_index_arrow
            if np.any(cond):
                index_arrow[cond] -= N_angles
 
        # Color mapping 
        if ds_param is not None:
            param_arr = np.asarray(ds_param).flatten()
            norm      = mcolors.Normalize(vmin=param_arr.min(), vmax=param_arr.max())
            cmap_obj  = cm.get_cmap(ds_cmap)
            sm        = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
 
            def get_color(ind):
                return cmap_obj(norm(param_arr[ind]))
        else:
            def get_color(ind):
                return colors[name_colors[(ind + ds_idx * 10) % 10]]
 
        # Plot dataset ellipses
        for ind in range(E.size):
            if not is_pol[ind]:
                continue
 
            color  = get_color(ind)
            label  = f"{ds_pname}: {ds_param[ind]:.3g}" if ds_param is not None else (
                     E.name if E.size == 1 else f"{E.name}[{ind}]")
 
            axis.plot(Ex[ind, :], Ey[ind, :], color=color, label=label, **kwargs)
 
            if draw_arrow and not is_linear[ind]:
                axis.arrow(
                    Ex[ind, index_arrow[ind]],
                    Ey[ind, index_arrow[ind]],
                    Ex[ind, index_arrow[ind] + dif_index_arrow] - Ex[ind, index_arrow[ind]],
                    Ey[ind, index_arrow[ind] + dif_index_arrow] - Ey[ind, index_arrow[ind]],
                    width=0,
                    head_width=arrow_head(sp_idx),
                    linewidth=0,
                    color=color,
                    length_includes_head=True,
                )
 
        # colorbars
        if ds_param is not None:
            cbar = fig.colorbar(sm, ax=axis, orientation='horizontal',
                                pad=0.12, fraction=0.046)
            cbar.set_label(ds_pname, fontsize=11)
 
    # Axis formatting and titles
    for sp_idx, axis in enumerate(axes_grid):
        axis.grid(True)
        axis.set_xlim(*subplot_xlim[sp_idx])
        axis.set_ylim(*subplot_ylim[sp_idx])
 
        if draw_titles:
            names = [datasets[i].name for i in range(n) if subplot_assignment[i] == sp_idx]
            axis.set_title(', '.join(names), fontsize=14)
 
        if draw_labels:
            axis.set_xlabel('$E_x$', fontsize=14)
            axis.set_ylabel('$E_y$', fontsize=14)
        else:
            axis.set_xticklabels([])
            axis.set_yticklabels([])
 
        ds_in_sp = [i for i in range(n) if subplot_assignment[i] == sp_idx]
        if any(param[i] is None for i in ds_in_sp):
            axis.legend(fontsize=8)
 
    plt.tight_layout()
    plt.show()
    return fig, list(axes_grid)