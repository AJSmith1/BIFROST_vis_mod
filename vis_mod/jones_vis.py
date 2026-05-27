import numpy as np
from py_pol.utils import degrees
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def azel_2_xyz(az, el, pol=1):
    phi = 2*az #NOTE: azimuth is multiplied by 2 to match the Poincare sphere convention
    chi = -2*el + 90*degrees
    x = pol * np.sin(chi) * np.cos(phi)
    y = pol * np.sin(chi) * np.sin(phi)
    z = pol * np.cos(chi)
    return x, y, z

def jones_to_mueller(J):
    U = np.array([
        [1,  0,  0,  1],
        [1,  0,  0, -1],
        [0,  1,  1,  0],
        [0, 1j, -1j, 0],
    ])
    M = (U @ np.kron(J, J.conj()) @ np.linalg.inv(U)).real
    return M

def mueller_to_axis_angle(M):
    R = M[1:4, 1:4]  # Stokes sub-block (S1,S2,S3)

    # Rotation angle from trace: tr(R) = 1 + 2*cos(theta)
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1, 1)
    angle = np.arccos(cos_angle)

    # Rotation axis from the skew-symmetric part
    skew = (R - R.T) / 2
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        axis = np.array([1, 0, 0])
    else:
        axis /= norm
    return axis, angle

def rotation_arc(axis, start_vec, angle, n_points=120):

    # Build an orthonormal frame: axis, u (start), v (u × axis)
    u = start_vec - np.dot(start_vec, axis) * axis
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-10:
        perp = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
        u = perp - np.dot(perp, axis) * axis
        u /= np.linalg.norm(u)
    else:
        u /= u_norm
    v = np.cross(axis, u)

    ts = np.linspace(0, angle, n_points)
    arc = np.outer(np.cos(ts), u) + np.outer(np.sin(ts), v)
    # Project onto the sphere (radius = distance of start_vec from axis)
    r = np.linalg.norm(start_vec - np.dot(start_vec, axis) * axis)
    arc *= r
    arc += np.dot(start_vec, axis) * axis[np.newaxis, :]
    return arc[:, 0], arc[:, 1], arc[:, 2]

def plot_jones(*datasets, 
               fig=None, 
                figsize=(6, 6), 
                draw_axes=True, 
                draw_guides=True,
                n_subplots=1,
                datasets_per_subplot=None, 
                axis_color="black",
                arc_color="crimson",
                axis_length=1.25,
                arc_width=5,
                axis_width=8,
                n_arrows=3,
                start_vec=None,
                show_angle_label=True,):
    
    n = len(datasets)

    # How many datasets go in each subplot
    # e.g. datasets_per_subplot=[1,1] means one dataset per subplot
    if datasets_per_subplot is None:
        datasets_per_subplot = [n // n_subplots] * n_subplots
        # distribute remainder
        for i in range(n % n_subplots):
            datasets_per_subplot[i] += 1

    # Build subplot assignment: which subplot does each dataset go into?
    subplot_assignment = []
    for subplot_idx, count in enumerate(datasets_per_subplot):
        subplot_assignment.extend([subplot_idx + 1] * count)  # 1-indexed for Plotly

    add_auxiliar = False
    if fig is None: # If no figure is provided, create a new one and add the axes and guides. If a figure is provided, it is assumed that the axes and guides have already been added.
        specs = [[{'type': 'surface'} for _ in range(n_subplots)]]
        fig = make_subplots(rows=1, cols=n_subplots, specs=specs, horizontal_spacing=0.05)
        add_auxiliar = True #This variable is used to avoid plotting the axes and guides multiple times when the function is called several times with the same figure.
    lighting = dict(ambient=0.9,
                    diffuse=0.,
                    roughness=0.5,
                    specular=0.05,
                    fresnel=0.2)
    hovertemplate = "S1: %{x:.3f}<br>S2: %{y:.3f}<br>S3: %{z:.3f}<br>Parameter: %{customdata:.3f}" 

    # Defining axes
    if draw_axes and add_auxiliar:
        line_p = dict(width=6, color="red")
        line_m = dict(width=6, color="red", dash="dash")
        marker = dict(size=1)
        axis_px = go.Scatter3d(x=[0, 1.2],
                               y=[0, 0],
                               z=[0, 0],
                               line=line_p,
                               marker=marker,
                               hoverinfo="skip",
                               name="S1")
        axis_mx = go.Scatter3d(x=[0, -1.2],
                               y=[0, 0],
                               z=[0, 0],
                               line=line_m,
                               marker=marker,
                               hoverinfo="skip",
                               name="-S1")
        line_p["color"], line_m["color"] = ("green", "green")
        axis_py = go.Scatter3d(x=[0, 0],
                               y=[0, 1.2],
                               z=[0, 0],
                               line=line_p,
                               marker=marker,
                               hoverinfo="skip",
                               name="S2")
        axis_my = go.Scatter3d(x=[0, 0],
                               y=[0, -1.2],
                               z=[0, 0],
                               line=line_m,
                               marker=marker,
                               hoverinfo="skip",
                               name="-S2")
        line_p["color"], line_m["color"] = ("blue", "blue")
        axis_pz = go.Scatter3d(x=[0, 0],
                               y=[0, 0],
                               z=[0, 1.2],
                               line=line_p,
                               marker=marker,
                               hoverinfo="skip",
                               name="S3")
        axis_mz = go.Scatter3d(x=[0, 0],
                               y=[0, 0],
                               z=[0, -1.2],
                               line=line_m,
                               marker=marker,
                               hoverinfo="skip",
                               name="-S3")
        size = 30
        axis_annotations = [
            dict(showarrow=False,
                 x=1.2,
                 y=0,
                 z=0,
                 text="S1",
                 font=dict(color="red", size=size, family="Times New Roman"),
                 xshift=15,
                 yshift=15),
            dict(showarrow=False,
                 x=-1.2,
                 y=0,
                 z=0,
                 text="-S1",
                 font=dict(color="red", size=size, family="Times New Roman"),
                 xshift=-15,
                 yshift=-15),
            dict(showarrow=False,
                 x=0,
                 y=1.2,
                 z=0,
                 text="S2",
                 font=dict(color="green", size=size, family="Times New Roman"),
                 xshift=15,
                 yshift=15),
            dict(showarrow=False,
                 x=0,
                 y=-1.2,
                 z=0,
                 text="-S2",
                 font=dict(color="green", size=size, family="Times New Roman"),
                 xshift=-15,
                 yshift=-15),
            dict(showarrow=False,
                 x=0,
                 y=0,
                 z=1.2,
                 text="S3",
                 font=dict(color="blue", size=size, family="Times New Roman"),
                 xshift=15,
                 yshift=15),
            dict(showarrow=False,
                 x=0,
                 y=0,
                 z=-1.2,
                 text="-S3",
                 font=dict(color="blue", size=size, family="Times New Roman"),
                 xshift=-15,
                 yshift=-15),
        ]
    else:
        axis_annotations = []

    # Defining curves
    if draw_guides and add_auxiliar:
        angle = np.linspace(0, 360 * degrees, 361)
        line = dict(width=2, color="darkslategrey", dash="dashdot")
        x = np.sin(angle)
        y = np.cos(angle)
        z = np.zeros_like(angle)
        circle_z = go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                line=line,
                                mode="lines",
                                name="S3=0")
        x = np.sin(angle)
        z = np.cos(angle)
        y = np.zeros_like(angle)
        circle_y = go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                line=line,
                                mode="lines",
                                name="S2=0")
        y = np.sin(angle)
        z = np.cos(angle)
        x = np.zeros_like(angle)
        circle_x = go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                line=line,
                                mode="lines",
                                name="S1=0")
        
        #Defining sphere    
    if add_auxiliar:
        el, az = np.mgrid[-45 * degrees:45 * degrees:100j,
                          0:180 * degrees:100j]
        x, y, z = azel_2_xyz(az, el)
        customdata = np.dstack(
            (az / degrees, el / degrees))

    # Add axes, guides, sphere to every subplot
    for col in range(1, n_subplots + 1):
        if draw_axes:
            fig.add_traces([axis_px, axis_mx, axis_py, axis_my, axis_pz, axis_mz],
                               rows=[1]*6, cols=[col]*6)
        if draw_guides:
            fig.add_traces([circle_z, circle_y, circle_x],
                               rows=[1]*3, cols=[col]*3)
        Psphere = go.Surface(x=x,
                             y=y,
                             z=z,
                             surfacecolor=np.ones_like(x) * 0.2,
                             cmin=0,
                             cmax=1,
                             opacity=0.7,
                             colorscale="Blues",
                             showscale=False,
                             lighting=lighting,
                             customdata=customdata,
                             hovertemplate=hovertemplate,
                             name="Sphere",
                             showlegend=False)
        fig.add_trace(Psphere, row=1, col=col)

    subplot_angle_annotations = {i+1: [] for i in range(n_subplots)}

    for ds_idx, J in enumerate(datasets):
        J = J.copy()
        col = subplot_assignment[ds_idx]

        M = jones_to_mueller(J)
        axis, angle = mueller_to_axis_angle(M)

        ax, ay, az = axis * axis_length
        axis_trace = go.Scatter3d(
            x=[-ax, ax], y=[-ay, ay], z=[-az, az],
            mode="lines",
            line=dict(color=axis_color, width=axis_width),
            name="Rotation axis",
            hoverinfo="skip",
            )
    
        sv = start_vec   

        if sv is None:
            perp = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
            sv = perp - np.dot(perp, axis) * axis
            sv /= np.linalg.norm(sv)

        x_arc, y_arc, z_arc = rotation_arc(axis, sv, angle)

        # Arc trace
        arc_trace = go.Scatter3d(
            x=x_arc, y=y_arc, z=z_arc,
            mode="lines",
            line=dict(color=arc_color, width=arc_width),
            name="Rotation arc (%.1f°)" % (angle / degrees),
            hoverinfo="name",
        )

            # Arrow traces
        arrow_traces = []
        if n_arrows > 0:
            indices = np.linspace(10, len(x_arc) - 2, n_arrows, dtype=int)
            for i in indices:
                # Tangent direction at this arc point
                tangent = np.array([
                    x_arc[i + 1] - x_arc[i - 1],
                    y_arc[i + 1] - y_arc[i - 1],
                    z_arc[i + 1] - z_arc[i - 1],
                ])
                tangent /= np.linalg.norm(tangent)
                cone = go.Cone(
                    x=[x_arc[i]], y=[y_arc[i]], z=[z_arc[i]],
                    u=[tangent[0]], v=[tangent[1]], w=[tangent[2]],
                    sizemode="absolute", sizeref=0.12,
                    colorscale=[[0, arc_color], [1, arc_color]],
                    showscale=False,
                    anchor="tip",
                    hoverinfo="skip",
                )
                arrow_traces.append(cone)

        # Angle annotations
        if show_angle_label:
            mid = len(x_arc) // 2
            subplot_angle_annotations[col].append(dict(
            x=x_arc[mid] * 1.1, y=y_arc[mid] * 1.1, z=z_arc[mid] * 1.1,
            text="%.1f°" % (angle / degrees),
            showarrow=False,
            font=dict(color=arc_color, size=18, family="Times New Roman"),
        ))

        subplot_width = 1.0 / n_subplots

        fig.add_trace(axis_trace, row=1, col=col)
        fig.add_trace(arc_trace, row=1, col=col)

        for t in arrow_traces:
            fig.add_trace(t, row=1, col=col)

    #Plot figure
    axis_dict = dict(showbackground=False, showgrid=False, zeroline=False, visible=False)
    camera = dict(up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.85, y=0.85, z=0.85))

    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                    width=int(figsize[0] * 100 * n_subplots),
                    height=int(figsize[1] * 100),
                    showlegend=False)

    for col in range(1, n_subplots + 1):
        scene_key = "scene" if col == 1 else f"scene{col}"
        annotations = axis_annotations + subplot_angle_annotations[col]
        fig.layout[scene_key].annotations = annotations
        fig.layout[scene_key].xaxis = axis_dict
        fig.layout[scene_key].yaxis = axis_dict
        fig.layout[scene_key].zaxis = axis_dict
        fig.layout[scene_key].camera = camera

    fig.show()
    return fig
