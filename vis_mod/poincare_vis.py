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

def obj_2_xyz(S, in_degrees=False):
    az, el = S.parameters.azimuth_ellipticity(out_number=False, use_nan=False)
    x, y, z = azel_2_xyz(az, el)
    if in_degrees:
        ret = [x, y, z, az/degrees, el/degrees]
    else:
        ret = [x, y, z, az, el]
    return ret

def plot_poincare(*datasets, 
                  fig=None, 
                  figsize=(6, 6), 
                  draw_axes=True, 
                  draw_guides=True, 
                  param=None, 
                  in_degrees=False, 
                  param_name=None,
                  log=False,
                  colormap="Blackbody",
                  n_subplots=1,
                  datasets_per_subplot=None):
    
    n = len(datasets)
    if not isinstance(param, list):
        param = [param] * n
    if not isinstance(param_name, list):
        param_name = [param_name] * n
    if not isinstance(colormap, list):
        colormap = [colormap] * n

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
        annotations = [
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
        annotations = []

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
                             colorscale="Greys",
                             showscale=False,
                             lighting=lighting,
                             customdata=customdata,
                             hovertemplate=hovertemplate,
                             name="Sphere",
                             showlegend=False)
        fig.add_trace(Psphere, row=1, col=col)
    
    subplot_colorbar_count = [0] * (n_subplots + 1)

    for ds_idx, S in enumerate(datasets):
        S = S.copy()
        col = subplot_assignment[ds_idx]
    
        ds_param = param[ds_idx]
        ds_param_name = param_name[ds_idx] if param_name[ds_idx] is not None else S.name
        ds_colormap = colormap[ds_idx]

        # Position within this subplot (0, 1, 2, ...)
        within_subplot_idx = subplot_colorbar_count[col]
        subplot_colorbar_count[col] += 1

        x, y, z, az, el = obj_2_xyz(S, in_degrees=True)
        customdata = np.squeeze(np.dstack((az, el)))

        subplot_width = 1.0 / n_subplots
        colorbar_x = (col - 1) * subplot_width + subplot_width / 2
        colorbar_y = -0.1 - within_subplot_idx * 0.15

        if isinstance(ds_param, str):
                        Scolor = eval("S.parameters." + ds_param + "(out_number=False)")
                        colorbar = dict(
                            title=ds_param,
                            orientation='h',
                            ticklabelposition='outside bottom',
                            y=colorbar_y,
                            x=colorbar_x,
                            len=subplot_width * 0.8,
                            anchor="center"
                        )
        elif ds_param is not None:
                        Scolor = np.array(ds_param).flatten()
                        colorbar = dict(
                            title=ds_param_name,
                            orientation='h',
                            ticklabelposition='outside bottom',
                            y=colorbar_y,
                            x=colorbar_x,
                            len=subplot_width * 0.8,
                            xanchor='center',
                            tickfont=dict(size=15)
                        )
        else:
            n_points = len(x)
            Scolor = np.ones(n_points) * (ds_idx + 1) / n
            colorbar = {}

        marker = dict(size=10, color=Scolor, colorbar=colorbar, colorscale=ds_colormap)
        Fdata = go.Scatter3d(
            x=x, y=y, z=z,
            marker=marker,
            name=S.name,
            mode="markers",
            customdata=customdata,
            hovertemplate=hovertemplate)
        fig.add_trace(Fdata, row=1, col=col)

    #Plot figure
    axis = dict(showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False)
    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=0.85, y=0.85, z=0.85))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      width=int(figsize[0] * 100 * n_subplots),
                      height=int(figsize[1] * 100),
                      showlegend=False)
    fig.update_scenes(annotations=annotations,
                      xaxis=axis,
                      yaxis=axis,
                      zaxis=axis,
                      camera=camera)
    

    fig.show()
    return fig
    
    