import numpy as np
from py_pol.stokes import Stokes
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
S0 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
S1 = np.array([0.93969262, 0.93969262, 0.93969262, 0.93969262, 0.93969262, 0.93969262,
               0.93969262, 0.93969262, 0.93969262, 0.93969262, 0.93969262])
S2 = np.array([-0.29373713, -0.29841252, -0.30283834, -0.30701584, -0.3109464, -0.3146316,
               -0.31807315, -0.32127293, -0.32423294, -0.32695533, -0.32944238])
S3 = np.array([0.17520353, 0.16711597, 0.15895508, 0.15072841, 0.14244337, 0.13410718,
               0.12572689, 0.11730936, 0.10886128, 0.10038918, 0.0918994])

S = Stokes("Source 0"); S.from_matrix(np.stack([S0, S1, S2, S3], axis=1))
####################################################################################


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
                  colormap="Blackbody"):
    
    n = len(datasets)
    if not isinstance(param, list):
        param = [param] * n
    if not isinstance(param_name, list):
        param_name = [param_name] * n
    if not isinstance(colormap, list):
        colormap = [colormap] * n
    
    #S = S.copy()
    #Ndim = S.ndim
    #Nrows = 1
    #Ncolumns = 1
    #if Ndim <= 1:
         #S.shape = [1, 1, 1, S.size]
    #else:
        #S.shape = [1, 1, S.shape[0]] + [np.prod(S.shape[1:])]
    #Ones = np.ones(S.shape[3])
    
    # Reshape param
    #if param is not None and not isinstance(param, str):
        #param = np.reshape(param, S.shape)
    
    add_auxiliar = False
    if fig is None: # If no figure is provided, create a new one and add the axes and guides. If a figure is provided, it is assumed that the axes and guides have already been added.
        specs = [[{'type': 'surface'}]]
        fig = make_subplots(rows=1, cols=1, specs=specs)
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
        level = 0.2
        Psphere = go.Surface(x=x,
                             y=y,
                             z=z,
                             surfacecolor=np.ones_like(x) * level,
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

    for ds_idx, S in enumerate(datasets):
        S = S.copy()
        Ndim = S.ndim
        if Ndim <= 1:
            S.shape = [1, 1, 1, S.size]
        else:
            S.shape = [1, 1, S.shape[0]] + [np.prod(S.shape[1:])]
        
        Ones = np.ones(S.shape[3])
        ds_param = param[ds_idx]
        ds_param_name = param_name[ds_idx] if param_name[ds_idx] is not None else S.name

        # Reshape per-dataset param array if provided
        if ds_param is not None and not isinstance(ds_param, str):
            ds_param = np.reshape(ds_param, S.shape)

        for indR, Srow in enumerate(S):
            for indC, Scol in enumerate(Srow):
    
                # Set colorbar once per dataset
                if isinstance(ds_param, str):
                        colorbar = dict(
                            title=ds_param,
                            orientation='h',
                            ticklabelposition='outside bottom',
                            y=-0.1 - ds_idx * 0.15,
                            x=0.5,
                            len=0.5
                        )
                elif ds_param is not None:
                        colorbar = dict(
                            title=ds_param_name,
                            orientation='h',
                            ticklabelposition='outside bottom',
                            y=-0.1 - ds_idx * 0.15,
                            x=0.5,
                            len=0.5,
                            tickfont=dict(size=15)
                        )
                else:
                        colorbar = {}

            # Plotting axes
                if draw_axes and add_auxiliar:
                        fig.add_traces(
                            [axis_px, axis_mx, axis_py, axis_my, axis_pz, axis_mz],
                            rows=1,
                            cols=1)

            # Plotting curves
            if draw_guides and add_auxiliar:
                fig.add_traces([circle_z, circle_y, circle_x],
                                    rows=1,
                                    cols=1)
            if add_auxiliar:
                fig.add_trace(Psphere, row=indR + 1, col=indC + 1)
                    
            # Loop in traces
            for indT, Strace in enumerate(Scol):
                    if isinstance(ds_param, str):
                        Scolor = eval("Strace.parameters." + ds_param +
                                    "(out_number=False)")
                    elif ds_param is not None:
                        Scolor = ds_param[indR, indC, indT, ...]
                    else:
                        Scolor = Ones * (indT + 1) / Scol.shape[0]
                    if ds_param is not None:
                        if log:
                            cond = Scolor <= 0
                            if np.any(cond):
                                Scolor[cond] = np.inf
                                Scolor[cond] = np.min(Scolor) / 10
                            Scolor = np.log10(Scolor)
                            colorbar["title"] = "log(" + colorbar["title"] + ")"
                            
                        elif in_degrees:
                            Scolor = Scolor / degrees
                            colorbar["title"] += " (deg)"

                    x, y, z, az, el = obj_2_xyz(Strace, in_degrees=True)
                    customdata = np.squeeze(np.dstack((az, el)))
                    ds_colormap = colormap[ds_idx]
                    marker = dict(size=10, color=Scolor, colorbar=colorbar, colorscale=ds_colormap)
                    Fdata = go.Scatter3d(x=x,
                                                y=y,
                                                z=z,
                                                marker=marker,
                                                name=S.name,
                                                mode="markers",
                                                customdata=customdata,
                                                hovertemplate=hovertemplate)
                    fig.add_trace(Fdata, row=indR + 1, col=indC + 1)

    #Plot figure
    axis = dict(showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False)
    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=0.85, y=0.85, z=0.85))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      width=int(figsize[0] * 100),
                      height=int(figsize[1] * 100),
                      showlegend=False)
    fig.update_scenes(annotations=annotations,
                      xaxis=axis,
                      yaxis=axis,
                      zaxis=axis,
                      camera=camera)
    

    fig.show()
    return fig
    
    
#plot_poincare(S, param="azimuth", in_degrees=True, colormap="Viridis")