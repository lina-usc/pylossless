import nibabel
import importlib
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse
import plotly.graph_objs as go
from utils.helper_functions import mesh_edges, smoothing_matrix

drc = importlib.import_module("utils.dash_reusable_components")
figs = importlib.import_module("utils.figures")

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server

DEFAULT_COLORSCALE = [[0, 'rgb(12,51,131)'], [0.25, 'rgb(10,136,186)'],
                      [0.5, 'rgb(242,211,56)'], [0.75, 'rgb(242,143,56)'], [1, 'rgb(217,30,30)']]

DEFAULT_COLORSCALE_NO_INDEX = [ea[1] for ea in DEFAULT_COLORSCALE]


def plotly_triangular_mesh(vertices, faces, intensities=None, colorscale="Viridis",
                           flatshading=False, showscale=False, reversescale=False, plot_edges=False):
    ''' vertices = a numpy array of shape (n_vertices, 3)
        faces = a numpy array of shape (n_faces, 3)
        intensities can be either a function of (x,y,z) or a list of values '''

    x, y, z = vertices.T
    I, J, K = faces.T

    mesh = dict(
        type='mesh3d',
        hoverinfo='none',
        x=x, y=y, z=z,
        colorscale=colorscale,
        intensity=intensities,
        flatshading=flatshading,
        i=I, j=J, k=K,
        name='',
        showscale=showscale
    )

    mesh.update(lighting=dict(ambient=0.8,
                              diffuse=1,
                              fresnel=0.1,
                              specular=1,
                              roughness=0.1,
                              facenormalsepsilon=1e-6,
                              vertexnormalsepsilon=1e-12))

    mesh.update(lightposition=dict(x=100,
                                   y=200,
                                   z=0))

    if showscale is True:
        mesh.update(colorbar=dict(thickness=20, ticklen=4, len=0.75))

    if plot_edges is False:  # the triangle sides are not plotted
        return [mesh]
    else:  # plot edges
        # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        # None separates data corresponding to two consecutive triangles
        tri_vertices = vertices[faces]
        Xe = []
        Ye = []
        Ze = []
        for T in tri_vertices:
            Xe += [T[k % 3][0] for k in range(4)] + [None]
            Ye += [T[k % 3][1] for k in range(4)] + [None]
            Ze += [T[k % 3][2] for k in range(4)] + [None]
        # define the lines to be plotted
        lines = dict(type='scatter3d',
                     x=Xe,
                     y=Ye,
                     z=Ze,
                     mode='lines',
                     name='',
                     line=dict(color='rgb(70,70,70)', width=1)
                     )
        return [mesh, lines]


data_path = mne.datasets.sample.data_path()

fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
freesurfer_path = data_path + "/subjects/sample/surf/"

lh = nibabel.freesurfer.io.read_geometry(freesurfer_path + "lh.inflated")[0]
rh = nibabel.freesurfer.io.read_geometry(freesurfer_path + "rh.inflated")[0]
rh[:, 0] = rh[:, 0] + 85

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
inverse_operator = read_inverse_operator(fname_inv)
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

src = inverse_operator['src']

lh_points = lh
rh_points = rh
points = np.r_[lh_points, rh_points]
points *= 170

vertices = np.r_[src[0]['vertno'], lh_points.shape[0] + src[1]['vertno']]

use_faces = np.r_[src[0]['tris'], lh_points.shape[0] + src[1]['tris']]

adj_mat = mesh_edges(use_faces)
smooth_mat = smoothing_matrix(vertices, adj_mat)

# Compute inverse solution
pick_ori = "normal"  # Get signed values to see the effect of sign filp
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_ori=pick_ori)
index_time = np.abs(stc.data).mean(0).argmax()

data = plotly_triangular_mesh(points, use_faces, smooth_mat * stc.data[:, index_time],
                              colorscale=DEFAULT_COLORSCALE, flatshading=False,
                              showscale=False, reversescale=False, plot_edges=False)

axis_template_time = dict(
    showspikes='across+toaxis',
    spikedash='solid',
    spikemode='across',
    spikesnap='cursor',
    showline=True,
    showgrid=True,
    zeroline=False,
    showbackground=False,
    backgroundcolor="rgb(200, 200, 230)",
    gridcolor="white",
    zerolinecolor="white")

axis_template = dict(
    showspikes=None,
    title="",
    zeroline=False,
    showline=False,
    showgrid=False,
    showticklabels=False,
    showlabel=False,
    showbackground=True,
    backgroundcolor="rgb(0, 0, 0)",
    gridcolor="rgb(0, 0, 0)",
    zerolinecolor="rgb(0, 0, 0)")

plot_layout = dict(
    title='',
    margin=dict(t=0, b=0, l=0, r=0),
    displayModeBar=False,
    font=dict(size=12, color='white'),
    width=650,
    height=650,
    showlegend=False,
    plot_bgcolor='black',
    paper_bgcolor='black',
    scene=dict(xaxis=axis_template,
               yaxis=axis_template,
               zaxis=axis_template,
               aspectratio=dict(x=1, y=1.2, z=1),
               camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
               annotations=[]
               )
)
plot_layout_time = dict(
    title='',
    font=dict(size=12, color='white'),
    width=650,
    height=650,
    showlegend=False,
    displayModeBar=False,
    plot_bgcolor='black',
    paper_bgcolor='black',
    hovermode='closest',
    xaxis={'showspikes': True},
    scene=dict(xaxis=axis_template_time,
               yaxis=axis_template_time,
               annotations=[]
               )

)

app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "MNE Source Space Explorer",
                                    href="https://github.com/mne-python",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[
                                html.Img(src=app.get_asset_url("mne_logo.png"))
                            ],
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Subject",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {"label": "sample", "value": "sample"},
                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value="sample",
                                        ),
                                        drc.NamedSlider(
                                            name="Time",
                                            id="slider-dataset-sample-size",
                                            min=evoked.times.min() * 1000,
                                            max=evoked.times.max() * 1000,
                                            step=len(evoked.times),
                                            marks={ii: '{0:.0f}'.format(ii) if ii == evoked.times[0] * 1000 else
                                                '{0:.0f}'.format(ii) if not (i_l % 100) else ''
                                                   for i_l, ii in enumerate(evoked.times * 1000)},
                                            value=int(len(evoked.times) / 2),
                                        ),
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-dataset-noise-level",
                                            min=0,
                                            max=1,
                                            marks={
                                                i: str(i)
                                                for i in [0, 0.25, 0.5, 0.75, 1]
                                            },
                                            step=0.1,
                                            value=0.2,
                                        ),
                                    ],
                                ),
                            ],
                        ), html.Div([
                            dcc.Graph(id='g1', figure={
                                'data': [go.Scatter(
                                    x=evoked.times * 1000,
                                    y=evoked.data[index, :].T,
                                    mode='lines',
                                    hoverinfo='x+y'
                                ) for index in np.arange(2, 306, 3)] + [go.Scatter(
                                    x=[evoked.times[index_time] * 1000, evoked.times[index_time] * 1000],
                                    y=[-np.abs(evoked.data[2:306:3]).max(), np.abs(evoked.data[2:306:3]).max()],
                                    mode='lines',
                                    line=dict(color='white', width=6),
                                    hoverinfo='skip'
                                )],
                                'layout': plot_layout_time, })
                        ], className="six columns"),

                        html.Div(
                            [
                                dcc.Graph(
                                    id="brain-graph",
                                    figure={
                                        "data": data,
                                        "layout": plot_layout,
                                    },
                                    config={"editable": True, "scrollZoom": False},
                                )
                            ],
                            className="graph__container",
                        ),

                    ],
                )
            ],
        ),
    ]
)


@app.callback(Output('brain-graph', 'figure'),
              [Input('slider-dataset-sample-size', 'value')],
              [State('brain-graph', 'figure')])
def update_graph(selected_dropdown_value, figure):
    index = (np.abs(stc.times * 1000 - selected_dropdown_value)).argmin()
    data = plotly_triangular_mesh(points, use_faces, smooth_mat * stc.data[:, index],
                                  colorscale=DEFAULT_COLORSCALE, flatshading=False,
                                  showscale=False, reversescale=False, plot_edges=False)
    figure["data"] = data
    figure["layout"] = plot_layout
    return figure


@app.callback(Output('g1', 'figure'),
              [Input('slider-dataset-sample-size', 'value')],
              [State('g1', 'figure')])
def update_graph(selected_dropdown_value, figure):
    index_time = (np.abs(stc.times * 1000 - selected_dropdown_value)).argmin()

    figure["data"][-1] = go.Scatter(
        x=[evoked.times[index_time] * 1000, evoked.times[index_time] * 1000],
        y=[-np.abs(evoked.data[2:306:3]).max(), np.abs(evoked.data[2:306:3]).max()],
        mode='lines',
        line=dict(color='white', width=6),
        hoverinfo='skip'
    )
    figure["layout"] = plot_layout_time
    return figure


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
