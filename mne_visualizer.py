from dash import dcc, html
from dash.dependencies import Input, Output

# time series plot
import plotly.graph_objects as go

import numpy as np

from collections import Iterable

class MNEVisualizer:

    def __init__(self, app, inst=None, dcc_graph_kwargs=None,
                 dash_ids=None, ch_slider=None, time_slider=None):
        self.app = app
        self.inst = None
        if inst is not None:
            self.set_inst(inst)
        self.n_sel_ch = 20  # n of channels to display in plot
        self.win_start = 0  # min time to disp on plot
        self.win_size = 10  # max time to disp on plot
        self.ystep = 1e-3
        self.layout = None
        self.traces = None
        self._ch_slider_val = 0
        default_ids = ['graph', 'ch-slider', 'time-slider', 'container-plot']
        self.dash_ids = {id_:id_ for id_ in default_ids}
        if dash_ids is not None:
            self.dash_ids.update(dash_ids)
        self.dcc_graph_kwargs = dict(id=self.dash_ids['graph'],
                                     className='dcc-graph',
                                     figure={'data': None,
                                             'layout': None})
        if dcc_graph_kwargs is not None:
            self.dcc_graph_kwargs.update(dcc_graph_kwargs)
        self.graph = dcc.Graph(**self.dcc_graph_kwargs)
        self.graph_div = html.Div([self.graph],
                                        className='dcc-graph-div')
        self.use_ch_slider = ch_slider
        self.use_time_slider = time_slider
        self.init_sliders()
        self.set_div()
        self.set_callback()
        self.initialize_layout()

    def set_inst(self, inst):
        self.inst = inst
        self.inst.load_data()


############################
# Create Timeseries Layouts
############################

    def initialize_layout(self):
        start, stop = self.inst.time_as_index([self.win_start, self.win_size])
        data, times = self.inst[:self.n_sel_ch, start:stop]

        self.layout = go.Layout(
                                width = 1200,
                                height=400,
                                xaxis={'zeroline': False,
                                       'showgrid': True,
                                       'title': "time (seconds)",
                                       'gridcolor':'white'},
                                yaxis={'showgrid': True,
                                       'showline': True,
                                       'zeroline': False,
                                       'autorange': False,  #'reversed',
                                       'scaleratio': 0.5,
                                       "tickmode": "array",
                                       "tickvals": np.arange(-self.n_sel_ch + 1, 1) * self.ystep,
                                       'ticktext': [''] * self.n_sel_ch,
                                       'range':[-self.ystep*self.n_sel_ch, self.ystep]},
                                showlegend=False,
                                margin={'t': 25,'b': 25,'l': 35, 'r': 25},
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="#EAEAF2",
                                )

        trace_kwargs = {'mode':'lines', 'line':dict(color='#222222', width=1)}
        # create objects for layout and traces
        self.traces = []

        # loop over the channels
        for ii in range(self.n_sel_ch):
            self.traces.append(go.Scatter(x=times, y=data.T[:, ii] - ii * self.ystep, **trace_kwargs))

        self.graph.figure['layout'] = self.layout
        self.graph.figure['data'] = self.traces

    def update_layout(self,
                      ch_slider_val=None,
                      time_slider_val=None):
        print(ch_slider_val)

        if ch_slider_val is not None:
            self._ch_slider_val = ch_slider_val
        if time_slider_val is not None:
            self.win_start = time_slider_val

        # Update selected channels
        first_sel_ch = self._ch_slider_val - self.n_sel_ch + 1
        last_sel_ch = self._ch_slider_val + 1  # +1 bc this is used in slicing below, & end is not inclued

        # Update times
        start_samp, stop_samp = self.inst.time_as_index([self.win_start, self.win_start + self.win_size])

        data, times = self.inst[::-1, start_samp:stop_samp]
        data = data[first_sel_ch:last_sel_ch, :]

        # Update the raw timeseries traces
        ch_names = self.inst.ch_names[::-1][first_sel_ch:last_sel_ch]
        self.layout.yaxis['ticktext'] = ch_names
        for i, (ch_name, signal, trace) in enumerate(zip(ch_names, data, self.traces)):
            trace.x = times
            trace.y = signal - (self.n_sel_ch - i - 1) * self.ystep
            trace.name = ch_name
        self.graph.figure['data'] = self.traces

    ###############
    # CALLBACKS
    ###############

    def set_callback(self):
        args = [Output(self.dash_ids['graph'], 'figure')]
       
        if self.use_ch_slider:
            args += [Input(self.use_ch_slider, 'value')]
        else: 
            args += [Input(self.dash_ids['ch-slider'], 'value')]
        if self.use_time_slider:
            args += [Input(self.use_time_slider, 'value')]
        else:
            args += [Input(self.dash_ids['time-slider'], 'value')]

        @self.app.callback(*args, suppress_callback_exceptions=True)
        def channel_slider_change(ch, time):
            
            self.update_layout(ch_slider_val=ch, time_slider_val=time)
            return self.graph.figure

    def init_sliders(self):
        self.channel_slider = dcc.Slider(id=self.dash_ids["ch-slider"],
                                         min=self.n_sel_ch -1,
                                         max=len(self.inst.ch_names) -1,
                                         step=1,
                                         marks=None,
                                         value=len(self.inst.ch_names) -1,
                                         included=False,
                                         updatemode='mouseup',
                                         vertical=True,
                                         verticalHeight=300)

        marks_keys = np.round(np.linspace(self.inst.times[0], self.inst.times[-1], 10))
        self.time_slider = dcc.Slider(id=self.dash_ids['time-slider'],
                                      min=self.inst.times[0],
                                      max=self.inst.times[-1] - self.win_size,
                                      marks={int(key):str(int(key)) for key in marks_keys},
                                      value=self.win_start,
                                      vertical=False,
                                      included=False,
                                      updatemode='mouseup')
    def set_div(self):
        if self.use_ch_slider is None:
            outer_ts_div = [html.Div(self.channel_slider), self.graph_div]
        else:
            outer_ts_div = [self.graph_div]
        if self.use_time_slider is None:
            ts_and_timeslider = [html.Div([
                                            html.Div(outer_ts_div,
                                                    className="outer-timeseries-div"),
                                            html.Div(self.time_slider,
                                                    style= {'width': '1200px'})],
                                                    className='timeseries-container')]
        else:
            ts_and_timeslider = [html.Div([html.Div(outer_ts_div,
                                                    className="outer-timeseries-div")],
                                                    className='timeseries-container')]
        self.container_plot = html.Div(id=self.dash_ids['container-plot'], 
                                       children=ts_and_timeslider)
