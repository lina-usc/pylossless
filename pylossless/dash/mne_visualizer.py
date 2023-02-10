import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output
from dash_extensions import EventListener

# time series plot
import plotly.graph_objects as go

import numpy as np

from mne.io import BaseRaw
from mne import  BaseEpochs, Evoked
import mne
from mne.utils import _validate_type


DEFAULT_LAYOUT_XAXIS = {'zeroline': False,
                        'showgrid': True,
                        'title': "time (seconds)",
                        'gridcolor':'white',
                        'fixedrange':True,
                        'showspikes' : True,
                        'spikemode': 'across',
                        'spikesnap': 'cursor',
                        'showline': True,
                        'spikecolor':'black',
                        'spikedash':'dash'
                        }

DEFAULT_LAYOUT_YAXIS = {'showgrid': True,
                        'showline': True,
                        'zeroline': False,
                        'autorange': False,  #'reversed',
                        'scaleratio': 0.5,
                        "tickmode": "array",
                        'fixedrange':True}

DEFAULT_LAYOUT = dict(width = 1200,
                      height=400,
                      xaxis=DEFAULT_LAYOUT_XAXIS,
                      yaxis=DEFAULT_LAYOUT_YAXIS,
                      showlegend=False,
                      margin={'t': 25,'b': 25, 'l': 35, 'r': 25},
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#EAEAF2",
                      shapes=[],
                      hovermode='closest')



class MNEVisualizer:

    def __init__(self, app, inst, dcc_graph_kwargs=None,
                 dash_id_suffix='', ch_slider=None, time_slider=None,
                 scalings='auto', zoom=2, remove_dc=True,
                 annot_created_callback=None):
        """ text
            Parameters
            ----------
            app : instance of Raw
                A raw object to use the data from.
            inst : int
                must be an instance of mne.Raw, mne.Epochs, mne.Evoked
            start : float
                text
            dcc_grpah_kwargs : float | None
                text
            dash_id_suffix : float
                each component id in the users app file needs to be unique.
                if using more than 1 MNEVisualizer object in a single application.
            ch_slider : bool
                text
            time_slider : float
                text
            Returns
            -------
            """

        self.app = app
        self.scalings_arg = scalings
        self.__inst = None        
        self.n_sel_ch = 20  # n of channels to display in plot
        self.win_start = 0  # min time to disp on plot
        self.win_size = 10  # max time to disp on plot
        self.zoom = zoom
        self.remove_dc = remove_dc
        #self.graph = None
        #self.layout = None
        self.traces = None
        self._ch_slider_val = 0
        self.annotating = False
        self.annotating_start = None
        self.annotation_inprogress = None
        self.annot_created_callback = annot_created_callback
        self.new_annot_desc = 'selected_time'

        # setting component ids based on dash_id_suffix
        default_ids = ['graph', 'ch-slider', 'time-slider', 'container-plot', 'keyboard', 'output']
        self.dash_ids = {id_:(id_ + dash_id_suffix) for id_ in default_ids}

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
        self.shift_down = False
        self.inst = inst

        # initialization subroutines          
        self.init_sliders()
        self.set_div()
        self.initialize_layout()
        self.set_callback()
        self.initialize_keyboard()

    @property
    def inst(self):
        return self.__inst

    @inst.setter
    def inst(self, inst):
        _validate_type(inst, (BaseEpochs, BaseRaw, Evoked), 'inst')
        self.__inst = inst
        self.inst.load_data()
        self.scalings = dict(mag=1e-12,
                             grad=4e-11,
                             eeg=20e-6,
                             eog=150e-6,
                             ecg=5e-4,
                             emg=1e-3,
                             ref_meg=1e-12,
                             misc=1e-3,
                             stim=1,
                             resp=1,
                             chpi=1e-4,
                             whitened=1e2)
        if self.scalings_arg == 'auto':
            for kind in np.unique(self.inst.get_channel_types()):
                self.scalings[kind] = np.percentile(self.inst.get_data(), 99.5)
        else:
            self.scalings.update(self.scalings_arg)

    def _get_norm_factor(self, ch_type):
        "will divide returned value to data for timeseries"
        return 2 * self.scalings[ch_type] / self.zoom


    def _get_annot_text(self, annotation):
        return dict(x=annotation['onset'] + annotation['duration'] /2,
                    y=self.layout.yaxis['range'][1],
                    text=annotation['description'],
                    showarrow=False,
                    yshift=10,
                    font={'color':'#F1F1F1'})

    def _get_annot_shape(self, annotation):
        return dict(name=annotation['description'],
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=annotation['onset'],
                    y0=self.layout.yaxis['range'][0],
                    x1=annotation['onset'] + annotation['duration'],
                    y1=self.layout.yaxis['range'][1],
                    fillcolor='red',
                    opacity=0.25 if annotation['duration'] else .75,
                    line_width=1,
                    line_color='black',
                    layer="below" if annotation['duration'] else 'above')

    def add_annot_shapes(self, annotations):
        self.layout.shapes = [self._get_annot_shape(annot)
                              for annot in annotations]
        self.layout.annotations = [self._get_annot_text(annot)
                                   for annot in annotations]
        if self.annotating:
            self.layout.shapes += (self.annotation_inprogress,)

    def refresh_annotations(self):
        tmin, tmax = self.win_start, self.win_start + self.win_size
        annots = self.inst.annotations.copy().crop(tmin, tmax, use_orig_time=False)
        self.add_annot_shapes(annots)



############################
# Create Timeseries Layouts
############################

    @property
    def layout(self):
        return self.graph.figure['layout']

    @layout.setter
    def layout(self, layout):
        self.graph.figure['layout'] = layout

    def initialize_layout(self):

        DEFAULT_LAYOUT['yaxis'].update({"tickvals": np.arange(-self.n_sel_ch + 1, 1),
                                     'ticktext': [''] * self.n_sel_ch,
                                     'range': [-self.n_sel_ch, 1]})
        self.layout = go.Layout(**DEFAULT_LAYOUT)

        trace_kwargs = {'x': [],
                        'y': [],
                        'mode': 'lines',
                        'line': dict(color='#222222', width=1)
                        }
        # create objects for layout and traces
        self.traces = [go.Scatter(name=ii, **trace_kwargs)
                       for ii in range(self.n_sel_ch)]

        self.update_layout(ch_slider_val=self.channel_slider.max, time_slider_val=0)

    def update_layout(self,
                      ch_slider_val=None,
                      time_slider_val=None):

        if ch_slider_val is not None:
            self._ch_slider_val = ch_slider_val
        if time_slider_val is not None:
            self.win_start = time_slider_val

        tmin, tmax = self.win_start, self.win_start + self.win_size

        # Update selected channels
        first_sel_ch = self._ch_slider_val - self.n_sel_ch + 1
        last_sel_ch = self._ch_slider_val + 1  # +1 bc this is used in slicing below, & end is not inclued

        # Update times
        start_samp, stop_samp = self.inst.time_as_index([tmin, tmax])

        data, times = self.inst[::-1, start_samp:stop_samp]
        data = data[first_sel_ch:last_sel_ch, :]
        if self.remove_dc:
            data -= np.nanmean(data, axis=1)[:, np.newaxis] 

        # Update the raw timeseries traces
        ch_names = self.inst.ch_names[::-1][first_sel_ch:last_sel_ch]
        self.layout.yaxis['ticktext'] = ch_names
        ch_types = self.inst.get_channel_types()[::-1][first_sel_ch:last_sel_ch]
        for i, (ch_name, signal, trace, ch_type) in enumerate(zip(ch_names, data, self.traces, ch_types)):
            trace.x = times
            trace.y = signal/self._get_norm_factor(ch_type) - (self.n_sel_ch - i - 1)
            trace.name = ch_name
            if ch_name in self.inst.info['bads']:
                trace.line.color = '#d3d3d3'
            else:
                trace.line.color = 'black'

        self.graph.figure['data'] = self.traces

        self.refresh_annotations()

    def initialize_keyboard(self):
        events =[{"event": "keydown", "props": ["key", "shiftKey"]},
                 {"event":"keyup", "props":["key","shiftKey"]}]
        event_listener = EventListener(id=self.dash_ids['keyboard'], events=events)
        self.app.layout.children.extend([event_listener, html.Div(id=self.dash_ids["output"])])
        @self.app.callback(Output(self.dash_ids['output'], "children"), [Input(self.dash_ids['keyboard'], "event")])
        def event_callback(event):
            if event is None:
                return ''
            if event['key'] == 'Shift':
                self.shift_down = event['shiftKey']
            return '' 

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
        args += [Input(self.dash_ids['graph'], "clickData"),
                 Input(self.dash_ids['graph'], "hoverData")]

    

        @self.app.callback(*args, suppress_callback_exceptions=False)
        def callback(ch, time, click_data, hover_data):
            ctx = dash.callback_context
            assert(len(ctx.triggered) == 1)
            if len(ctx.triggered[0]['prop_id'].split('.')) == 2:
                object_, dash_event = ctx.triggered[0]["prop_id"].split('.')
                if object_ == self.dash_ids['graph']:
                    if dash_event == 'clickData':
                        if self.shift_down:
                            if self.annotating:
                                # finishing an annotation
                                self.layout.xaxis['spikedash'] = 'dash'
                                self.layout.xaxis['spikecolor'] = 'black'
                                new_annot = mne.Annotations(onset=[self.annotating_start],
                                                        duration=[click_data['points'][0]['x'] - self.annotating_start],
                                                        description=self.new_annot_desc,
                                                        orig_time=self.inst.annotations.orig_time)
                                self.annotating = not self.annotating
                                if self.annot_created_callback is not None:
                                    self.annot_created_callback(new_annot)
                                else:
                                    self.inst.set_annotations(self.inst.annotations + new_annot)
                                    #self.refresh_annotations()
                                    self.update_layout() # TODO replace update_layout to ensure refresh without updating whole layout
                            else:
                                # starting an annotation
                                self.annotating_start = click_data['points'][0]['x']
                                self.layout.xaxis['spikedash'] = 'solid'
                                self.layout.xaxis['spikecolor'] = 'red'
                                shape = dict(type="rect",
                                            xref="x",
                                            yref="y",
                                            x0=click_data['points'][0]['x'],
                                            y0=self.layout.yaxis['range'][0],
                                            x1=click_data['points'][0]['x'],
                                            y1=self.layout.yaxis['range'][1],
                                            fillcolor="red",
                                            opacity=0.45,
                                            line_width=0,
                                            layer="below")
                                self.annotation_inprogress = shape
                                self.annotating = not self.annotating
                                self.update_layout() # TODO replace update_layout to ensure refresh without updating whole layout
                                #self.refresh_annotations()

                        else: # not shift_down
                            ch_name = self.traces[click_data["points"][0]["curveNumber"]].name
                            if ch_name in self.inst.info['bads']:
                                self.inst.info['bads'].pop()
                            else:
                                self.inst.info['bads'].append(ch_name)
                            self.update_layout()

                    elif dash_event == 'hoverData':
                        if self.annotating:
                            #self.annotating_current = hover_data['points'][0]['x']
                            self.annotation_inprogress['x1'] = hover_data['points'][0]['x']
                            self.update_layout() # TODO replace update_layout to ensure refresh without updating whole layout
                        else:
                            return no_update
                    else:
                        #self.select_trace()
                        pass # for selecting traces                    
                elif object_ in [self.dash_ids['ch-slider'],
                                 self.dash_ids['time-slider'],
                                 self.use_time_slider, self.use_ch_slider]:
                    # if object_ in [self.dash_ids['ch-slider'], self.use_ch_slider]:
                    #    pass

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


class ICVisualizer(MNEVisualizer):
    
    def __init__(self, *args, cmap=None, **kwargs):

        
        if cmap is not None:
            self.cmap = cmap
        else:
            self.cmap = dict()
        super(ICVisualizer, self).__init__(*args, **kwargs)
    
    def update_layout(self,
                      ch_slider_val=None,
                      time_slider_val=None):

        super(ICVisualizer, self).update_layout(ch_slider_val,
                                                time_slider_val)

        # Update selected channels
        first_sel_ch = self._ch_slider_val - self.n_sel_ch + 1
        last_sel_ch = self._ch_slider_val + 1  # +1 bc this is used in slicing below, & end is not inclued

        # Update the raw timeseries traces
        ch_names = self.inst.ch_names[::-1][first_sel_ch:last_sel_ch]
        for ch_name, trace in zip(ch_names, self.traces):
            if ch_name in self.inst.info['bads']:
                trace.line.color = '#d3d3d3'
            else:
                trace.line.color = self.cmap[ch_name]

        self.graph.figure['data'] = self.traces