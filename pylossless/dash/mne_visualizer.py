import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output
from uuid import uuid1

# time series plot
import plotly.graph_objects as go

import numpy as np

from mne.io import BaseRaw
from mne import BaseEpochs, Evoked
import mne
from mne.utils import _validate_type, logger

from .css_defaults import DEFAULT_LAYOUT, CSS, STYLE


def _add_watermark_annot():
    from .css_defaults import WATERMARK_ANNOT
    return WATERMARK_ANNOT


class MNEVisualizer:

    def __init__(self, app, inst, dcc_graph_kwargs=None,
                 dash_id_suffix='',
                 show_time_slider=True, show_ch_slider=True,
                 scalings='auto', zoom=2, remove_dc=True,
                 annot_created_callback=None, refresh_inputs=None,
                 show_n_channels=20, set_callbacks=True):
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
                if using more than 1 MNEVisualizer object in a single
                application.
            ch_slider : bool
                text
            time_slider : float
                text
            Returns
            -------
            """

        if not isinstance(refresh_inputs, list):
            refresh_inputs = [refresh_inputs]
        self.refresh_inputs = refresh_inputs
        self.app = app
        self.scalings_arg = scalings
        self._inst = None
        self.n_sel_ch = show_n_channels
        self.win_start = 0  # min time to disp on plot
        self.win_size = 10  # max time to disp on plot
        self.zoom = zoom
        self.remove_dc = remove_dc
        # self.graph = None
        # self.layout = None
        self.traces = None
        self._ch_slider_val = 0
        self.annotating = False
        self.annotating_start = None
        self.annotation_inprogress = None
        self.annot_created_callback = annot_created_callback
        self.new_annot_desc = 'selected_time'
        self.mne_annots = None

        # setting component ids based on dash_id_suffix
        default_ids = ['graph', 'ch-slider', 'time-slider', 'container-plot', 'output', 'mne-annotations']
        self.dash_ids = {id_: (id_ + f'_{dash_id_suffix}') for id_ in default_ids}
        self.dcc_graph_kwargs = dict(id=self.dash_ids['graph'],
                                     className=CSS['timeseries'],
                                     style=STYLE['timeseries'],
                                     figure={'data': None,
                                             'layout': None},
                                     config={'modeBarButtonsToAdd': ["eraseshape"],
                                             'modeBarButtonsToRemove': ['zoom','pan']}
                                     )
        if dcc_graph_kwargs is not None:
            self.dcc_graph_kwargs.update(dcc_graph_kwargs)
        self.graph = dcc.Graph(**self.dcc_graph_kwargs)
        self.graph_div = html.Div([self.graph],
                                  style=STYLE['timeseries-div'],
                                  className=CSS['timeseries-div'])
        self.show_time_slider = show_time_slider
        self.show_ch_slider = show_ch_slider
        self.inst = inst

        # initialization subroutines
        self.init_sliders()
        self.init_annot_store()
        self.set_div()
        self.initialize_layout()
        if set_callbacks:
            self.set_callback()

    def load_recording(self, raw):
        """ """
        self.inst = raw
        self.channel_slider.max = self.nb_channels - 1
        self.channel_slider.value = self.nb_channels - 1
        marks_keys = np.round(np.linspace(self.times[0], self.times[-1], 10))
        self.time_slider.min = self.times[0]
        self.time_slider.max = self.times[-1] - self.win_size
        self.time_slider.marks = {int(key): str(int(key))
                                  for key in marks_keys}
        self.initialize_shapes()
        self.update_layout()

    @property
    def inst(self):
        return self._inst

    @inst.setter
    def inst(self, inst):
        if not inst:
            return
        _validate_type(inst, (BaseEpochs, BaseRaw, Evoked), 'inst')
        self._inst = inst
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

    
    ###########################################################
    # Methods for converting MNE annots to Plotly Shapes/annots
    ###########################################################
    def _get_annot_text(self, annotation, name=None):
        """Get description from mne.annotation.
        
        Parameters
        ----------
        annotation : mne.annotation
            an mne.annotation from a raw object.
        returns
        -------
        a dict that can be used as a plotly annotation (text).
        """
        return go.layout.Annotation(
                    dict(x=annotation['onset'] + annotation['duration'] /2,
                        y=self.layout.yaxis['range'][1],
                        text=annotation['description'],
                        name=name,
                        showarrow=False,
                        yshift=10,
                        font={'color': '#F1F1F1'})
                        )

    def _get_annot_shape(self, annotation, name='description'):
        """Make a plotly shape from an mne.annotation."""
        editable = True if 'bad' in annotation['description'] else False
        return go.layout.Shape(dict(name=name,
                    editable=editable,
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=annotation['onset'],
                    y0=self.layout.yaxis['range'][0],
                    x1=annotation['onset'] + annotation['duration'],
                    y1=self.layout.yaxis['range'][1],
                    fillcolor='red',
                    opacity=0.51 if annotation['duration'] else .75,
                    line_width=1,
                    line_color='black',
                    layer="below" if annotation['duration'] else 'above'))
    
    def initialize_shapes(self):
        if self.inst:

            ids = [str(uuid1()) for _ in range(len(self.inst.annotations))]
            annots = {id_: {"shape": self._get_annot_shape(annot, name=id_), 
                            "description": self._get_annot_text(annot, name=id_)} 
                      for id_, annot in zip(ids, self.inst.annotations)}
            self.mne_annots.data = annots

    def refresh_shapes(self):
        """Identify shapes that are viewable in the current time-window."""
        if not self.inst:
            return
        tmin, tmax = self.win_start, self.win_start + self.win_size

        annots = tuple(zip(*[(annot["shape"], annot["description"]) for annot
                             in self.mne_annots.data.values()
                             if tmin < annot["shape"]['x0'] < tmax]))
        if len(annots):
            self.layout.shapes, self.layout.annotations = annots
        else:
            self.layout.shapes, self.layout.annotations = [], []


    def _shape_from_selection(self, selections, name):
        """Make a new plotly shape from a user-drawn selection."""
        logger.debug('id for new drawn shape: ', name)
        return go.layout.Shape(dict(name=name,
                                editable=True,
                                type="rect",
                                xref="x",
                                yref="y",
                                x0=selections[0]['x0'],
                                y0=self.layout.yaxis['range'][0],
                                x1=selections[0]['x1'],
                                y1=self.layout.yaxis['range'][1],
                                fillcolor='red',
                                opacity=0.51,
                                line_width=1,
                                line_color='black',
                                layer="below"))

    def _plotly_annot_from_selection(self, selections, name):
        """Make a new plotly annotation for a user-drawn shape."""
        desc = self.new_annot_desc
        dur = selections[0]['x1'] - selections[0]['x0']
        logger.debug('id for new drawn text: ', name)
        return go.layout.Annotation(dict(x=selections[0]['x0'] + dur / 2,
                    y=self.layout.yaxis['range'][1],
                    text=desc,
                    name=name,
                    showarrow=False,
                    yshift=10,
                    font={'color': '#F1F1F1'}))

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

        if not self.inst:
            DEFAULT_LAYOUT['annotations'] = _add_watermark_annot()

        tickvals_handler = np.arange(-self.n_sel_ch + 1, 1)
        DEFAULT_LAYOUT['yaxis'].update({"tickvals": tickvals_handler,
                                        'ticktext': [''] * self.n_sel_ch,
                                        'range': [-self.n_sel_ch, 1]})
        
        DEFAULT_LAYOUT['xaxis'].update({'range': [self.win_start,
                                                  self.win_start + self.win_size]})
        
        
        self.layout = go.Layout(**DEFAULT_LAYOUT)

        trace_kwargs = {'x': [],
                        'y': [],
                        'mode': 'lines',
                        'line': dict(color='#2c2c2c', width=1)
                        }
        # create objects for layout and traces
        self.traces = [go.Scatter(name=ii, **trace_kwargs)
                       for ii in range(self.n_sel_ch)]

        self.update_layout(ch_slider_val=self.channel_slider.max,
                           time_slider_val=0)

    def update_layout(self,
                      ch_slider_val=None,
                      time_slider_val=None):
        if not self.inst:
            return
        if ch_slider_val is not None:
            self._ch_slider_val = ch_slider_val
        if time_slider_val is not None:
            self.win_start = time_slider_val

        tmin, tmax = self.win_start, self.win_start + self.win_size
        self.layout.xaxis.update({'range': [tmin, tmax]})

        # Update selected channels
        first_sel_ch = self._ch_slider_val - self.n_sel_ch + 1
        # +1 bc this is used in slicing below, & end is not inclued
        last_sel_ch = self._ch_slider_val + 1

        # Update times
        start_samp, stop_samp = self.inst.time_as_index([tmin, tmax])

        data, times = self.inst[::-1, start_samp:stop_samp]
        data = data[first_sel_ch:last_sel_ch, :]
        if self.remove_dc:
            data -= np.nanmean(data, axis=1)[:, np.newaxis]

        # Update the raw timeseries traces
        ch_names = self.inst.ch_names[::-1][first_sel_ch:last_sel_ch]
        self.layout.yaxis['ticktext'] = ch_names
        ch_types_list = self.inst.get_channel_types()
        ch_types = ch_types_list[::-1][first_sel_ch:last_sel_ch]
        ch_zip = zip(ch_names, data, self.traces, ch_types)
        for i, (ch_name, signal, trace, ch_type) in enumerate(ch_zip):
            trace.x = np.round(times, 3)
            step_trace = signal / self._get_norm_factor(ch_type)
            trace.y = step_trace - (self.n_sel_ch - i - 1)
            trace.name = ch_name
            if ch_name in self.inst.info['bads']:
                trace.line.color = '#d3d3d3'
            else:
                trace.line.color = '#2c2c2c'
            # Hover template will show Channel number and Time
            trace.text = np.round(signal * 1e6, 3)  # Volts to microvolts
            trace.hovertemplate = (f'<b>Channel</b>: {ch_name}<br>' +
                                   '<b>Time</b>: %{x}s<br>' +
                                   '<b>Amplitude</b>: %{text}uV<br>' +
                                   '<extra></extra>')

        self.graph.figure['data'] = self.traces

        self.refresh_shapes()

    ###############
    # CALLBACKS
    ###############

    def set_callback(self):
        args = [Output(self.dash_ids['graph'], 'figure'),
                Input(self.dash_ids['ch-slider'], 'value'),
                Input(self.dash_ids['time-slider'], 'value'),
                Input(self.dash_ids['graph'], "clickData"),
                Input(self.dash_ids['graph'], "relayoutData"),
                ]
        if self.refresh_inputs:
            args += self.refresh_inputs

        @self.app.callback(*args, suppress_callback_exceptions=False)
        def callback(ch, time, click_data, relayout_data, *args):
            print("mega callback")
            if not self.inst:
                return dash.no_update

            update_layout_ids = [self.dash_ids['ch-slider'],
                                 self.dash_ids['time-slider'],
                                 ]
            if self.refresh_inputs:
                update_layout_ids.extend([inp.component_id for inp in self.refresh_inputs])

            ctx = dash.callback_context
            if len(ctx.triggered[0]['prop_id'].split('.')) == 2:
                object_, dash_event = ctx.triggered[0]["prop_id"].split('.')
                if object_ == self.dash_ids['graph']:
                    if dash_event == 'clickData':
                        c_index = click_data["points"][0]["curveNumber"]
                        ch_name = self.traces[c_index].name
                        if ch_name in self.inst.info['bads']:
                            self.inst.info['bads'].pop()
                        else:
                            self.inst.info['bads'].append(ch_name)
                        self.update_layout()
                    elif dash_event == 'relayoutData':
                        if "selections" in relayout_data:
                            print('shape creation!!!')
                            id_ = str(uuid1())
                            self.mne_annots.data[id_] = {
                                "shape":  self._shape_from_selection(relayout_data['selections'], name=id_), 
                                "description": self._plotly_annot_from_selection(relayout_data['selections'], name=id_)
                            }
                            self.refresh_shapes()
                            # print('id ', new_shape.name, ' assigned to new shape')
                            # print('id ', new_text.name, 'assigned to new text')
                        elif "shapes" in relayout_data:
                            print('shape was deleted!!!')
                            updated_shapes = relayout_data['shapes']
                            if len(updated_shapes) < len(self.layout.shapes):
                                # Shape (i.e. annotation) was deleted
                                previous_names = [shape['name'] for
                                                  shape in self.layout.shapes]
                                new_names = [shape['name'] for
                                             shape in updated_shapes]
                                print(f'previous: {previous_names}, new: {new_names}')
                                deleted = (set(previous_names) - set(new_names)).pop()
                                #print(deleted, ' was deleted')
                                #print('## n shapes b4 deletion ', len(self.mne_annots.data['shapes']))
                                del self.mne_annots.data[deleted]
                                #print('## n shapes AFTER deletion ', len(self.mne_annots.data['shapes']))
                                #print('## n texts AFTER deletion ', len(self.mne_annots.data['descriptions']))
                            self.refresh_shapes()
                        elif any([key.endswith('x0')
                                  for key in relayout_data.keys()]):
                            new_x_strt_key = [key for key in relayout_data.keys()
                                              if key.endswith('x0')][0]
                            new_x_end_key = [key for key in relayout_data.keys()
                                             if key.endswith('x1')][0]
                            new_x_strt_val = relayout_data[new_x_strt_key]
                            new_x_end_val = relayout_data[new_x_end_key]
                            shape_i = (int(new_x_strt_key.split('[', 1)[1]
                                                         .split(']', 1)[0])
                                       )
                            edited_shape = self.layout.shapes[shape_i]
                            edited_shape['x0'] = new_x_strt_val
                            edited_shape['x1'] = new_x_end_val
                            edited_text = self.layout.annotations[shape_i]
                            dur = edited_shape['x1'] - edited_shape['x0']
                            text_new_x = edited_shape['x0'] + dur / 2
                            edited_text['x'] = text_new_x
                            name = edited_shape['name']
                            text_name = edited_text['name']
                            print(f'shape with id {name} was edited')
                            print(f'text with id {text_name} was edited')
                            for annot_name, annot in self.mne_annots.data.items():
                                if annot_name == name:
                                    print(f"found {name} in annotations.data['shapes']")
                                    annot["shape"]['x0'] = new_x_strt_val
                                    annot["shape"]['x1'] = new_x_end_val
                                    annot["description"]['x'] = text_new_x
                                    break
                            # Existing shape was amended.
                        else:
                            return no_update
                    else:
                        pass  # for selecting traces
                elif object_ in update_layout_ids:
                    print("mega callback - all other components")
                    self.update_layout(ch_slider_val=ch, time_slider_val=time)

            return self.graph.figure

    @property
    def nb_channels(self):
        if self.inst:
            return len(self.inst.ch_names)
        return self.n_sel_ch

    @property
    def times(self):
        if self.inst:
            return self.inst.times
        return [0]

    def init_sliders(self):
        self.channel_slider = dcc.Slider(id=self.dash_ids["ch-slider"],
                                         min=self.n_sel_ch - 1,
                                         max=self.nb_channels - 1,
                                         step=1,
                                         marks=None,
                                         value=self.nb_channels - 1,
                                         included=False,
                                         updatemode='mouseup',
                                         vertical=True,
                                         verticalHeight=300)
        self.channel_slider_div = html.Div(self.channel_slider,
                                           className=CSS['ch-slider-div'],
                                           style={})
        if not self.show_ch_slider:
            self.channel_slider_div.style.update({'display': 'none'})

        marks_keys = np.round(np.linspace(self.times[0], self.times[-1], 10))
        marks_dict = {int(key): str(int(key)) for key in marks_keys}
        max_ = self.times[-1] - self.win_size
        self.time_slider = dcc.Slider(id=self.dash_ids['time-slider'],
                                      min=self.times[0],
                                      max=max_ if max_ > 0 else 0,
                                      marks=marks_dict,
                                      value=self.win_start,
                                      vertical=False,
                                      included=False,
                                      updatemode='mouseup')
        self.time_slider_div = html.Div(self.time_slider,
                                        className=CSS['time-slider-div'],
                                        style={})
        if not self.show_time_slider:
            self.time_slider_div.style.update({'display': 'none'})
    
    def init_annot_store(self):
        self.mne_annots = dcc.Store(id=self.dash_ids["mne-annotations"])

    def set_div(self):
        """build the final hmtl.Div to be returned to user."""
        # include both the timeseries graph and the sliders
        # note that the order of components is important
        graph_components = [self.channel_slider_div,
                            self.graph_div,
                            self.time_slider_div,
                            self.mne_annots]
        # pass the list of components into an html.Div
        self.container_plot = html.Div(id=self.dash_ids['container-plot'],
                                       className=CSS['timeseries-container'],
                                       children=graph_components)


class ICVisualizer(MNEVisualizer):

    def __init__(self, raw, *args, cmap=None, ic_types=None, **kwargs):

        """ """
        self.ic_types = ic_types
        if cmap is not None:
            self.cmap = cmap
        else:
            self.cmap = dict()
        super(ICVisualizer, self).__init__(raw, *args, **kwargs)

    def load_recording(self, raw, cmap=None, ic_types=None):
        """ """
        self.ic_types = ic_types
        if cmap is not None:
            self.cmap = cmap
        else:
            self.cmap = dict()

        super(ICVisualizer, self).load_recording(raw)

    def update_layout(self,
                      ch_slider_val=None,
                      time_slider_val=None):
        """Update raw timeseries layout"""
        if not self.inst:
            return
        super(ICVisualizer, self).update_layout(ch_slider_val,
                                                time_slider_val)

        # Update selected channels
        first_sel_ch = self._ch_slider_val - self.n_sel_ch + 1
        # +1 bc this is used in slicing below, & end is not inclued
        last_sel_ch = self._ch_slider_val + 1

        # Update the raw timeseries traces
        self.ic_types
        ch_names = self.inst.ch_names[::-1][first_sel_ch:last_sel_ch]
        for ch_name, trace in zip(ch_names, self.traces):
            if ch_name in self.inst.info['bads']:
                trace.line.color = '#d3d3d3'
            else:
                trace.line.color = self.cmap[ch_name]
            # IC Hover template will show IC number and Time by default
            trace.hovertemplate = (f'<b>Component</b>: {ch_name}' +
                                   '<br><b>Time</b>: %{x}s<br>' +
                                   '<extra></extra>')
            if self.ic_types:
                # update hovertemplate with IC label
                label = self.ic_types[ch_name]
                trace.hovertemplate += f'<b>Label</b>: {label}<br>'
        self.graph.figure['data'] = self.traces
