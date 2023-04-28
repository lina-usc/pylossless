# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT

"""class to wrap an mne.raw object in a dash plot."""

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output

# time series plot
import plotly.graph_objects as go

import numpy as np

from mne.io import BaseRaw
from mne import BaseEpochs, Evoked
from mne.utils import _validate_type, logger

from .css_defaults import DEFAULT_LAYOUT, CSS, STYLE
from .qcannotations import EEGAnnotationList, EEGAnnotation


def _add_watermark_annot():
    from .css_defaults import WATERMARK_ANNOT
    return WATERMARK_ANNOT


class MNEVisualizer:
    """Visualize an mne.io.raw object in a dash graph."""

    def __init__(self, app, inst, dcc_graph_kwargs=None,
                 dash_id_suffix=None,
                 show_time_slider=True, show_ch_slider=True,
                 scalings='auto', zoom=2, remove_dc=True,
                 annot_created_callback=None, refresh_inputs=None,
                 show_n_channels=20, set_callbacks=True):
        """Initialize class.

        Parameters
        ----------
        app : instance of Dash.app
            The dash app object to place the plot within.
        inst : mne.io.Raw
            An instance of mne.io.Raw
        dcc_graph_kwargs : str | None
            keyword arguments to be passed to dcc.graph when
            creating the MNEVisualizer time-series plot from the
            mne.io.raw object. Must be a valid keyword argument
            for dcc.graph.
        dash_id_suffix : str
            string to append to the end of the MNEVisualizer.graph
            dash component ID. Each component id in the users app file
            needs to be unique. If using more than 1 MNEVisualizer
            object in a single, application. You must pass a suffix
            to at least one of the objects to make their dash-ID
            unique.
        show_ch_slider : bool
            Whether to show the channel slider with the MNEVIsualizer
            time-series graph. Defaults to True.
        show_time_slider : bool
            Whether to show the channel slider with the MNEVIsualizer
            time-series graph. Defaults to True.
        Returns
        -------
        an instance of MNEVisualizer.
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
        self.loading_div = None

        # setting component ids based on dash_id_suffix
        default_ids = ['graph', 'ch-slider', 'time-slider',
                       'container-plot', 'output', 'mne-annotations',
                       'loading', "loading-output"]
        self.dash_ids = {id_: (id_ + f'_{dash_id_suffix}')
                         if dash_id_suffix else id_
                         for id_ in default_ids}
        modebar_buttons = {'modeBarButtonsToAdd': ["eraseshape"],
                           'modeBarButtonsToRemove': ['zoom', 'pan']}
        self.dcc_graph_kwargs = dict(id=self.dash_ids['graph'],
                                     className=CSS['timeseries'],
                                     style=STYLE['timeseries'],
                                     figure={'data': None,
                                             'layout': None},
                                     config=modebar_buttons)
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
        self._init_sliders()
        self._init_annot_store()
        self._set_loading_icon()
        self._set_div()

        self.initialize_layout()
        if set_callbacks:
            self.set_callback()

    def load_recording(self, raw):
        """Load the mne.io.raw object and initialize the graph layout.

        Parameters
        ----------
        raw : mne.io.raw
            An instance of mne.io.Raw
        """
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
        """Property that returns the mne.io.raw object."""
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
        """Divide returned value to data for timeseries."""
        return 2 * self.scalings[ch_type] / self.zoom

    ###########################################################
    # Methods for converting MNE annots to Plotly Shapes/annots
    ###########################################################

    def initialize_shapes(self):
        """Make graph.layout.shapes for each mne.io.raw.annotation."""
        if not self.inst:
            return
        self.mne_annots.data = EEGAnnotationList.from_mne_inst(self.inst,
                                                               self.layout)

    def refresh_shapes(self):
        """Identify shapes that are viewable in the current time-window."""
        if not self.inst:
            return
        tmin, tmax = self.win_start, self.win_start + self.win_size

        annots = self.mne_annots.data.pick(tmin, tmax).set_editable(True)

        if len(annots):
            self.layout.shapes = list(annots.dash_shapes.values)
            self.layout.annotations = list(annots.dash_descriptions.values)
        else:
            self.layout.shapes, self.layout.annotations = [], []

    def update_inst_annnotations(self):
        """Set mne.io.raw.annotations from plotly shapes."""
        annots = self.mne_annots.data.to_mne_annotation()
        self.inst.set_annotations(annots)

############################
# Create Timeseries Layouts
############################

    @property
    def layout(self):
        """Return MNEVIsualizer.graph.figure.layout."""
        return self.graph.figure['layout']

    @layout.setter
    def layout(self, layout):
        self.graph.figure['layout'] = layout

    def initialize_layout(self):
        """Create MNEVisualizer.graph.figure.layout."""
        if not self.inst:
            DEFAULT_LAYOUT['annotations'] = _add_watermark_annot()

        tickvals_handler = np.arange(-self.n_sel_ch + 1, 1)
        DEFAULT_LAYOUT['yaxis'].update({"tickvals": tickvals_handler,
                                        'ticktext': [''] * self.n_sel_ch,
                                        'range': [-self.n_sel_ch, 1]})
        tmin = self.win_start
        tmax = self.win_start + self.win_size
        DEFAULT_LAYOUT['xaxis'].update({'range': [tmin, tmax]})
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
        """Update MNEVisualizer.graph.figure.layout."""
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
        # +1 bc this is used in slicing below, & end is not included
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
        """Set the dash callback for the MNE.Visualizer object."""
        args = [Output(self.dash_ids['graph'], 'figure'),
                Input(self.dash_ids['ch-slider'], 'value'),
                Input(self.dash_ids['time-slider'], 'value'),
                Input(self.dash_ids['graph'], "clickData"),
                Input(self.dash_ids['graph'], "relayoutData"),
                ]
        if self.refresh_inputs:
            args += self.refresh_inputs

        @self.app.callback(*args, suppress_callback_exceptions=False,
                           prevent_initial_call=False)
        def callback(ch, time, click_data, relayout_data, *args):
            if not self.inst:
                return dash.no_update

            update_layout_ids = [self.dash_ids['ch-slider'],
                                 self.dash_ids['time-slider'],
                                 ]
            if self.refresh_inputs:
                update_layout_ids.extend([inp.component_id
                                          for inp
                                          in self.refresh_inputs])

            update_layout = False
            ctx = dash.callback_context

            events = [event['prop_id'].split('.') for event in ctx.triggered
                      if len(event['prop_id'].split('.')) == 2]
            for object_, dash_event in events:

                if object_ == self.dash_ids['graph']:
                    if dash_event == 'clickData':
                        # Working on traces
                        logger.debug('** Trace selected')
                        c_index = click_data["points"][0]["curveNumber"]
                        ch_name = self.traces[c_index].name
                        if ch_name in self.inst.info['bads']:
                            self.inst.info['bads'].pop()
                        else:
                            self.inst.info['bads'].append(ch_name)
                        update_layout = True

                    elif dash_event == 'relayoutData':
                        # Working on annotations
                        logger.debug(f'** relayoutData: {relayout_data}')
                        if "selections" in relayout_data:
                            # shape creation
                            logger.debug('** shape created')
                            onset = relayout_data["selections"][0]['x0']
                            offset = relayout_data["selections"][0]['x1']
                            description = self.new_annot_desc
                            annot = EEGAnnotation(onset, offset-onset,
                                                  description, self.layout)
                            self.mne_annots.data.append(annot)

                        elif "shapes" in relayout_data:
                            # shape was deleted
                            logger.debug('** shape deleted')
                            updated_shapes = relayout_data['shapes']
                            if len(updated_shapes) < len(self.layout.shapes):
                                # Shape (i.e. annotation) was deleted
                                previous_names = [shape['name'] for
                                                  shape in self.layout.shapes]
                                new_names = [shape['name'] for
                                             shape in updated_shapes]
                                deleted = set(previous_names) - set(new_names)
                                self.mne_annots.data.remove(deleted.pop())

                        elif any([key.endswith('x0')
                                  for key in relayout_data.keys()]):
                            # shape was modified
                            logger.debug('** shape modified')
                            shape_str = (list(relayout_data.keys())[0]
                                         .split(".")[0]
                                         )
                            x0 = relayout_data[f"{shape_str}.x0"]
                            x1 = relayout_data[f"{shape_str}.x1"]
                            shape_i = int(shape_str.split('[', 1)[1][:-1])
                            name = self.layout.shapes[shape_i]['name']
                            if name in self.mne_annots.data:
                                annot = self.mne_annots.data[name]
                                annot.onset = x0
                                annot.duration = x1 - x0
                        else:
                            continue
                        update_layout = True

                elif object_ in update_layout_ids:
                    update_layout = True

            if update_layout:
                self.update_layout(ch_slider_val=ch, time_slider_val=time)
                return self.graph.figure

            return no_update

    @property
    def nb_channels(self):
        """Return the number of channel names in the mne.io.Raw object."""
        if self.inst:
            return len(self.inst.ch_names)
        return self.n_sel_ch

    @property
    def times(self):
        """Return the times of the mne.io.Raw object in MNEVisualizer.inst.

        Returns
        -------
        an np.array of the times.
        """
        if self.inst:
            return self.inst.times
        return [0]

    def _init_sliders(self):
        """Initialize the Channel and Time dcc.Slider components."""
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

    def _init_annot_store(self):
        """Initialize the dcc.Store component of mne annotations."""
        self.mne_annots = dcc.Store(id=self.dash_ids["mne-annotations"])

    def _set_loading_icon(self):
        """Add the loading icon."""
        loading = dcc.Loading(
            id=self.dash_ids['loading'],
            type="circle",
            children=html.Div(id=self.dash_ids['loading-output'])
        )
        self.loading_div = html.Div(loading)
        self.graph_div.children.append(self.loading_div)

    def _set_div(self):
        """Build the final html.Div component to be returned."""
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
    """Class to plot an mne.io.Raw object made of IC signals."""

    def __init__(self, raw, *args, cmap=None, ic_types=None, **kwargs):
        """Initialize class.

        Parameters
        ----------
        app : instance of Dash.app
            The dash app object to place the plot within.
        inst : mne.io.Raw
            An instance of mne.io.Raw
        dcc_graph_kwargs : str | None
            keyword arguments to be passed to dcc.graph when
            creating the MNEVisualizer time-series plot from the
            mne.io.raw object. Must be a valid keyword argument
            for dcc.graph.
        dash_id_suffix : str
            string to append to the end of the MNEVisualizer.graph
            dash component ID. Each component id in the users app file
            needs to be unique. If using more than 1 MNEVisualizer
            object in a single, application. You must pass a suffix
            to at least one of the objects to make their dash-ID
            unique.
        show_ch_slider : bool
            Whether to show the channel slider with the MNEVIsualizer
            time-series graph. Defaults to True.
        show_time_slider : bool
            Whether to show the channel slider with the MNEVIsualizer
            time-series graph. Defaults to True.
        cmap : dict.
            a mapping where the Keys are the IC name, and the values are a
            compatible rgba or HEX string, to color the IC traces.

        Returns
        -------
        an instance of ICVisualizer.

        Notes
        ----
        Any arguments that can be passed to MNEVisualizer can also be passed
        to ICVisualizer.
        """
        self.ic_types = ic_types
        if cmap is not None:
            self.cmap = cmap
        else:
            self.cmap = dict()
        super(ICVisualizer, self).__init__(raw, *args, **kwargs)

    def load_recording(self, raw, cmap=None, ic_types=None):
        """Load the mne.io.raw object and initialize the graph layout.

        Parameters
        ----------
        raw : mne.io.raw
            An instance of mne.io.Raw
        """
        self.ic_types = ic_types
        if cmap is not None:
            self.cmap = cmap
        else:
            self.cmap = dict()

        super(ICVisualizer, self).load_recording(raw)

    def update_layout(self,
                      ch_slider_val=None,
                      time_slider_val=None):
        """Update raw timeseries layout."""
        if not self.inst:
            return
        super(ICVisualizer, self).update_layout(ch_slider_val,
                                                time_slider_val)

        # Update selected channels
        first_sel_ch = self._ch_slider_val - self.n_sel_ch + 1
        # +1 bc this is used in slicing below, & end is not included
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
