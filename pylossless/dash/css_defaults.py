"""CSS styling constant for Lossless Pipeline in Dash."""

# Author: Scott Huberty <seh33@uw.edu>

# Default Layout for individual scatter plots within timeseries
###############################################################
DEFAULT_LAYOUT_XAXIS = {'zeroline': False,
                        'showgrid': True,
                        'title': "time (seconds)",
                        'gridcolor': 'white',
                        'fixedrange': True,
                        'showspikes': True,
                        'spikemode': 'across',
                        'spikesnap': 'cursor',
                        'showline': True,
                        'spikecolor': 'black',
                        'titlefont': dict(color='#ADB5BD'),
                        'tickfont': dict(color='#ADB5BD'),
                        'spikedash': 'dash',
                        'automargin': True
                        }

DEFAULT_LAYOUT_YAXIS = {'showgrid': True,
                        'showline': True,
                        'zeroline': False,
                        'autorange': False,  # 'reversed',
                        'scaleratio': 0.5,
                        "tickmode": "array",
                        'titlefont': dict(color='#ADB5BD'),
                        'tickfont': dict(color='#ADB5BD'),
                        'fixedrange': True,
                        'automargin': True}

DEFAULT_LAYOUT = dict(  # height=400,
                      # width=850,
                      xaxis=DEFAULT_LAYOUT_XAXIS,
                      yaxis=DEFAULT_LAYOUT_YAXIS,
                      showlegend=False,
                      margin={'t': 15, 'b': 0, 'l': 35, 'r': 5},
                      # {'t': 15,'b': 25, 'l': 35, 'r': 5},
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#EAEAF2",
                      shapes=[],
                      hovermode='closest')

WATERMARK_ANNOT = (dict(text='NO FILE SELECTED',
                        textangle=0, opacity=0.1,
                        font=dict(color='red', size=80),
                        xref='paper', yref='paper', x=.5, y=.5,
                        showarrow=False),
                   )

CSS = dict()
STYLE = dict()


####################################################
# TIMESERIES
####################################################

# bootstrap for channel slider: self.channel_slider
# Empty

# bootstrap format for channel slider div: self.channel_slider_div
CSS['ch-slider-div'] = "d-inline-block align-top"

# bootstrap format for time slider
# Empty

# bootstrap for time slider div
CSS['time-slider-div'] = "w-100"

# bootstrap for timeseries: self.graph
CSS['timeseries'] = "w-100 d-inline-block"  # border border-info
STYLE['timeseries'] = {'height': '40vh'}

# bootstrap for timeseries-div: self.graph_div; border border-warnings
CSS['timeseries-div'] = "mh-100 d-inline-block shadow-lg"
STYLE['timeseries-div'] = {'width': '95%'}

# bootstrap for timeseries-container: self.container_plot

CSS['timeseries-container'] = "w-100"  # border border-success

############################################################
# TOPO PLOTS
############################################################

# bootstrap for topo dcc-graph
CSS['topo-dcc'] = "bg-secondary bg-opacity-50 border rounded"  # border-info

# bootstrap for div containing topo-dcc; border border-warning
CSS['topo-dcc-div'] = 'bg-secondary bg-opacity-50 align-top'

# boostrap for final container: self.container_plot
CSS['topo-container'] = 'topo-div shadow-lg'  # border border-success

#####################
# QC DASHBOARD LAYOUT
#####################

# QC container
STYLE['qc-container'] = {}  # {'border': '2px solid yellow'}

# File control Row
CSS['file-row'] = 'h-20'
# STYLE['file-row'] = {'border': '2px solid orange'}

# Select Folder & Save Button Col
# STYLE['folder-col'] = {'border': '2px solid red'}

# Select Folder & Save Buttons
CSS['button'] = "d-md-inline-block"

# Dropdown Col
# STYLE['dropdown-col'] = {'border': '2px dashed purple'}

# Dropdown
CSS['dropdown'] = "d-md-inline-block w-100"

# Logo Col
# Empty

# Logo
CSS['logo'] = 'ms-5 mh-100'

# Timeseries & Topo boostrap Row
CSS['plots-row'] = 'h-80'
STYLE['plots-row'] = {}  # {'border': '2px dashed pink'}

# Timeseries plots Col
CSS['timeseries-col'] = 'w-100 h-100 mh-100'

# Topoplots Col
CSS['topo-col'] = 'h-100 mt-2 mb-2'  # border border-danger
