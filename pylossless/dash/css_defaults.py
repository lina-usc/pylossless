# Default Layout for individual scatter plots in timeseries
###########################################################
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
                        }

DEFAULT_LAYOUT_YAXIS = {'showgrid': True,
                        'showline': True,
                        'zeroline': False,
                        'autorange': False,  # 'reversed',
                        'scaleratio': 0.5,
                        "tickmode": "array",
                        'titlefont': dict(color='#ADB5BD'),
                        'tickfont': dict(color='#ADB5BD'),
                        'fixedrange': True}

DEFAULT_LAYOUT = dict(  # height=400,
                      # width=850,
                      xaxis=DEFAULT_LAYOUT_XAXIS,
                      yaxis=DEFAULT_LAYOUT_YAXIS,
                      showlegend=False,
                      margin={'t': 5, 'b': 5, 'l': 35, 'r': 5},
                      # {'t': 15,'b': 25, 'l': 35, 'r': 5},
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#EAEAF2",
                      shapes=[],
                      hovermode='closest')

CSS = dict()
STYLE = dict()


#############
# TIMESERIES
#############

# bootstrap for channel slider: self.channel_slider
###################################################

# bootstrap format for channel slider div: self.channel_slider_div
##################################################################
CSS['ch-slider-div'] = "d-inline-block align-top"

# bootstrap format for time slider
# ################################

# bootstrap for time slider div
################################
CSS['time-slider-div'] = "w-100"

# bootstrap for timeseries: self.graph
######################################
CSS['timeseries'] = "mh-100 d-inline-block"
STYLE['timeseries'] = {'width': '100%', 'height': '40vh'}

# bootstrap for timeseries-div: self.graph_div
##############################################
CSS['timeseries-div'] = "h-100 d-inline-block shadow-lg border border-warning"
STYLE['timeseries-div'] = {'width': '95%', 'height': '50%'}

# bootstrap for timeseries-container: self.container_plot
#########################################################
CSS['timeseries-container'] = "gx-1 mt-1 w-100 border border-success"

############
# TOPO PLOTS
############

# bootstrap for topo dcc-graph
CSS['topo-dcc'] = "bg-secondary border rounded border-info"

# bootstrap for div containing topo-dcc
CSS['topo-dcc-div'] = 'bg-secondary border border-warning align-top'

# boostrap for final container: self.container_plot
CSS['topo-container'] = 'topo-div border border-success align-top'

#####################
# QC DASHBOARD LAYOUT
#####################

# QC container
STYLE['qc-container'] = {'border': '2px solid yellow'}

# File control Row
CSS['file-row'] = 'h-20'
STYLE['file-row'] = {'border': '2px solid orange'}

# Select Folder Button Col
STYLE['folder-col'] = {'border': '2px solid red'}

# Select Folder Button
CSS['folder-button'] = "d-md-inline-block me-1"

# Dropdown Col
STYLE['dropdown-col'] = {'border': '2px dashed purple'}

# Dropdown
CSS['dropdown'] = "d-md-inline-block w-100"

# Timeseries & Topo boostrap Row
CSS['plots-row'] = 'h-80'
STYLE['plots-row'] = {'border': '2px dashed pink'}

# Timeseries plots Col
CSS['timeseries-col'] = 'w-100 h-100 mh-100'

# Topoplots Col
CSS['topo-col'] = 'h-100 border border-danger'
