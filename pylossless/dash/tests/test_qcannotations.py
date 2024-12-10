# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
# License: MIT

import json
import plotly.graph_objects as go

from ..qcannotations import EEGAnnotationList, EEGAnnotation


def test_EEGAnnotationList_serialization():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[1, 2, 3], y=[1, 2, 3]))
    layout = fig.layout
    layout.xaxis.update({"range": [0, 1]})
    layout.yaxis.update({"range": [0, 1]})
    annotation = EEGAnnotation(0.0, 1.0, "test", layout)
    annotation_list = EEGAnnotationList(annotation)

    # Check if json serializable
    json.dumps(annotation_list)
