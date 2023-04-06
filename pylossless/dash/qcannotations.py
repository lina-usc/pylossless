import mne
import numpy as np
import pandas as pd
from uuid import uuid1


class EEGAnnotation:

    def __init__(self, onset, duration, description_str, layout):
        self._id = str(uuid1())
        self._onset = onset
        self._duration = duration
        self._description = description_str

        self._dash_layout = layout
        self._dash_shape = dict(name=self.id,
                                type="rect",
                                xref="x",
                                yref="y",
                                x0=self.onset,
                                y0=self._dash_layout.yaxis['range'][0],
                                x1=self.onset + self.duration,
                                y1=self._dash_layout.yaxis['range'][1],
                                fillcolor='red',
                                opacity=0.25 if self.duration else .75,
                                line_width=1,
                                line_color='black',
                                layer="below" if self.duration else 'above')
        self._dash_description = dict(x=self.onset + self.duration / 2,
                                      y=self._dash_layout.yaxis['range'][1],
                                      text=self.description,
                                      showarrow=False,
                                      yshift=10,
                                      font={'color': '#F1F1F1'})

    def update_dash_objects(self):
        self._dash_shape["x0"] = self.onset
        self._dash_shape["x1"] = self.onset + self.duration
        self._dash_shape["opacity"] = 0.25 if self.duration else .75

        self._dash_description["x"] = self.onset + self.duration / 2
        self._dash_description["text"] = self.description

    def to_mne_annotation(self):
        return mne.Annotations(self.onset, self.duration, self.description)

    @property
    def id(self):
        return self._id

    @property
    def dash_shape(self):
        return self._dash_shape

    @property
    def dash_description(self):
        return self._dash_description

    @property
    def onset(self):
        return self._onset

    @onset.setter
    def onset(self, onset):
        self._onset = onset
        self.update_dash_objects()

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration
        self.update_dash_objects()

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description
        self.update_dash_objects()

    @staticmethod
    def from_mne_annotation(annot, layout):
        return EEGAnnotation(annot["onset"],
                             annot["duration"],
                             annot["description"],
                             layout)

    def set_editable(self, editable=True):
        self._dash_shape["editable"] = editable
        self._dash_shape["opacity"] = 0.51


class EEGAnnotationList:

    def __init__(self, annotations=None):
        if annotations is not None:
            if isinstance(annotations, list):
                self.annotations = pd.Series({annot.id: annot
                                              for annot in annotations})
            else:
                self.annotations = pd.Series(annotations)
        else:
            self.annotations = pd.Series()

    def __get_series(self, attr):
        return pd.Series({annot.id: getattr(annot, attr)
                          for annot in self.annotations.values})

    @property
    def durations(self):
        return self.__get_series("duration")

    @property
    def onsets(self):
        return self.__get_series("onset")

    @property
    def descriptions(self):
        return self.__get_series("description")

    @property
    def dash_shapes(self):
        return self.__get_series("dash_shape")

    @property
    def dash_descriptions(self):
        return self.__get_series("dash_description")

    def pick(self, tmin=0, tmax=np.inf):

        annot_tmin = self.onsets
        annot_tmax = annot_tmin + self.durations

        mask = (((tmin <= annot_tmin) & (annot_tmin < tmax)) |
                ((tmin < annot_tmax) & (annot_tmax <= tmax)) |
                ((annot_tmin < tmin) & (annot_tmax > tmax))
                )
        return EEGAnnotationList(self.annotations[mask])

    def remove(self, id_):
        self.annotations = self.annotations.drop(index=id_)

    @staticmethod
    def from_mne_inst(inst, layout):
        annots = [EEGAnnotation.from_mne_annotation(annot, layout)
                  for annot in inst.annotations]
        return EEGAnnotationList(annots)

    def __len__(self):
        return len(self.annotations)

    def append(self, annot):
        self.annotations[annot.id] = annot

    def __setitem__(self, key, value):
        self.annotations[key] = value

    def __getitem__(self, key):
        return self.annotations[key]

    def __contains__(self, key):
        return key in self.annotations

    def to_mne_annotation(self):
        return mne.Annotations(self.onsets, self.durations, self.descriptions)

    def set_editable(self, editable=True):
        for annot in self.annotations:
            annot.set_editable(editable)
        return self
