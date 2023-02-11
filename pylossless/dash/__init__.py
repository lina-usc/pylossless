from mne_icalabel.config import ICLABEL_LABELS_TO_MNE

IC_COLORS = ['green', 'blue', 'cyan', 'goldenrod', 'magenta', 'red',
             'purple','brown', 'yellowgreen', 'burlywood', 'plum']

ic_label_cmap = dict(zip(ICLABEL_LABELS_TO_MNE.values(), IC_COLORS))
