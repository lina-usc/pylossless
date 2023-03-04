from mne_icalabel.config import ICLABEL_LABELS_TO_MNE

IC_COLORS = ['#2c2c2c', '#003e83', 'cyan', 'goldenrod', 'magenta', '#b08699',
             '#96bfe6','brown', 'yellowgreen', 'burlywood', 'plum']

ic_label_cmap = dict(zip(ICLABEL_LABELS_TO_MNE.values(), IC_COLORS))
