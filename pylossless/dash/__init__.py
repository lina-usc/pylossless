from mne.utils import check_version
import mne_icalabel.config as iclabel_config

if not check_version("mne_icalabel", "0.5.0"):
    raise ImportError(
        "mne_icalabel version 0.5.0 or higher is required. "
        "Please upgrade mne_icalabel."
    )
IC_COLORS = [
# New Colors
    "#000000", # BRAIN, Black
    "#0173b2", # Muscle, Dark Blue
    "#cc78bc", # EOG, Purple
    "#de8f05", # ECG, Orange
    "#56b4e9", # LINE NOISE, Cyan
    "#ca9161", # CHANNEL NOISE, Brown
    "#029e73", # OTHER, Green
    "#d55e00",  # Unclassified, Reddish-Brown
    "#fbafe4",  # Unclassified, Bright Pink
    "#ece1334",  # Unclassified, Yellow

""" Original Colors
    "#2c2c2c",  # Brain, Dark Gray
    "#cc78bc",  # EOG, Pink
    "#d55e00",  # ECG, Red
    "#029e73",  # Line Noise, Green
    "#de8f05",  # Channel Noise, Orange
    "#0173b2",  # Other, Blue
    "#56b4e9",  # Unclassified, Light Blue
    "#ece133",  # Unclassified, Yellow
    "#949494",  # Unclassified, Gray
"""

ic_label_cmap = dict(zip(iclabel_config.ICA_LABELS_TO_MNE.values(), IC_COLORS))
