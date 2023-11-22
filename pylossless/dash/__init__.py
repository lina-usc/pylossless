from mne.utils import check_version
import mne_icalabel.config as iclabel_config

if not check_version("mne_icalabel", "0.5.0"):
    raise ImportError(
        "mne_icalabel version 0.5.0 or higher is required. "
        "Please upgrade mne_icalabel."
    )
IC_COLORS = [
    "#2c2c2c",
    "#003e83",
    "cyan",
    "goldenrod",
    "magenta",
    "#b08699",
    "#96bfe6",
    "brown",
    "yellowgreen",
    "burlywood",
    "plum",
]

ic_label_cmap = dict(zip(iclabel_config.ICA_LABELS_TO_MNE.values(), IC_COLORS))
