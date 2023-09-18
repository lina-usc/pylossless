from time import sleep
from pathlib import Path

import mne
import mne_bids
import openneuro

import pylossless as ll


def load_openneuro_bids():
    """Load a BIDS dataset from OpenNeuro."""
    config = ll.config.Config()
    config.load_default()
    config["project"]["bids_montage"] = ""
    config["project"]["analysis_montage"] = "standard_1020"
    config["project"]["set_montage_kwargs"]["on_missing"] = "warn"

    # Shamelessly copied from
    # https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
    # pip install openneuro-py

    dataset = "ds002778"
    subject = "pd6"

    # Download one subject's data from each dataset
    bids_root = Path(".") / dataset
    # TODO: Delete this directory after test otherwise MNE will think the
    # sample directory is outdated, and will re-download it the next time
    # data_path() is called, which is annoying for users.
    bids_root.mkdir(exist_ok=True)

    openneuro.download(
        dataset=dataset, target_dir=bids_root, include=[f"sub-{subject}"]
    )

    datatype = "eeg"
    session = "off"
    task = "rest"
    suffix = "eeg"
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        suffix=suffix,
        datatype=datatype,
        root=bids_root,
    )

    while not bids_path.fpath.with_suffix(".bdf").exists():
        print(list(bids_path.fpath.glob("*")))
        sleep(1)
    raw = mne_bids.read_raw_bids(bids_path, verbose="ERROR")
    annots = mne.Annotations(
        onset=[1, 15], duration=[1, 1], description=["test_annot", "test_annot"]
    )
    raw.set_annotations(annots)
    return raw, config, bids_root
