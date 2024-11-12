from time import sleep
from pathlib import Path

import mne_bids

import pylossless as ll


def load_openneuro_bids(subject="pd6", timeout=20):
    """Download and Load BIDS dataset ds002778 from OpenNeuro.

    Parameters
    ----------
    subject : str
        The subject to load. Defaults to ``"pd6"``.

    Returns
    -------
    raw : mne.io.Raw
        The :class:`~mne.io.Raw` data.
    config : pylossless.config.Config
        The :class:`~pylossless.config.Config` object that can be used to run the
        pipeline.
    bids_path : mne_bids.BIDSPath
        The :class:``~mne_bids.BIDSPath`` instance for the dataset.

    Notes
    -----
    This function will download the dataset to the current working directory. The
    dataset is from the "UC San Diego Resting State EEG Data from Patients with
    Parkinson's Disease" study, curated by Alexander Rockhill.
    """
    openneuro = ll.utils.import_optional_dependency("openneuro")

    config = ll.config.Config()
    # TODO: Tests were developed using this configuration. The adult version
    # should probably be used instead, but this change will require fixing the
    # tests accordingly.
    config.load_default("infants")
    config["project"]["bids_montage"] = ""
    config["project"]["analysis_montage"] = "standard_1020"
    config["project"]["set_montage_kwargs"]["on_missing"] = "warn"

    # Shamelessly copied from
    # https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
    # pip install openneuro-py

    dataset = "ds002778"
    subject = subject

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

    for _ in range(timeout):
        if bids_path.fpath.with_suffix(".bdf").exists():
            break
        print("Waiting for .bdf files to be created. Current files available:",
              list(bids_path.fpath.glob("*")))
        sleep(1)
    else:
        raise TimeoutError("OpenNeuro failed to create the .bdf files.")

    raw = mne_bids.read_raw_bids(bids_path, verbose="ERROR")
    return raw, config, bids_path
