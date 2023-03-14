"""Helper functions for extending mne_bids functionality."""

from mne_bids import BIDSPath, write_raw_bids


# TODO: Add parameters and return.
def get_bids_path(bids_path_kwargs, datatype='eeg', bids_root='./bids_dataset'):
    """Getter method for BIDS path from BIDS recording.
    """
    if "datatype" not in bids_path_kwargs:
        bids_path_kwargs["datatype"] = datatype
    if "root" not in bids_path_kwargs:
        bids_path_kwargs["root"] = bids_root

    return BIDSPath(**bids_path_kwargs)


# TODO: Add parameters and return.
def get_dataset_bids_path(bids_path_args, datatype='eeg', bids_root='./bids_dataset'):
    """Getter method for BIDS path from BIDS dataset.
    """
    return [get_bids_path(bids_path_kwargs, datatype, bids_root)
            for bids_path_kwargs in bids_path_args]


def convert_recording_to_bids(import_func, import_kwargs, bids_path_kwargs,
                    datatype='eeg', bids_root='./bids_dataset',
                    import_events=True, **write_kwargs):
    """This functions convert a dataset to BIDS.

    Parameters
    ----------
    import_func : function
       This function when called, will take as keyword arguments the
       dictionary `import_args`. If `import_events` set to True, this
       function must return 1) an object of type `mne.io.Raw`, 2) a structure
       of events (specified as usual in MNE-Python), and 3) a event_id
       dictionary (specified as usual in MNE-Python). If `import_events` is set to False,
       this function must return an object of the `mne.io.Raw` class. 
    import_kwargs : dict
       Dictionary of the keyword  arguments necessary to be passed to 
       `import_fct` to successfully import the corresponding recording.
    bids_path_kwargs : dict
       Dictionary of the keyword arguments necessary to be passed to the constructor of the 
       `mne_bids.BIDSPath` class.
    import events: boolean
        Whether to import a provided events object
    Returns
    -------
    bids_paths : list of instance of `mne_bids.BIDSPath`
      `mne_bids.BIDSPath` for the different recordings
    """

    bids_path = get_bids_path(bids_path_kwargs, datatype, bids_root)
    if import_events:
        raw, events, event_id = import_func(**import_kwargs)
    else:
        raw = import_func(**import_kwargs)
        events = None
        event_id = None

    if "format" not in write_kwargs:
        write_kwargs["format"] = "EDF"
    if "allow_preload" not in write_kwargs:
        write_kwargs["allow_preload"] = True

    write_raw_bids(raw, bids_path=bids_path,
                    events_data=events, event_id=event_id,
                    **write_kwargs)

    return bids_path


def convert_dataset_to_bids(import_funcs, import_args, bids_path_args,
                    datatype='eeg', bids_root='./bids_dataset',
                    import_events=True, **write_kwargs):
    """This functions convert a dataset to BIDS.

    Parameters
    ----------
    import_fct : function or list of functions
       This function when called, will take as keyword arguments one of the
       dictionaries in the `import_args` list. If `import_events` set to True, this
       function must return 1) an object of type `mne.io.Raw`, 2) a structure
       of events (specified as usual in MNE-Python), and 3) a event_id
       dictionary (specified as usual in MNE-Python). if `import_events` set to False,
       this function must return an object of the `mne.io.Raw` class. 
       This argument can also be a list of functions, in case where,
       for example, different recordings have different formats and require different
       import functions. In this case, the length of this attribute must be
       the same as `import_args`.
    import_args : list of dict
       Each item of this list needs to be a dictionary of the keyword 
       arguments necessary to be passed to `import_fct` to successfully import
       the corresponding recording. This list needs to be of the same length 
       as bids_path_args.
    bids_path_args : list of dict
       Each item of this list needs to be a dictionary of the keyword 
       arguments necessary to be passed to the constructor of the 
       `mne_bids.BIDSPath` class. This list needs to be of the same length 
       as import_args. 
    import events: boolean
        Whether to import a provided events object.
    Returns
    -------
    bids_paths : list of instance of `mne_bids.BIDSPath`
      `mne_bids.BIDSPath` for the different recordings
    """

    assert(len(import_args) == len(bids_path_args))
    if isinstance(import_funcs, list):
        assert(len(import_args) == len(import_funcs))
    else:
        import_funcs = [import_funcs]*len(import_args)

    bids_paths = []
    for import_kwargs, bids_path_kwargs, func in zip(import_args, bids_path_args, import_funcs):
        bids_paths.append(convert_recording_to_bids(func, import_kwargs, bids_path_kwargs,
                    datatype=datatype, bids_root=bids_root,
                    import_events=import_events, **write_kwargs))

    return bids_paths
