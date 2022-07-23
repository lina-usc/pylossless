from mne_bids import BIDSPath, write_raw_bids


def convert_to_bids(import_fct, import_args, bids_path_args,
                    datatype='eeg', bids_root='./bids_dataset', 
                    **write_kwags):
    """This functions convert a dataset to BIDS.

    Parameters
    ----------
    import_fct : function or list of functions
       This functions must, when provided keyword arguments for one of the 
       dictionary of the `import_args` list of dictionaries load the 
       corresponding recording and return, in that order, an object of type 
       `mne.io.Raw`, a structure of events (specified as usual in MNE-Python),
       and a event_id dictionnary (specified as usual in MNE-Python).
       This argument can also be a list of functions, in case where, e.g., 
       different recordings have different formats and require different
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
    overwrite : bool
       Specify whether existing output files should be overwritten or
       an exception should be raised. 
    Returns
    -------
    bids_paths : list of instance of `mne_bids.BIDSPath`
      `mne_bids.BIDSPath` for the different recordings
    """

    assert(len(import_args) == len(bids_path_args))
    if isinstance(import_fct, list):
            assert(len(import_args) == len(import_fct))
    else:
        import_fct = [import_fct]*len(import_args)

    bids_paths = []
    for import_kwargs, bids_path_kwargs, fct in zip(import_args, bids_path_args, import_fct):

        if "datatype" not in bids_path_kwargs:
            bids_path_kwargs["datatype"] = datatype
        if "root" not in bids_path_kwargs:
            bids_path_kwargs["root"] = bids_root

        print(bids_path_kwargs)
        bids_paths.append(BIDSPath(**bids_path_kwargs))

        raw, events, event_id = fct(**import_kwargs)

        if "format" not in write_kwags:
            write_kwags["format"] = "EDF"
        if "allow_preload" not in write_kwags:
            write_kwags["allow_preload"] = True

        write_raw_bids(raw, bids_path=bids_paths[-1],
                       events_data=events, event_id=event_id,
                       **write_kwags)

    return bids_paths
