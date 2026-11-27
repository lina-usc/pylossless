from pathlib import Path

import mne
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd

import pytest

import pylossless as ll


@pytest.mark.parametrize("clean_ch_mode", [None, "drop", "interpolate"])
def test_rejection_policy(clean_ch_mode, pipeline_fixture):
    """Test the rejection policy."""
    rejection_config_path = Path("test_rejection_config.yaml")
    rejection_config = ll.RejectionPolicy(ch_cleaning_mode=clean_ch_mode)
    rejection_config.save(rejection_config_path)
    want_flags = ["noisy", "uncorrelated", "bridged"]
    assert rejection_config["ch_flags_to_reject"] == want_flags

    pipeline_fixture.config["version"] = "-1"
    with pytest.raises(RuntimeError, match="The output of the pipeline was"):
        raw, ica = rejection_config.apply(pipeline_fixture,
                                          version_mismatch="raise")
    with pytest.raises(RuntimeWarning, match="The output of the pipeline was"):
        raw, ica = rejection_config.apply(pipeline_fixture,
                                          version_mismatch="warning")
    with pytest.raises(ValueError, match="version_mismatch can take values"):
        raw, ica = rejection_config.apply(pipeline_fixture,
                                          version_mismatch="sdfdf")
    raw, ica = rejection_config.apply(pipeline_fixture, return_ica=True,
                                      version_mismatch="ignore")

    flagged_chs = []
    for key in rejection_config["ch_flags_to_reject"]:
        flagged_chs.extend(pipeline_fixture.flags["ch"][key].tolist())
    assert flagged_chs == ["PO3", "Oz", "O2"]
    if clean_ch_mode is None:
        assert_array_equal(flagged_chs, raw.info["bads"])
    elif clean_ch_mode == "drop":
        assert len(list(set(flagged_chs) - set(raw.ch_names))) == 3
    elif clean_ch_mode == "interpolate":
        # interpolate_bads will drop ch_names from raw.info["bads"]
        # so to make sure that the channel was interpolated, lets check it
        assert len(list(set(flagged_chs) - set(raw.info["bads"]))) == 3

    df = pipeline_fixture.flags["ic"]
    assert not df.loc[ica.exclude]["ic_type"].str.contains("brain").any()
    assert not df.loc[ica.exclude]["ic_type"].str.contains("other").any()
    assert df.loc[ica.exclude]["ic_type"].str.contains("muscle").any()
    assert df.loc[ica.exclude]["ic_type"].str.contains("eog").any()
    assert df.loc[ica.exclude]["ic_type"].str.contains("line_noise").any()
    threshold = rejection_config["ic_rejection_threshold"]
    assert (df.loc[ica.exclude]["confidence"] > threshold).all()
    rejection_config_path.unlink()


def test_rejection_policy_remove_flagged_ics_false():
    """Test that remove_flagged_ics=False skips IC subtraction."""
    rng = np.random.default_rng(42)
    sfreq = 100.0
    ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = rng.normal(0.0, 1e-5, size=(len(ch_names), int(sfreq * 10)))
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(l_freq=1.0, h_freq=None, verbose=False)
    ica = mne.preprocessing.ICA(n_components=4, random_state=42, verbose=False)
    ica.fit(raw, verbose=False)

    pipeline = ll.LosslessPipeline()
    pipeline.config = ll.Config().load_default()
    pipeline.raw = raw
    pipeline.ica2 = ica.copy()
    pipeline.flags["ic"] = ll.flagging.FlaggedICs(
        pd.DataFrame(
            {
                "component": ica._ica_names,
                "annotator": ["ic_label"] * 4,
                "ic_type": ["eog", "muscle", "brain", "brain"],
                "confidence": [0.95, 0.90, 0.85, 0.80],
            }
        )
    )

    policy_keep = ll.RejectionPolicy(remove_flagged_ics=False)
    cleaned_keep, returned_ica = policy_keep.apply(pipeline, return_ica=True)
    assert_allclose(cleaned_keep.get_data(), raw.get_data())
    assert returned_ica.exclude == []

    # Also works when ICA was skipped (pipeline.ica2 is None)
    pipeline.ica2 = None
    cleaned_no_ica = policy_keep.apply(pipeline)
    assert_allclose(cleaned_no_ica.get_data(), raw.get_data())

    # With remove_flagged_ics=True, flagged ICs are subtracted as expected
    pipeline.ica2 = ica.copy()
    policy_remove = ll.RejectionPolicy(remove_flagged_ics=True)
    cleaned_remove, removed_ica = policy_remove.apply(pipeline, return_ica=True)
    assert removed_ica.exclude == [0, 1]
    assert not np.allclose(cleaned_remove.get_data(), raw.get_data())
