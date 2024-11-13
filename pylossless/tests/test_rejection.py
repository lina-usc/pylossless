from pathlib import Path

from numpy.testing import assert_array_equal

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
