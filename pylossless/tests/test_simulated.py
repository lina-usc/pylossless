import pytest

from pathlib import Path

from numpy.testing import assert_array_equal


import pylossless as ll

raw_sim = ll.datasets.load_simulated_raw()

# LOAD DEFAULT CONFIG
config = ll.config.Config()
config.load_default("infants")
config["noisy_channels"]["outliers_kwargs"]["lower"] = 0.25
config["noisy_channels"]["outliers_kwargs"]["upper"] = 0.75
# short file, raise threshold so epochs w/ blinks dont cause flag
config["noisy_channels"]["flag_crit"] = 0.30
config.save("sample_audvis_config.yaml")
# GENERATE PIPELINE
pipeline = ll.LosslessPipeline("sample_audvis_config.yaml")
pipeline.raw = raw_sim


# TEST
@pytest.mark.parametrize("pipeline", [(pipeline)])
def test_simulated_raw(pipeline):
    """Test pipeline on simulated EEG."""
    with pytest.warns(RuntimeWarning, match="sampling frequency"):
        pipeline._check_sfreq()
    # This file should have been downsampled
    assert pipeline.raw.info["sfreq"] == 600
    # FIND NOISY EPOCHS
    pipeline.flag_noisy_epochs()
    # Epoch 2 was made noisy and should be flagged.
    assert_array_equal(pipeline.flags["epoch"]["noisy"], [2])
    epochs = pipeline.get_epochs()
    # only epoch at indice 2 should have been dropped
    assert all(not tup or i == 2 for i, tup in enumerate(epochs.drop_log))

    # Flag noisy channels
    pipeline.flag_noisy_channels()
    noisy_chs = ["EEG 001", "EEG 002"]
    assert_array_equal(pipeline.flags["ch"]["noisy"], noisy_chs)

    # FIND UNCORRELATED CHS
    data_r_ch = pipeline.flag_uncorrelated_channels()
    # Previously flagged chs should not be in the correlation array
    assert all(
        [name not in data_r_ch.coords["ch"] for name in pipeline.flags["ch"]["noisy"]]
    )
    # EEG 024 was made random and should be flagged.
    # https://github.com/lina-usc/pylossless/issues/141

    # RUN FLAG_CH_BRIDGE
    pipeline.flag_bridged_channels(data_r_ch)
    # Channels below are duplicates and should be flagged.
    assert "EEG 053" in pipeline.flags["ch"]["bridged"]
    assert "EEG 054" in pipeline.flags["ch"]["bridged"]

    # Delete temp config file
    tmp_config_fname = Path(pipeline.config_fname).absolute()
    tmp_config_fname.unlink()
