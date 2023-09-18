from pathlib import Path
import shutil

import pylossless as ll
from pylossless.tests.test_datasets import load_openneuro_bids


def test_rejection_policy():
    """Test the rejection policy."""
    raw, config, bids_root = load_openneuro_bids()

    config.save("test_config.yaml")
    pipeline = ll.LosslessPipeline("test_config.yaml")
    not_in_1020 = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
    pipeline.raw = raw.pick("eeg", exclude=not_in_1020).load_data()
    pipeline.run_with_raw(pipeline.raw)

    rejection_config_path = Path("test_rejection_config.yaml")
    rejection_config = ll.Config()
    rejection_config["ch_flags_to_reject"] = ["ch_sd", "low_r"]
    rejection_config.save(rejection_config_path)

    raw = ll.RejectionPolicy(rejection_config_path).apply(pipeline)
    rejection_config_path.unlink()

    Path("test_config.yaml").unlink()  # delete config file
    shutil.rmtree(bids_root)
