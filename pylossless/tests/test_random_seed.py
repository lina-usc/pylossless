from copy import deepcopy

import mne
import numpy as np
import pytest

import pylossless as ll


def _pipeline(random_seed=97):
    config = ll.Config().load_default()
    config["random_seed"] = random_seed
    return ll.LosslessPipeline(config=config)


def test_default_configs_define_random_seed():
    """Adult and infant defaults should opt into reproducible ICA."""
    assert ll.Config().load_default("adults")["random_seed"] == 97
    assert ll.Config().load_default("infants")["random_seed"] == 97


@pytest.mark.parametrize("run", ["run1", "run2"])
def test_global_random_seed_is_inherited_by_ica_runs(run):
    """Both ICA runs should inherit the pipeline-level seed."""
    _, kwargs = _pipeline(random_seed=123)._get_ica_kwargs(run)
    assert kwargs["random_state"] == 123


def test_run_specific_random_state_takes_precedence():
    """A deliberate local override should win over the global seed."""
    pipeline = _pipeline(random_seed=123)
    pipeline.config["ica"]["ica_args"]["run2"]["random_state"] = 456

    _, run1_kwargs = pipeline._get_ica_kwargs("run1")
    _, run2_kwargs = pipeline._get_ica_kwargs("run2")

    assert run1_kwargs["random_state"] == 123
    assert run2_kwargs["random_state"] == 456


def test_legacy_config_retains_historical_seed():
    """Configs without the new key should preserve the former seed of 97."""
    config = ll.Config().load_default()
    del config["random_seed"]
    pipeline = ll.LosslessPipeline(config=config)

    _, kwargs = pipeline._get_ica_kwargs("run1")
    assert kwargs["random_state"] == 97


def test_none_requests_nondeterministic_initialization():
    """None should be forwarded rather than replaced by the fallback seed."""
    _, kwargs = _pipeline(random_seed=None)._get_ica_kwargs("run1")
    assert kwargs["random_state"] is None


@pytest.mark.parametrize("bad_seed", [True, 1.5, "97", [97]])
def test_invalid_global_random_seed_raises(bad_seed):
    """Invalid YAML-compatible seed values should fail before ICA fitting."""
    pipeline = _pipeline(random_seed=bad_seed)
    with pytest.raises(TypeError, match="random_seed.*integer or None"):
        pipeline._get_ica_kwargs("run1")


def test_numpy_integer_seed_is_normalized():
    """NumPy integer scalars should be accepted and normalized to Python int."""
    _, kwargs = _pipeline(random_seed=np.int64(12))._get_ica_kwargs("run1")
    assert kwargs["random_state"] == 12
    assert type(kwargs["random_state"]) is int


def test_resolving_ica_kwargs_does_not_mutate_config():
    """Runtime defaults must not modify the user-provided configuration."""
    pipeline = _pipeline(random_seed=123)
    before = deepcopy(pipeline.config)

    pipeline._get_ica_kwargs("run1")
    pipeline._get_ica_kwargs("run2")

    assert pipeline.config == before
    assert "random_state" not in pipeline.config["ica"]["ica_args"]["run1"]


def test_random_seed_round_trip(tmp_path):
    """The global seed should survive YAML serialization."""
    config = ll.Config().load_default()
    config["random_seed"] = 31415
    path = tmp_path / "config.yaml"

    config.save(path)
    loaded = ll.Config().read(path)

    assert loaded["random_seed"] == 31415


def test_invalid_ica_run_fails_with_clear_error():
    """An invalid run name should fail before indexing the configuration."""
    with pytest.raises(ValueError, match="run1.*run2"):
        _pipeline()._get_ica_kwargs("run3")

@pytest.mark.filterwarnings(
    "ignore:FastICA did not converge:sklearn.exceptions.ConvergenceWarning"
)
def test_global_seed_reproduces_mne_ica_fit():
    """The same global seed should reproduce a real MNE ICA fit."""
    rng = np.random.default_rng(123)
    sfreq = 128.0
    times = np.arange(0, 8, 1 / sfreq)
    sources = np.vstack(
        [
            np.sin(2 * np.pi * 7 * times),
            np.sign(np.sin(2 * np.pi * 3 * times)),
            rng.normal(size=times.size),
            np.sin(2 * np.pi * 13 * times),
        ]
    )
    mixing = np.array(
        [
            [1.0, 0.3, 0.2, 0.4],
            [0.2, 1.0, 0.4, 0.1],
            [0.5, 0.2, 1.0, 0.3],
            [0.1, 0.4, 0.2, 1.0],
        ]
    )
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg")
    raw = mne.io.RawArray(mixing @ sources, info, verbose=False)
    raw.filter(1, 40, verbose=False)

    unmixing_matrices = []
    for _ in range(2):
        pipeline = _pipeline(random_seed=123)
        pipeline.config["epoching"]["epochs_args"].update(tmax=1, baseline=None)
        pipeline.config["ica"]["ica_args"]["run1"].update(
            n_components=3, max_iter=1000
        )
        pipeline.raw = raw.copy()
        pipeline.run_ica("run1")
        assert pipeline.ica1.random_state == 123
        unmixing_matrices.append(pipeline.ica1.unmixing_matrix_)

    np.testing.assert_allclose(unmixing_matrices[0], unmixing_matrices[1])
