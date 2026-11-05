import gc

import pytest

from axiom_prediction.dataset import load_axiom_dataset_shards

def load_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_axiom_dataset_shards([tmp_path / "missing.csv"], dataset_path = tmp_path)

def test_loading_restores_garbage_collector_state(tmp_path):
    gc.enable()
    load_missing(tmp_path)
    assert gc.isenabled()

    gc.disable()
    try:
        load_missing(tmp_path)
        assert not gc.isenabled()
    finally:
        gc.enable()
        gc.unfreeze()
