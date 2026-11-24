import numpy as np
import pytest

from dakd import data
from dakd.config import all_tasks, load_base

TASKS = all_tasks()


@pytest.fixture(scope="module", params=TASKS)
def sp(request):
    return request.param, data.load_splits(request.param)


def test_nested(sp):
    _, s = sp
    assert s["student_5k"][:2000] == s["student_2k"]
    assert s["student_10k"][:5000] == s["student_5k"]
    assert len(set(s["student_10k"])) == 10000


def test_dev_test_partition_validation(sp):
    _, s = sp
    assert not set(s["dev"]) & set(s["test"])
    assert set(s["dev"]) | set(s["test"]) == set(range(s["dataset_sizes"]["validation"]))


def test_class_ratio_within_2pp(sp):
    task, s = sp
    y = data.load_raw(task)["train"][1]
    full = np.bincount(y) / len(y)
    for k in ("student_2k", "student_5k", "student_10k"):
        ratio = np.bincount(y[s[k]], minlength=len(full)) / len(s[k])
        assert np.abs(ratio - full).max() < 0.02, k
    yv = data.load_raw(task)["validation"][1]
    fullv = np.bincount(yv) / len(yv)
    for k in ("dev", "test"):
        ratio = np.bincount(yv[s[k]], minlength=len(fullv)) / len(s[k])
        assert np.abs(ratio - fullv).max() < 0.02, k


def test_regenerate_identical(sp):
    task, s = sp
    new = data.compute_splits(task)
    for k in ("student_2k", "student_5k", "student_10k", "dev", "test"):
        assert new[k] == s[k]


def test_seed_independent_of_train_seed():
    assert "seed" not in str(load_base()["split_seed"]) and load_base()["split_seed"] == 0
