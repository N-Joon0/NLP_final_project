import pytest

from dakd.config import all_tasks, load_base
from dakd.models import build_student, build_teacher, count_params, resolve_checkpoint


@pytest.mark.parametrize("task", all_tasks())
def test_student_layers_and_vocab(task):
    from dakd.config import load_task
    nl = load_task(task)["num_labels"]
    ck = resolve_checkpoint(task)
    n = load_base()["n_layers"]
    t, s = build_teacher(ck, nl), build_student(ck, nl, n)
    assert len(t.bert.encoder.layer) == 12
    assert len(s.bert.encoder.layer) == n and s.config.num_hidden_layers == n
    assert t.config.vocab_size == s.config.vocab_size
    assert t.classifier.out_features == s.classifier.out_features == nl
    print(f"{task}: teacher {count_params(t) / 1e6:.1f}M, student {count_params(s) / 1e6:.1f}M")
    assert count_params(s) < count_params(t)
