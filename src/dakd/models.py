from __future__ import annotations

from .config import load_task


def resolve_checkpoint(task: str) -> str:
    from transformers import AutoConfig

    tc = load_task(task)
    try:
        AutoConfig.from_pretrained(tc["checkpoint"])
        return tc["checkpoint"]
    except Exception:
        fb = tc.get("checkpoint_fallback")
        if not fb:
            raise
        AutoConfig.from_pretrained(fb)
        return fb


def build_teacher(ckpt: str, num_labels: int):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=num_labels)


def build_student(ckpt: str, num_labels: int, n_layers: int = 6):
    model = build_teacher(ckpt, num_labels)
    model.bert.encoder.layer = model.bert.encoder.layer[:n_layers]
    model.config.num_hidden_layers = n_layers
    return model


def build_for_condition(task: str, condition: str, n_layers: int, ckpt: str | None = None):
    tc = load_task(task)
    ckpt = ckpt or resolve_checkpoint(task)
    if condition == "teacher":
        return build_teacher(ckpt, tc["num_labels"])
    return build_student(ckpt, tc["num_labels"], n_layers)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
