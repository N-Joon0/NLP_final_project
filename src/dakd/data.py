from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from . import paths
from .config import load_base, load_task

SPLIT_NAMES = ["teacher_train", "student_2k", "student_5k", "student_10k", "dev", "test"]


@lru_cache(maxsize=None)
def load_raw(task: str):
    from datasets import load_dataset

    tc = load_task(task)
    last = None
    for _ in range(3):
        try:
            ds = load_dataset(tc["hf_dataset"], tc["hf_config"])
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"{tc['hf_dataset']}/{tc['hf_config']} 다운로드 실패(3회 재시도): {last}")
    out = {}
    for key, split in (("train", tc["train_split"]), ("validation", tc["val_split"])):
        d = ds[split]
        out[key] = ([str(x) for x in d[tc["text_field"]]], np.asarray(d[tc["label_field"]], dtype=np.int64))
    return out


def dataset_versions() -> dict:
    import datasets
    import sklearn
    return {"datasets": datasets.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__}


def _nested_order(labels: np.ndarray, n_total: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    classes = sorted(set(labels.tolist()))
    per_class = {c: rng.permutation(np.where(labels == c)[0]).tolist() for c in classes}
    frac = {c: len(per_class[c]) / len(labels) for c in classes}
    taken = {c: 0 for c in classes}
    order: list[int] = []
    for t in range(n_total):
        best = max((c for c in classes if taken[c] < len(per_class[c])),
                   key=lambda c: (frac[c] * (t + 1) - taken[c], -c))
        order.append(int(per_class[best][taken[best]]))
        taken[best] += 1
    return order


def compute_splits(task: str) -> dict:
    from sklearn.model_selection import train_test_split

    base = load_base()
    raw = load_raw(task)
    y_train, y_val = raw["train"][1], raw["validation"][1]
    seed = base["split_seed"]
    budgets = base["data"]["budgets"]
    order = _nested_order(y_train, max(budgets), seed)
    val_idx = np.arange(len(y_val))
    dev_idx, test_idx = train_test_split(
        val_idx, test_size=base["data"]["dev_fraction"], stratify=y_val, random_state=seed)
    dev_idx, test_idx = sorted(dev_idx.tolist()), sorted(test_idx.tolist())

    def dist(idx, y):
        c = np.bincount(y[np.asarray(idx)], minlength=load_task(task)["num_labels"])
        return [int(x) for x in c]

    subsets = {f"student_{b // 1000}k": order[:b] for b in budgets}
    return {
        "task": task,
        "split_seed": seed,
        "dataset_sizes": {"train": int(len(y_train)), "validation": int(len(y_val))},
        "versions": dataset_versions(),
        "teacher_train": {"note": "official train, all indices", "size": int(len(y_train))},
        **{k: v for k, v in subsets.items()},
        "dev": dev_idx,
        "test": test_idx,
        "dev_labels": [int(x) for x in y_val[dev_idx]],
        "test_labels": [int(x) for x in y_val[test_idx]],
        "class_distribution": {
            "train": dist(np.arange(len(y_train)), y_train),
            **{k: dist(v, y_train) for k, v in subsets.items()},
            "dev": dist(dev_idx, y_val),
            "test": dist(test_idx, y_val),
        },
    }


def load_splits(task: str) -> dict:
    p = paths.splits_path(task)
    if not p.exists():
        raise FileNotFoundError(
            f"분할 파일이 없습니다: {p}\n  → `bash scripts/01_prepare_data.sh` 를 실행하거나 git pull 로 받으세요.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def prepare_splits(task: str, force: bool = False) -> str:
    p = paths.splits_path(task)
    new = compute_splits(task)
    if p.exists() and not force:
        old = json.load(open(p, encoding="utf-8"))
        keys = [k for k in new if k not in ("versions",)]
        diff = [k for k in keys if old.get(k) != new[k]]
        if diff:
            raise RuntimeError(
                f"[{task}] 기존 분할 JSON 과 재계산 결과가 다릅니다 (다른 키: {diff}).\n"
                f"  덮어쓰지 않습니다. 의도한 변경이면 --force-regenerate 를 사용하세요.")
        verify_splits(task, old)
        return "verified"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False)
    verify_splits(task, new)
    return "created"


def verify_splits(task: str, sp: dict | None = None) -> None:
    sp = sp or load_splits(task)
    base = load_base()
    s2, s5, s10 = sp["student_2k"], sp["student_5k"], sp["student_10k"]
    assert s5[:len(s2)] == s2 and s10[:len(s5)] == s5, "student 부분집합이 중첩되지 않습니다"
    assert len(set(s10)) == len(s10), "student 인덱스에 중복이 있습니다"
    dev, test = set(sp["dev"]), set(sp["test"])
    assert not (dev & test), "dev 와 test 가 겹칩니다"
    assert len(dev | test) == sp["dataset_sizes"]["validation"], "dev ∪ test != validation"
    assert [len(s2), len(s5), len(s10)] == [b for b in base["data"]["budgets"]], "budget 크기 불일치"
    exp = load_task(task).get("official_sizes")
    if exp and exp != sp["dataset_sizes"]:
        raise RuntimeError(f"[{task}] 데이터셋 크기가 명세와 다릅니다: {sp['dataset_sizes']} vs {exp}")


def split_indices(task: str, name: str) -> np.ndarray:
    sp = load_splits(task)
    if name == "teacher_train":
        return np.arange(sp["dataset_sizes"]["train"])
    if name in ("dev", "test") or name.startswith("student_"):
        return np.asarray(sp[name])
    raise ValueError(f"알 수 없는 split: {name}")


def budget_split_name(budget) -> str:
    return "teacher_train" if budget in (None, "full") else f"student_{int(budget) // 1000}k"


def get_texts_labels(task: str, name: str):
    idx = split_indices(task, name)
    raw = load_raw(task)["validation" if name in ("dev", "test") else "train"]
    texts = [raw[0][i] for i in idx]
    return texts, raw[1][idx], idx


def checkpoint_name(task: str) -> str:
    return load_task(task)["checkpoint"]


def get_tokenizer(task: str, path: str | None = None):
    from transformers import AutoTokenizer
    from .models import resolve_checkpoint
    return AutoTokenizer.from_pretrained(path or resolve_checkpoint(task))


class TextDataset:
    def __init__(self, input_ids, labels, idx, teacher_logits=None):
        self.input_ids, self.labels, self.idx, self.tl = input_ids, labels, idx, teacher_logits

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        item = {"input_ids": self.input_ids[i], "label": int(self.labels[i]), "idx": int(self.idx[i])}
        if self.tl is not None:
            item["teacher_logits"] = self.tl[i]
        return item


def build_dataset(task: str, split: str, tokenizer, max_length: int, teacher_logits=None) -> TextDataset:
    texts, labels, idx = get_texts_labels(task, split)
    enc = tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]
    return TextDataset(enc, labels, idx, teacher_logits)


def make_collate(pad_id: int):
    import torch

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            att[i, :n] = 1
        out = {"input_ids": ids, "attention_mask": att,
               "labels": torch.tensor([b["label"] for b in batch]),
               "idx": torch.tensor([b["idx"] for b in batch])}
        if "teacher_logits" in batch[0]:
            out["teacher_logits"] = torch.tensor(np.stack([b["teacher_logits"] for b in batch]), dtype=torch.float32)
        return out

    return collate
