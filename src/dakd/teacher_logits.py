from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from . import paths, weights as W


def _sha256(p, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def cache_teacher_logits(task: str, force: bool = False) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    from . import data as D
    from .config import load_task

    wdir = paths.weights_dir("s01", f"{task}__teacher__full__s42", "teacher", task)
    out = paths.ensure(paths.teacher_logits_dir(task))
    if (out / "DONE").exists() and not force:
        print(f"[skip] {task}: teacher logits 이미 있음")
        return
    tmeta = W.require(wdir, f"[{task}] teacher")
    (out / "DONE").unlink(missing_ok=True)
    ml = tmeta["max_length"]
    device = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(wdir).to(device).eval()
    tok = D.get_tokenizer(task, str(wdir))
    collate = D.make_collate(tok.pad_token_id)
    K = load_task(task)["num_labels"]

    def logits_for(split):
        ds = D.build_dataset(task, split, tok, ml)
        outs = []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=256, collate_fn=collate):
                outs.append(model(input_ids=b["input_ids"].to(device),
                                  attention_mask=b["attention_mask"].to(device)).logits.float().cpu().numpy())
        return ds, np.concatenate(outs).astype(np.float32)

    n_train = D.load_splits(task)["dataset_sizes"]["train"]
    train_all = np.full((n_train, K), np.nan, dtype=np.float32)
    correct = np.full(n_train, -1, dtype=np.int8)
    needed = D.split_indices(task, "teacher_train")
    ds_idx = _SubsetDS(task, needed, tok, ml)
    outs = []
    with torch.no_grad():
        for b in DataLoader(ds_idx.ds, batch_size=256, collate_fn=collate):
            outs.append(model(input_ids=b["input_ids"].to(device),
                              attention_mask=b["attention_mask"].to(device)).logits.float().cpu().numpy())
    lg = np.concatenate(outs).astype(np.float32)
    train_all[ds_idx.ds.idx] = lg
    correct[ds_idx.ds.idx] = (lg.argmax(1) == ds_idx.ds.labels).astype(np.int8)
    np.save(out / "train.npy", train_all)
    np.save(out / "train_correct.npy", correct)
    for sp in ("dev", "test"):
        ds, lg = logits_for(sp)
        np.save(out / f"{sp}.npy", lg)
    meta = {"task": task, "teacher_checkpoint_sha256": tmeta["sha256"], "teacher_weights_dir": str(wdir),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_rows_computed": int((correct >= 0).sum()),
            "teacher_train_acc": float((correct[correct >= 0] == 1).mean())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "DONE").write_text("ok")
    print(f"[teacher_logits] {task}: train acc={meta['teacher_train_acc']:.4f}, rows={meta['train_rows_computed']}")
    del model
    torch.cuda.empty_cache()


class _SubsetDS:
    def __init__(self, task, idx, tok, max_len):
        from . import data as D
        raw = D.load_raw(task)["train"]
        idx = np.asarray(idx)
        enc = tok([raw[0][i] for i in idx], truncation=True, max_length=max_len)["input_ids"]
        self.ds = D.TextDataset(enc, raw[1][idx], idx)
