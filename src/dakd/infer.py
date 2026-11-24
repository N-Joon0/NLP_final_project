from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths, weights as W
from .registry import RunSpec


def spec_from_meta(meta: dict) -> RunSpec:
    s = {k: v for k, v in meta["spec"].items() if k not in ("run_id", "missing")}
    return RunSpec(**s)


def infer_run(spec: RunSpec, splits: list[str] | None = None, batch_size: int = 128, overwrite: bool = False) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    from . import data as D
    from .config import load_base, load_task
    from .metrics import compute_metrics
    from .train import _amp_dtype, pick_precision, predict_logits, softmax_np

    splits = splits or ["test"]
    wdir = paths.weights_dir(spec.stage, spec.run_id, spec.condition, spec.task)
    meta = W.require(wdir, f"[{spec.run_id}]")
    run_dir = spec.run_dir
    todo = [sp for sp in splits if overwrite or not (sp == "test" and (run_dir / "DONE_INFER").exists())]
    if not todo:
        print(f"[skip] {spec.run_id}: DONE_INFER 있음")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    base, tc = load_base(), load_task(spec.task)
    device = torch.device("cuda")
    amp = _amp_dtype(pick_precision(meta.get("precision", "auto")))
    model = AutoModelForSequenceClassification.from_pretrained(wdir).to(device)
    tok = D.get_tokenizer(spec.task, str(wdir))
    collate = D.make_collate(tok.pad_token_id)
    for sp in todo:
        ds = D.build_dataset(spec.task, sp, tok, meta["max_length"])
        logits = predict_logits(model, DataLoader(ds, batch_size=batch_size, collate_fn=collate), device, amp)
        probs = softmax_np(logits)
        m = compute_metrics(probs, ds.labels, tc["primary_metric"], base["eval"]["ece_bins"])
        m["weights_sha256"] = meta["sha256"]
        np.save(run_dir / f"{sp}_preds.npy", probs)
        np.save(run_dir / f"{sp}_logits.npy", logits)
        (run_dir / f"{sp}_metrics.json").write_text(json.dumps(m, indent=1))
        if sp == "test":
            (run_dir / "DONE_INFER").write_text("ok")
        print(f"[infer] {spec.run_id} {sp}: {tc['primary_metric']}={m['primary']:.4f} (n={m['n']})", flush=True)
    del model
    torch.cuda.empty_cache()


def infer_stage(stage_arg: str, task_arg: str = "all", splits: list[str] | None = None, overwrite: bool = False) -> int:
    from .config import parse_stage_list, parse_tasks
    from .registry import expand_stage
    missing, done, ran = [], 0, 0
    for st in parse_stage_list(stage_arg):
        for spec in expand_stage(st, parse_tasks(task_arg)):
            if not spec.test_inference:
                continue
            wdir = paths.weights_dir(spec.stage, spec.run_id, spec.condition, spec.task)
            if (spec.run_dir / "DONE_INFER").exists() and not overwrite:
                done += 1
                continue
            ok, why = W.check(wdir)
            if not ok:
                missing.append(f"{st}/{spec.run_id}  ({why})")
                continue
            infer_run(spec, splits, overwrite=overwrite)
            ran += 1
    print(f"[infer-stage] 추론 {ran}건, 이미 완료 {done}건, 가중치 없음 {len(missing)}건")
    for m in missing:
        print(f"  - 가중치 없음: {m}")
    if missing:
        print(f"  → Google Drive 에서 받은 가중치를 {paths.weights_root()} 아래 같은 구조로 넣고 다시 실행하세요.")
    return 1 if missing else 0
