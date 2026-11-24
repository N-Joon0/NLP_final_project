from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from . import paths, weights as W
from .config import effective_epochs, load_base, load_task
from .registry import RunSpec

EXIT_OOM = 75
EXIT_NAN = 76


class NonFiniteLoss(RuntimeError):
    pass


def is_done_train(d: Path) -> bool:
    return (d / "DONE_TRAIN").exists()


def is_done(spec: RunSpec) -> bool:
    d = spec.run_dir
    return is_done_train(d) and (not spec.test_inference or (d / "DONE_INFER").exists())


def running_elsewhere(d: Path, stale_s: int = 900) -> bool:
    f = d / "RUNNING"
    if not f.exists():
        return False
    try:
        host, pid = f.read_text().split(":")
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0)
                return int(pid) != os.getpid()
            except OSError:
                return False
        return time.time() - f.stat().st_mtime < stale_s
    except Exception:
        return False


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def pick_precision(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def _amp_dtype(precision: str):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def status_file(spec: RunSpec) -> Path:
    return paths.logs_dir() / spec.stage / f"{spec.run_id}.status"


def predict_logits(model, loader, device, amp_dtype):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                z = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
            out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Dist:
    def __init__(self):
        import torch
        import torch.distributed as dist
        self.dist = dist
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.local = int(os.environ.get("LOCAL_RANK", 0))
        self.main = self.rank == 0
        self.device = torch.device("cuda", self.local % max(1, torch.cuda.device_count()))
        torch.cuda.set_device(self.device)
        if self.world > 1:
            backend = os.environ.get("DAKD_DIST_BACKEND", "nccl")
            dist.init_process_group(backend, timeout=timedelta(minutes=60))

    def barrier(self):
        if self.world > 1:
            self.dist.barrier()

    def sum(self, t):
        if self.world > 1:
            self.dist.all_reduce(t)
        return t

    def gather_obj(self, obj):
        if self.world == 1:
            return [obj]
        out = [None] * self.world
        self.dist.all_gather_object(out, obj)
        return out

    def close(self):
        if self.world > 1 and self.dist.is_initialized():
            self.dist.destroy_process_group()


def rank_slice(n: int, rank: int, world: int) -> tuple[int, int]:
    base, rem = divmod(n, world)
    a = rank * base + min(rank, rem)
    return a, a + base + (1 if rank < rem else 0)


def train_run(spec: RunSpec, micro_batch: int | None = None, precision: str = "auto",
              no_infer: bool = False, max_steps: int | None = None) -> Path:
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from . import data as D
    from .losses import distill_loss, instance_weights
    from .metrics import compute_metrics
    from .models import build_for_condition, count_params, resolve_checkpoint
    from .seed import set_seed

    base, tc = load_base(), load_task(spec.task)
    tr = base["train"]
    d = spec.run_dir
    wdir = paths.weights_dir(spec.stage, spec.run_id, spec.condition, spec.task)

    if is_done_train(d):
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"[skip] {spec.run_id}: DONE_TRAIN 있음")
            if not no_infer and spec.test_inference and not (d / "DONE_INFER").exists():
                from .infer import infer_run
                infer_run(spec)
        return d
    if running_elsewhere(d):
        raise RuntimeError(f"{spec.run_id} 은(는) 다른 프로세스에서 실행 중입니다 ({d / 'RUNNING'})")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 를 찾지 못했습니다. (CPU 학습은 지원하지 않습니다)")

    dd = Dist()
    world, rank, device = dd.world, dd.rank, dd.device
    if dd.main:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        (d / "RUNNING").write_text(f"{socket.gethostname()}:{os.getpid()}")
        W.prepare(wdir)
    dd.barrier()

    precision = pick_precision(precision if precision != "auto" else tr["precision"])
    amp = _amp_dtype(precision)
    is_teacher = spec.condition == "teacher"
    epochs = effective_epochs(base, is_teacher)
    bs = tr["batch_size"]
    per_rank = rank_slice(bs, 0, world)[1]
    micro = max(1, min(micro_batch or per_rank, per_rank))

    set_seed(spec.seed)
    ckpt = resolve_checkpoint(spec.task)
    model = build_for_condition(spec.task, spec.condition, base["n_layers"], ckpt).to(device)
    n_layers = model.config.num_hidden_layers
    n_params = count_params(model)
    tok = D.get_tokenizer(spec.task, ckpt)
    torch.manual_seed(spec.seed * 1000 + rank)
    net = DDP(model, device_ids=[device.index]) if world > 1 else model

    lam = 0.0 if spec.condition in ("teacher", "student_ft") else float(spec.lam)
    kd_kw = dict(lam=lam, T=spec.T or 1.0, gamma=spec.gamma or 0.0, beta=spec.beta or 0.0, weight_norm=spec.weight_norm)
    global_norm = spec.weight_norm == "batch_mean" and lam > 0 and micro >= per_rank
    cfg_json = {
        "spec": spec.to_dict(), "task": spec.task, "n_layers": n_layers, "epochs": epochs, "batch_size": bs,
        "world_size": world, "per_rank_batch": per_rank, "micro_batch": micro, "precision": precision,
        "norm_scope": ("global_batch" if global_norm else "micro_batch") if spec.weight_norm == "batch_mean" else None,
        "lr": tr["lr"], "weight_decay": tr["weight_decay"], "warmup_ratio": tr["warmup_ratio"],
        "max_length": tr["max_length"], "checkpoint": ckpt, "git_commit": _git_commit(),
        "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(device), "gpu_count": world,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": __import__("transformers").__version__, "params": n_params,
        "weights_dir": str(wdir), "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if micro < per_rank and spec.weight_norm == "batch_mean" and dd.main:
        print("[경고] OOM fallback(micro-batch<per-rank batch): batch_mean 정규화가 micro-batch 단위로 계산됩니다.")
    if dd.main:
        (d / "config.json").write_text(json.dumps(cfg_json, indent=1, ensure_ascii=False))

    split = D.budget_split_name(spec.budget)
    train_correct = None
    train_ds = D.build_dataset(spec.task, split, tok, tr["max_length"])
    if spec.is_kd:
        tdir = paths.teacher_logits_dir(spec.task)
        if not (tdir / "DONE").exists():
            raise FileNotFoundError(f"teacher logits 없음: {tdir} (04 단계 실행 또는 receive_shared.sh)")
        all_logits = np.load(tdir / "train.npy")
        train_correct = np.load(tdir / "train_correct.npy")
        tl = all_logits[train_ds.idx]
        if np.isnan(tl).any():
            raise RuntimeError("teacher logits 에 NaN(미계산 행) 포함 — 04 단계(teacher logits)를 다시 실행하세요")
        train_ds.tl = tl
    dev_ds = D.build_dataset(spec.task, "dev", tok, tr["max_length"])
    collate = D.make_collate(tok.pad_token_id)
    g = torch.Generator()
    g.manual_seed(spec.seed)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": tr["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=tr["lr"])
    total = epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(tr["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    primary = tc["primary_metric"]
    log_f = open(d / "train_log.jsonl", "a") if dd.main else None
    torch.cuda.reset_peak_memory_stats()
    t0, step, best, best_ep = time.time(), 0, -1.0, -1
    epoch_log, weight_log, last_rec = [], [], None

    for ep in range(1, epochs + 1):
        net.train()
        ep_t0 = time.time()
        acc = torch.zeros(4, device=device)
        w_rec = {"idx": [], "w": [], "c": [], "d": [], "kd": [], "ce": []}
        for batch in train_loader:
            B = batch["labels"].shape[0]
            a, b_ = rank_slice(B, rank, world)
            n_loc = b_ - a
            if n_loc == 0:
                a, b_, scale_fix = 0, 1, 0.0
            else:
                scale_fix = 1.0
            loc = {k: v[a:b_] for k, v in batch.items()}
            chunks = {k: v.split(micro) for k, v in loc.items()}
            nchunk = len(chunks["labels"])
            opt.zero_grad(set_to_none=True)
            for ci in range(nchunk):
                mb = {k: v[ci].to(device, non_blocking=True) for k, v in chunks.items()}
                k_n = mb["labels"].shape[0]
                sync = ci == nchunk - 1 or world == 1
                with (contextlib.nullcontext() if sync else net.no_sync()):
                    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                        z = net(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"]).logits
                    norm_mean = None
                    if global_norm:
                        w_raw, _, _ = instance_weights(z.float(), mb["teacher_logits"], kd_kw["gamma"], kd_kw["beta"], "none")
                        s = dd.sum((w_raw.sum() * scale_fix).reshape(1))
                        norm_mean = s[0] / B
                    loss, info = distill_loss(z, mb["labels"], mb.get("teacher_logits"), norm_mean=norm_mean, **kd_kw)
                    if not torch.isfinite(loss):
                        raise NonFiniteLoss(f"non-finite loss at step {step} (rank {rank})")
                    coef = scale_fix * world * k_n / B if n_loc else 0.0
                    scaler.scale(loss * coef).backward()
                if scale_fix:
                    acc[0] += loss.detach() * k_n
                    acc[1] += info["ce"].sum()
                    if "kd" in info:
                        acc[2] += info["kd"].sum()
                    acc[3] += k_n
                    if "w" in info:
                        w_rec["idx"].append(mb["idx"].cpu().numpy())
                        for k in ("w", "c", "d", "kd", "ce"):
                            w_rec[k].append(info[k].cpu().numpy())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if dd.main:
                (d / "RUNNING").touch()
                if step % tr["log_every"] == 0:
                    log_f.write(json.dumps({"step": step, "epoch": ep, "lr": sched.get_last_lr()[0],
                                            "loss_rank0": float(acc[0] / max(acc[3], 1))}) + "\n")
                    log_f.flush()
            if max_steps and step >= max_steps:
                break

        tot = dd.sum(acc.clone()).cpu().numpy()
        gathered = dd.gather_obj({k: (np.concatenate(v) if v else np.zeros(0)) for k, v in w_rec.items()})
        ep_time = time.time() - ep_t0
        if dd.main:
            if lam > 0:
                rec = {k: np.concatenate([x[k] for x in gathered]) for k in w_rec}
                weight_log.append(_weight_stats(ep, rec, train_correct))
                last_rec = rec
            net.eval()
            dl = predict_logits(model, dev_loader, device, amp)
            probs = softmax_np(dl)
            m = compute_metrics(probs, dev_ds.labels, primary, base["eval"]["ece_bins"])
            e = {"epoch": ep, "train_loss": float(tot[0] / max(tot[3], 1)), "train_ce": float(tot[1] / max(tot[3], 1)),
                 "train_kd": float(tot[2] / max(tot[3], 1)) if lam > 0 else None, "epoch_time_s": ep_time,
                 "dev": {k: v for k, v in m.items() if k not in ("reliability",)}}
            epoch_log.append(e)
            print(f"[{spec.run_id}] epoch {ep}/{epochs} loss={e['train_loss']:.4f} dev_{primary}={m['primary']:.4f} "
                  f"({ep_time:.0f}s)", flush=True)
            if m["primary"] > best:
                best, best_ep = m["primary"], ep
                W.save(model, tok, wdir, {})
                np.save(d / "dev_preds.npy", probs)
                np.save(d / "dev_logits.npy", dl)
                (d / "best_dev_metrics.json").write_text(json.dumps(m))
        dd.barrier()
        if max_steps and step >= max_steps:
            break

    if dd.main:
        log_f.close()
        metrics = {
            "best_epoch": best_ep, "best_dev": best, "primary_metric": primary, "epochs": epoch_log,
            "train_time_s": time.time() - t0, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
            "world_size": world, "total_steps": step, "weight_stats": weight_log,
            "train_examples": len(train_ds), "train_samples_per_s": len(train_ds) * len(epoch_log) / (time.time() - t0),
        }
        (d / "train_metrics.json").write_text(json.dumps(metrics, indent=1))
        if last_rec is not None and len(last_rec["idx"]):
            np.savez_compressed(d / "train_example_stats.npz", **last_rec)
        W.finalize(wdir, {"run_id": spec.run_id, "stage": spec.stage, "spec": spec.to_dict(), "task": spec.task,
                          "condition": spec.condition, "n_layers": n_layers, "num_labels": tc["num_labels"],
                          "max_length": tr["max_length"], "precision": precision, "best_epoch": best_ep,
                          "best_dev": best, "primary_metric": primary, "git_commit": _git_commit(),
                          "host": socket.gethostname(), "world_size": world})
        if is_teacher:
            (d / "DONE").write_text("ok")
        (d / "DONE_TRAIN").write_text("ok")
        (d / "RUNNING").unlink(missing_ok=True)
        print(f"[{spec.run_id}] done: best dev {primary}={best:.4f} @epoch {best_ep}, {metrics['train_time_s']:.0f}s, "
              f"peak {metrics['peak_mem_mb']:.0f}MB/GPU, world={world}, weights → {wdir}", flush=True)
    dd.barrier()
    del net, model, opt
    torch.cuda.empty_cache()
    dd.close()
    if dd.main and not no_infer and spec.test_inference:
        from .infer import infer_run
        infer_run(spec)
    return d


def _weight_stats(ep: int, rec: dict, train_correct: np.ndarray | None) -> dict:
    idx, w, c, dd = rec["idx"].astype(int), rec["w"], rec["c"], rec["d"]
    out = {"epoch": ep, "w_mean": float(w.mean()), "w_std": float(w.std()), "w_min": float(w.min()),
           "w_max": float(w.max()), "c_mean": float(c.mean()), "d_mean": float(dd.mean()),
           "kd_mean": float(rec["kd"].mean())}
    if train_correct is not None:
        corr, wrong = train_correct[idx] == 1, train_correct[idx] == 0
        out["w_mean_teacher_correct"] = float(w[corr].mean()) if corr.any() else None
        out["w_mean_teacher_wrong"] = float(w[wrong].mean()) if wrong.any() else None
        out["c_mean_teacher_wrong"] = float(c[wrong].mean()) if wrong.any() else None
        out["n_teacher_wrong"] = int(wrong.sum())
    return out


def main_train(spec: RunSpec, **kw) -> int:
    import torch
    try:
        train_run(spec, **kw)
        return 0
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] {spec.run_id}: {e}", file=sys.stderr)
        _write_status(spec, "oom")
        return EXIT_OOM
    except NonFiniteLoss as e:
        print(f"[NaN] {spec.run_id}: {e}", file=sys.stderr)
        _write_status(spec, "nan")
        return EXIT_NAN


def _write_status(spec: RunSpec, reason: str) -> None:
    p = status_file(spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(reason)
