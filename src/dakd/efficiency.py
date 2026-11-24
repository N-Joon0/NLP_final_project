from __future__ import annotations

import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from . import paths
from .config import load_base, load_task, parse_tasks


def _gpu_busy(gpu_index: int) -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-i", str(gpu_index), "--query-gpu=utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True)
        return int(out.strip())
    except Exception:
        return 0


def _size_mb(model) -> float:
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td, safe_serialization=True)
        return sum(f.stat().st_size for f in Path(td).glob("*.safetensors")) / 2**20


def measure(task: str, kind: str, precision: str, gpu: int = 0, seq_len: int = 128) -> dict:
    import torch
    from .models import build_for_condition, count_params

    base = load_base()
    dev = torch.device(f"cuda:{gpu}")
    cond = "teacher" if kind == "teacher" else "student_ft"
    model = build_for_condition(task, cond, base["n_layers"]).to(dev).eval()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model = model.to(dtype)
    row = {"task": task, "model": kind, "precision": precision, "n_layers": model.config.num_hidden_layers,
           "params": count_params(model), "size_mb": _size_mb(model), "seq_len": seq_len, "latency_batch": 1, "throughput_batch": 64,
           "latency_warmup": 20, "latency_iters": 200, "throughput_warmup": 10, "throughput_iters": 50}

    def batch(b):
        ids = torch.randint(1000, 5000, (b, seq_len), device=dev)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    with torch.no_grad():
        x = batch(1)
        for _ in range(20):
            model(**x)
        torch.cuda.synchronize(dev)
        times = []
        for _ in range(200):
            t = time.perf_counter()
            model(**x)
            torch.cuda.synchronize(dev)
            times.append((time.perf_counter() - t) * 1000)
        row["latency_ms_bs1"] = statistics.median(times)

        x = batch(64)
        for _ in range(10):
            model(**x)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t = time.perf_counter()
        for _ in range(50):
            model(**x)
        torch.cuda.synchronize(dev)
        row["throughput_samples_s_bs64"] = 64 * 50 / (time.perf_counter() - t)
        row["peak_mem_mb_bs64"] = torch.cuda.max_memory_allocated(dev) / 2**20
    del model
    torch.cuda.empty_cache()
    return row


def run_efficiency(task_arg: str = "all", gpu: int = 0) -> Path:
    import torch
    util = _gpu_busy(gpu)
    if util > 10:
        print(f"[경고] GPU {gpu} 사용률 {util}% — 다른 run 이 돌고 있으면 latency 가 부정확합니다. 빈 GPU 에서 측정하세요.")
    rows = []
    for task in parse_tasks(task_arg):
        for kind in ("teacher", "student"):
            for prec in ("fp32", "bf16"):
                r = measure(task, kind, prec, gpu)
                r.update({"gpu_name": torch.cuda.get_device_name(gpu), "torch": torch.__version__,
                          "cuda": torch.version.cuda})
                rows.append(r)
                print(f"[eff] {task} {kind} {prec}: {r['latency_ms_bs1']:.2f}ms, {r['throughput_samples_s_bs64']:.0f}/s, "
                      f"{r['peak_mem_mb_bs64']:.0f}MB, {r['params'] / 1e6:.1f}M, {r['size_mb']:.0f}MB")
    out = paths.ensure(paths.results_dir()) / "efficiency.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"저장: {out}")
    return out
