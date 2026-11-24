from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from .config import load_base


@dataclass
class GpuInfo:
    index: int
    name: str
    free_gb: float
    total_gb: float


def detect_gpus() -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                i, name, free, total = [x.strip() for x in line.split(",")]
                gpus.append(GpuInfo(int(i), name, float(free) / 1024, float(total) / 1024))
        except Exception:
            pass
    return gpus or _detect_via_torch()


_TORCH_PROBE = (
    "import torch, json\n"
    "out = []\n"
    "for i in range(torch.cuda.device_count()):\n"
    "    free, total = torch.cuda.mem_get_info(i)\n"
    "    out.append([i, torch.cuda.get_device_name(i), free / 2**30, total / 2**30])\n"
    "print(json.dumps(out))\n"
)


def _detect_via_torch() -> list[GpuInfo]:
    import json
    import sys
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    try:
        out = subprocess.check_output([sys.executable, "-c", _TORCH_PROBE], text=True, env=env,
                                      stderr=subprocess.DEVNULL, timeout=120)
        return [GpuInfo(int(i), n, float(f), float(t)) for i, n, f, t in json.loads(out.strip().splitlines()[-1])]
    except Exception:
        return []


def parse_gpu_ids(arg: str | None) -> list[int]:
    arg = arg or os.environ.get("DAKD_GPUS") or load_base()["gpu"]["default_ids"]
    try:
        ids = [int(x) for x in str(arg).split(",") if x.strip() != ""]
    except ValueError:
        raise RuntimeError(f"--gpus 형식이 잘못되었습니다: '{arg}' (예: 0,1,2,3)")
    if not ids or len(set(ids)) != len(ids):
        raise RuntimeError(f"--gpus 에 중복/빈 값이 있습니다: '{arg}'")
    return ids


def select_gpus(arg: str | None = None) -> list[GpuInfo]:
    required = load_base()["gpu"]["world_size"]
    ids = parse_gpu_ids(arg)
    if len(ids) != required:
        raise RuntimeError(
            f"GPU {len(ids)}장({ids})이 지정되었습니다. 실험은 재현성을 위해 항상 {required}장(DDP)으로 통일합니다.\n"
            f"  → 예: --gpus {','.join(str(i) for i in range(required))}")
    all_g = {g.index: g for g in detect_gpus()}
    if not all_g:
        raise RuntimeError("사용 가능한 NVIDIA GPU 를 찾지 못했습니다. (CPU 학습은 지원하지 않습니다)\n"
                           "  → 컨테이너는 `--gpus all` 로 실행해야 하고, `ls /dev/nvidia*` 와 "
                           "`python -c 'import torch; print(torch.cuda.is_available())'` 로 확인하세요.")
    missing = [i for i in ids if i not in all_g]
    if missing:
        raise RuntimeError(f"요청한 GPU {missing} 를 찾을 수 없습니다. 감지된 GPU: {sorted(all_g)}")
    return [all_g[i] for i in ids]


def describe(gpus: list[GpuInfo]) -> str:
    return ", ".join(f"[{g.index}] {g.name} {g.free_gb:.0f}/{g.total_gb:.0f}GB free" for g in gpus)
