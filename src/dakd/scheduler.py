from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from . import deps, gpu as G, paths
from .config import load_base, load_stage, parse_stage_list, parse_tasks
from .registry import RunSpec, expand_stage
from .train import EXIT_NAN, EXIT_OOM, is_done, running_elsewhere, status_file


@dataclass
class Job:
    spec: RunSpec
    status: str = "pending"
    attempts: int = 0
    micro_batch: int | None = None
    precision: str = "auto"


def _ints(arg: str | None) -> set[int] | None:
    return {int(x) for x in arg.split(",") if x.strip()} if arg else None


def build_jobs(stages: list[str], tasks: list[str], allow_missing: bool = False,
               only_condition: str | None = None, only_budget: str | None = None,
               only_seed: str | None = None) -> list[RunSpec]:
    budgets, seeds = _ints(only_budget), _ints(only_seed)
    specs: list[RunSpec] = []
    for st in stages:
        for s in expand_stage(st, tasks, allow_missing=allow_missing):
            if only_condition and s.condition != only_condition:
                continue
            if budgets and s.budget not in budgets:
                continue
            if seeds and s.seed not in seeds:
                continue
            specs.append(s)
    return sorted(specs, key=lambda s: (s.stage, s.run_id))


def apply_shard(specs: list[RunSpec], shard: str | None) -> list[RunSpec]:
    if not shard:
        return specs
    k, n = (int(x) for x in shard.split("/"))
    assert 1 <= k <= n, "--shard k/N 은 1 <= k <= N"
    ordered = sorted(specs, key=lambda s: (s.stage, s.run_id))
    return [s for i, s in enumerate(ordered) if i % n == k - 1]


def dry_run(stages: list[str], tasks: list[str], shard: str | None, only_condition: str | None,
            only_budget: str | None = None, only_seed: str | None = None) -> int:
    total = 0
    for st in stages:
        cfg = load_stage(st)
        msgs = deps.check_stage(st, tasks)
        specs = apply_shard(build_jobs([st], tasks, True, only_condition, only_budget, only_seed), shard)
        done = [s for s in specs if is_done(s)]
        print(f"{st} ({cfg['name']}): {len(specs)} runs, {len(done)} done, {len(specs) - len(done)} to run")
        if msgs:
            print("   의존성 없음:")
            for m in msgs:
                print(f"     - {m}")
        total += len(specs)
        for s in specs[:3]:
            print(f"     e.g. {s.run_id}" + ("  (선택값 대기)" if s.missing else ""))
    print(f"합계: {total} runs")
    return total


class Scheduler:
    def __init__(self, jobs: list[Job], gpus: list[G.GpuInfo]):
        self.jobs = jobs
        self.gpus = gpus
        self.ids = [g.index for g in gpus]
        base = load_base()
        self.min_micro = base["gpu"]["min_micro_batch"]
        self.cpu_threads = max(1, (os.cpu_count() or 4) // len(self.ids))
        self.t0 = time.time()

    def _log_path(self, spec: RunSpec):
        return paths.ensure(paths.logs_dir() / spec.stage) / f"{spec.run_id}.log"

    def _command(self, job: Job) -> list[str]:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(self.ids)}",
               "-m", "dakd", "train"] + job.spec.to_cli_args()
        if job.micro_batch:
            cmd += ["--micro-batch", str(job.micro_batch)]
        if job.precision != "auto":
            cmd += ["--precision", job.precision]
        return cmd

    def _run_job(self, job: Job) -> None:
        spec = job.spec
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, self.ids)),
                   OMP_NUM_THREADS=str(self.cpu_threads), MKL_NUM_THREADS=str(self.cpu_threads))
        while True:
            job.attempts += 1
            status_file(spec).unlink(missing_ok=True)
            with open(self._log_path(spec), "a") as lf:
                lf.write(f"\n===== attempt {job.attempts} gpus={self.ids} micro_batch={job.micro_batch} "
                         f"precision={job.precision} {time.strftime('%F %T')} =====\n")
                lf.flush()
                rc = subprocess.call(self._command(job), env=env, stdout=lf, stderr=subprocess.STDOUT)
            if rc == 0:
                job.status = "done"
                return
            reason = status_file(spec).read_text().strip() if status_file(spec).exists() else ""
            if reason == "oom" or rc == EXIT_OOM:
                cur = job.micro_batch or load_base()["train"]["batch_size"] // len(self.ids)
                if cur // 2 >= self.min_micro:
                    job.micro_batch = cur // 2
                    job.attempts -= 1
                    continue
            elif (reason == "nan" or rc == EXIT_NAN) and job.precision != "fp32":
                job.precision = "fp32"
                continue
            if job.attempts >= 2:
                job.status = "failed"
                self._record_failure(job, rc, reason)
                return

    def _record_failure(self, job: Job, rc: int, reason: str) -> None:
        spec = job.spec
        paths.ensure(paths.logs_dir())
        with open(paths.logs_dir() / "FAILED_RUNS.md", "a") as f:
            f.write(f"- {time.strftime('%F %T')} `{spec.stage}/{spec.run_id}` exit={rc} reason={reason or '?'} "
                    f"log: `{self._log_path(spec)}`\n")
        if spec.run_dir.exists():
            (spec.run_dir / "FAILED").write_text(f"exit={rc} {reason}")

    def _progress(self) -> str:
        c = {k: 0 for k in ("done", "running", "pending", "failed", "skipped")}
        for j in self.jobs:
            c[j.status] += 1
        return (f"[{time.strftime('%H:%M:%S')}] done {c['done']} | running {c['running']} | pending {c['pending']} | "
                f"failed {c['failed']} | skipped {c['skipped']} | elapsed {(time.time() - self.t0) / 60:.1f}min")

    def run(self) -> int:
        todo = [j for j in self.jobs if j.status == "pending"]
        todo.sort(key=lambda j: (j.spec.condition != "teacher", j.spec.stage, j.spec.run_id))
        print(f"각 run 을 GPU {self.ids} ({len(self.ids)}장, DDP)로 순차 실행합니다. 프로세스당 CPU threads={self.cpu_threads}", flush=True)
        try:
            for j in todo:
                j.status = "running"
                print(self._progress() + f"  → {j.spec.stage}/{j.spec.run_id}", flush=True)
                try:
                    self._run_job(j)
                except Exception as e:
                    j.status = "failed"
                    self._record_failure(j, -1, str(e))
                    print(f"[scheduler] {j.spec.run_id}: {e}", file=sys.stderr)
                if j.status == "failed":
                    print(f"[실패] {j.spec.run_id} (로그: {self._log_path(j.spec)})", flush=True)
        except KeyboardInterrupt:
            print("중단됨. 다시 실행하면 완료된 run 은 건너뛰고 이어서 진행합니다.", flush=True)
        print(self._progress(), flush=True)
        return sum(1 for j in self.jobs if j.status == "failed")


def run_stages(stage_arg: str, task_arg: str = "all", gpus: str | None = None, shard: str | None = None,
               only_condition: str | None = None, dry: bool = False,
               only_budget: str | None = None, only_seed: str | None = None) -> int:
    stages, tasks = parse_stage_list(stage_arg), parse_tasks(task_arg)
    if dry:
        dry_run(stages, tasks, shard, only_condition, only_budget, only_seed)
        return 0
    errs = []
    for st in stages:
        errs += deps.check_stage(st, tasks)
    if errs:
        print("실행할 수 없습니다. 먼저 필요한 의존성을 준비하세요:", file=sys.stderr)
        for m in dict.fromkeys(errs):
            print(f"  - {m}", file=sys.stderr)
        return 1
    specs = apply_shard(build_jobs(stages, tasks, False, only_condition, only_budget, only_seed), shard)
    jobs: list[Job] = []
    for s in specs:
        j = Job(s)
        if is_done(s):
            j.status = "skipped"
        elif running_elsewhere(s.run_dir):
            j.status = "skipped"
            print(f"[skip] {s.run_id}: 다른 프로세스가 실행 중")
        jobs.append(j)
    pend = [j for j in jobs if j.status == "pending"]
    print(f"stages={stages} tasks={tasks} shard={shard or '-'}: 총 {len(jobs)} runs, "
          f"{len(jobs) - len(pend)} 건너뜀, {len(pend)} 실행 예정")
    paths.ensure(paths.jobs_dir())
    tag = "_".join(stages) + ("" if task_arg == "all" else "__" + task_arg.replace(",", "+"))
    with open(paths.jobs_dir() / f"{tag}.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps({**j.spec.to_dict(), "status": j.status}) + "\n")
    if not pend:
        return 0
    try:
        sel = G.select_gpus(gpus)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print("사용 GPU:", G.describe(sel), flush=True)
    return 1 if Scheduler(jobs, sel).run() else 0
