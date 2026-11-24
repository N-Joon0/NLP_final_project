from __future__ import annotations

import argparse
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dakd", description="Difficulty-Aware KD 실험 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare-data", help="데이터 다운로드, 분할 생성(없을 때)/검증(있을 때)")
    s.add_argument("--task", default="all")
    s.add_argument("--force-regenerate", action="store_true")

    s = sub.add_parser("download-models", help="사전학습 체크포인트·tokenizer 캐시")
    s.add_argument("--task", default="all")

    s = sub.add_parser("train", help="단일 run 학습 (+ test 추론)")
    s.add_argument("--stage", default="manual")
    s.add_argument("--task", required=True)
    s.add_argument("--condition", required=True,
                   choices=["teacher", "student_ft", "std_kd", "conf_kd", "ours", "ours_norm"])
    s.add_argument("--budget", default="full", help="2000|5000|10000|full")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--T", type=float)
    s.add_argument("--lam", type=float)
    s.add_argument("--gamma", type=float)
    s.add_argument("--beta", type=float)
    s.add_argument("--weight-norm", default="none", choices=["none", "batch_mean"])
    s.add_argument("--run-id")
    s.add_argument("--test-inference", type=int, default=1)
    s.add_argument("--no-infer", action="store_true", help="학습 후 test 추론 생략")
    s.add_argument("--micro-batch", type=int, help="per-device batch (유효 batch 는 grad accumulation 으로 유지)")
    s.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    s.add_argument("--max-steps", type=int, help="디버그용 step 제한")

    s = sub.add_parser("infer", help="weights/ 의 가중치로 dev/test 추론 (단일 GPU)")
    s.add_argument("--weights-dir", help="weights/ 아래 run 폴더 (메타에서 run 정보를 읽음)")
    s.add_argument("--stage"), s.add_argument("--run-id")
    s.add_argument("--splits", default="test")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--gpu", type=int, default=None)

    s = sub.add_parser("infer-stage", help="stage 중 weights/ 에 가중치가 있는 run 만 test 추론 (없는 run 은 목록 보고)")
    s.add_argument("--stage", required=True)
    s.add_argument("--task", default="all")
    s.add_argument("--splits", default="test")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--gpu", type=int, default=None)

    s = sub.add_parser("check-weights", help="weights/ 에 어떤 run 의 가중치가 있는지 점검")
    s.add_argument("--stage", default="all")
    s.add_argument("--task", default="all")
    s.add_argument("--verify-sha", action="store_true", help="sha256 까지 검증 (느림)")

    s = sub.add_parser("verify-outputs", help="프로포절 기준 산출물(학습/추론/효율/분석) 형식·누락 점검")
    s.add_argument("--stage", default="all")
    s.add_argument("--task", default="all")
    s.add_argument("--no-weights", action="store_true", help="weights/ 검사 생략 (가중치 없이 결과만 받아 합친 경우)")

    s = sub.add_parser("cache-teacher-logits", help="teacher logits 저장 (weights/teachers/{task} 사용, 단일 GPU)")
    s.add_argument("--task", default="all")
    s.add_argument("--gpu", type=int, default=None)
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("run-stage", help="Stage 의 모든 run 생성·병렬 실행")
    s.add_argument("--stage", required=True, help="예: s05  /  s05,s06,s08  /  all")
    s.add_argument("--task", default="all")
    s.add_argument("--gpus", default=None,
                   help="사용할 GPU 번호 (예: 0,1,2,3). 모든 run 이 이 GPU 전체로 DDP 학습. 생략 시 $DAKD_GPUS 또는 base.yaml 기본값")
    s.add_argument("--shard", help="k/N: 팀원 간 job 분배")
    s.add_argument("--only-condition")
    s.add_argument("--only-budget", help="예: 2000 또는 2000,5000 (gate 용 run 만 먼저 돌릴 때)")
    s.add_argument("--only-seed", help="예: 42 또는 13,42")
    s.add_argument("--only-task", help="--task 의 별칭")
    s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("select", help="튜닝 Stage 결과에서 최적값 선택 → configs/selected/")
    s.add_argument("--stage", required=True)
    s.add_argument("--task", default="all")

    sub.add_parser("gate", help="teacher–student 성능 차이 점검")

    s = sub.add_parser("efficiency", help="크기·속도·메모리 측정")
    s.add_argument("--task", default="all")
    s.add_argument("--gpu", type=int, default=0)

    sub.add_parser("analyze", help="수집·통계·표·그림·리포트")

    s = sub.add_parser("status", help="Stage 별 진행 현황")
    s.add_argument("--stage", default="all")
    s.add_argument("--task", default="all")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(argv)
    if getattr(args, "gpu", None) is not None and args.cmd in ("infer", "infer-stage", "cache-teacher-logits"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from . import paths
    from .config import parse_tasks

    c = args.cmd
    if c == "prepare-data":
        from . import data
        for t in parse_tasks(args.task):
            r = data.prepare_splits(t, force=args.force_regenerate)
            sp = data.load_splits(t)
            print(f"[{t}] 분할 {r}: " + ", ".join(f"{k}={len(sp[k])}" for k in ("student_2k", "student_5k", "student_10k", "dev", "test")))
        return 0
    if c == "download-models":
        from . import data
        from .models import resolve_checkpoint
        for t in parse_tasks(args.task):
            ck = resolve_checkpoint(t)
            data.get_tokenizer(t, ck)
            from transformers import AutoModelForSequenceClassification
            AutoModelForSequenceClassification.from_pretrained(ck, num_labels=2)
            print(f"[{t}] 캐시 완료: {ck}")
        return 0
    if c == "train":
        from .registry import RunSpec
        from .train import main_train
        budget = None if args.budget == "full" else int(args.budget)
        spec = RunSpec(stage=args.stage, task=args.task, condition=args.condition, budget=budget, seed=args.seed,
                       T=args.T, lam=args.lam, gamma=args.gamma, beta=args.beta, weight_norm=args.weight_norm,
                       test_inference=bool(args.test_inference))
        if spec.is_kd and None in (spec.T, spec.lam, spec.gamma, spec.beta):
            print("KD 조건에는 --T --lam --gamma --beta 가 모두 필요합니다.", file=sys.stderr)
            return 2
        if args.run_id and args.run_id != spec.run_id:
            print(f"[경고] --run-id({args.run_id}) 가 규칙({spec.run_id}) 과 다릅니다.", file=sys.stderr)
        return main_train(spec, micro_batch=args.micro_batch, precision=args.precision,
                          no_infer=args.no_infer, max_steps=args.max_steps)
    if c == "infer":
        from pathlib import Path
        from . import weights as W
        from .infer import infer_run, spec_from_meta
        from .registry import RunSpec
        if args.weights_dir:
            spec = spec_from_meta(W.require(Path(args.weights_dir), "지정한"))
        elif args.stage and args.run_id:
            wd = paths.weights_root() / ("teachers" if "__teacher__" in args.run_id else f"runs/{args.stage}")
            wd = wd / (args.run_id.split("__")[0] if "__teacher__" in args.run_id else args.run_id)
            spec = spec_from_meta(W.require(wd, args.run_id))
        else:
            print("--weights-dir 또는 --stage/--run-id 가 필요합니다.", file=sys.stderr)
            return 2
        infer_run(spec, args.splits.split(","), overwrite=args.overwrite)
        return 0
    if c == "infer-stage":
        from .infer import infer_stage
        return infer_stage(args.stage, args.task, args.splits.split(","), args.overwrite)
    if c == "check-weights":
        from .weights_report import check_weights
        return check_weights(args.stage, args.task, args.verify_sha)
    if c == "verify-outputs":
        from .verify import run_verify
        return run_verify(args.stage, args.task, not args.no_weights)
    if c == "cache-teacher-logits":
        from .teacher_logits import cache_teacher_logits
        for t in parse_tasks(args.task):
            cache_teacher_logits(t, force=args.force)
        return 0
    if c == "run-stage":
        from .scheduler import run_stages
        return run_stages(args.stage, args.only_task or args.task, args.gpus, args.shard,
                          args.only_condition, args.dry_run,
                          args.only_budget, args.only_seed)
    if c == "select":
        from .select import select_stage
        from .config import parse_stage_list
        for st in parse_stage_list(args.stage):
            select_stage(st, args.task)
        return 0
    if c == "gate":
        from .gate import run_gate
        return run_gate()
    if c == "efficiency":
        from .efficiency import run_efficiency
        run_efficiency(args.task, args.gpu)
        return 0
    if c == "analyze":
        from .analysis.report import run_analysis
        run_analysis()
        return 0
    if c == "status":
        from .status import show_status
        show_status(args.stage, args.task)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
