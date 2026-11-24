from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import paths

OK_FILE = "WEIGHTS_OK"
META_FILE = "weights_meta.json"
MODEL_FILE = "model.safetensors"


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def prepare(wdir: Path) -> None:
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / OK_FILE).unlink(missing_ok=True)


def save(model, tokenizer, wdir: Path, meta: dict) -> None:
    wdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(wdir, safe_serialization=True)
    tokenizer.save_pretrained(wdir)


def finalize(wdir: Path, meta: dict) -> None:
    meta = dict(meta, sha256=sha256_file(wdir / MODEL_FILE), size_bytes=(wdir / MODEL_FILE).stat().st_size,
                saved=time.strftime("%F %T"))
    (wdir / META_FILE).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (wdir / OK_FILE).write_text("ok")


def check(wdir: Path, verify_sha: bool = False) -> tuple[bool, str]:
    wdir = Path(wdir)
    if not wdir.exists() or not any(f.name != ".gitkeep" for f in wdir.iterdir()):
        return False, "폴더 없음"
    if not (wdir / MODEL_FILE).exists():
        return False, f"{MODEL_FILE} 없음"
    if not (wdir / OK_FILE).exists() or not (wdir / META_FILE).exists():
        return False, f"{OK_FILE}/{META_FILE} 없음 (불완전한 다운로드/학습 중단?)"
    if verify_sha:
        meta = json.loads((wdir / META_FILE).read_text())
        if sha256_file(wdir / MODEL_FILE) != meta.get("sha256"):
            return False, "sha256 불일치 (파일 손상 또는 다운로드 미완료)"
    return True, "ok"


def require(wdir: Path, what: str = "") -> dict:
    wdir = Path(wdir)
    ok, why = check(wdir)
    if not ok:
        raise FileNotFoundError(
            f"{what} 가중치를 찾을 수 없습니다: {wdir}  ({why})\n"
            f"  → 직접 학습했다면 해당 학습 단계가 끝났는지, 팀원이 학습했다면 Google Drive 에서 받은 폴더를\n"
            f"    {paths.weights_root()} 아래 같은 구조(teachers/{{task}}/, runs/{{stage}}/{{run_id}}/)로 넣으세요.\n"
            f"    확인: `bash scripts/check_weights.sh`")
    return json.loads((wdir / META_FILE).read_text())
