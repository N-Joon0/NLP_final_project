# Difficulty-Aware KD

CSE710 (NLP/NLU) Team 1 최종 프로젝트. SST-2(영어)와 KLUE-YNAT(한국어)에서 **표준 KD / Confidence KD / Confidence+Disagreement KD(Ours)** 를 비교한다.

- 이 README 는 환경 설정, 실험 설계, 실행·검증 방법을 다룬다.

## 0. 필요한 환경

| 항목 | 요구사항 |
|---|---|
| GPU | **RTX A6000(48GB) 4장** 이 한 번의 실험 단위다. 모든 학습 run 은 4장 DDP 로 통일하며 GPU 수가 4장이 아니면 에러로 멈춘다. 8장 서버는 `0,1,2,3` / `4,5,6,7` 두 스트림을 동시에 쓸 수 있다. |
| 드라이버 | 검증 환경: 드라이버 580.105.08 (CUDA 13.0) + torch 2.14.1+cu130. 드라이버가 580 미만이면 [문제 해결](#7-문제-해결)의 cu124 방법을 쓴다. |
| 디스크 | 저장소가 있는 파티션에 **여유 100GB 이상** 권장 (.venv 6GB, HF 캐시 1GB, 가중치 약 30GB, logits·로그·결과 수 GB). 모든 산출물은 저장소 안(`workspace/`, `weights/`)에 저장된다. |
| OS / Python | Ubuntu 22.04, Python 3.10 이상 (검증: 3.10.12) |
| 도구 | `git`, `python3` + `venv`, `tmux`, `curl` (없으면 `scripts/bootstrap.sh` 가 설치. 설치에는 root 또는 sudo 가 필요) |
| 네트워크 | HuggingFace(모델·데이터), GitHub, PyPI, pytorch.org 접근 |

규칙(바꾸지 말 것)
- seed 는 랜덤이 아닌 고정값: 본 실험 `[13, 42, 87]`, 튜닝·teacher `42`, 분할 `0` (`configs/base.yaml`).
- `data/splits/*.json` 은 절대 재생성·수정하지 않는다. `01_prepare_data.sh` 는 `verified` 만 나와야 하고, `created` 가 나오거나 "기존 분할 JSON 과 재계산 결과가 다릅니다" 에러가 나면 즉시 중단하고 팀에 알린다.
- 데이터·모델 캐시·가중치·결과는 **저장소 안에만** 저장된다. 위치를 바꾸는 환경변수는 없다. `workspace/`(로그·예측·지표·캐시) 와 `weights/`(가중치) 는 git 에 올라가지 않는다.
- 토큰, SSH 개인키, `HF_TOKEN` 같은 비밀값은 출력·커밋하지 않는다.

## 1. 서버 설정

서버에서 바로 쓰면 아래 명령만으로 된다. Docker 컨테이너를 쓰는 경우는 [컨테이너 사용 시(선택)](#컨테이너-사용-시-선택)를 먼저 본다.

```bash
git clone git@github.com:N-Joon0/NLP_final_project_team1.git      # 저장소 접근 권한이 있는 GitHub 계정(SSH 키)으로
cd NLP_final_project_team1
git log -1 --oneline            # 모든 서버가 같은 커밋을 써야 한다 (팀 채팅의 커밋과 비교)

bash scripts/bootstrap.sh       # 도구 확인(없으면 설치), GPU·공유메모리·디스크 점검, .venv 생성과 패키지 설치 (약 8분)
```
`[경고]` 가 나오면 [문제 해결](#7-문제-해결)을 먼저 본다. 환경 설치만 다시 하려면 `bash scripts/00_setup_env.sh`.
`requirements.lock` 은 Python 3.12 기준이라 **3.10 에서는 설치에 실패하는 것이 정상**이고, 스크립트가 자동으로 `requirements.txt` 로 다시 설치한다 (`ERROR: ... accelerate==1.16.0` 메시지는 무시).

환경 점검 (각각 통과해야 다음으로):
```bash
source .venv/bin/activate        # 새 터미널을 열 때마다 (scripts/*.sh 는 자동으로 활성화)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True 여야 함
bash scripts/check_nccl.sh       # 4개 rank 모두 10.0 이면 PASS
bash scripts/01_prepare_data.sh  # "[sst2] 분할 verified ..." "[ynat] 분할 verified ..." (created 가 나오면 중단!)
bash scripts/02_download_models.sh
python -m pytest -q              # 30 passed
```
이후 실험 실행은 3절.

### 컨테이너 사용 시 (선택)
컨테이너를 만들 때 아래 세 가지만 지키면 된다.
- `--gpus all` : GPU 사용
- `--shm-size=32g` : 공유메모리가 작으면 NCCL/DataLoader 가 멈춘다
- 저장소 경로에 **컨테이너를 지워도 남는 호스트 디스크를 마운트** (모든 결과가 저장소 안에 저장되므로)
```bash
docker run -it --gpus all --shm-size=32g -v <호스트 디스크 경로>/NLP:/root/project --name dakd ubuntu:22.04 bash
apt-get update && apt-get install -y git openssh-client ca-certificates     # clone 에 필요한 최소 패키지
```
이후는 위 "서버 설정"과 같다. private 저장소 clone 을 위한 SSH 키는 컨테이너 안에서 `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519` 로 만들고 공개키(`~/.ssh/id_ed25519.pub`)만 저장소 관리자에게 전달해 등록한다 (개인키는 공유 금지).

## 2. 실험 요약

- **질문**: RQ1 난이도 인지 KD(`ours`)가 표준 KD 보다 좋은가? RQ2 student 데이터가 적을 때(2k/5k/10k) 효과가 달라지는가? RQ3 영어·한국어에서 경향이 일관적인가?
- **데이터**: SST-2(`nyu-mll/glue:sst2`, 2클래스, accuracy), KLUE-YNAT(`klue/klue:ynat`, 7클래스, macro-F1). 공식 test 는 쓰지 않고 공식 validation 을 층화 50/50 으로 `dev`(선택용)/`test`(최종 평가 전용) 로 나눴다 (split_seed 0). student 부분집합은 2k ⊂ 5k ⊂ 10k (층화·중첩), teacher 는 train 전체.
- **모델**: teacher = `google-bert/bert-base-uncased`(SST-2, 12L) / `klue/bert-base`(YNAT, 12L). student = 같은 체크포인트의 하위 6개 encoder layer(약 67M/68M).
- **손실**
  ```
  KD_i = T² · KL(softmax(z_t/T) ‖ softmax(z_s/T))
  c_i = max softmax(z_t),  d_i = ½ Σ|q_t − q_s|,  w_i = c_i^γ (1 + β d_i)   (stop-gradient)
  L = mean_i [ CE_i + λ · w̃_i · KD_i ]       w̃ = w  또는  w / (mean_batch(w)+1e-8)
  ```
  | condition | λ | γ | β | norm |
  |---|---|---|---|---|
  | teacher | – | – | – | – |
  | student_ft | 0 | – | – | – |
  | std_kd | 튜닝 | 0 | 0 | none |
  | conf_kd | std_kd 값 | 튜닝 | 0 | none |
  | ours | std_kd 값 | 튜닝 | 튜닝 | none |
  | ours_norm | std_kd 값 | ours 값 | ours 값 | batch_mean |
- **하이퍼파라미터**: AdamW(wd 0.01), lr 2e-5, linear warmup 10%, 유효 batch 32, max_len 128(dynamic padding), grad clip 1.0, bf16(미지원 시 fp16→fp32). Teacher 3 epoch / Student 5 epoch, 매 epoch dev 평가로 best 체크포인트 선택. **test 는 어떤 선택에도 쓰지 않는다.**
- **학습 118회 / test 추론 92회**

  | stage | 내용 | run 수 | 스크립트 |
  |---|---|---|---|
  | s01 | Teacher FT | 2 | `03` |
  | – | teacher logits 저장 | – | `04` |
  | s03 | Student FT (2k/5k/10k × seed 3 × 2 task) | 18 | `05` |
  | – | Gate (teacher 와 student_ft@2k 의 dev 차이) | – | `06` |
  | s04 | Std KD 튜닝 (T∈{2,4} × λ∈{0.5,1}) | 8 | `07` |
  | s05 | Std KD 본 실험 | 18 | `08` |
  | s06 | Conf KD 튜닝 (γ) | 6 | `09` |
  | s07 | Conf KD 본 실험 | 18 | `10` |
  | s08 | Ours 튜닝 (γ, β) | 12 | `11` |
  | s09 | Ours 본 실험 | 18 | `12` |
  | s10 | Ours 정규화 ablation | 18 | `13` |
  | – | 효율 측정 / 분석 | – | `14` / `99` |

  Gate: 두 task 중 하나라도 teacher 와 `student_ft@2k(seed 42)` 의 dev 주 지표 차이가 1.0%p 미만이면 `n_layers=4` 로 바뀌고(종료코드 2), s03 을 다시 실행한다.
- **분석**: paired bootstrap(B=10,000, seed 0), teacher 정답/오답·confidence 분위별 정확도, inheritance rate, ECE, 가중치 진단, 효율(파라미터·크기·latency·throughput·peak memory). 효과가 없거나 나빠도 그대로 보고한다.

## 3. 전체 실행 순서 (참고: 한 서버에서 처음부터 끝까지)

한 서버에서 전부 돌릴 때의 순서다. 장시간 실행은 `tmux` 안에서 한다 (`tmux new -s exp`, 빠져나오기 `Ctrl+b d`, 다시 붙기 `tmux attach -t exp`). 중간에 끊겨도 같은 명령을 다시 실행하면 **완료된 run 은 건너뛴다**. 모든 명령은 저장소 루트에서, `--gpus` 는 항상 4장.

```bash
bash scripts/status.sh                                  # 118 run 이 pending 인지

bash scripts/03_train_teacher.sh --gpus 0,1,2,3         # s01: teacher 2개
bash scripts/04_cache_teacher_logits.sh --gpu 0         # teacher logits (단일 GPU)
bash scripts/05_train_student_ft.sh --gpus 0,1,2,3      # s03: student FT 18개
bash scripts/06_gate_check.sh                           # 종료코드 2 면 n_layers=4 로 바뀜 → 05 → 06 재실행

bash scripts/07_tune_std_kd.sh --gpus 0,1,2,3           # s04 튜닝 + select → configs/selected/*_std_kd.json
bash scripts/08_train_std_kd.sh --gpus 0,1,2,3          # s05
bash scripts/09_tune_conf_kd.sh --gpus 0,1,2,3          # s06 + select
bash scripts/10_train_conf_kd.sh --gpus 0,1,2,3         # s07
bash scripts/11_tune_ours.sh --gpus 0,1,2,3             # s08 + select
bash scripts/12_train_ours.sh --gpus 0,1,2,3            # s09
bash scripts/13_train_ours_norm.sh --gpus 0,1,2,3       # s10

bash scripts/14_measure_efficiency.sh --gpu 0           # 다른 run 이 없는 빈 GPU 에서
bash scripts/99_analyze.sh                              # workspace/results/RESULTS.md, tables/, figures/
bash scripts/verify_outputs.sh                          # 산출물 검증 (world_size==4 도 검사)
```
- **`bash scripts/run_all.sh`** 은 위 과정을 의존성 순서대로 **자동으로 끝까지** 실행한다: 데이터·모델 준비 → GPU 세트별 NCCL 점검 → teacher·logits·gate용 student FT → gate(종료코드 2 면 gate용 s03 을 다시 돌리고 재확인) → s03~s10(튜닝·select 포함) → 효율 측정 → 분석·검증. GPU 4장 세트 2개(기본 `0,1,2,3` / `4,5,6,7`)로 sst2 와 ynat 를 **동시에** 진행하고, 세트를 하나만 주면(`--sets 0,1,2,3`) 차례로 진행한다. 이미 끝난 run·logits 는 건너뛰므로 끊겨도 같은 명령으로 이어 간다. 로그는 `workspace/logs/run_all.log`.
- 공통 옵션(`--gpus`/`--shard`/`--dry-run` 등)은 `03`, `05`, `07`~`13` 에 사용할 수 있다: `--shard k/N`(job 분배), `--dry-run`(실행 없이 목록·의존성만), `--only-task sst2|ynat`, `--only-budget 2000`, `--only-seed 42`. 직접 호출은 `python -m dakd --help`.
- 튜닝 stage(07/09/11)는 한 사람이 끝까지 실행한다 (select 는 튜닝 run 이 모두 한 workspace 에 있어야 한다). 선택값 `configs/selected/*.json` 은 담당자(A)가 커밋해 두고 나머지는 `git pull` 로 받는다.
- 8장 서버는 `--gpus 0,1,2,3` 과 `--gpus 4,5,6,7` 을 **서로 다른 stage 또는 `--only-task`/`--shard`** 로 동시에 돌릴 수 있다. 같은 run 은 `RUNNING` 잠금으로 중복 실행되지 않는다.
- 실패한 run 은 1회 재시도하고 그래도 실패하면 `workspace/logs/FAILED_RUNS.md` 에 기록하고 계속한다. 로그는 `workspace/logs/<stage>/<run_id>.log`. OOM 이면 rank 당 micro-batch 를 절반으로(최소 4), NaN 이면 fp32 로 1회 재실행한다.

시간 측정: teacher 1개(sst2)를 먼저 돌려 `workspace/teachers/sst2/train_metrics.json` 의 `epochs[*].epoch_time_s`(epoch 별 시간), `train_time_s`, `peak_mem_mb`(GPU 당 메모리), `train_samples_per_s` 로 전체 118회 소요를 추정한다.

### 재현하기 (팀원용: clone → Drive 꾸러미 → 실행)

팀원이 결과를 재현해 보고 싶을 때의 순서다. 튜닝 값(`configs/selected/*.json`, `gate.json`, `base.yaml`)은 **git clone 에 들어 있고**, 가중치·결과는 Google Drive 공유 폴더(`dakd_shared/`)의 두 꾸러미로 받는다. 브라우저로 폴더를 내려받아 그 경로를 스크립트에 넘기면 압축 해제와 제 위치 배치는 자동이다.

| 꾸러미 | 내용 |
|---|---|
| `dakd_shared/teacher/` | teacher 가중치(압축) + teacher logits + teacher 결과 |
| `dakd_shared/results/` | student 가중치(s03·s05·s07·s09·s10, stage 별 압축) + 전체 run 결과 + 분석 결과 + 튜닝 값 사본 |

```bash
git clone git@github.com:N-Joon0/NLP_final_project_team1.git && cd NLP_final_project_team1

# (a) 학습부터 재현: teacher 와 공개된 튜닝 값을 그대로 쓰고 s03~s10 을 다시 학습 (수 시간)
bash scripts/reproduce.sh --teacher ~/Downloads/teacher                       # GPU 4장 (0,1,2,3)
bash scripts/reproduce.sh --teacher ~/Downloads/teacher --sets 0,1,2,3:4,5,6,7   # 8장이면 두 세트로 동시에

# (b) 결과 재현: 받은 결과로 분석·검증만 다시 실행 (학습 없음, 빠름)
bash scripts/reproduce.sh --results ~/Downloads/results
```
`reproduce.sh` 가 하는 일: 환경 설정(`bootstrap.sh`) → 데이터 분할 `verified` 확인과 모델 다운로드 → 꾸러미를 `weights/`·`workspace/` 에 배치(sha256 검증, 이미 있는 파일은 덮어쓰지 않음) → (a) `check_nccl.sh` 후 `run_all.sh --skip-tuning` / (b) 분석·검증.
`--skip-tuning` 은 teacher·gate·튜닝을 건너뛰고 clone 에 들어 있는 튜닝 값으로 학습한다 (다시 튜닝하면 값이 달라질 수 있어서). 필요한 튜닝 값이나 `gate.json` 이 없으면 시작 전에 알려 준다.
꾸러미를 만드는 쪽(실험을 돌린 서버)은 실험이 끝난 뒤 `bash scripts/share_results.sh` 로 `workspace/share/teacher/`, `workspace/share/results/` 를 만들고 각각 Drive 의 `dakd_shared/teacher/`, `dakd_shared/results/` 에 폴더째 올린다 (`weights/` 안은 압축 파일, `results/` 안은 그대로).

## 4. 산출물과 검증

`bash scripts/verify_outputs.sh [--stage s01,s03] [--task all] [--no-weights]` 가 다음 항목이 빠짐없이 형식에 맞게 만들어졌는지 검사하고 문제가 있으면 `[FAIL]` 과 종료코드 1 을 낸다. (미완료 run 은 실패가 아니라 "미완료"로 집계)

| 구분 | 산출물 |
|---|---|
| 학습 | epoch 별 train loss/CE/KD, dev 지표(accuracy, macro-F1, per-class F1, confusion matrix, ECE, NLL, Brier), best epoch, 학습 시간, step 수, 처리량, peak memory, `config.json`(seed, GPU, 정밀도, git commit, 버전) |
| 학습(KD) | epoch 별 w 평균/표준편차/최소/최대, c·d 평균, teacher 정답/오답별 평균 가중치, 오답 수, 예제별 `w,c,d,kd,ce` (`train_example_stats.npz`) |
| 추론 | test 지표 전체 + `test_preds.npy`(확률) + `test_logits.npy` + 사용한 가중치 sha256. **튜닝 run(s04/s06/s08)에는 test 추론이 없어야 한다** |
| Teacher logits | `train/dev/test.npy`, `train_correct.npy`, `meta.json`(teacher sha256) |
| 효율 | 파라미터 수, 저장 크기, latency(batch 1), throughput(batch 64), peak GPU 메모리, 하드웨어·정밀도·길이·워밍업 (`results/efficiency.csv`, task × teacher/student × fp32/bf16 = 8행) |
| 분석 | main_results, delta_vs_std_kd, significance(paired bootstrap), teacher_error_analysis(+inheritance), confidence_bins, calibration(ECE), weight_diagnostics, tuning_results, efficiency (표 9개 `results/tables/*.csv`), 그림 7개 `results/figures/*.png`, `results/RESULTS.md` |

가중치 용량: 학습 1 run 당 student 약 0.26GB, teacher 약 0.42GB, 전체 118 run 약 30GB. 튜닝 run(s04/s06/s08)의 가중치는 select 이후 필요 없으므로 `bash scripts/prune_checkpoints.sh`(목록만) → `--yes`(삭제)로 지울 수 있다.

## 5. 디렉토리와 스크립트

- `configs/` 설정 (`base.yaml`, `tasks/`, `stages/`, **`selected/`** = 튜닝 결과) · `data/splits/` 분할 JSON · `src/dakd/` 구현 · `scripts/` 진입점 · `tests/` 단위 테스트
- `workspace/` (git 제외): `hf_cache/`, `teachers/`, `teacher_logits/`, `runs/{stage}/{run_id}/`, `logs/`, `jobs/`, `results/`, `share/`
- `weights/` (내용은 git 제외, 폴더 구조만 `.gitkeep`): `teachers/{task}/`, `runs/{stage}/{run_id}/` — 각 폴더에 `model.safetensors`, `weights_meta.json`, `WEIGHTS_OK`

| 스크립트 | 내용 |
|---|---|
| `bootstrap.sh` | 서버 초기 설정 (도구 확인·설치, GPU/공유메모리/디스크 점검, `00` 실행) |
| `00_setup_env.sh` | `.venv` 생성, 패키지 설치, GPU/bf16 확인 |
| `check_nccl.sh` | GPU 4장 NCCL all_reduce 점검 |
| `01_prepare_data.sh` / `02_download_models.sh` | 데이터 다운로드와 분할 검증 / 사전학습 모델·tokenizer 캐시 |
| `03`~`13` | 위 표의 stage 별 학습·튜닝 |
| `14_measure_efficiency.sh` / `99_analyze.sh` | 효율 측정(빈 GPU 에서) / 표·그림·리포트 |
| `share_results.sh` | 검증 후 공유용 꾸러미 생성: `workspace/share/teacher/`, `workspace/share/results/` (가중치는 압축) |
| `receive_shared.sh` | 내려받은 꾸러미 폴더를 `weights/`·`workspace/` 의 제 위치에 넣고 가중치 점검 |
| `reproduce.sh` | 팀원 재현: 환경 설정 → 데이터·모델 → 꾸러미 배치 → 학습 재현 또는 분석 재현 |
| `15_infer_from_weights.sh` | `weights/` 에 있는 run 만 test 추론 |
| `status.sh` / `check_weights.sh` / `verify_outputs.sh` | 진행 현황 / 가중치 점검 / 산출물 검증 |
| `prune_checkpoints.sh` | 튜닝 run 가중치 정리 |
| `run_all.sh` | 전체를 의존성 순서대로 자동 실행 (GPU 세트 2개로 task 별 동시 진행, 재현용 `--skip-tuning`) |

## 6. 테스트
```bash
python -m pytest -q   # 단위 테스트 (splits, models, losses, registry, DDP gradient 동등성). GPU 불필요
```

## 7. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `bootstrap.sh` 가 GPU 를 못 본다 | 컨테이너면 `--gpus all` 로 만들었는지 호스트에서 확인 (`nvidia-smi` 가 안에서 동작해야 함). 서버면 드라이버 설치 상태 확인 |
| `bootstrap.sh` 가 "root 도 sudo 도 없습니다" | 서버 관리자에게 출력된 패키지(git, tmux, curl, python3-venv 등) 설치를 요청 |
| `torch.cuda.is_available()` 이 False | 드라이버가 CUDA 13 을 지원하지 않는 경우. `rm -rf .venv && TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 DAKD_IGNORE_LOCK=1 bash scripts/00_setup_env.sh` (드라이버에 맞는 cu 버전으로). 바꿨다면 팀에 알린다. |
| `00_setup_env.sh` 에 `ERROR: ... accelerate==1.16.0` | Python 3.10/3.11 에서 `requirements.lock` 이 안 맞는 것이며 정상. 자동으로 `requirements.txt` 로 재설치된다. |
| `check_nccl.sh` 가 멈추거나 FAIL | `df -h /dev/shm` 이 충분한지 확인 (컨테이너는 `--shm-size=32g`). `NCCL_SHM_DISABLE=1` 은 임시 우회일 뿐이니 원인을 해결하고 팀에 알린다. |
| `ProcessGroupNCCL ... Guessing device ID` 경고 | 모든 로그에 나오는 경고. rank→GPU 매핑이 균일하면 무해하다. |
| "GPU 2장(...)이 지정되었습니다 ... 4장으로 통일" | 의도된 가드. `--gpus 0,1,2,3` 처럼 정확히 4장을 지정한다. |
| 디스크 부족 | `df -h .` 로 저장소 파티션 확인. 튜닝 가중치는 `prune_checkpoints.sh --yes`. |
| HF 다운로드·추론이 느림 / "unauthenticated requests" 경고 | rate limit 일 수 있다. 셸에서만 `export HF_TOKEN=...` (파일에 쓰거나 커밋 금지). |
| run 이 중간에 죽었다 | 같은 명령을 다시 실행하면 완료된 run 은 건너뛴다. `bash scripts/status.sh` 로 현황, `workspace/logs/<stage>/<run_id>.log` 로 원인 확인. |
| `receive_shared.sh` 의 "sha256 불일치" | 다운로드가 깨진 것이다. 해당 파일을 다시 내려받는다 |
| 새 터미널에서 `python -m dakd` 가 안 된다 | `source .venv/bin/activate` (scripts/*.sh 는 자동 활성화) |

## 8. 검증된 환경 (참고)

| 항목 | 값 |
|---|---|
| GPU / 드라이버 | RTX A6000 x8 (48GB), 드라이버 580.105.08, CUDA 13.0 |
| Python | 3.10.12 |
| torch | 2.14.1+cu130 (NCCL 2.30.7) |
| transformers / datasets | 5.20.2 / 5.2.0 |
| 실행 환경 | Docker (Ubuntu 22.04), `--gpus all --shm-size=32g` |
| 단위 테스트 | `python -m pytest -q` → 30 passed |

`requirements.txt` 는 버전을 고정하지 않아 설치 시점의 최신이 설치된다. 서로 다른 서버의 결과를 합치기 전에 `config.json` 의 `torch`, `transformers` 버전이 서로 같은지 확인한다.
