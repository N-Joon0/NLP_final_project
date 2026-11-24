.PHONY: setup data models teacher logits student_ft gate analyze status test
setup:      ; bash scripts/00_setup_env.sh
data:       ; bash scripts/01_prepare_data.sh
models:     ; bash scripts/02_download_models.sh
teacher:    ; bash scripts/03_train_teacher.sh
logits:     ; bash scripts/04_cache_teacher_logits.sh
student_ft: ; bash scripts/05_train_student_ft.sh
gate:       ; bash scripts/06_gate_check.sh
analyze:    ; bash scripts/99_analyze.sh
status:     ; bash scripts/status.sh
test:       ; python -m pytest -q
