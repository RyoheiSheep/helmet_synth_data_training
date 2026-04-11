# Implementation Plan — Helmet Chinstrap VLM Pipeline

Living document. Update as tasks move between sections.
Authoritative spec: [helmet_chinstrap_vlm_design_v03_reviewed.md](helmet_chinstrap_vlm_design_v03_reviewed.md)
Coding rules: [CLAUDE.md](CLAUDE.md)

**Status legend:** ✅ done · 🟡 partial · ⬜ not started · ⏸️ deferred (v0.4+)

**Overall state (2026-04-12):** Phase 0 complete. Phase 1.B (real LoRA fine-tuning) and 1.C (predict.py) code implemented — 72 tests pass. Seed data (33 images) and eval data (test_real: 8, edge_cases: 4) acquired. GPU tests skipped by default (`addopts = "-m 'not gpu'"`). **Next milestone: GPU smoke tests on RunPod (1.D/1.E), then Loop 1 execution (1.F).**

---

## Phase 0 — Skeleton & local-testable pipeline ✅

Every step is importable, has a dummy/dry-run mode, and is covered by unit tests. This phase is the "v0.3 1-loop completion" prerequisite.

| # | Task | Status | Files | Notes |
|---|---|---|---|---|
| 0.1 | Repo structure per design §8 | ✅ | `seeds/`, `generated/`, `screened/`, `dataset/`, `models/`, `eval/`, `config/`, `docker/`, `scripts/`, `tests/` | `eval/` subdirs exist but empty |
| 0.2 | `pyproject.toml` + uv lock (minimal deps) | ✅ | [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Only `pyyaml` + `pytest`; heavy deps isolated to Docker |
| 0.3 | `.env.example` | ✅ | [.env.example](.env.example) | Has `HF_TOKEN`, `TEACHER_MODEL_PATH`, `LOOP_NUM` |
| 0.4 | `docker-compose.yml` (3 services) | ✅ | [docker-compose.yml](docker-compose.yml) | step_a, step_b, step_d wired |
| 0.5 | Step A: pure prompt generator + dummy backend | ✅ | [docker/step_a_imagegen/generate.py](docker/step_a_imagegen/generate.py), [config/step_a.yaml](config/step_a.yaml) | `api.provider: "dummy"` copies seeds; diffusers path deferred-imported |
| 0.6 | Step A: Dockerfile + requirements | ✅ | [docker/step_a_imagegen/Dockerfile](docker/step_a_imagegen/Dockerfile) | CUDA 12.4 base |
| 0.7 | Step B: screen_and_label + precomputed backend | ✅ | [docker/step_b_screening/screen_and_label.py](docker/step_b_screening/screen_and_label.py) | `--provider precomputed` reads fixture JSONL |
| 0.8 | Step B: vLLM Teacher wrapper (deferred import) | ✅ | [docker/step_b_screening/vllm_teacher.py](docker/step_b_screening/vllm_teacher.py) | `StructuredOutputsParams` with JSON schema |
| 0.9 | Step B: Dockerfile + requirements | ✅ | [docker/step_b_screening/Dockerfile](docker/step_b_screening/Dockerfile) | CUDA 12.4 base, installs vllm |
| 0.10 | Step B config | ✅ | [config/step_b.yaml](config/step_b.yaml) | Qwen3.5-27B default |
| 0.11 | Step C: build_dataset (no container) | ✅ | [scripts/build_dataset.py](scripts/build_dataset.py) | Host-side per v0.3 |
| 0.12 | Step D: finetune scaffold + dry-run | ✅ | [docker/step_d_finetune/finetune.py](docker/step_d_finetune/finetune.py), [config/step_d.yaml](config/step_d.yaml) | Dry-run writes fake adapter_config.json + loss curve |
| 0.13 | Evaluate: accuracy + McNemar | ✅ | [scripts/evaluate.py](scripts/evaluate.py) | McNemar uses coarse chi² lookup |
| 0.14 | Unit tests — all 5 modules | ✅ | `tests/test_{imagegen,screening,build_dataset,finetune,evaluate}.py` | |
| 0.15 | System test — full A→B→C→D→Eval | ✅ | [tests/test_system.py](tests/test_system.py) | Multi-loop dummy run + McNemar on same test set |
| 0.16 | Test fixtures per step | ✅ | `tests/fixtures/step_{a,b,c,d}/` | |

**Baseline verification:** `uv run pytest` → 72 tests collected, all passing (as of 2026-04-11).

---

## GPU Environment

**Platform:** RunPod Community Cloud — **1× A40 48 GB** on-demand ($0.35/hr).

Chosen as the cheapest money × time option for our small dataset (~330 generated images, ~200 training samples):

| Step | Model | VRAM estimate | Fits A40? |
|---|---|---|---|
| A — Image Gen | FLUX.2 Klein 9B (FP16) | ~20 GB | Yes |
| B — Teacher | Qwen3.5-27B (**FP8 quantized**) | ~28 GB | Yes |
| D — Fine-tune | Qwen3.5-8B + LoRA (FP16) | ~22 GB | Yes |
| Predict | Qwen3.5-8B + LoRA (FP16) | ~18 GB | Yes |

Step B requires FP8 quantization — the 27B model at FP16 (~54 GB) exceeds 48 GB. vLLM supports FP8 natively via `--quantization fp8` or `--dtype float8`. Update `config/step_b.yaml` with `dtype: float8` when running on GPU.

**Estimated Loop 1 cost:** ~$1–2 (2–4 hours active compute including setup).

**Alternatives considered:**
- A100 80 GB ($1.39/hr): fits everything at full precision, but 4× the price. Not worth it for our dataset size.
- RTX 4090 24 GB ($0.59/hr): too tight for FLUX 9B + fine-tuning overheads; risk of OOM.

---

## Phase 1 — Loop 1 on real data 🟡

Goal: complete ONE real loop end-to-end with actual seeds, real Teacher VLM, real fine-tuning, and `test_real` evaluation. This is the v0.3 success criterion.

### 1.A Data acquisition 🟡

| # | Task | Status | Files / Path | Definition of done |
|---|---|---|---|---|
| 1.A.1 | Collect ≤100 seed images (tight & loose) | ✅ | `seeds/images/*.png` (33 images: 15 tight, 18 loose) | Balanced tight/loose count; diverse enough that Flux edits preserve labels |
| 1.A.2 | Hand-label seeds | ✅ | `seeds/labels.csv` | Each image_id mapped to `tight`/`loose` |
| 1.A.3 | Collect ≥50 real test images | 🟡 | `eval/test_real/images/` (8 images), `eval/test_real/labels.csv` | Real-environment photos, NOT from seeds; hand-labeled. Currently 8, target ≥50. Enough for smoke test. |
| 1.A.4 | Collect ≥20 edge-case images | 🟡 | `eval/edge_cases/images/` (4 images), `eval/edge_cases/labels.csv` | Half-fastened, chin tucked, backlit, etc. Currently 4, target ≥20. Enough for smoke test. |
| 1.A.5 | Decide on val_generated strategy | ⬜ | `eval/val_generated/` | Design says 10–20/loop with hand labels. Decide: annotate a sample of Step A output per loop? |

### 1.B Step D — real LoRA fine-tuning ✅ (code) / ⬜ (GPU verification)

Real training loop implemented. GPU smoke test still needed.

| # | Task | Status | Files | Definition of done |
|---|---|---|---|---|
| 1.B.1 | Step D `Dockerfile` + `requirements.txt` | ✅ | [docker/step_d_finetune/Dockerfile](docker/step_d_finetune/Dockerfile), [docker/step_d_finetune/requirements.txt](docker/step_d_finetune/requirements.txt) | CUDA 12.4 base; pinned torch/transformers/peft/accelerate/datasets |
| 1.B.2 | Implement Qwen3.5-8B LoRA training loop | ✅ | [docker/step_d_finetune/finetune.py](docker/step_d_finetune/finetune.py) | `_run_live_training()` loads model, attaches LoRA via peft, trains, saves adapter. All heavy imports deferred per CLAUDE.md §4.1. `format_answer()` added for stable JSON serialization. |
| 1.B.3 | Dataset collator for image+JSON answer | ✅ | Part of 1.B.2 | `_build_collator()` converts samples to model inputs with prompt-masked labels (answer-only loss) |
| 1.B.4 | Smoke test on 1 GPU with tiny subset | ⬜ | — | Run on ~20 samples, 1 epoch. Verify loss decreases and adapter_config.json is real peft format. |
| 1.B.5 | Add integration test with `--dry-run=False` marked `@pytest.mark.gpu` | ✅ | [tests/test_finetune.py](tests/test_finetune.py) | `TestRunFinetuningGPU` class; `gpu` marker registered in pyproject.toml; skipped by default |

### 1.C Inference script for evaluation ✅

| # | Task | Status | Files | Definition of done |
|---|---|---|---|---|
| 1.C.1 | New `scripts/predict.py` (host side, no container) | ✅ | [scripts/predict.py](scripts/predict.py) | `run_prediction()` with `dummy` and `vllm` providers. vLLM path loads base model + LoRA adapter via `LoRARequest`, uses structured output. Deferred imports for vllm. |
| 1.C.2 | Fake backend for predict.py | ✅ | Same file | `predict_dummy()` with configurable accuracy + seed; mirrors Step A/B/D pattern |
| 1.C.3 | Unit tests for `predict.py` | ✅ | [tests/test_predict.py](tests/test_predict.py) | 10 tests: schema, label constraints, dummy accuracy, error paths. Fixtures in `tests/fixtures/eval/` |
| 1.C.4 | Wire `predict.py` into system test | ✅ | [tests/test_system.py](tests/test_system.py) | `TestPredictIntegration` class runs A→B→C→D→Predict(dummy)→Evaluate |

### 1.D Step A — Flux Kontext verification ⬜

| # | Task | Status | Files | Definition of done |
|---|---|---|---|---|
| 1.D.1 | Smoke test `api.provider: "diffusers"` on GPU | ⬜ | — | Generate 10 images from 2 seeds, inspect by eye: chinstrap state preserved, background/clothing changed |
| 1.D.2 | Tune `num_inference_steps`, `guidance_scale` if needed | ⬜ | [config/step_a.yaml](config/step_a.yaml) | Record chosen values; trade-off: speed vs fidelity |
| 1.D.3 | Measure label-preservation rate on a small sample | ⬜ | — | Manual check: of 20 generated images, how many kept the chinstrap state? Target ≥80% (basic assumption 2 in design §1) |

### 1.E Step B — Teacher VLM verification ⬜

| # | Task | Status | Files | Definition of done |
|---|---|---|---|---|
| 1.E.1 | Verify Teacher model repo ID resolves | ⬜ | [config/step_b.yaml](config/step_b.yaml) | Teacher fixed to `Qwen/Qwen3.5-27B` (v0.3 decision). Confirm HF repo ID downloads and loads under vLLM on the target GPU host. |
| 1.E.2 | Verify `vllm.LLM.chat` + `StructuredOutputsParams` API versions | ⬜ | [docker/step_b_screening/vllm_teacher.py](docker/step_b_screening/vllm_teacher.py) | vLLM API surface for multimodal + structured output shifts between versions. Pin vllm version in `requirements.txt`. |
| 1.E.3 | Smoke test Teacher on ~20 generated images | ⬜ | — | Run real vLLM path, confirm all outputs conform to JSON schema, no parse failures |
| 1.E.4 | Baseline Teacher accuracy on `test_real` | ⬜ | — | Run Teacher on eval/test_real. Records the **noise ceiling** for our training labels (CLAUDE.md §10) |

### 1.F Loop 1 orchestration & execution ⬜

| # | Task | Status | Files | Definition of done |
|---|---|---|---|---|
| 1.F.1 | Run Step A (Loop 1) on real seeds | ⬜ | `generated/loop_1/` | ≥ `variations_per_seed × num_seeds` images + meta.csv |
| 1.F.2 | Run Step B (Loop 1) with real Teacher | ⬜ | `screened/loop_1/` | screening.csv + labeled.jsonl; **record keep_rate** (must be >50% to proceed) |
| 1.F.3 | Run Step C (Loop 1) | ⬜ | `dataset/loop_1/train.jsonl` | Matches schema in CLAUDE.md §3.2 |
| 1.F.4 | Run Step D (Loop 1) real fine-tune | ⬜ | `models/loop_1/lora_weights/` | Real adapter; final loss < starting loss |
| 1.F.5 | Run predict.py on `eval/test_real` | ⬜ | `eval_out/loop_1/test_real.jsonl` | Predictions JSONL |
| 1.F.6 | Run evaluate.py → record baseline accuracy | ⬜ | `eval_out/loop_1/metrics.json` | Accuracy on test_real. This is the Loop 1 number to beat. |

**Loop 1 is "done" when:** 1.F.6 produces a real accuracy number on real test data, and `keep_rate` (Step B) + `loss_history` (Step D) are all recorded.

---

## Phase 2 — Multi-loop self-improvement ⬜

Run only after Phase 1 is complete and the 1-loop hypothesis is validated.

| # | Task | Status | Files | Definition of done |
|---|---|---|---|---|
| 2.1 | Loop 2 config (angle_deg=15, medium diversity) | ⬜ | [config/step_a.yaml](config/step_a.yaml) variant | Per design §2 loop strategy table |
| 2.2 | Loop 2 end-to-end run | ⬜ | `{generated,screened,dataset,models}/loop_2/` | All artifacts present |
| 2.3 | McNemar: Loop 2 vs Loop 1 | ⬜ | `eval_out/loop_2/metrics.json` | p-value recorded; check significance |
| 2.4 | Loop 3 (angle_deg=30, high diversity) — IF Loop 2 improved | ⬜ | — | Conditional on 2.3 |
| 2.5 | Termination-gate script | ⬜ | `scripts/check_loop_gate.py` | Implements design §4 termination conditions (success, stagnation, quality degradation, upper limit). Returns exit code so a shell/Make driver can branch. |
| 2.6 | Replace coarse McNemar p-value lookup | ⬜ | [scripts/evaluate.py](scripts/evaluate.py#L119) | Swap `_chi2_p_value_approx` for `scipy.stats.chi2.sf(chi2, 1)`. Add scipy only when needed — NOT to project root pyproject.toml; host-side eval script can import locally via an `eval` extras group, or put it in a requirements file. |

---

## Phase 3 — Hardening & observability ⬜

| # | Task | Status | Files | Definition of done |
|---|---|---|---|---|
| 3.1 | Run-level logging (stdout is fine) with loop_num + step | ⬜ | All entry points | Each step prints structured `[stepX loop=N]` lines |
| 3.2 | Per-loop `summary.json` aggregator | ⬜ | `scripts/summarize_loop.py` | Combines keep_rate, training_log, metrics into one file per loop |
| 3.3 | Dockerfile for step_d | ✅ | [docker/step_d_finetune/Dockerfile](docker/step_d_finetune/Dockerfile) | Done in 1.B.1 |
| 3.4 | Smoke-test all three containers build | ⬜ | — | `docker compose build` succeeds on the target GPU host |
| 3.5 | Pin vllm/torch/transformers versions | ⬜ | All three `requirements.txt` | Reproducibility |
| 3.6 | Pre-commit hook: `uv run pytest` | ⬜ | `.pre-commit-config.yaml` (or simpler shell hook) | Catches regressions before commit |

---

## Phase 4 — Deferred (v0.4+) ⏸️

Per design §10 and [CLAUDE.md §3.1](CLAUDE.md) these stay out of v0.3. **Do not implement unless Loop 1/2/3 results justify it.**

| # | Task | Trigger to reconsider |
|---|---|---|
| 4.1 | Self-Screening (old Step F) — Fine-Tuned VLM re-validates training data | Loop 2+ with stable keep_rate |
| 4.2 | Multi-model voting (Teacher + FT VLM) | Teacher keep_rate < 70% |
| 4.3 | Rationale ablation study | After Loop 1 complete |
| 4.4 | Hard-negative mining | Loop 3+ |
| 4.5 | Quantized LoRA (QLoRA) for cheaper training | If full-precision LoRA is too slow on available GPU |
| 4.6 | Switch from Qwen3.5-8B to moondream2 / PaliGemma2 for edge deployment | Once accuracy target is met |

---

## Definition of "complete codebase"

The codebase is considered complete when **all of Phase 1 and Phase 2 tasks are ✅**, AND:

- [ ] `uv run pytest` passes (including any new tests added in 1.B.5, 1.C.3)
- [ ] `docker compose build` succeeds for all 3 services
- [ ] Loops 1 and 2 have real artifacts in `generated/`, `screened/`, `dataset/`, `models/`, `eval_out/`
- [ ] `eval_out/loop_2/metrics.json` shows `test_real` accuracy + McNemar p-value vs Loop 1
- [ ] A single command (e.g. `scripts/run_loop.sh N` or `make loop-N`) can re-run any loop from scratch
- [ ] This PLAN.md is updated with final numbers in a new "Results" section

Phase 3 items are quality-of-life — nice to have but not blockers for "complete."
Phase 4 items are explicitly out of scope for v0.3.

---

## Changelog

- **2026-04-09** — PLAN.md created. Phase 0 marked complete (55 tests passing, all dummy/dry-run backends working). Phase 1 broken into A–F sub-phases. Identified missing Step D Dockerfile/requirements and missing predict.py as the two biggest code gaps.
- **2026-04-11** — Phase 1.B (code) and 1.C completed. Step D: real LoRA training loop (`_run_live_training`, `_build_collator`, `format_answer`), Dockerfile + pinned requirements. predict.py: dummy + vLLM providers, 10 unit tests, wired into system test. 72 tests passing. Remaining blockers: data acquisition (1.A) and GPU smoke tests (1.B.4, 1.D, 1.E).
- **2026-04-12** — Seed data acquired (33 images: 15 tight, 18 loose). Eval data acquired (test_real: 8, edge_cases: 4) — below design targets but sufficient for smoke tests. Added `addopts = "-m 'not gpu'"` to pyproject.toml so GPU tests are skipped by default. Added eval directory scaffolding (images/, labels.csv). GPU environment decided: **RunPod 1× A40 48 GB** ($0.35/hr) — cheapest option that fits all steps (Step B uses FP8 quantization for 27B Teacher).
