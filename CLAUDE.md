# CLAUDE.md

Project: **Helmet chinstrap VLM self-improvement pipeline**
Design spec: [helmet_chinstrap_vlm_design_v03_reviewed.md](helmet_chinstrap_vlm_design_v03_reviewed.md) (v0.3, authoritative — read before changing architecture)

This file captures the conventions, constraints, and mental model required to keep implementation quality high. Read it before touching code.

---

## 1. What this project is

A self-improving VLM fine-tuning loop for binary classification of helmet chinstrap state (`tight` / `loose`). The pipeline synthesizes training data from a small seed set (≤100 images), uses a Teacher VLM to screen and label generated images, and fine-tunes a lightweight student VLM with LoRA. Each loop increases diversity/difficulty.

**The v0.3 priority is to complete ONE loop end-to-end.** Do not add features for multi-loop optimization, self-screening, or hard-negative mining until Loop 1 is validated.

---

## 2. Pipeline (4 steps)

```
Step A  Image Generation        docker/step_a_imagegen/generate.py
  │     seeds/ → generated/loop_{N}/{images,meta.csv}
  ▼
Step B  Teacher Screen & Label  docker/step_b_screening/screen_and_label.py + vllm_teacher.py
  │     generated/loop_{N}/ → screened/loop_{N}/{screening.csv, labeled.jsonl}
  ▼
Step C  Dataset Build           scripts/build_dataset.py  (NOT containerized)
  │     screened/loop_{N}/ → dataset/loop_{N}/{train.jsonl, stats.json}
  ▼
Step D  Fine-tuning (LoRA)      docker/step_d_finetune/finetune.py
        dataset/loop_{N}/ → models/loop_{N}/{lora_weights/, training_log.json}

Evaluate  scripts/evaluate.py   accuracy + McNemar p-value between loops
```

Only Step A, B, D are containerized. Step C is a plain script — don't containerize it.

---

## 3. Non-negotiable design rules

### 3.1 KISS — do not expand scope
- **Do not implement Self-Screening, Multi-Model Voting, Rationale ablation, or Hard Negative Mining.** They are explicitly deferred to v0.4+ (design §10).
- **Do not add configurability for future use cases.** Add a knob only when the current step needs it.
- **Teacher screening is single-model.** Do not reintroduce the 3-model voting from v0.2.
- **One fixed question prompt.** See design §5. Do not introduce per-sample prompt variation.

### 3.2 Schema is load-bearing
The fixed JSON I/O schema between steps is the contract:

| File | Required columns/keys |
|---|---|
| `seeds/labels.csv` | `image_id, label` (label ∈ {`tight`, `loose`}) |
| `generated/loop_{N}/meta.csv` | `image_id, seed_id, seed_label, prompt, loop_num` |
| `screened/loop_{N}/screening.csv` | `image_id, seed_label, pred_teacher, keep` |
| `screened/loop_{N}/labeled.jsonl` | `{image_id, label[, rationale]}` (keep=True only) |
| `dataset/loop_{N}/train.jsonl` | `{image_path, question, answer:{label[, rationale]}}` |
| `models/loop_{N}/lora_weights/` | LoRA adapter files |
| `models/loop_{N}/training_log.json` | `{base_model, lora_config, epochs, batch_size, learning_rate, num_samples, loss_history, status}` |

**If you change any of these, update its test fixture and the downstream step's reader in the same commit.**

### 3.3 Labels are a closed set
`label ∈ {"tight", "loose"}` everywhere. Enforce with `enum` in JSON schemas and with explicit `assert`/`ValueError` at boundaries. Never silently coerce unknown labels.

### 3.4 Structured output is enforced, not parsed defensively
Teacher VLM output is constrained by vLLM's `StructuredOutputsParams` with a JSON schema (see [docker/step_b_screening/vllm_teacher.py](docker/step_b_screening/vllm_teacher.py)). **Do not add elaborate retry/recovery logic for malformed JSON** — if the schema fails, log and skip. The design assumes guided decoding makes parse errors effectively non-existent.

---

## 4. Coding conventions (observed in existing code — keep doing these)

### 4.1 Deferred imports for heavy/GPU-only dependencies
Torch, diffusers, vllm, peft, transformers imports are **inside the function that uses them**, never at module top. This lets `pytest` run on machines without GPU/ML stacks. Example: [docker/step_a_imagegen/generate.py:137](docker/step_a_imagegen/generate.py#L137), [docker/step_b_screening/screen_and_label.py:128](docker/step_b_screening/screen_and_label.py#L128).

**When you add a new GPU-dependent code path, defer its imports the same way.**

### 4.2 Every step has a local-testable mode
- Step A: `api.provider: "dummy"` copies seed images (no GPU)
- Step B: `--provider precomputed` reads teacher responses from a JSONL fixture
- Step D: `--dry-run` simulates a loss curve and writes a dummy adapter config

Pattern: real backend + fake backend, selected by config/flag. Add a fake mode whenever you add a real one.

### 4.3 Test-first container workflow (design §7)
Per the design doc: *"write a script that mimics the container I/O, test it locally, THEN write the Dockerfile."* The reason is that the local dev machine can't run VLMs.

- All Python logic lives in `.py` files that can be imported and tested directly.
- The Dockerfile is a thin wrapper — `ENTRYPOINT ["python3.11", "<script>.py"]`.
- Tests import the module and call its functions directly — they do NOT spin up containers.

### 4.4 Pure functions + thin `main()`
Business logic is in importable functions (`run_step_a`, `screen_and_label`, `build_dataset`, `run_finetuning`, `evaluate`). `main()` only parses args, loads config, calls the function, and prints a one-line summary. **Don't inline logic into `main()`.**

### 4.5 Config lives in `config/step_*.yaml`
Never hardcode hyperparameters in the script. Thread them through the function signature with sensible defaults so tests can override.

### 4.6 Prompts are constants at the top of the file
See `TEACHER_PROMPT` / `TEACHER_SYSTEM_PROMPT` / `QUESTION_WITH_RATIONALE`. If a prompt changes, it changes in exactly one place. Never build prompts from scratch inline inside loops.

### 4.7 Paths are `pathlib.Path`, strings at I/O boundaries only
All internal APIs take `Path`. Only `argparse` and final file writes convert to `str`.

---

## 5. Testing rules

### 5.1 Run tests with `uv`
```
uv run pytest                     # all tests
uv run pytest tests/test_screening.py -v
uv run pytest -k mcnemar
```

### 5.2 Every step must have unit tests covering:
- Output schema (column names, JSON keys)
- Label value constraints (`tight` / `loose` only)
- Required-field / invalid-input errors raise
- Happy-path numerical correctness (counts, keep_rate, accuracy)

See [tests/test_screening.py](tests/test_screening.py) as the template. Fixtures live in `tests/fixtures/step_{a,b,c,d}/`.

### 5.3 System test is the integration gate
[tests/test_system.py](tests/test_system.py) runs A→B→C→D→Evaluate end-to-end in dummy/dry-run mode. **It must stay green after every change.** If a change breaks it, fix the change, not the test (unless the schema intentionally evolved).

### 5.4 Don't mock what you can fake
Prefer a `dummy`/`precomputed`/`dry_run` provider with real I/O over `unittest.mock`. Real file I/O through tmp_path catches more bugs.

---

## 6. Commands cheatsheet

```bash
# Python: always uv, never bare python/python3 (except inside Docker)
uv run pytest
uv run python scripts/build_dataset.py --loop 1
uv run python scripts/evaluate.py --predictions eval/loop_1.jsonl

# Docker (GPU required for real runs)
docker compose build step_a step_b step_d
LOOP_NUM=1 docker compose run --rm step_a
LOOP_NUM=1 docker compose run --rm step_b
# Step C runs on host (no container)
uv run python scripts/build_dataset.py --loop 1
LOOP_NUM=1 docker compose run --rm step_d
```

---

## 7. Evaluation contract (design §4)

| Dataset | Size | Purpose | Never used for |
|---|---|---|---|
| `eval/test_real/` | ≥50 real images, hand-labeled | **Primary metric.** Must be real-environment photos. | Training. Ever. |
| `eval/val_generated/` | 10–20 per loop, hand-labeled | Validation of generated data quality | Training |
| `eval/edge_cases/` | ≥20 hand-labeled | Half-fastened, chin tucked, backlit, etc. | Training |

**Loop termination** is decided by `test_real` accuracy + McNemar p-value between adjacent loops (see [scripts/evaluate.py](scripts/evaluate.py)). Success: `test_real` ≥85% AND McNemar p<0.05 vs previous loop. Stop also on: 2 consecutive loops <+2% improvement, Step B keep_rate <50%, or Loop 5 reached.

The current McNemar implementation uses a coarse chi² lookup table (`_chi2_p_value_approx`). If precision matters, swap to `scipy.stats.chi2.sf(chi2, 1)` — but add scipy as a dependency only when that precision is actually needed.

---

## 8. Things NOT to do

- Do NOT add scipy, pandas, numpy, or torch to the project-level `pyproject.toml`. Heavy deps belong in the Docker `requirements.txt` of the step that needs them. The project venv stays minimal so `uv run pytest` is fast.
- Do NOT create new top-level directories without updating the design doc's §8 directory structure.
- Do NOT rename `docker/step_*` directories or their Python entry points — the `docker-compose.yml` and Dockerfiles depend on them.
- Do NOT add `--no-verify` to git commits or `--force` to pushes.
- Do NOT mix Step C into a container. It's deliberately host-side per v0.3.
- Do NOT introduce a second question prompt variant beyond rationale/label-only. The design calls for a single fixed question.
- Do NOT write README-style markdown files, changelogs, or summary docs unless the user asks. The design doc is the single source of truth.
- Do NOT use `python` / `python3` directly. Always `uv run python ...`. (Docker images are exempt — they use `python3.11` inside the container.)

---

## 9. When modifying code

1. **Read the design doc section for the step you're touching.** v0.3 explicitly simplified several things (no 3-model voting, no Step F); don't reintroduce them.
2. **Read the existing module and its tests** before editing. Match the style.
3. **Update tests in the same change.** If you break a test, either the code is wrong or the contract changed — in the latter case, update the fixture + the downstream reader + the schema table in this file.
4. **Run `uv run pytest` before declaring done.** The system test must pass.
5. **Keep commits focused.** Don't bundle a bug fix with a refactor.

---

## 10. Open risks (from design §9) — keep in mind

- **Label drift in generation:** Flux2. Klein may change chinstrap state despite "keep as-is" prompt. Step B catches this via Teacher disagreement → `keep=False`. Monitor `keep_rate` per loop.
- **Teacher self-error:** Teacher VLM is not ground truth. Measure Teacher accuracy on `eval/test_real` to know the noise ceiling on your training labels.
- **Synthetic→real domain gap:** `test_real` MUST be real photos, not generated ones, or evaluation is invalid.
- **Overfitting to synthetic distribution:** The only real metric that matters is `test_real` accuracy.
