# LLM Training Pipeline

This repository is the active training and serving pipeline for local data preparation, Modal GPU training, and Docker-based inference.

The intended execution model is:

`Local validation + local preprocessing + Modal GPU training + Docker inference`

That split keeps CPU-heavy and data-heavy work on your machine, uses Modal only where GPU time matters, and exports a stable model bundle for serving from Docker.

## Operating Model

### Local CPU stages

Run these locally before spending on GPU:

- configuration validation
- preprocessing and tokenization
- smoke tests
- cost estimation

### Remote GPU stage

Use Modal only for training:

- processed artifacts are uploaded after local preparation
- training runs on the selected Modal GPU
- checkpoints and best-model export are managed remotely

### Serving stage

Use Docker for inference:

- the inference container reads from `models/current/`
- the serving contract is stable and deployment-focused
- Kubernetes is not part of the active local deployment path

## Canonical Flow

1. Validate the config and environment locally.
2. Preprocess the dataset into `data/processed/`.
3. Estimate cost and confirm the GPU plan.
4. Train on Modal GPU.
5. Export the best model bundle into:
   - `models/trained_models/<dataset>/`
   - `models/current/`
6. Start the Docker inference server.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to serve locally:

```bash
pip install -r inference_requirements.txt
```

### 2. Configure environment

Create a `.env` file with the credentials you need:

```env
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
HF_TOKEN=...
```

### 3. Validate before training

```bash
python scripts/validate_pipeline.py --config configs/config.yaml
```

This runs a small local validation pass so you can catch broken configs before using Modal credits.

### 4. Preprocess locally

```bash
python scripts/preprocess_only.py --config configs/config.yaml
```

### 5. Estimate cost

```bash
python main.py --config configs/config.yaml --estimate-cost
```

### 6. Train on Modal

```bash
python scripts/train_with_modal.py --config configs/config.yaml --single-dataset roneneldan/TinyStories
```

For the full orchestrated path:

```bash
python main.py --config configs/config.yaml --yes
```

### 7. Serve with Docker

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

The inference service exposes port `8000` by default.

## Common Workflows

### Full pipeline

```bash
python main.py --config configs/config.yaml --yes
```

### Preprocess only

```bash
python main.py --config configs/config.yaml --preprocess-only
```

### Local smoke training

```bash
python main.py --config configs/config.yaml --local
```

### Override training steps and GPU

```bash
python main.py --config configs/config.yaml --steps 5000 --gpu A10G --yes
```

### Skip validation when you already verified the config

```bash
python main.py --config configs/config.yaml --skip-validation --yes
```

### Monitor training artifacts

```bash
python scripts/monitor.py
```

## Model Profiles

Prebuilt profile configs are available under `configs/models/`:

- `configs/models/small_85m.yaml`
- `configs/models/medium_200m.yaml`
- `configs/models/large_350m.yaml`
- `configs/models/xlarge_500m.yaml`

These profiles are normalized into the canonical runtime config by the orchestration layer.

## Artifact Contract

The inference path is built around `models/current/`.

Expected files:

- `models/current/best_model.pt`
- `models/current/tokenizer.json`
- `models/current/manifest.json`

The per-dataset export path remains:

- `models/trained_models/<dataset>/`

## Reliability and Cost Controls

The pipeline is designed to reduce wasted runs and wasted spend:

- validation happens locally before remote training
- preprocessing reuse depends on matching saved state and metadata
- cost estimation runs before Modal submission
- the best-model export is validation-driven
- checkpoint/resume flow is designed to preserve the best checkpoint
- Docker serving reads from one stable model bundle instead of ad hoc paths

## Key Entry Points

| Command | Purpose |
| --- | --- |
| `python main.py --config configs/config.yaml --yes` | Full pipeline |
| `python main.py --config configs/config.yaml --local` | Local training smoke run |
| `python main.py --config configs/config.yaml --estimate-cost` | Cost estimation only |
| `python scripts/preprocess_only.py --config configs/config.yaml` | Local preprocessing |
| `python scripts/validate_pipeline.py --config configs/config.yaml` | Pre-training validation |
| `python scripts/train_with_modal.py --config configs/config.yaml --single-dataset roneneldan/TinyStories` | Direct Modal training wrapper |
| `python scripts/train_cli.py` | Interactive CLI launcher |
| `python scripts/train_interactive.py` | Alternate interactive launcher |
| `python scripts/monitor.py` | Training status and artifact monitoring |

## Repository Layout

```text
configs/                  Runtime configs and model profiles
data/cache/               Hugging Face cache
data/processed/           Local preprocessed artifacts
docker/                   Docker inference runtime
logs/                     Local logs
models/checkpoints/       Local checkpoints
models/trained_models/    Per-dataset exported bundles
models/current/           Stable inference contract
scripts/                  CLI wrappers and validation helpers
src/core/                 Models, trainer, datasets, tokenization
src/pipeline/             Preprocessing, checkpointing, Modal training
src/orchestration/        Validation, budgeting, artifact export
src/inference/            Inference loading and Modal/Docker serving
tests/                    Test suite
validation_results/       Saved validation reports
```

## Practical Notes

- Start with `configs/initial_training.yaml` if you want a smaller first run.
- Use local validation and preprocessing first if you are changing configs.
- Prefer `A10G` as the balanced default unless your model size requires more memory.
- Treat `models/current/` as the deployment target and `models/trained_models/` as the archive.
- The active runtime path is this repository: `D:\LLM\llm_training_pipeline`.
