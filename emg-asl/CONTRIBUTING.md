# Contributing to MAIA EMG-ASL

## Branching

```
main          → production (auto-deploys to Railway on push)
feature/*     → new features and model improvements
fix/*         → bug fixes
experiment/*  → exploratory work, not meant to merge
```

Never push directly to `main` without testing. All pushes to `main` trigger a Railway redeploy.

## Commit Style

Keep commits small and focused. Use this prefix convention:

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature or model architecture |
| `fix:` | Bug fix |
| `train:` | Training script or config changes |
| `data:` | Data pipeline, scripts, or protocol changes |
| `docs:` | Documentation only |
| `chore:` | Dependency updates, tooling, CI |
| `refactor:` | Code restructuring without behavior change |

Examples:
```
feat: add Conformer cross-modal pre-training stage
fix: correct ONNX input shape in /predict endpoint (was 80-dim, must be (1,40,8))
train: increase Conformer batch size to 1024 for IYA A100
docs: flesh out data collection protocol and IRB guidance
```

## Code Style

- Python: follow PEP 8, 4-space indent, 100-char line limit
- TypeScript: strict mode, 2-space indent
- No emojis in code or comments
- Type hints required for all function signatures in `src/`
- No print statements in `src/` — use `structlog` logger

## Running Tests

```bash
# Full test suite (21 tests)
pytest tests/ -v

# Single file
pytest tests/test_pipeline.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

All tests must pass before merging to `main`.

## Adding a New Model Architecture

1. Create `src/models/your_model.py` with a class that:
   - Accepts `input_shape=(40, 8)` and `n_classes=26`
   - Has a `forward(x: Tensor) -> Tensor` method (input `(B, 40, 8)`, output `(B, 26)`)
   - Exports cleanly to ONNX opset 17: `torch.onnx.export(model, dummy, "out.onnx", opset_version=17)`

2. Add a SLURM training script in `scripts/slurm/`

3. Add benchmark results to `docs/model_card.md`

4. Add a test in `tests/test_pipeline.py` covering forward pass + ONNX export

## Model Artifacts

Never commit `.pt` checkpoint files (excluded in `.gitignore`). Use Cloudflare R2:

```bash
python scripts/upload_artifact.py --file models/your_model.onnx --set-latest
```

The 8.2KB baseline ONNX (`models/asl_emg_classifier.onnx`) is committed directly because it's small and needed by Railway.

## Secrets

Never commit:
- `.env` files with real API keys
- `r2_credentials.sh`
- Any file containing Railway tokens, R2 secrets, or API keys

Use Railway's Variables tab for production secrets.

## Questions

Open an issue on GitHub or ping Caleb Newton directly.
