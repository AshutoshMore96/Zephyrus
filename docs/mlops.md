# MLOps & deployment (M7)

How Zephyrus keeps its models honest in production and how to ship it.

## Model registry

`zephyrus.registry.ModelRegistry` is a dependency-free, file-based registry under
`data/registry/<name>/vNNNN.json`. Each entry (`ModelVersion`) records the metrics and
params a version was scored on. Serving can ask for `best(name)` (by primary metric) or
`latest(name)`. MLflow's registry can replace it later behind the same interface.

```python
from zephyrus.registry import ModelRegistry
reg = ModelRegistry()
reg.register("demand", {"mape": 7.4, "pinball": 0.004}, primary_metric="mape")
best = reg.best("demand")  # lowest MAPE
```

## Scheduled retrain

`scripts/retrain.py` backtests the demand forecaster (logging to MLflow when installed),
registers the new version, and checks it against the previous best for **performance
regression** — failing the run if the model has decayed. It runs weekly via
`.github/workflows/retrain.yml` (Mondays 06:00 UTC) and on demand.

```bash
python scripts/retrain.py --weeks 8 --folds 4
```

## Drift & performance monitoring

`zephyrus.monitoring` provides two guards that return a typed `DriftReport` and log an
alert when breached (the hook a real PagerDuty/Slack alerter would replace):

- `detect_feature_drift(reference, current)` — Population Stability Index (PSI); > 0.2
  signals a material distribution shift worth a retrain.
- `detect_performance_regression(live_mape, baseline_mape)` — flags decay when live error
  exceeds `tolerance × baseline`.

## Deployment (free tiers)

- **API** — containerised via the repo `Dockerfile`; `fly.toml` ships it to Fly.io
  (`fly deploy`) with scale-to-zero and a `/health` check. Set `ZEPHYRUS_API_KEY` as a
  secret to require `X-API-Key`.
- **Dashboard** — Streamlit Community Cloud or Hugging Face Spaces: point at
  `src/zephyrus/dashboard/app.py`.
- **Config** — every knob is env-driven (`ZEPHYRUS_*`); see `.env.example`.
