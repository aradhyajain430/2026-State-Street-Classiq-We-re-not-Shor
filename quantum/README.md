# Quantum VaR (IQAE)

Run the Gaussian state prep + IQAE VaR estimation end to end:
```
uv run python -m quantum.iqae_var
```

Classical-only sanity check (no Classiq synthesis/execution):
```
uv run python -m quantum.iqae_var --no-quantum
```

Notes:
- The IQAE run requires network access to the Classiq service.

Scaling vs num_shots (log-log error with fit line):
```
uv run python -m quantum.iqae_shots_scaling
```
Defaults use 12 points spaced by a factor of 1.4 for smoother scaling curves.

Tune IQAE epsilon/confidence_alpha at fixed shots (default 64):
```
uv run python -m quantum.iqae_eps_alpha_tuning
```
