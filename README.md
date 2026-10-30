# Rasp

Closed-model generator for the 404-GEN subnet (SN17) competition, round 1.
Takes a reference photo, writes a single-file three.js scene, checks it through the
witness renders and returns the JavaScript source.

## Layout

- `main.py` — entry point: `generate(image: bytes, seed: int, witness, renderer) -> str | None`.
- `sn17gen/` — the pipeline:
  - `pipeline.py` — generation → render → refine chains, candidate panel, final selection;
  - `prompts.py` — system/user prompts, colour and material rules;
  - `views.py` — render view sets (default grid, judge front views);
  - `orient.py` — photo-aware tilt for flat objects (silhouette comparison on renders only);
  - `jsutil.py` — source helpers (normalization wrapper, tilt, size guards);
  - `budget.py` — call and credit accounting against the per-prompt limits;
  - `config.py` — replay-safe knobs (all committed, see below).
- `miner_reference/` — local batch runner and render checkpoints used for dry runs; not used by `main.py`.
- `tests/` — flag behaviour tests (`pytest tests -q`).

## Knobs (`sn17gen/config.py`)

| Flag | Default | Effect |
|---|---|---|
| `KEEP_GEN` | on | keep each chain's pre-refine module as a candidate in the final panel |
| `PHOTO_TILT` | on | tilt flat objects toward the camera by comparing the judge front view with the photo |
| `KEEP_POSE` | on | refinement keeps the module's pose unless renders contradict the photo |
| `JUDGE_REFINE` | on | refinement inspects the judge's four front views and lists discrepancies first |
| `COLOUR_RULES` | on | numeric colour/material rules in the generation prompt |
| `ASTRA_GEN` | off | first generation of each chain on the high-reasoning model |

## Limits

Per prompt the pipeline stays within 60 witness calls and 5,000,000 credits
(measured: 7 calls, ≈0.7 M credits, final source ≤ 30 KiB). Every decision depends only on
the returned bytes, so replay reproduces the same selection.

## Run

```
python -m pytest tests -q
```

`generate` is called by the subnet runtime with a witness and a renderer; see `main.py`.
