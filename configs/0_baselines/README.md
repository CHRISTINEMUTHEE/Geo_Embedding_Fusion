# Per-source baselines (legacy full-data random split, subset smoke runs, and
# full-data DataModule runs). Copy one of these and change only the knob under test.

- `01_alphaearth_lightunet.yaml` / `02_tessera_lightunet.yaml` — full-data LightUNet,
  legacy random split (`datasets.py`). Kept for reproducibility; prefer 09/10.
- `03`–`08` `*_subset_datamodule.yaml` — region-grouped split on `data/subset/`
  (laptop smoke: AlphaEarth/Tessera LightUNet, THOR/TerraMind S1/S2 EfficientEncoderDecoder).
- `09`–`14` `*_datamodule.yaml` — same models and sources as 03–08, but `data_root: data`
  is the full training split from `scripts/acquire.py` (~2024 tiles, `data/manifest.csv`).
