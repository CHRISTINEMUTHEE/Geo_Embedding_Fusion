# Channel-fusion experiments

Pixel-aligned sources only (AlphaEarth, Tessera) -- concatenated along the channel
dimension by `datamodule.FusedTileDataset` into one tile, before cropping. No
upsampling needed since both sources are already 256x256, same grid as the label.
Patch-token sources (THOR/TerraMind) are not supported here yet -- see
`emb2heights/README.md`'s `FusedTileDataset` entry for why that's a separate,
harder problem (resolution mismatch needs an explicit upsample/crop strategy).

- `01_alphaearth_tessera_fusion_subset.yaml` -- AlphaEarth (64ch) + Tessera (128ch)
  -> 192ch, `model_name: lightunet` (not a dedicated fusion model -- fused channels are
  just a wider input to the exact same architecture the single-source baselines use,
  so the comparison isolates "does fusion help" instead of also changing model
  capacity). Region-grouped smoke run on `data/subset/`. Compare against
  `configs/0_baselines/old_yamls/03_alphaearth_subset_datamodule.yaml` /
  `04_tessera_subset_datamodule.yaml` (same recipe, model, and data_root -- single
  source each).

Config keys unique to fusion: `fusion_sources: [source, ...]` (>=2 pixel-aligned
source names to concatenate) drives the actual data loading; `embedding_source` stays
a plain descriptive label for logging/CSV rows (e.g. `"alphaearth_tessera_fusion"`),
not something the DataModule reads when `fusion_sources` is set.
