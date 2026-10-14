# Data layout

Create `selfmf/data/train/` and `selfmf/data/eval/` locally, or set `SELFMF_DATA_ROOT` and `SELFMF_EVAL_DATA_ROOT` to the prepared input folders. The `.gitignore` excludes data/checkpoint files.

- `ixi` adapter: MAT files with an `Img2` field, matching the preprocessing used by the supplied adapter.
- `siat` adapter: paired 256 × 256 complex k-space `T1` and `T2` arrays per MAT file.
- Frequency weight: MAT file with a `weight` array.
- Mask files: MAT files with a `mask` array.
- Reconstruction: input MAT file with `T1_img` (auxiliary) and `T2_img` (target) unless `SELFMF_REFERENCE_KEY` and `SELFMF_TARGET_KEY` are specified.

All inputs must follow the normalization, alignment, coil-channel layout and Fourier conventions of the original acquisition/preprocessing.
