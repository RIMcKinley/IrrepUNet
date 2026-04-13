# irrepunet

A resolution-adaptive, rotation-equivariant 3D UNet for medical image
segmentation. Convolution kernels are built in physical units (millimetres)
using [e3nn](https://github.com/e3nn/e3nn) spherical harmonics, so a single
trained model can be run at arbitrary voxel spacings.

Two training backends are provided:

| Backend | Script        | Environment needs        |
|---------|---------------|--------------------------|
| PyTorch | `train.py`    | PyTorch, e3nn            |
| JAX     | `train_jax.py`| JAX, Flax NNX, cuequivariance, cuequivariance_jax |

The same configuration files and multi-resolution data pipeline are shared
between the two backends.

---

## Installation

The PyTorch and JAX paths need separate environments. Minimum dependencies:

**PyTorch env:**

```
pytorch torchvision e3nn batchgenerators monai nibabel numpy scipy matplotlib
```

**JAX env:**

```
jax jaxlib flax optax cuequivariance cuequivariance-jax batchgenerators nibabel numpy scipy matplotlib
```

Both paths share the package under `irrepunet/`. After cloning, make the
repository importable from the working directory:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

No installer / `pyproject` package build is provided — scripts are run
directly from the checkout.

---

## Data layout

Training operates on nnUNet-preprocessed volumes that have been further
reduced to `.npy` + `.pkl` pairs.

1. Convert your raw NIfTI data to the nnUNet raw format (see
   `prepare_nnunet_dataset.py`):
   ```
   nnUNet_raw/DatasetXXX_<name>/
     imagesTr/case_0001_0000.nii.gz
     labelsTr/case_0001.nii.gz
     dataset.json
   ```
2. Run nnUNet's standard preprocessing:
   ```bash
   nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity
   ```
3. Run the irrepunet preprocessing step, which reorients to RAS, crops the
   foreground, normalises intensities, and writes `.npy` + `.pkl` metadata
   under the `irrepunet/` subdirectory:
   ```bash
   python preprocess.py preprocess \
       --nnunet_raw  $nnUNet_raw \
       --nnunet_preprocessed $nnUNet_preprocessed \
       --dataset DatasetXXX_<name>
   ```
4. (Optional) Generate anisotropic subsampled variants used as additional
   training cases (`--subsample_weight` in training):
   ```bash
   python preprocess.py subsample --preprocessed_dir $PREP_DIR
   ```

The loaders read from `$nnUNet_preprocessed/DatasetXXX_<name>/irrepunet/`.

---

## Training

### Two-phase workflow

1. **Plan**: `train.py --plan_only` generates `config.json`,
   `loader_config.txt` (review the receptive-field verification table), and
   `run_training.sh`.
2. **Run**: `train.py --config <dir>/config.json` starts training using the
   planned configuration.

### Recipe: pyramid-curriculum (PyTorch)

This is the reference configuration used for our recent resolution-adaptive
experiments. It combines pyramid-kernel convolution (coarser scales are weighted higher than finer), curriculum batch-size
tiers (train first on slarge batches/coarse kernels, small batches/fine scales enter training later):

```bash
python train.py --plan_only \
    --preprocessed_dir $nnUNet_preprocessed/DatasetXXX/irrepunet \
    --fold 0 \
    --output_dir ./experiments/pyramid_curriculum \
    --patch_size_mm 128 128 128 \
    --n_base_filters 2 --n_downsample 6 \
    --diameter 5.0 --scale 2.0 \
    --kernel_trim_cross_section 0.3 --kernel_growth 2.0 \
    --pyramid scatter \
    --subsample_weight 0.2 \
    --curriculum_bs_tiers 12 8 4 2 1 --curriculum_phase_len 200 \
    --target_memory_mb 24000 \
    --grad_clip 1.0 --epochs 1000
```


Then launch (single GPU):

```bash
python -u train.py --config ./experiments/pyramid_curriculum/config.json \
    > /tmp/irrepunet_training.log 2>&1 &
```

### Multi-GPU training (DDP)

Multi-GPU training uses PyTorch DistributedDataParallel via `torchrun`.
No extra flags are needed — DDP is auto-detected from the `RANK` /
`LOCAL_RANK` / `WORLD_SIZE` environment variables that `torchrun` sets:

```bash
torchrun --nproc_per_node=2 train.py \
    --config ./experiments/pyramid_curriculum/config.json \
    > /tmp/irrepunet_training.log 2>&1 &
```

**Async group selection (default).** Each rank independently picks a spacing
group every step and draws different random patches.  The two GPUs may train
on different resolutions in the same step, increasing per-step resolution
diversity — useful when the dataset spans many spacing groups.  Constraints:

- Gradient accumulation is disabled in async mode (each rank must call
  `backward()` exactly once per optimiser step to keep the all-reduce in sync).
- The training loop counts **steps** rather than patches; the step count is
  `round(patches_per_epoch / mean_batch_size)` so the total number of patches
  seen per epoch stays close to `--patches_per_epoch`.

**Sync group selection (opt-in).** Pass `--ddp_sync_groups` to force every
rank onto the same spacing group every step (replicates single-GPU behaviour
across ranks):

```bash
torchrun --nproc_per_node=2 train.py --config <cfg> --ddp_sync_groups
```

Effective global batch size = `per_gpu_batch * num_gpus` in both modes.
Validation, logging, and checkpointing run on rank 0 only.

A NaN guard wraps the optimiser step: any non-finite loss or grad-norm on any
rank causes all ranks to skip the step, with a `[NaN GUARD]` log line
recording the spacing/patch/batch_size that triggered it.  If model
parameters themselves go non-finite the run aborts cleanly with a
`RuntimeError` rather than continuing on a dead model.

Key knobs:

| Flag                          | Meaning                                     |
|-------------------------------|---------------------------------------------|
| `--patch_size_mm`             | Physical patch size — independent of native image spacing|
| `--n_base_filters`            | Multipler of the base mix of irreps, controls feature depth              |
| `--pyramid scatter`           | Enable pyramid kernel convolution (more weight on coarse scales)           |
| `--curriculum_bs_tiers`       | Effective batch size per curriculum phase   |
| `--curriculum_phase_len`      | Epochs per phase                            |
| `--subsample_weight`          | Undersample anisotropic skip-subsampled cases|
| `--target_memory_mb`          | Memory budget for the profiler              |

### JAX training

The jax implementation lags behinf the main branch pytorch implementation.  `train_jax.py` accepts the same `config.json` as `train.py`, so you can plan
with the PyTorch script and run with the JAX script:

```bash
python train_jax.py --config ./experiments/pyramid_curriculum/config.json
```

---

## Validation

### Full validation set

```bash
python validate.py experiments/pyramid_curriculum \
    --checkpoint model_best.pt --gpu 0
```

`validate.py` distils the trained e3nn model into pure PyTorch Conv3d
kernels per architecture group (via `project_to_spacing`), then runs sliding
window inference and reports per-case dice stratified by resolution and
imaging plane.

The JAX equivalent is `validate_jax.py`.

### Single-case inference

```bash
python infer_case.py \
    --checkpoint experiments/pyramid_curriculum/model_best.pt \
    --input  /path/to/case.nii.gz \
    --output /path/to/pred.nii.gz \
    --mirror_axes 0 1 2 --overlap 0.69
```

JAX: `infer_case_jax.py`.

### Sanity checks

`validate_projection.py` verifies that the distilled Conv3d model produces
the same output as the native e3nn model across a handful of validation
cases — useful after changes to `models/distill.py`.

---

## Package layout

```
irrepunet/
├── inference.py                 # Sliding-window + TTA
├── data/
│   ├── spacing.py               # Canonical spacing grid, case grouping
│   ├── batchgen_dataset.py      # E3nnDataset / E3nnDataLoader
│   ├── batchgen_transforms.py   # nnUNet-style augmentation pipeline
│   ├── multi_resolution_loader.py  # Dynamic batch / RF verification utils
│   ├── dataloader.py            # MultiResolutionLoader (PyTorch DataLoader)
│   └── jax_adapter.py           # JAX-friendly transforms
├── models/
│   ├── layers.py                # VoxelConvolution, PyramidVoxelConvolution,
│   │                            #   Encoder/Decoder, EquivariantPool3d, FusedGate
│   ├── unet.py                  # E3nnUNet
│   ├── distill.py               # project_to_spacing(), hierarchical bundle
│   └── radial_basis.py
├── models_jax/                  # JAX/Flax NNX port of the model and
│   │                            #   training step (inc. split-JIT)
│   ├── unet.py, layers.py, radial_basis.py, tp_right.py, bands.py,
│   ├── split_jit.py, train.py, weight_transfer.py, inference_layers.py
└── training/
    ├── losses.py                # DiceLoss, DiceCELoss, DeepSupervisionLoss
    └── utils.py                 # Shared config and RF-plot helpers
```

The model and the data pipeline are shared between PyTorch and JAX, so a
change to `data/` or `models/layers.py` (shapes, RF math, spacing grouping)
should be kept in sync between the two backends when relevant.

---

## Tests

```bash
pytest tests/       -m "not gpu and not data"
pytest tests_jax/   -m "not gpu and not data"
```

GPU-dependent tests are gated behind `-m gpu`; tests that need preprocessed
data on disk use `-m data`. The JAX `test_layers.py` and `test_radial_basis.py`
cross-validate against the PyTorch `e3nn` library and therefore need both
libraries installed in the same environment.


