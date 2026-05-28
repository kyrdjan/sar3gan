# SAR3GAN

SAR3GAN is a PyTorch super-resolution GAN for learning a paired `64x64 -> 256x256`
image upscaler. The active training code lives in `sar3gan_training/`; each
subfolder is a self-contained experiment with its own `trained.py`, saved EMA
generator weights, and training log.

This repository is derived from the R3GAN/StyleGAN codebase, but the practical
entry point for this project is the SAR3GAN training workflow below.

## Repository Layout

```text
SAR3GAN/
|-- datasets/
|   |-- 64klr.zip                 # low-resolution training images
|   `-- 256khr.zip                # matching high-resolution training images
|-- sar3gan_training/
|   |-- baseline/
|   |   |-- trained.py            # baseline SAR3GAN training script
|   |   |-- 1st_best_generator_ema.pth
|   |   |-- test_inference.py     # single-image inference utility
|   |   `-- evaulate_psnr_ssim.py # PSNR/SSIM evaluation utility
|   |-- train1/
|   |   |-- trained.py
|   |   `-- best_generator_ema.pth
|   |-- train2/
|   |   |-- trained.py
|   |   `-- best_generator_ema.pth
|   `-- train3/
|       |-- trained.py
|       `-- best_generator_ema.pth
|-- torch_utils/                  # custom CUDA/PyTorch ops used by training
|-- requirements.txt
`-- README.md
```

## Which SAR3GAN Model Should I Start With?

Use `sar3gan_training/train2/trained.py` as the recommended training script. It
is the strongest logged experiment in this repo:

| Experiment | Main difference | Batch size | Best logged FID |
| --- | --- | ---: | ---: |
| `baseline` | baseline residual SAR3GAN setup | 8 | 10.65 |
| `train1` | attention enabled at the 64x64 generator/discriminator stage | 8 | 8.68 |
| `train2` | adjusted attention placement before/after resampling | 4 | 7.42 |
| `train3` | alternate attention placement | 8 | 10.23 |

All four experiments train the same task: convert RGB low-resolution inputs to
RGB high-resolution outputs using paired images from `datasets/64klr.zip` and
`datasets/256khr.zip`.

## Requirements

This project expects a CUDA-capable PyTorch environment. The pinned requirements
install PyTorch 2.5.1 with CUDA 12.1 wheels.

```powershell
python -m venv sar3ganve
.\sar3ganve\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torchinfo pytorch-fid torchmetrics
```

If you use Linux/macOS, activate the environment with:

```bash
python -m venv sar3ganve
source sar3ganve/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torchinfo pytorch-fid torchmetrics
```

## Dataset Preparation

The training scripts expect two ZIP files in `datasets/`:

```text
datasets/64klr.zip
datasets/256khr.zip
```

The ZIPs must contain matching image pairs:

- `64klr.zip` contains the low-resolution source images.
- `256khr.zip` contains the corresponding high-resolution targets.
- Valid extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, and `.tiff`.
- Files are paired by sorted ZIP filename order, so keep names consistent across
  both archives.
- During training, LR images are resized to `64x64`; HR images are resized to
  `256x256`.

Example naming:

```text
64klr.zip:  000001.png, 000002.png, 000003.png
256khr.zip: 000001.png, 000002.png, 000003.png
```

## Train SAR3GAN

Run training from the repository root, not from inside `sar3gan_training/`. The
scripts import `torch_utils` with paths relative to the repo root.

Recommended run:

```powershell
python sar3gan_training\train2\trained.py
```

Equivalent Linux/macOS command:

```bash
python sar3gan_training/train2/trained.py
```

By default the training script uses:

- input size: `64x64`
- output size: `256x256`
- paired ZIP dataset: `datasets/64klr.zip` and `datasets/256khr.zip`
- total epochs: `50`
- optimizer: Adam with `lr=1e-4`, `betas=(0.0, 0.99)`
- scheduler: warmup cosine with 500 warmup steps
- EMA decay: `0.999`
- sample image interval: every `500` steps
- FID interval: every `1000` steps
- checkpoint interval: every `2000` steps

Training output is written to:

```text
training_snapshots_64to256/
```

Important outputs:

```text
training_snapshots_64to256/log.txt
training_snapshots_64to256/latest_checkpoint.pth
training_snapshots_64to256/checkpoint_step_<step>.pth
training_snapshots_64to256/best_generator_ema.pth
training_snapshots_64to256/final_generator.pth
training_snapshots_64to256/final_generator_ema.pth
training_snapshots_64to256/final_discriminator.pth
training_snapshots_64to256/step_<step>.png
```

Use `best_generator_ema.pth` for inference and evaluation unless you have a
specific reason to inspect the non-EMA generator.

## Resume Training

Resume a full training state, including generator, discriminator, optimizers,
schedulers, EMA state, epoch, step, and best FID:

```powershell
python sar3gan_training\train2\trained.py --resume training_snapshots_64to256\latest_checkpoint.pth
```

Warm-start from generator weights only:

```powershell
python sar3gan_training\train2\trained.py `
  --weights-only sar3gan_training\train2\best_generator_ema.pth `
  --start-step 0 `
  --start-epoch 0
```

Use `--weights-only` when you have a `.pth` generator state dict but no matching
optimizer/scheduler checkpoint. Use `--resume` when you are continuing a run
created by the same `trained.py`.

## Adjust Training Settings

The training scripts currently keep their configuration inside `main()` rather
than exposing every value as a command-line flag. Edit the relevant variables in
the selected `trained.py`:

```python
dataset = ImagePairZipDataset("datasets/256khr.zip", "datasets/64klr.zip", ...)
batch_size = 4
width = [64, 128, 256]
gen_blocks = [2, 2, 2]
disc_blocks = [2, 2, 2]
total_epochs = 50
```

Common changes:

- Lower `batch_size` if you run out of GPU memory.
- Change the dataset ZIP paths if your archives have different names.
- Reduce `num_images` in `evaluate_fid(...)` for faster development runs.
- Change `save_dir` if you want to keep multiple runs separate.

## Run Single-Image Inference

The included inference utility is in the `baseline` folder and matches the
baseline/train1/train3 generator layout. For best compatibility, use it with the
baseline-style weights unless you copy the exact generator definition from the
experiment you trained.

```powershell
python sar3gan_training\baseline\test_inference.py `
  --generator sar3gan_training\baseline\1st_best_generator_ema.pth `
  --lr-image path\to\input_64x64.png `
  --save-path outputs\sr_output.png
```

The script loads one RGB image, resizes it to `64x64`, runs the generator, and
saves a `256x256` super-resolved output.

## Evaluate PSNR and SSIM

The PSNR/SSIM utility is also in `sar3gan_training/baseline/`:

```powershell
python sar3gan_training\baseline\evaulate_psnr_ssim.py
```

Before running it, edit the bottom of the file if needed:

```python
evaluate(
    best_model_path="training_snapshots_64to256-FINAL-TRAIN-3/best_generator_ema.pth",
    hr_zip="datasets/256khr.zip",
    lr_zip="datasets/64klr.zip",
    max_images=500
)
```

For meaningful results, evaluate on images that were not used for training.

## Troubleshooting

- `FileNotFoundError` for dataset ZIPs: confirm that `datasets/64klr.zip` and
  `datasets/256khr.zip` exist and that you launched Python from the repo root.
- CUDA out of memory: lower `batch_size` in `trained.py`. `train2` already uses
  `batch_size = 4`.
- Import errors for `torchinfo`, `pytorch_fid`, or `torchmetrics`: install the
  extra packages shown in the requirements section.
- Bad image pairing: make sure both ZIP files sort into the same pair order.
- Slow FID: the default FID pass can generate up to 50,000 images. Lower the
  `num_images` argument in `evaluate_fid(...)` while iterating.

## Notes for Contributors

Keep new experiments under `sar3gan_training/<experiment-name>/` with the same
basic artifacts:

```text
trained.py
log.txt
best_generator_ema.pth
```

If the generator architecture changes, copy the matching generator definition
into any inference or evaluation script you plan to use. PyTorch state dicts are
architecture-specific; a checkpoint from one experiment may not load into a
different generator definition.
