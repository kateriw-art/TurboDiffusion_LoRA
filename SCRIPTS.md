# Scripts Reference

This document summarizes every runnable script in the TurboDiffusion repository, organized by location and purpose.

---

## End-to-End Workflow

The scripts form a sequential fine-tuning and deployment pipeline. The diagram below shows the order of operations and the file that is produced at each step.

```
HuggingFace safetensors weights
        │
        ▼  (1) safetensors_to_pth.py
base_model.pth   ◄──── also keep a copy as "diff_base" for the merge step
        │
        ▼  (2) train.py  (torchrun)
            [trainer saves DCP checkpoints automatically]
        │
        ▼  (3) dcp_to_pth.py
finetuned_model.pth   ◄──── "diff_target" for the merge step
        │
        ▼  (4) merge_models.py
merged_model.pth   =  base + w × (finetuned − base)
        │
        ▼  (5) modify_model.py  (called by quantize.sh)
            [replace attention with SLA/SageSLA, optionally quantize linears]
deployment_model.pth
        │
        ▼  (6) inference  (wan2.1_t2v_infer.py or wan2.2_i2v_infer.py)
output_video.mp4
```

### Step-by-step explanation

| Step | Script | What it does |
|------|--------|--------------|
| **1** | `turbodiffusion/scripts/safetensors_to_pth.py` | Converts a locally saved pretrained Wan base model from HuggingFace sharded `.safetensors` format into a single `base_model.pth`. Download the model weights first (e.g. via `huggingface-cli download` or `wget`), then point `--model_dir` at the directory. Add `--prefix net.` so key names match the training framework. Keep this file — it serves as both the training starting point and the `--diff_base` argument in the merge step. |
| **2** | `turbodiffusion/scripts/train.py` | Fine-tunes the model (rCM distillation / LoRA / full fine-tune). The trainer loads `base_model.pth` via the config and **automatically saves checkpoints in PyTorch Distributed Checkpoint (DCP) format** — no separate conversion is needed before training. |
| **3** | `turbodiffusion/scripts/dcp_to_pth.py` | Converts the DCP checkpoint directory produced by training into a single `finetuned_model.pth`. Extracts the EMA weights (`net_ema.*` → `net.*`) and saves in `bfloat16`. |
| **4** | `turbodiffusion/scripts/merge_models.py` | Applies the trained delta back onto the original base model using vector arithmetic: `merged = base + w × (finetuned − base)`. Pass `base_model.pth` as both `--base` and `--diff_base`, and `finetuned_model.pth` as `--diff_target`. Adjust `--w` (default `1.0`) to control how strongly the fine-tuning is applied. |
| **5** | `turbodiffusion/inference/modify_model.py` (via `scripts/quantize.sh`) | Prepares `merged_model.pth` for fast deployment: replaces self-attention with SLA or SageSLA, swaps in fused LayerNorm/RMSNorm, and optionally quantizes linear layers to Int8. Produces `deployment_model.pth`. |
| **6** | `scripts/inference_wan2.1_t2v.sh` / `scripts/inference_wan2.2_i2v.sh` | Runs the TurboDiffusion inference pipeline on `deployment_model.pth` and writes the output video. |

> **Note on the "convert to DCP" step:** there is no separate script to convert a `.pth` file *into* DCP format. The training framework (`train.py`) handles this automatically — it reads `base_model.pth` at startup and writes DCP checkpoints during training. The only explicit checkpoint conversion scripts are `safetensors_to_pth.py` (HuggingFace → `.pth`, done *before* training) and `dcp_to_pth.py` (DCP → `.pth`, done *after* training).

### Concrete example (Wan2.1-T2V-1.3B fine-tune)

```bash
export PYTHONPATH=turbodiffusion

# 1. Convert HuggingFace base model to .pth
python turbodiffusion/scripts/safetensors_to_pth.py \
    --model_dir /path/to/Wan2.1-T2V-1.3B \
    --output_path checkpoints/base_model.pth \
    --prefix net.

# 2. Fine-tune  (the trainer saves DCP checkpoints to checkpoints/iter_*/model/)
torchrun --nproc_per_node=8 -m scripts.train \
    --config configs/wan2.1_t2v_1.3B.py

# 3. Convert the best DCP checkpoint back to .pth
python turbodiffusion/scripts/dcp_to_pth.py \
    --dcp_checkpoint_dir checkpoints/iter_000010000/model \
    --save_path checkpoints/finetuned_model.pth

# 4. Merge fine-tuned delta onto the base model
python turbodiffusion/scripts/merge_models.py \
    --base      checkpoints/base_model.pth \
    --diff_base checkpoints/base_model.pth \
    --diff_target checkpoints/finetuned_model.pth \
    --w 1.0 \
    --output checkpoints/merged_model.pth

# 5. Prepare for deployment (replace attention + quantize)
python turbodiffusion/inference/modify_model.py \
    --model Wan2.1-1.3B \
    --input_path  checkpoints/merged_model.pth \
    --output_path checkpoints/deployment_model.pth \
    --attention_type sla \
    --quant_linear

# 6. Run inference
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --model Wan2.1-1.3B \
    --dit_path checkpoints/deployment_model.pth \
    --prompt "Your prompt here" \
    --resolution 480p \
    --num_steps 4 \
    --quant_linear \
    --attention_type sagesla
```

---

## Shell Scripts (`scripts/`)

### `scripts/inference_wan2.1_t2v.sh`
**Purpose:** Run text-to-video (T2V) inference using a TurboWan2.1 checkpoint.

Sets `PYTHONPATH=turbodiffusion` and invokes `turbodiffusion/inference/wan2.1_t2v_infer.py` with example arguments.

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dit_path` | *(required)* | Path to the finetuned TurboDiffusion checkpoint |
| `--model` | `Wan2.1-1.3B` | Model variant: `Wan2.1-1.3B` or `Wan2.1-14B` |
| `--prompt` | *(required)* | Text prompt for video generation |
| `--resolution` | `480p` | Output resolution: `480p` or `720p` |
| `--aspect_ratio` | `16:9` | Aspect ratio in `W:H` format |
| `--num_frames` | `77` | Number of frames to generate |
| `--num_steps` | `4` | Sampling steps (1–4) |
| `--num_samples` | `1` | Number of videos to generate |
| `--sigma_max` | `80` | Initial sigma for rCM; larger values reduce diversity but may improve quality |
| `--seed` | `0` | Random seed for reproducibility |
| `--save_path` | `output/generated_video.mp4` | Output file path (include extension) |
| `--attention_type` | `sagesla` | Attention module: `original`, `sla`, or `sagesla` |
| `--sla_topk` | `0.1` | Top-k ratio for SLA/SageSLA attention (0.15 recommended for higher quality) |
| `--vae_path` | `checkpoints/Wan2.1_VAE.pth` | Path to Wan2.1 VAE |
| `--text_encoder_path` | `checkpoints/models_t5_umt5-xxl-enc-bf16.pth` | Path to umT5 text encoder |
| `--quant_linear` | *(flag)* | Enable quantization for linear layers (use with quantized checkpoints) |
| `--default_norm` | *(flag)* | Use original LayerNorm/RMSNorm instead of fast replacements |

**Example:**
```bash
bash scripts/inference_wan2.1_t2v.sh
```

---

### `scripts/inference_wan2.2_i2v.sh`
**Purpose:** Run image-to-video (I2V) inference using a dual-checkpoint TurboWan2.2 setup (separate high-noise and low-noise models).

Sets `PYTHONPATH=turbodiffusion` and invokes `turbodiffusion/inference/wan2.2_i2v_infer.py` with example arguments.

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--image_path` | *(required)* | Path to the input image |
| `--high_noise_model_path` | *(required)* | Path to the high-noise TurboDiffusion checkpoint |
| `--low_noise_model_path` | *(required)* | Path to the low-noise TurboDiffusion checkpoint |
| `--boundary` | `0.9` | Timestep boundary for switching from high to low noise model |
| `--model` | `Wan2.2-A14B` | Model variant (currently only `Wan2.2-A14B`) |
| `--prompt` | *(required)* | Text prompt for video generation |
| `--resolution` | `720p` | Output resolution: `480p` or `720p` |
| `--aspect_ratio` | `16:9` | Aspect ratio in `W:H` format |
| `--adaptive_resolution` | *(flag)* | Adapt output resolution to match the input image's aspect ratio |
| `--ode` | *(flag)* | Use ODE sampling (sharper but less robust than SDE) |
| `--num_frames` | `77` | Number of frames to generate |
| `--num_steps` | `4` | Sampling steps (1–4) |
| `--num_samples` | `1` | Number of videos to generate |
| `--sigma_max` | `200` | Initial sigma for rCM |
| `--seed` | `0` | Random seed for reproducibility |
| `--save_path` | `output/generated_video.mp4` | Output file path (include extension) |
| `--attention_type` | `sagesla` | Attention module: `original`, `sla`, or `sagesla` |
| `--sla_topk` | `0.1` | Top-k ratio for SLA/SageSLA attention |
| `--vae_path` | `checkpoints/Wan2.2_VAE.pth` | Path to Wan2.2 VAE |
| `--text_encoder_path` | `checkpoints/models_t5_umt5-xxl-enc-bf16.pth` | Path to umT5 text encoder |
| `--quant_linear` | *(flag)* | Enable quantization for linear layers |
| `--default_norm` | *(flag)* | Use original LayerNorm/RMSNorm |

**Example:**
```bash
bash scripts/inference_wan2.2_i2v.sh
```

---

### `scripts/quantize.sh`
**Purpose:** Convert raw rCM-format training checkpoints into deployment-ready `.pth` files by optionally replacing attention modules (SLA/SageSLA) and/or quantizing linear layers.

Calls `turbodiffusion/inference/modify_model.py` once per target checkpoint variant. The script ships with commented-out examples for Wan2.1 models and active commands for the Wan2.2-A14B low/high noise models.

**Example:**
```bash
bash scripts/quantize.sh
```

---

## Python Scripts (`turbodiffusion/scripts/`)

### `turbodiffusion/scripts/train.py`
**Purpose:** Main entry point for distributed model training. Reads a Python-based lazy config file, initializes the distributed environment, builds the model and data loaders, and starts the training loop.

**Usage:**
```bash
torchrun --nproc_per_node=<N> -m scripts.train \
    --config <path/to/config.py> \
    [key=value overrides ...]
```

**Key arguments:**

| Argument | Description |
|---|---|
| `--config` | *(required)* Path to the Python config file |
| `opts` | Key=value config overrides (LazyConfig style) |
| `--dryrun` | Validate and print the config without running training |

---

### `turbodiffusion/scripts/merge_models.py`
**Purpose:** Merge three PyTorch model checkpoints using vector arithmetic:

```
Result = Base + w × (Diff_Target − Diff_Base)
```

Useful for applying a trained delta (e.g., a LoRA-style difference) to a base model with a configurable interpolation weight.

**Usage:**
```bash
python turbodiffusion/scripts/merge_models.py \
    --base <base.pt> \
    --diff_base <diff_base.pt> \
    --diff_target <diff_target.pt> \
    --w 1.0 \
    --output merged_model.pt
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--base` | *(required)* | Path to the base model `.pt` file |
| `--diff_base` | *(required)* | Path to the diff-base model `.pt` file |
| `--diff_target` | *(required)* | Path to the diff-target model `.pt` file |
| `--w` | `1.0` | Interpolation weight |
| `--output` | `merged_model.pt` | Output path for the merged checkpoint |

---

### `turbodiffusion/scripts/dcp_to_pth.py`
**Purpose:** Convert a PyTorch Distributed Checkpoint (DCP) directory—as saved by `torch.distributed.checkpoint`—into a single `.pth` file. Extracts the EMA weights (`net_ema.*`), renames them to `net.*`, and saves in `bfloat16`.

**Usage:**
```bash
python turbodiffusion/scripts/dcp_to_pth.py \
    --dcp_checkpoint_dir checkpoints/iter_000010000/model \
    --save_path saved_model.pth
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dcp_checkpoint_dir` | `checkpoints/iter_000010000/model` | Path to the DCP checkpoint directory |
| `--save_path` | `saved_model.pt` | Path for the output `.pth` file |

---

### `turbodiffusion/scripts/safetensors_to_pth.py`
**Purpose:** Merge a sharded HuggingFace-style `.safetensors` model (described by a `diffusion_pytorch_model.safetensors.index.json` index file) into a single `.pth` file. Automatically converts weights to `bfloat16` and reshapes `patch_embedding.weight` from Conv3d to Linear format.

**Usage:**
```bash
python turbodiffusion/scripts/safetensors_to_pth.py \
    --model_dir <path/to/safetensors_dir> \
    --output_path <output.pth> \
    [--prefix net.]
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | *(required)* | Directory containing the index JSON and shard files |
| `--output_path` | *(required)* | Output path for the merged `.pth` file |
| `--prefix` | `None` | Optional prefix prepended to all state-dict keys (e.g., `net.`) |

---

## Python Inference Modules (`turbodiffusion/inference/`)

These modules are invoked by the shell scripts above and can also be called directly.

### `turbodiffusion/inference/wan2.1_t2v_infer.py`
**Purpose:** Core text-to-video inference pipeline for Wan2.1 models. Loads the DiT, VAE, and text encoder; embeds the prompt with umT5; runs rCM-based sampling; and saves the result as a video file. Supports an optional interactive `--serve` TUI mode that keeps the model loaded between generations.

**Invoked by:** `scripts/inference_wan2.1_t2v.sh`

---

### `turbodiffusion/inference/wan2.2_i2v_infer.py`
**Purpose:** Core image-to-video inference pipeline for Wan2.2. Accepts an input image plus a text prompt, runs a dual-model rCM sampling strategy (switching from the high-noise to the low-noise model at a configurable `--boundary` timestep), and saves the output video. Also supports `--serve` TUI mode.

**Invoked by:** `scripts/inference_wan2.2_i2v.sh`

---

### `turbodiffusion/inference/modify_model.py`
**Purpose:** Modify a trained checkpoint for deployment by:
1. Replacing self-attention modules with SLA or SageSLA variants.
2. Optionally quantizing `nn.Linear` layers to `Int8Linear`.
3. Replacing `LayerNorm`/`RMSNorm` with faster fused implementations.
4. Saving the modified state dict as a standalone `.pth` file.

**Invoked by:** `scripts/quantize.sh`

**Usage:**
```bash
python turbodiffusion/inference/modify_model.py \
    --model Wan2.1-1.3B \
    --input_path checkpoints/merge_rcm_format.pth \
    --output_path checkpoints/modified/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type sla \
    [--quant_linear] \
    [--default_norm]
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `Wan2.1-1.3B` | Model variant: `Wan2.1-1.3B`, `Wan2.1-14B`, or `Wan2.2-A14B` |
| `--input_path` | *(required)* | Input checkpoint path (rCM-format `.pth`) |
| `--output_path` | *(required)* | Output path for the modified checkpoint |
| `--attention_type` | `original` | Attention type: `original`, `sla`, or `sagesla` |
| `--sla_topk` | `0.2` | Top-k ratio for SLA/SageSLA |
| `--quant_linear` | *(flag)* | Replace linear layers with quantized (`Int8`) versions |
| `--default_norm` | *(flag)* | Keep original LayerNorm/RMSNorm (skip fast replacements) |
