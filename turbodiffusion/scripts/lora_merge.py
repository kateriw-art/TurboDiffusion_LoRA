"""Merge a LoRA checkpoint into a base model .pth file for inference.

Two LoRA key formats are supported:

  Diffusers/PEFT  — keys end in ``.lora_A.weight`` / ``.lora_B.weight``
  Kohya           — keys end in ``.lora_down.weight`` / ``.lora_up.weight``

The merged weight delta for each LoRA layer is computed as::

    delta_W = lora_up @ lora_down * (alpha / rank) * scale

where ``rank`` is inferred from the inner dimension of lora_down,
``alpha`` is read from the ``<base_key>.alpha`` key in the LoRA file
(defaults to ``rank`` when absent, which gives ``alpha/rank = 1``),
and ``scale`` is a user-supplied multiplier (default ``1.0``).

Key-prefix remapping:

  LoRA files trained on the original Hugging Face Wan model (e.g. via
  diffusers/PEFT) typically use ``transformer.`` as the top-level module
  prefix, while TurboDiffusion base models store weights under ``net.``.
  Use ``--lora_key_prefix transformer.`` and ``--model_key_prefix net.``
  (the defaults) to map between them automatically.

  If your LoRA already uses ``net.`` keys, pass ``--lora_key_prefix net.``.
  If there is no prefix at all, pass ``--lora_key_prefix ""``.

Usage::

    python turbodiffusion/scripts/lora_merge.py \\
        --base_model checkpoints/base_model.pth \\
        --lora       my_lora.safetensors \\
        --output     checkpoints/lora_merged.pth \\
        [--scale 1.0] \\
        [--lora_key_prefix transformer.] \\
        [--model_key_prefix net.]
"""

import argparse
import sys

import torch
from safetensors import safe_open


def load_lora(path: str) -> dict:
    """Load a LoRA checkpoint from a ``.safetensors`` or ``.pth`` / ``.pt`` file."""
    if path.endswith(".safetensors"):
        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return sd
    return torch.load(path, map_location="cpu")


def detect_lora_format(sd: dict) -> str:
    """Return ``'peft'`` or ``'kohya'`` based on the key naming in *sd*."""
    for key in sd:
        if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            return "peft"
        if key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight"):
            return "kohya"
    raise ValueError(
        "Could not detect LoRA format. Expected keys ending in "
        "'.lora_A.weight' / '.lora_B.weight' (Diffusers/PEFT) or "
        "'.lora_down.weight' / '.lora_up.weight' (Kohya)."
    )


def collect_lora_layers(sd: dict, fmt: str) -> dict:
    """
    Parse all LoRA layer pairs from *sd* and return a mapping::

        base_key -> {"down": tensor, "up": tensor, "alpha": float | None}

    ``base_key`` is the LoRA key with the format-specific suffix stripped.
    """
    down_suffix = ".lora_A.weight" if fmt == "peft" else ".lora_down.weight"
    up_suffix = ".lora_B.weight" if fmt == "peft" else ".lora_up.weight"

    layers: dict = {}
    for key, val in sd.items():
        if key.endswith(down_suffix):
            base = key[: -len(down_suffix)]
            layers.setdefault(base, {})["down"] = val
        elif key.endswith(up_suffix):
            base = key[: -len(up_suffix)]
            layers.setdefault(base, {})["up"] = val
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            layers.setdefault(base, {})["alpha"] = float(val.item())
    return layers


def remap_key(lora_base_key: str, lora_prefix: str, model_prefix: str) -> str:
    """Strip *lora_prefix* and prepend *model_prefix* to produce a model state-dict key."""
    if lora_prefix and lora_base_key.startswith(lora_prefix):
        lora_base_key = lora_base_key[len(lora_prefix):]
    return f"{model_prefix}{lora_base_key}"


def merge_lora(base_path, lora_path, output_path, scale, lora_key_prefix, model_key_prefix):
    print(f"Loading base model: {base_path}")
    base_sd = torch.load(base_path, map_location="cpu")

    print(f"Loading LoRA: {lora_path}")
    lora_sd = load_lora(lora_path)

    fmt = detect_lora_format(lora_sd)
    print(f"Detected LoRA format: {fmt}")

    layers = collect_lora_layers(lora_sd, fmt)
    print(f"Found {len(layers)} LoRA layer pair(s).")

    merged_sd = {k: v.clone() for k, v in base_sd.items()}

    applied = 0
    skipped = []
    for lora_base_key, components in layers.items():
        if "down" not in components or "up" not in components:
            print(f"  [WARNING] Incomplete LoRA pair for '{lora_base_key}', skipping.")
            continue

        model_weight_key = remap_key(lora_base_key, lora_key_prefix, model_key_prefix) + ".weight"
        if model_weight_key not in merged_sd:
            skipped.append(model_weight_key)
            continue

        lora_down = components["down"].float()   # shape: (rank, in_features)
        lora_up = components["up"].float()        # shape: (out_features, rank)
        rank = lora_down.shape[0]
        alpha = components.get("alpha", rank)     # alpha == rank  →  effective scale = user scale
        effective_scale = scale * (alpha / rank)

        delta = (lora_up @ lora_down) * effective_scale   # (out_features, in_features)

        target = merged_sd[model_weight_key]
        if delta.shape != target.shape:
            print(
                f"  [WARNING] Shape mismatch for '{model_weight_key}': "
                f"LoRA delta {tuple(delta.shape)} vs base weight {tuple(target.shape)}. Skipping."
            )
            continue

        with torch.no_grad():
            merged_sd[model_weight_key] = (target.float() + delta).to(target.dtype)
        applied += 1

    if skipped:
        print(f"\n  [INFO] {len(skipped)} LoRA key(s) had no matching base-model weight and were skipped:")
        for k in skipped[:10]:
            print(f"    {k}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more.")
        print(
            "  If many keys were skipped, check --lora_key_prefix / --model_key_prefix.\n"
            "  Run with a LoRA key printed above (minus the '.weight' suffix) to verify the mapping."
        )

    print(f"\nApplied {applied} / {len(layers)} LoRA layer(s) to the base model.")
    print(f"Saving merged model to: {output_path}")
    torch.save(merged_sd, output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the base model .pth file (e.g. produced by safetensors_to_pth.py).",
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="Path to the LoRA file (.safetensors, .pth, or .pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the merged .pth file.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Global LoRA strength multiplier applied on top of alpha/rank (default: 1.0).",
    )
    parser.add_argument(
        "--lora_key_prefix",
        type=str,
        default="transformer.",
        help=(
            "Prefix that appears at the start of LoRA keys but NOT in the base model. "
            "It is stripped before --model_key_prefix is prepended. "
            "Use 'transformer.' (default) for Diffusers/PEFT LoRAs trained on the original Wan model, "
            "or 'net.' if the LoRA was already trained against the TurboDiffusion key naming."
        ),
    )
    parser.add_argument(
        "--model_key_prefix",
        type=str,
        default="net.",
        help=(
            "Prefix used in the base model state dict (default: 'net.'). "
            "This is prepended after --lora_key_prefix is stripped."
        ),
    )
    args = parser.parse_args()

    if not args.output.endswith((".pth", ".pt")):
        print("[WARNING] Output path does not end with .pth or .pt. Continuing anyway.")

    merge_lora(
        base_path=args.base_model,
        lora_path=args.lora,
        output_path=args.output,
        scale=args.scale,
        lora_key_prefix=args.lora_key_prefix,
        model_key_prefix=args.model_key_prefix,
    )
