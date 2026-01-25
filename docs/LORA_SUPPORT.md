# LoRA Support for Mixtral Router Layers

This document describes how to use LoRA (Low-Rank Adaptation) with Mixtral models in OmniQuant.

## Overview

The LoRA implementation targets the **Router (gate) layers** in Mixtral's Mixture-of-Experts architecture. These gate layers determine which experts process each token, making them critical for model performance.

### Key Features

- **Automatic Detection**: The system automatically detects Mixtral models and applies LoRA to router layers
- **Frozen Base Model**: Original model weights remain frozen; only LoRA adapters are trained/loaded
- **Flexible Loading**: LoRA weights can be loaded from checkpoints during evaluation
- **Batch Experiments**: Support for running multiple LoRA configurations automatically

## Files Added/Modified

| File | Description |
|------|-------------|
| `models/lora_utils.py` | Core LoRA utilities: `LoraLinear` class, weight loading, layer replacement |
| `models/LMClass.py` | Modified to auto-detect and load LoRA for Mixtral evaluation |
| `run_lora_experiments.py` | Batch experiment script for hyperparameter sweeps |
| `config/lora_experiments.yaml` | Example YAML configuration |
| `config/lora_experiments.json` | Example JSON configuration |

## Usage

### 1. Training with LoRA

Training automatically applies LoRA to router layers for Mixtral models:

```bash
python main.py \
    --model /path/to/Mixtral-8x7B-v0.1 \
    --net mixtral-8x7b \
    --epochs 10 \
    --output_dir ./log/mixtral_lora \
    --wbits 4 --abits 16 --group_size 128 \
    --lwc \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_lr 5e-3 \
    --eval_ppl
```

### 2. Evaluation with LoRA Checkpoint

Load a trained LoRA checkpoint for evaluation:

```bash
python main.py \
    --model /path/to/Mixtral-8x7B-v0.1 \
    --net mixtral-8x7b \
    --epochs 0 \
    --output_dir ./log/eval \
    --wbits 4 --abits 16 --group_size 128 \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_checkpoint_path ./log/mixtral_lora/omni_parameters.pth \
    --eval_ppl
```

### 3. Batch Experiments

Run multiple experiments with different LoRA configurations:

```bash
# Using YAML config (requires PyYAML)
python run_lora_experiments.py --config config/lora_experiments.yaml

# Using JSON config
python run_lora_experiments.py --config config/lora_experiments.json

# Generate a parameter sweep config
python run_lora_experiments.py --generate-sweep \
    --model /path/to/Mixtral-8x7B-v0.1 \
    --net mixtral-8x7b \
    --sweep-output config/my_sweep.yaml

# Dry run (show commands without executing)
python run_lora_experiments.py --config config/lora_experiments.yaml --dry-run

# Skip training, only evaluate existing checkpoints
python run_lora_experiments.py --config config/lora_experiments.yaml --skip-training
```

## Configuration File Format

### YAML Format

```yaml
global:
  model_path: "/path/to/Mixtral-8x7B-v0.1"
  net: "mixtral-8x7b"
  output_base_dir: "./experiments/lora_mixtral"
  cache_dir: "./cache"
  calib_dataset: "wikitext2"
  eval_ppl: true

defaults:
  epochs: 10
  batch_size: 1
  wbits: 4
  abits: 16
  group_size: 128
  lwc: true

experiments:
  - name: "lora_rank8_alpha16"
    lora_rank: 8
    lora_alpha: 16.0
    lora_lr: 0.005
    
  - name: "lora_rank16_alpha32"
    lora_rank: 16
    lora_alpha: 32.0
    lora_lr: 0.005
```

### JSON Format

```json
{
  "global": {
    "model_path": "/path/to/Mixtral-8x7B-v0.1",
    "net": "mixtral-8x7b",
    "output_base_dir": "./experiments/lora_mixtral",
    "eval_ppl": true
  },
  "defaults": {
    "epochs": 10,
    "wbits": 4,
    "lwc": true
  },
  "experiments": [
    {
      "name": "lora_rank8_alpha16",
      "lora_rank": 8,
      "lora_alpha": 16.0,
      "lora_lr": 0.005
    }
  ]
}
```

## LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_rank` | 8 | Rank of the low-rank matrices |
| `--lora_alpha` | 16.0 | Scaling factor (scaling = alpha / rank) |
| `--lora_lr` | 5e-3 | Learning rate for LoRA parameters |
| `--lora_checkpoint_path` | None | Path to load LoRA weights from |
| `--init_lora` | False | Initialize LoRA without pretrained weights |

## Output Files

After training, the following files are generated:

- `omni_parameters.pth`: Contains all learned parameters including LoRA weights
- Evaluation results are saved in the experiment's `eval_logs` directory
- Aggregated results in CSV format when using batch experiments

## Technical Details

### LoRA Architecture

For each router layer, we add:
- `lora_A`: Matrix of shape `(in_features, rank)`
- `lora_B`: Matrix of shape `(rank, out_features)`

The forward pass becomes:
```
output = base_output + (x @ lora_A @ lora_B) * (alpha / rank)
```

### Weight Loading

LoRA weights are stored in `omni_parameters.pth` with the following structure:
```python
{
    layer_idx: {
        'block_sparse_moe.gate.lora_A': tensor,
        'block_sparse_moe.gate.lora_B': tensor,
        # ... other OmniQuant parameters
    }
}
```

### Base Model Freezing

During evaluation:
1. Base model weights are registered as buffers (non-trainable)
2. Only LoRA parameters (`lora_A`, `lora_B`) are nn.Parameters
3. `freeze_base_model()` sets `requires_grad=False` for all non-LoRA parameters

## Dependencies

- `torch`: Required
- `transformers`: Required
- `pyyaml`: Optional (for YAML config files)

## Troubleshooting

### "Checkpoint not found" warning
- Ensure the checkpoint path is correct
- Check if training completed successfully

### LoRA not applied
- Verify the model name contains "mixtral"
- Check that `--lora_checkpoint_path` points to a valid file

### Mismatched dimensions
- Ensure `--lora_rank` and `--lora_alpha` match between training and evaluation
