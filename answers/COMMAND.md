# EMG2QWERTY Commands Reference

All commands assume you're in the project root directory and have activated the conda environment.

## Environment Setup

```bash
# Required for Apple Silicon (M1/M2/M3) MPS acceleration
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Training Commands

### Single-User Training (Personalized Model)

```bash
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=mps trainer.devices=1
```

### Generic User Training (Multi-User Model)

```bash
python -m emg2qwerty.train \
  user="generic" \
  trainer.accelerator=mps trainer.devices=1
```

### Training with Custom Model Architecture

```bash
# Transformer small with sinusoidal positional encoding
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_small_sinusoidal \
  trainer.accelerator=mps trainer.devices=1

# Transformer small with learnable positional encoding
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_small_learnable \
  trainer.accelerator=mps trainer.devices=1

# Transformer large with sinusoidal positional encoding
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_large_sinusoidal \
  trainer.accelerator=mps trainer.devices=1

# Transformer large with learnable positional encoding
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_large_learnable \
  trainer.accelerator=mps trainer.devices=1
```

### Training with Custom Hyperparameters

```bash
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=mps trainer.devices=1 \
  batch_size=16 \
  trainer.max_epochs=200
```

## Testing Commands

### Greedy Decoding (Fast)

```bash
python -m emg2qwerty.train \
  user="single_user" \
  checkpoint="path/to/checkpoint.ckpt" \
  train=False \
  trainer.accelerator=mps \
  decoder=ctc_greedy
```

### Beam Search Decoding (Better Accuracy)

```bash
python -m emg2qwerty.train \
  user="single_user" \
  checkpoint="path/to/checkpoint.ckpt" \
  train=False \
  trainer.accelerator=mps \
  decoder=ctc_beam
```

### Testing with Specific Checkpoint

Replace `YYYY-MM-DD/HH-MM-SS` with your actual log directory:

```bash
python -m emg2qwerty.train \
  user="single_user" \
  checkpoint="logs/YYYY-MM-DD/HH-MM-SS/checkpoints/last.ckpt" \
  train=False \
  trainer.accelerator=mps \
  decoder=ctc_greedy
```

## Utility Commands

### Print Dataset Statistics

```bash
python scripts/print_dataset_stats.py
```

### Generate Data Splits

```bash
python scripts/generate_splits.py
```

### Build Character-Level Language Model (for Beam Search)

```bash
./scripts/lm/build_char_lm.sh 6
```

## Configuration Override Reference

| Parameter | Options | Description |
|-----------|---------|-------------|
| `user` | `single_user`, `generic` | Dataset split configuration |
| `model` | `tds_conv_ctc`, `transformer_small_sinusoidal`, `transformer_small_learnable`, `transformer_large_sinusoidal`, `transformer_large_learnable` | Model architecture |
| `decoder` | `ctc_greedy`, `ctc_beam` | Decoding strategy |
| `trainer.accelerator` | `mps`, `cpu`, `gpu` | Hardware accelerator |
| `trainer.devices` | `1`, `2`, etc. | Number of devices |
| `trainer.max_epochs` | integer | Maximum training epochs |
| `batch_size` | integer | Batch size |
| `train` | `True`, `False` | Whether to train or just test |
| `checkpoint` | path string | Path to checkpoint file |

## Log Locations

- Training logs: `logs/YYYY-MM-DD/HH-MM-SS/`
- Checkpoints: `logs/YYYY-MM-DD/HH-MM-SS/checkpoints/`
- Hydra configs: `logs/YYYY-MM-DD/HH-MM-SS/hydra_configs/`
