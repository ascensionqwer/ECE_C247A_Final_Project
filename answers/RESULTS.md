# Training Results (2026-03-11, 09:00+)

## Note on High CER Values
**Baseline CER is higher than expected (~30 per spec).** Possible causes:
- MPS (Apple Silicon) vs CUDA numerical differences
- Different training dynamics on Apple hardware
- **Action item**: Compare test CER (not val CER) for final comparison

## Validation CER Results

| Model Type | Best Val CER | Checkpoint | Notes |
|------------|--------------|------------|-------|
| TDS Conv CTC (Baseline) | 53.37 | epoch=74-step=9000 | Higher than expected |
| Transformer Small Sinusoidal | 66.97 | epoch=92-step=11160 | |
| Transformer Small Learnable | 99.14 | epoch=56-step=6840 | Poor convergence |
| **Transformer Large Sinusoidal** | **34.38** | epoch=23-step=2880 | **Best so far** |
| Transformer Large Learnable | 87.13* | epoch=9-step=1200 | Still training |

## Model Configurations

### TDS Conv CTC (Baseline)
- Module: TDSConvCTCModule
- in_features: 528
- mlp_features: [384]
- block_channels: [24, 24, 24, 24]
- kernel_width: 32

### Transformer Small (Sinusoidal/Learnable)
- Module: TransformerCTCModule
- d_model: 128
- nhead: 4
- num_layers: 2
- dim_feedforward: 256
- dropout: 0.1

### Transformer Large (Sinusoidal/Learnable)
- Module: TransformerCTCModule
- d_model: 512
- nhead: 8
- num_layers: 4
- dim_feedforward: 1024
- dropout: 0.1
