# Branch Analysis Report

Analysis of commits after `c3f0e422f9e048bcefbcf1cb4cc17f529637f889` across branches (excluding `feat/sk-work`).

---

## Branch: `origin/main`

**Commit:** `7f8d63e`  
**Author:** Wesley Gunawan (wesleygwn@gmail.com)  
**Date:** Tue Mar 10 00:27:35 2026 -0700  
**Message:** "Experiment Done"

### Files Modified/Added

| File | Status |
|------|--------|
| `Colab_setup.ipynb` | Modified |
| `config/model/cnn_lstm_ctc.yaml` | Added |
| `config/model/gru_ctc.yaml` | Added |
| `config/model/lstm_1channel.yaml` | Added |
| `config/model/lstm_4channels.yaml` | Added |
| `config/model/lstm_8channels.yaml` | Added |
| `config/model/lstm_augmented.yaml` | Added |
| `config/model/lstm_ctc.yaml` | Added |
| `config/model/lstm_downsample_2x.yaml` | Added |
| `config/model/lstm_downsample_4x.yaml` | Added |
| `config/user/single_user.yaml` | Modified |
| `config/user/single_user_25percent.yaml` | Added |
| `config/user/single_user_50percent.yaml` | Added |
| `emg2qwerty/lightning.py` | Modified |
| `emg2qwerty/modules.py` | Modified |
| `emg2qwerty/transforms.py` | Modified |

### Work Done

#### 1. Model Implementations (`emg2qwerty/modules.py`)

Added three new encoder classes:

**LSTMEncoder**
- Bidirectional LSTM encoder
- Parameters: `in_features`, `hidden_size=256`, `num_layers=2`, `dropout=0.3`, `bidirectional=True`
- Flattens input `(T, N, bands, C, freq)` → `(T, N, features)`
- Uses `output_proj` linear layer to project bidirectional output back to `hidden_size`
- Applies LayerNorm + ReLU

**GRUEncoder**
- Bidirectional GRU encoder
- Same architecture pattern as LSTMEncoder
- Uses `nn.GRU` instead of `nn.LSTM`

**CNNLSTMEncoder**
- Hybrid CNN+LSTM architecture
- Parameters: `in_features`, `cnn_channels=(64, 128)`, `kernel_size=3`, `hidden_size=128`, `num_layers=1`
- 1D CNN layers with Conv1d + ReLU + BatchNorm1d blocks
- Followed by bidirectional LSTM
- Output projection + LayerNorm + ReLU

#### 2. Lightning Modules (`emg2qwerty/lightning.py`)

Added three new LightningModule classes:

**LSTMCTCModule**
- Uses `SpectrogramNorm` + `LSTMEncoder` + `Linear` classifier
- CTC loss with greedy decoding
- Hyperparams: `hidden_size`, `num_layers`, `dropout`, `electrode_channels=16`

**GRUCTCModule**
- Same structure as LSTMCTCModule
- Uses `GRUEncoder` instead of LSTM

**CNNLSTMCTCModule**
- Uses `SpectrogramNorm` + `CNNLSTMEncoder` + `Linear` classifier
- Hyperparams: `cnn_channels`, `kernel_size`, `lstm_hidden`, `lstm_layers`

#### 3. Data Augmentation Transforms (`emg2qwerty/transforms.py`)

Added four new transform classes:

**AddGaussianNoise**
- Adds Gaussian noise with configurable `std` (default 0.1)

**SelectChannels**
- Selects subset of electrode channels via tuple of indices
- For ablation studies on channel count

**Downsample**
- Downsamples EMG signal along time axis by integer factor
- For ablation studies on sampling rate

**RandomChannelDropout**
- Randomly drops electrode channels during training
- Configurable `drop_prob` (default 0.1)

#### 4. Model Configurations

| Config | Description |
|--------|-------------|
| `cnn_lstm_ctc.yaml` | CNN+LSTM hybrid, `cnn_channels=[64,128]`, `lstm_hidden=128`, `lstm_layers=1` |
| `gru_ctc.yaml` | GRU-based, `hidden_size=256`, `num_layers=2`, `dropout=0.3` |
| `lstm_ctc.yaml` | LSTM-based, `hidden_size=256`, `num_layers=2`, `dropout=0.3` |
| `lstm_1channel.yaml` | LSTM with 1 channel |
| `lstm_4channels.yaml` | LSTM with 4 channels |
| `lstm_8channels.yaml` | LSTM with 8 channels |
| `lstm_augmented.yaml` | LSTM with enhanced augmentation (SpecAugment, GaussianNoise, band rotation, temporal jitter) |
| `lstm_downsample_2x.yaml` | LSTM with 2x downsampling |
| `lstm_downsample_4x.yaml` | LSTM with 4x downsampling |

#### 5. Training Data Configurations

| Config | Description |
|--------|-------------|
| `single_user_25percent.yaml` | 4 training sessions (25% of data) |
| `single_user_50percent.yaml` | 8 training sessions (50% of data) |
| `single_user.yaml` | Modified (details not shown) |

#### 6. Notebook

- `Colab_setup.ipynb` - Significantly expanded (22,170+ lines added)

---

## Branch: `origin/ahjc`

**Commit:** `90cd96c`  
**Author:** ahjcgit (ahjcuniv@gmail.com)  
**Date:** Mon Mar 9 01:29:08 2026 -0700  
**Message:** "Added code implementation for multiple models and run data"

### Files Modified/Added

| File | Status |
|------|--------|
| `.gitignore` | Modified |
| `README.md` | Modified |
| `config/base.yaml` | Modified |
| `config/experiment/aug_logspec_only.yaml` | Added |
| `config/experiment/aug_logspec_specaug.yaml` | Added |
| `config/experiment/channels_4.yaml` | Added |
| `config/experiment/channels_8.yaml` | Added |
| `config/experiment/downsample_2x.yaml` | Added |
| `config/experiment/downsample_4x.yaml` | Added |
| `config/experiment/fast.yaml` | Added |
| `config/experiment/train_frac_25.yaml` | Added |
| `config/experiment/train_frac_50.yaml` | Added |
| `config/model/cnn_rnn_ctc.yaml` | Added |
| `config/model/gru_ctc.yaml` | Added |
| `config/model/lstm_ctc.yaml` | Added |
| `config/transforms/log_spectrogram.yaml` | Modified |
| `config/transforms/logspec_only.yaml` | Added |
| `config/transforms/logspec_specaug.yaml` | Added |
| `emg2qwerty/decoder.py` | Modified |
| `emg2qwerty/lightning.py` | Modified |
| `emg2qwerty/train.py` | Modified |
| `emg2qwerty/transforms.py` | Modified |
| `requirements-gpu.txt` | Added |
| `runs/03-36-27/*` | Added (training logs and results) |
| `scripts/summarize_runs.py` | Added |

### Work Done

#### 1. Model Implementations (`emg2qwerty/lightning.py`)

Added two new LightningModule classes:

**LSTMCTCModule**
- Frontend: `SpectrogramNorm` + `MultiBandRotationInvariantMLP` + `Flatten`
- RNN: `nn.LSTM` with configurable `rnn_hidden_size`, `rnn_num_layers`, `rnn_bidirectional`, `rnn_dropout`
- Output: `Linear` + `LogSoftmax`
- Uses existing `utils.instantiate_optimizer_and_scheduler`

**CNNRNNCTCModule**
- Frontend: Same as LSTMCTCModule
- Temporal Conv: Stack of 1D Conv1d + ReLU + Dropout blocks
  - Configurable: `conv_channels`, `conv_layers`, `conv_kernel_size`, `conv_dropout`
- RNN: `nn.GRU` (not LSTM) for temporal modeling
- Output: `Linear` + `LogSoftmax`

#### 2. Data Transforms (`emg2qwerty/transforms.py`)

Added two new transform classes:

**SelectEMGChannels**
- Selects fixed subset of EMG channels
- Parameters: `n_channels` (None = keep all), `channel_dim=-1`
- Uses `tensor.narrow()` for selection

**TemporalDownsample**
- Downsamples along temporal axis by integer factor
- Parameters: `factor=1`, `time_dim=0`
- Uses slicing with `step=factor`

#### 3. Experiment Configurations (`config/experiment/`)

**fast.yaml**
- `batch_size: 16`
- `num_workers: 4`
- `max_epochs: 300`
- `gradient_clip_val: 1.0`
- Callbacks: LR monitor, ModelCheckpoint, EarlyStopping (patience=25)

**Ablation Configs:**
| Config | Purpose |
|--------|---------|
| `aug_logspec_only.yaml` | Log spectrogram only (no SpecAugment) |
| `aug_logspec_specaug.yaml` | Log spectrogram + SpecAugment |
| `channels_4.yaml` | 4 channels per wrist |
| `channels_8.yaml` | 8 channels per wrist |
| `downsample_2x.yaml` | 2x temporal downsampling |
| `downsample_4x.yaml` | 4x temporal downsampling |
| `train_frac_25.yaml` | 25% training data |
| `train_frac_50.yaml` | 50% training data |

#### 4. Model Configurations (`config/model/`)

| Config | Model | Key Params |
|--------|-------|------------|
| `gru_ctc.yaml` | GRUCTCModule | `mlp_features=[256]`, `rnn_hidden_size=256`, `rnn_num_layers=1`, `rnn_bidirectional=true` |
| `lstm_ctc.yaml` | LSTMCTCModule | `mlp_features=[256]`, `rnn_hidden_size=256`, `rnn_num_layers=1`, `rnn_bidirectional=true` |
| `cnn_rnn_ctc.yaml` | CNNRNNCTCModule | `conv_channels=256`, `conv_layers=2`, `conv_kernel_size=5`, `rnn_hidden_size=256`, `rnn_num_layers=1` |

#### 5. Transform Configurations (`config/transforms/`)

**logspec_only.yaml**
- ToTensor → SelectEMGChannels → TemporalDownsample → LogSpectrogram
- No augmentation

**logspec_specaug.yaml**
- Same as above + SpecAugment (time/frequency masking)

#### 6. Results (`runs/03-36-27/`)

Executed 4 training jobs with results:

| Job | Model | Val CER | Test CER | Test IER | Test DER | Test SER |
|-----|-------|---------|----------|----------|----------|----------|
| job0 | tds_conv_ctc | 19.76% | 22.89% | 3.05% | 3.00% | 16.84% |
| job1 | gru_ctc | 21.09% | 21.24% | 3.61% | 2.98% | 14.65% |
| job2 | lstm_ctc | 20.12% | 21.31% | 2.90% | 3.57% | 14.85% |
| job3 | cnn_rnn_ctc | 21.71% | 23.32% | 4.78% | 2.68% | 15.86% |

#### 7. Utility Script (`scripts/summarize_runs.py`)

- Aggregates CER metrics from Hydra run directories
- Outputs CSV summary
- Optional plotting of test CER comparison

#### 8. Documentation (`README.md`)

Updated README with:
- Architecture diagram (Mermaid flowchart)
- Quickstart commands for all models
- Ablation experiment commands
- Expected impact descriptions for each model type

---

## Summary Comparison

| Aspect | origin/main | origin/ahjc |
|--------|-------------|-------------|
| Models Added | 3 (LSTM, GRU, CNN+LSTM) | 2 (LSTM, CNN+GRU) |
| Encoder Location | `modules.py` | Inline in `lightning.py` |
| Architecture Style | Direct encoder on flattened input | Frontend (SpectrogramNorm + MLP) + RNN |
| Transforms Added | 4 (GaussianNoise, SelectChannels, Downsample, ChannelDropout) | 2 (SelectEMGChannels, TemporalDownsample) |
| Ablation Configs | Via separate model configs | Via experiment overrides |
| Training Results | Not included | 4 model results logged |
| Documentation | Notebook only | README + script |
