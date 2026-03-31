# AI Music Attribution

Pairwise attribution system for AI-generated music. Given two audio tracks, it outputs a similarity score, an AI detection index per track, and an attribution score indicating whether one track is a derivative of the other.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `librosa`, `torch`, `transformers`, `numpy`, `soundfile`, `tqdm`, `sentence-transformers`, `openai-whisper`, `laion-clap`, `datasets`.

All audio must be **16 kHz, mono, PCM_16 WAV**:

```bash
python validate_data_format.py
```

## Datasets

Place datasets under `data/`. Fetch scripts are in `data/`:

```bash
python data/fetch_sonics.py
python data/fetch_fmc.py
python data/fetch_smp.py

python data/fetch_echoes.py
python data/fetch_echoes_fma.py
python data/process_echoes.py
python data/process_fma.py
```

Dataset paths and schemas: `data_config.yaml`.

## Feature Extraction

Features are precomputed and stored as `.npz` files per track.

**CPU-only** (classical + AI detection + fakeprint):

```bash
python precompute_all.py --mode cpu
```

**Full** (adds MERT + CLAP embeddings — GPU recommended):

```bash
python precompute_all.py --mode all
```

**Merge caches** if previous were done separately:

```bash
python merge_caches.py 
```

**Add lyric embeddings** to an existing cache:

```bash
python add_lyrics_to_cache.py --cache_dir feature_cache_cpu
```

SONICS lyrics are loaded from the metadata CSVs directly. All other tracks are transcribed via Whisper. The cache directory must contain `feature_metadata.csv` (written by `precompute_all.py`).

## Dataset Label Design

Pair labels are generated automatically during dataset construction with two refinements over naive binary labels:

**Metadata-aware soft similarity for same-generator negatives.** Two AI tracks from the same generator (different content) are not truly zero-similar — they share generation artifacts and stylistic tendencies. The dataset uses genre and topic metadata from `extra_fake_metadata.csv` to assign soft similarity targets: same genre -> 0.15, same topic -> 0.10, otherwise -> 0.05. Cross-generator negatives remain at 0.0.

**Cross-dataset negative pairs.** In addition to within-dataset negatives, the pipeline builds two cross-dataset pair types (up to 200 each):

- SONICS real ↔ FMC fake: unrelated real track vs arbitrary AI clip
- SONICS fake ↔ FMC fake: AI tracks from different generators and corpora

These extend negative diversity and prevent the model from over-fitting to within-dataset artifact distributions.

---

## Training

```bash
# Default: advanced feature set (classical + artifacts + MERT/CLAP)
python -m model.train

# With lyric encoder (requires lyric embeddings in cache)
python -m model.train --use_lyrics --feature_set mix

# CPU-only feature set
python -m model.train --feature_set basic

# Ablation: trains basic and advanced, writes ablation_results.json
python -m model.train --ablation
```

Key training arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--data_dir` | `data` | Dataset root |
| `--cache_dir` | `feature_cache_cpu` | Feature cache directory |
| `--checkpoint_dir` | `checkpoints` | Where to save checkpoints |
| `--epochs` | 50 | Max training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--patience` | 10 | Early stopping patience (0 = off) |
| `--hidden_dim` | 256 | Model hidden dimension |
| `--feature_set` | `advanced` | `basic`, `advanced`, `mix` |
| `--use_lyrics` | off | Enable lyric encoder |
| `--neg_ratio` | 1.5 | Negative-to-positive pair ratio |

Best checkpoint saved to `checkpoints/<feature_set>/best.pt`.

## Inference

Compare two tracks:

```bash
python compare_tracks.py track_a.wav track_b.wav
```

With a specific checkpoint and lyric analysis:

```bash
python compare_tracks.py track_a.wav track_b.wav \
    --checkpoint checkpoints/advanced/best.pt \
    --use_lyrics
```

Output:

```json
{
  "similarity_score": 0.87,
  "ai_index_a": 0.12,
  "ai_index_b": 0.94,
  "attribution_score": 0.85
}
```

If no checkpoint is found, a heuristic fallback (cosine similarity on classical features + fakeprint peak regularity) runs automatically.

### Python API

```python
from compare import compare_tracks

result = compare_tracks("track_a.wav", "track_b.wav", use_lyrics=True)
print(result["attribution_score"])
```

## Score Interpretation

| Case | Similarity | AI Index A | AI Index B | Attribution |
| --- | --- | --- | --- | --- |
| Real vs AI derivative | High | Low | High | High |
| Real vs unrelated AI | Low | Low | High | Low |
| AI vs AI (same origin) | High | High | High | High |
| AI vs AI (unrelated) | Low | High | High | Low |
| Real vs cover | High | Low | Low | High |

## Project Structure

```text
compare_tracks.py        — inference CLI and Python API
precompute_all.py        — batch feature extraction
add_lyrics_to_cache.py   — add Whisper+SBERT lyric embeddings to cache
validate_data_format.py  — audio format checker
feature_extraction/
    classical.py         — 432-d classical audio features
    ai_detection.py      — 22-d phase/spectral/Fourier AI indicators
    fakeprint.py         — 897-d deconvolution artifact fingerprint
    embeddings.py        — MERT (1024-d) + CLAP (512-d) embeddings
    lyrics.py            — Whisper transcription + SBERT embeddings
model/
    dataset.py           — pair construction and dataset classes
    network.py           — model architecture
    losses.py            — multi-task loss
    train.py             — training loop
data/                    — dataset fetch and processing scripts
eda/                     — EDA notebooks and figures
```
