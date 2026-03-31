"""
Feature extraction module for AI-Original Pairwise Audio Similarity.

Feature Groups (designed for TWO separate downstream heads):

═══════════════════════════════════════════════════════════════════════════
SIMILARITY FEATURES — used by the pairwise similarity head.
Must be INVARIANT to generation method (no AI artifact leakage).
═══════════════════════════════════════════════════════════════════════════

Group 1: Timbral (MFCCs + deltas)                         →  120 dims
Group 2: Spectral shape (mel-spectrogram stats)            →  256 dims
Group 3: Melodic/harmonic (chroma + tonnetz)               →   36 dims
Group 4: Spectral contrast                                 →   14 dims
Group 5: Temporal dynamics (ZCR, RMS, tempo, onset rate)   →    6 dims
Group 6: MERT embedding (track-level, mean-pooled)         → 1024 dims
Group 7: CLAP embedding (track-level)                      →  512 dims
                                                    ─────────────────
                                    Similarity total: 1968 dims
    (of which 432 are classical hand-crafted, 1536 are learned embeddings)

═══════════════════════════════════════════════════════════════════════════
AI-DETECTION FEATURES — used by the per-track AI-index head.
Detect artifacts from neural vocoders / generators.
═══════════════════════════════════════════════════════════════════════════

Group 8:  Fakeprint (Deezer/Afchar et al. ISMIR 2025)     →  897 dims
          Average spectrum → subtract lower envelope →
          extract 1-8 kHz band peaks. This is the core
          feature from the best-paper ISMIR 2025.
          Output: raw fakeprint vector (n_bins in 1-8kHz)     897 dims
                                                              (sr=16kHz,
                                                              n_fft=2048)

Group 9:  Phase continuity features                        →    7 dims
          - Phase Continuity Index (PCI)                      1
          - Phase deviation (mean, std)                       2
          - Instantaneous freq stability (mean, std)          2
          - Group delay deviation (mean, std)                 2

Group 10: Spectral AI indicators                           →    5 dims
          - Harmonic-to-Noise Ratio (HNR)                     1
          - Spectral flatness (mean, std)                     2
          - High-freq rolloff ratio (85th/95th)               1
          - SSM novelty score                                 1

Group 11: Fourier artifact summary statistics              →   10 dims
          Peak-to-background ratios at 2×,4×,8× upsampling   3
          Peak counts at 2×,4×,8×                             3
          Peak regularity (CV of inter-peak spacing)          1
          Max peak-to-background ratio                        1
          Spectral periodicity                                1
          Artifact energy ratio                               1
                                                    ─────────────────
                                   AI-detection total:  919 dims

═══════════════════════════════════════════════════════════════════════════
DESIGN RATIONALE
═══════════════════════════════════════════════════════════════════════════

Why two separate feature sets?
  The assignment requires a system where:
    - Real + AI derivative of same song → HIGH similarity, different AI index
    - Two unrelated AI songs from same generator → LOW similarity, same AI index
  If AI-artifact features leak into the similarity computation, two unrelated
  Suno songs will appear "similar" because they share the same vocoder
  watermark. Decoupling prevents this.

Why fakeprints (Group 8)?
  Afchar et al. (ISMIR 2025, best paper) proved mathematically that
  deconvolution layers in neural vocoders produce periodic spectral peaks
  at frequencies n·fs for stride factor k. These depend ONLY on the model
  architecture (stride config), not on training data or weights. A simple
  logistic regression on fakeprints matches deep-learning detectors.
  The method: compute average spectrum, subtract lower envelope (sliding
  window local minima), analyze 1-8 kHz band where artifacts are prominent.

Why MERT + CLAP together?
  MERT (self-supervised on music, 24kHz, masked acoustic modeling) captures
  tonal/rhythmic/structural patterns. CLAP (contrastive language-audio,
  48kHz) captures cross-modal semantics (genre, mood, instrumentation).
  They have different cluster geometries — combining them improves
  similarity discrimination.

Why not include MERT/CLAP in the AI-detection features?
  Foundation model embeddings capture high-level musical semantics. They
  are not designed to detect low-level vocoder artifacts. Using them for
  AI detection would make the detector less interpretable and potentially
  less robust to generator changes.
"""

from .classical import extract_classical_features, CLASSICAL_DIM
from .ai_detection import extract_ai_detection_features, AI_DETECTION_DIM
from .fakeprint import extract_fakeprint, FAKEPRINT_DIM
from .precompute import precompute_all_features

# Embeddings require torch — import lazily
MERT_DIM = 1024
CLAP_DIM = 512

def extract_mert_embedding(*args, **kwargs):
    from .embeddings import extract_mert_embedding as _fn
    return _fn(*args, **kwargs)

def extract_clap_embedding(*args, **kwargs):
    from .embeddings import extract_clap_embedding as _fn
    return _fn(*args, **kwargs)

__all__ = [
    "extract_classical_features",
    "extract_ai_detection_features",
    "extract_fakeprint",
    "extract_mert_embedding",
    "extract_clap_embedding",
    "precompute_all_features",
    "CLASSICAL_DIM",
    "AI_DETECTION_DIM",
    "FAKEPRINT_DIM",
    "MERT_DIM",
    "CLAP_DIM",
]
