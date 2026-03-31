"""
Single-file feature precomputation.

Provides precompute_all_features() for extracting all feature groups
from a single audio waveform and returning them as a dict of ndarrays.
"""

import numpy as np

from .classical import extract_classical_features, extract_classical_chunked
from .ai_detection import extract_ai_detection_features
from .fakeprint import extract_fakeprint


def precompute_all_features(
    y: np.ndarray,
    sr: int = 16000,
    chunk_sec: float = 10.0,
    hop_sec: float = 5.0,
    include_embeddings: bool = False,
) -> dict:
    """Extract all feature groups from a waveform.

    Args:
        y: Audio waveform, mono, float32.
        sr: Sample rate.
        chunk_sec: Chunk length in seconds for chunked extraction.
        hop_sec: Hop length in seconds for overlapping chunks.
        include_embeddings: If True, also extract MERT and CLAP embeddings.

    Returns:
        Dict with keys: classical, classical_chunks, ai_detection, fakeprint,
        and optionally mert_chunks, clap_chunks.
    """
    result = {}

    # Track-level classical features
    result["classical"] = extract_classical_features(y, sr=sr)

    # Per-chunk classical features (10s window, 5s hop)
    result["classical_chunks"] = extract_classical_chunked(
        y, sr=sr, chunk_sec=chunk_sec, hop_sec=hop_sec
    )

    # AI detection features (track-level)
    result["ai_detection"] = extract_ai_detection_features(y, sr=sr)

    # Fakeprint vector (track-level, 897-d)
    result["fakeprint"] = extract_fakeprint(y, sr=sr)

    # Optional: learned embeddings (require GPU)
    if include_embeddings:
        from .embeddings import (
            extract_mert_embeddings_sequence,
            extract_clap_embeddings_sequence,
        )
        result["mert_chunks"] = extract_mert_embeddings_sequence(
            y, sr=sr, chunk_sec=chunk_sec
        )
        result["clap_chunks"] = extract_clap_embeddings_sequence(
            y, sr=sr, chunk_sec=chunk_sec
        )

    return result
