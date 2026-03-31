"""
Lyric transcription and similarity via Whisper + sentence-transformers.

Extracts lyrics from audio tracks using Whisper, computes sentence embeddings,
and provides pairwise lyric similarity as an additional model feature.

Usage:
    from feature_extraction.lyrics import extract_lyrics, compute_lyric_similarity

    text_a = extract_lyrics("track_a.wav")
    text_b = extract_lyrics("track_b.wav")
    sim = compute_lyric_similarity(text_a, text_b)
"""

import numpy as np
from pathlib import Path
from typing import Optional

_whisper_model = None
_sbert_model = None

LYRIC_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _load_whisper(model_size: str = "base"):
    """Lazy-load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def _load_sbert(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load sentence-transformers model."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model


def extract_lyrics(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
) -> str:
    """Transcribe lyrics from an audio file using Whisper.

    Args:
        audio_path: Path to audio file (any format Whisper supports).
        model_size: Whisper model size (tiny/base/small/medium/large).
        language: Force language (None = auto-detect).

    Returns:
        Transcribed text string.
    """
    model = _load_whisper(model_size)
    opts = {"fp16": False}
    if language:
        opts["language"] = language
    result = model.transcribe(str(audio_path), **opts)
    return result["text"].strip()


def extract_lyric_embedding(text: str) -> np.ndarray:
    """Compute sentence embedding for a text string.

    Returns:
        np.ndarray of shape (384,) — sentence embedding.
    """
    if not text or text.strip() == "":
        return np.zeros(LYRIC_EMBEDDING_DIM, dtype=np.float32)

    model = _load_sbert()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32)


def compute_lyric_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two lyric transcriptions.

    Returns:
        float in [-1, 1], typically [0, 1] for music lyrics.
    """
    emb_a = extract_lyric_embedding(text_a)
    emb_b = extract_lyric_embedding(text_b)

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def extract_lyrics_and_embedding(
    audio_path: str,
    model_size: str = "base",
) -> dict:
    """Extract both lyrics text and embedding from audio.

    Returns:
        dict with 'lyrics' (str) and 'lyric_embedding' (np.ndarray).
    """
    text = extract_lyrics(audio_path, model_size=model_size)
    embedding = extract_lyric_embedding(text)
    return {
        "lyrics": text,
        "lyric_embedding": embedding,
    }


def compute_pairwise_lyric_similarity(
    audio_path_a: str,
    audio_path_b: str,
    cache_dir: Optional[str] = None,
) -> float:
    """End-to-end: transcribe both tracks and compute lyric similarity.

    If cache_dir is provided, will cache transcriptions to avoid re-processing.
    """
    if cache_dir:
        import hashlib
        cache_dir = Path(cache_dir)

        def _cached_lyrics(path):
            h = hashlib.sha256(str(path).encode()).hexdigest()[:16]
            cache_file = cache_dir / f"lyrics_{h}.txt"
            if cache_file.exists():
                return cache_file.read_text()
            text = extract_lyrics(path)
            cache_file.write_text(text)
            return text

        text_a = _cached_lyrics(audio_path_a)
        text_b = _cached_lyrics(audio_path_b)
    else:
        text_a = extract_lyrics(audio_path_a)
        text_b = extract_lyrics(audio_path_b)

    return compute_lyric_similarity(text_a, text_b)


def unload_models():
    """Free memory by unloading cached models."""
    global _whisper_model, _sbert_model
    _whisper_model = None
    _sbert_model = None
