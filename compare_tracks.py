"""
compare_tracks — CLI and function for pairwise audio attribution.

Usage:
    python compare.py track_a.wav track_b.wav
    python compare.py track_a.wav track_b.wav --checkpoint checkpoints/best.pt
    python compare.py track_a.wav track_b.wav --use_lyrics

    from compare import compare_tracks
    result = compare_tracks("track_a.wav", "track_b.wav")
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

from feature_extraction.classical import extract_classical_features, extract_classical_chunked
from feature_extraction.ai_detection import extract_ai_detection_features
from feature_extraction.fakeprint import extract_fakeprint
from model.network import AttributionModel

TARGET_SR = 16000
CACHE_DIR = Path("feature_cache")
DEFAULT_CHECKPOINT = Path("checkpoints/best.pt")


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _extract_or_load(audio_path: Path) -> dict:
    """Extract features for a single track, using cache if available."""
    fid = _file_hash(audio_path)
    cache_path = CACHE_DIR / f"{fid}.npz"

    if cache_path.exists():
        return dict(np.load(cache_path, allow_pickle=True))

    y, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)

    features = {}
    features["classical"] = extract_classical_features(y, sr=TARGET_SR)
    features["classical_chunks"] = extract_classical_chunked(y, sr=TARGET_SR)
    features["ai_detection"] = extract_ai_detection_features(y, sr=TARGET_SR)
    features["fakeprint"] = extract_fakeprint(y, sr=TARGET_SR)

    try:
        from feature_extraction.embeddings import (
            extract_mert_embedding,
            extract_clap_embedding,
        )
        features["mert"] = extract_mert_embedding(y, sr=TARGET_SR)
        features["clap"] = extract_clap_embedding(y, sr=TARGET_SR)
    except Exception:
        pass  # embeddings are optional

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **features)

    return features


def _extract_lyric_embedding(audio_path: Path, feat: dict) -> np.ndarray:
    """Return lyric embedding from cache or transcribe via Whisper."""
    if "lyric_embedding" in feat:
        return feat["lyric_embedding"]
    try:
        from feature_extraction.lyrics import extract_lyrics, extract_lyric_embedding
        text = extract_lyrics(str(audio_path))
        return extract_lyric_embedding(text)
    except Exception:
        return np.zeros(384, dtype=np.float32)


def _features_to_batch(
    feat_a: dict,
    feat_b: dict,
    device: torch.device,
    use_lyrics: bool = False,
    audio_a: Path = None,
    audio_b: Path = None,
) -> dict:
    """Convert two feature dicts into a model-ready batch dict."""
    def _to_tensor(arr):
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    chunks_a = feat_a["classical_chunks"]
    chunks_b = feat_b["classical_chunks"]

    batch = {
        "chunks_a": _to_tensor(chunks_a),
        "mask_a": torch.ones(1, chunks_a.shape[0], device=device),
        "chunks_b": _to_tensor(chunks_b),
        "mask_b": torch.ones(1, chunks_b.shape[0], device=device),
        "ai_det_a": _to_tensor(feat_a["ai_detection"]),
        "ai_det_b": _to_tensor(feat_b["ai_detection"]),
        "fakeprint_a": _to_tensor(feat_a["fakeprint"]),
        "fakeprint_b": _to_tensor(feat_b["fakeprint"]),
    }

    if "mert" in feat_a and "mert" in feat_b:
        batch["mert_a"] = _to_tensor(feat_a["mert"])
        batch["mert_b"] = _to_tensor(feat_b["mert"])
    if "clap" in feat_a and "clap" in feat_b:
        batch["clap_a"] = _to_tensor(feat_a["clap"])
        batch["clap_b"] = _to_tensor(feat_b["clap"])

    if use_lyrics:
        emb_a = _extract_lyric_embedding(audio_a, feat_a)
        emb_b = _extract_lyric_embedding(audio_b, feat_b)
        batch["lyric_emb_a"] = torch.from_numpy(emb_a).unsqueeze(0).to(device)
        batch["lyric_emb_b"] = torch.from_numpy(emb_b).unsqueeze(0).to(device)

    return batch


def compare_tracks(
    track_a: str,
    track_b: str,
    checkpoint: str = None,
    use_lyrics: bool = False,
) -> dict:
    """Compare two audio tracks and return attribution scores.

    Args:
        track_a: Path to first audio file.
        track_b: Path to second audio file.
        checkpoint: Path to model checkpoint. Uses default if None.
        use_lyrics: If True, also compute lyric embeddings via Whisper.

    Returns:
        dict with similarity_score, ai_index_a, ai_index_b, attribution_score.
    """
    track_a = Path(track_a)
    track_b = Path(track_b)
    checkpoint = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    feat_a = _extract_or_load(track_a)
    feat_b = _extract_or_load(track_b)

    if not checkpoint.exists():
        return _heuristic_compare(feat_a, feat_b)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    config = ckpt.get("config", {})
    model_use_lyrics = config.get("use_lyrics", False)

    model = AttributionModel(
        hidden_dim=config.get("hidden_dim", 256),
        feature_set=config.get("feature_set", "advanced"),
        use_lyrics=model_use_lyrics,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    batch = _features_to_batch(
        feat_a, feat_b, device,
        use_lyrics=(use_lyrics and model_use_lyrics),
        audio_a=track_a,
        audio_b=track_b,
    )

    preds = model.predict(batch)

    # Lyric gate: if lyric cosine similarity is very high, pull attribution up.
    # This handles cases where the model's attribution head is uncertain but lyrics
    # are an unambiguous signal (e.g. prompt-hijacked tracks with identical lyrics).
    _LYRIC_GATE_THRESHOLD = 0.78
    _LYRIC_GATE_STRENGTH = 0.5

    lyric_cos = None
    if "lyric_emb_a" in batch and "lyric_emb_b" in batch:
        lyric_cos = float(
            torch.nn.functional.cosine_similarity(
                batch["lyric_emb_a"], batch["lyric_emb_b"], dim=-1
            ).item()
        )

    attr = float(preds["attribution_score"].item())
    if lyric_cos is not None and lyric_cos > _LYRIC_GATE_THRESHOLD:
        gate = (lyric_cos - _LYRIC_GATE_THRESHOLD) / (1.0 - _LYRIC_GATE_THRESHOLD)
        attr = attr + gate * (1.0 - attr) * _LYRIC_GATE_STRENGTH

    # Extract the raw predictions straight from the model
    result = {
        "similarity_score": round(float(preds["similarity_score"].item()), 4),
        "ai_index_a": round(float(preds["ai_index_a"].item()), 4),
        "ai_index_b": round(float(preds["ai_index_b"].item()), 4),
        "attribution_score": round(float(preds["attribution_score"].item()), 4),
    }
    
    # We could use high lyric similarity to pull attribution up. This is a simple heuristic "lyric gate" that boosts attribution if the lyric cosine similarity is above a threshold. The strength of the boost increases as the lyric similarity approaches 1.0.
    if "lyric_emb_a" in batch and "lyric_emb_b" in batch:
        lyric_cos = float(
            torch.nn.functional.cosine_similarity(
                batch["lyric_emb_a"], batch["lyric_emb_b"], dim=-1
            ).item()
        )
        result["lyric_similarity"] = round(lyric_cos, 4)
        
    return result   


def _heuristic_compare(feat_a: dict, feat_b: dict) -> dict:
    """Baseline heuristic when no trained model is available.

    Uses cosine similarity on classical features and fakeprint peak statistics.
    """
    def _cosine_sim(a, b):
        a, b = a.flatten(), b.flatten()
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-10:
            return 0.0
        return float(np.dot(a, b) / norm)

    def _ai_score_from_fakeprint(fp):
        threshold = np.mean(fp) + 2 * np.std(fp)
        peaks = np.where(fp > threshold)[0]
        if len(peaks) < 2:
            return 0.1
        spacings = np.diff(peaks)
        regularity = float(np.std(spacings) / (np.mean(spacings) + 1e-10))
        ai_prob = max(0.0, min(1.0, 1.0 - regularity))
        peak_density = min(1.0, len(peaks) / 50.0)
        return float(0.6 * ai_prob + 0.4 * peak_density)

    similarity = max(0.0, _cosine_sim(feat_a["classical"], feat_b["classical"]))
    ai_a = _ai_score_from_fakeprint(feat_a["fakeprint"])
    ai_b = _ai_score_from_fakeprint(feat_b["fakeprint"])
    attribution = similarity * max(ai_a, ai_b, 0.3)

    return {
        "similarity_score": round(similarity, 4),
        "ai_index_a": round(ai_a, 4),
        "ai_index_b": round(ai_b, 4),
        "attribution_score": round(attribution, 4),
        "_note": "heuristic (no trained model)",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two audio tracks for attribution"
    )
    parser.add_argument("track_a", help="Path to first audio file")
    parser.add_argument("track_b", help="Path to second audio file")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (default: checkpoints/best.pt)"
    )
    parser.add_argument(
        "--use_lyrics", action="store_true",
        help="Compute lyric embeddings via Whisper + sentence-transformers"
    )
    args = parser.parse_args()

    for path in [args.track_a, args.track_b]:
        if not Path(path).exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    result = compare_tracks(
        args.track_a, args.track_b, args.checkpoint, args.use_lyrics
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
