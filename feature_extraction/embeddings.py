"""
Foundation model embeddings — MERT and CLAP, optimized for variable-length audio.
"""

import math
import numpy as np
import torch
import torchaudio.transforms as T
from typing import Optional
from contextlib import nullcontext

MERT_DIM = 1024   # MERT-v1-330M hidden size
CLAP_DIM = 512    # CLAP projection dimension

_mert_model = None
_mert_processor = None
_clap_model = None
_clap_processor = None
_resamplers = {}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_resampler(src_sr: int, tgt_sr: int):
    key = (src_sr, tgt_sr)
    if key not in _resamplers:
        _resamplers[key] = T.Resample(src_sr, tgt_sr)
    return _resamplers[key]


def _load_mert(device: Optional[torch.device] = None):
    global _mert_model, _mert_processor
    if _mert_model is None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        if device is None:
            device = _get_device()

        _mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        _mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        ).to(device).eval()

    return _mert_model, _mert_processor


def _load_clap(device: Optional[torch.device] = None):
    global _clap_model, _clap_processor
    if _clap_model is None:
        from transformers import ClapModel, ClapProcessor

        if device is None:
            device = _get_device()

        _clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
        _clap_model = ClapModel.from_pretrained(
            "laion/larger_clap_music"
        ).to(device).eval()

    return _clap_model, _clap_processor


def _chunk_audio(
    y: np.ndarray,
    sr: int,
    chunk_sec: float,
    hop_sec: Optional[float] = None,
) -> np.ndarray:
    chunk = int(chunk_sec * sr)
    hop = int((hop_sec or chunk_sec) * sr)

    if len(y) == 0:
        return np.zeros((1, chunk), dtype=np.float32)

    chunks = []
    for i in range(0, max(len(y) - chunk + 1, 1), hop):
        segment = y[i:i + chunk]
        if len(segment) < chunk:
            segment = np.pad(segment, (0, chunk - len(segment)))
        chunks.append(segment)

    return np.stack(chunks)


def extract_mert_embeddings_sequence(
    y: np.ndarray,
    sr: int = 16000,
    device: Optional[torch.device] = None,
    chunk_sec: float = 10.0,
    hop_sec: Optional[float] = None,
    batch_size: int = 4,
) -> np.ndarray:

    if device is None:
        device = _get_device()

    model, processor = _load_mert(device)
    target_sr = processor.sampling_rate

    if sr != target_sr:
        resampler = _get_resampler(sr, target_sr)
        y = resampler(torch.from_numpy(y).float()).numpy()

    chunks = _chunk_audio(y, target_sr, chunk_sec, hop_sec)
    embeddings = []

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch_chunks = list(chunks[i:i + batch_size])

            inputs = processor(
                batch_chunks,
                sampling_rate=target_sr,
                return_tensors="pt"
            ).to(device)

            ctx = torch.autocast("cuda", torch.float16) if use_amp else nullcontext()

            with ctx:
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                layer = hidden_states[-1]
                batch_embs = layer.mean(dim=1)

            embeddings.append(batch_embs.cpu().numpy().astype(np.float32))

            del inputs, outputs, batch_embs

    return np.concatenate(embeddings, axis=0)


def extract_clap_embeddings_sequence(
    y: np.ndarray,
    sr: int = 16000,
    device: Optional[torch.device] = None,
    chunk_sec: float = 10.0,
    hop_sec: Optional[float] = None,
    batch_size: int = 4,
) -> np.ndarray:

    if device is None:
        device = _get_device()

    model, processor = _load_clap(device)
    target_sr = getattr(processor.feature_extractor, "sampling_rate", 48000)

    if sr != target_sr:
        resampler = _get_resampler(sr, target_sr)
        y = resampler(torch.from_numpy(y).float()).numpy()

    chunks = _chunk_audio(y, target_sr, chunk_sec, hop_sec)
    embeddings = []

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch_chunks = list(chunks[i:i + batch_size])

            inputs = processor(
                audio=batch_chunks,
                sampling_rate=target_sr,
                return_tensors="pt"
            ).to(device)

            ctx = torch.autocast("cuda", torch.float16) if use_amp else nullcontext()

            with ctx:
                out = model.get_audio_features(**inputs)

                if isinstance(out, torch.Tensor):
                    batch_embs = out
                elif hasattr(out, "audio_embeds"):
                    batch_embs = out.audio_embeds
                elif hasattr(out, "pooler_output"):
                    batch_embs = out.pooler_output
                    if batch_embs.shape[-1] != CLAP_DIM:
                        if hasattr(model, "audio_projection") and model.audio_projection is not None:
                            batch_embs = model.audio_projection(batch_embs)
                else:
                    batch_embs = out[0]

                if batch_embs.shape[-1] != CLAP_DIM:
                    raise RuntimeError(
                        f"Unexpected CLAP embedding dim: {batch_embs.shape}"
                    )

            embeddings.append(batch_embs.cpu().numpy().astype(np.float32))

            del inputs, out, batch_embs

    return np.concatenate(embeddings, axis=0)


def extract_mert_embedding(
    y: np.ndarray,
    sr: int = 16000,
    device: Optional[torch.device] = None,
    chunk_sec: float = 10.0,
    hop_sec: Optional[float] = None,
) -> np.ndarray:
    seq = extract_mert_embeddings_sequence(
        y, sr=sr, device=device, chunk_sec=chunk_sec, hop_sec=hop_sec
    )
    return seq.mean(axis=0).astype(np.float32)


def extract_clap_embedding(
    y: np.ndarray,
    sr: int = 16000,
    device: Optional[torch.device] = None,
    chunk_sec: float = 10.0,
    hop_sec: Optional[float] = None,
) -> np.ndarray:
    seq = extract_clap_embeddings_sequence(
        y, sr=sr, device=device, chunk_sec=chunk_sec, hop_sec=hop_sec
    )
    return seq.mean(axis=0).astype(np.float32)


def unload_models():
    """Free memory by unloading cached models."""
    global _mert_model, _mert_processor, _clap_model, _clap_processor

    _mert_model = None
    _mert_processor = None
    _clap_model = None
    _clap_processor = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
