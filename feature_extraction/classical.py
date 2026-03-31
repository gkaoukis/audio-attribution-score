"""
Classical audio features for track-level and chunk-level representation.
"""

import numpy as np
import librosa

CLASSICAL_DIM = 432


def extract_classical_features(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 20,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract all classical similarity features from a waveform.

    Args:
        y: Audio waveform, mono, float32.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        n_mels: Number of mel bands.
        n_fft: FFT window size.
        hop_length: Hop length in samples.

    Returns:
        np.ndarray of shape (432,) — fixed-length feature vector.
    """
    features = []

    # Compute magnitude and power spectrograms once and reuse below.
    S_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_power = S_mag ** 2

    # Mel-spectrogram stats (256-d)
    mel = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend([np.mean(mel_db, axis=1), np.std(mel_db, axis=1)])

    # MFCCs + deltas (120-d)
    # mfcc(S=...) expects a log-power mel spectrogram — mel_db is correct here.
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for m in [mfcc, mfcc_delta, mfcc_delta2]:
        features.extend([np.mean(m, axis=1), np.std(m, axis=1)])

    # Chroma + Tonnetz (36-d)
    # chroma_stft(S=...) expects a power spectrogram — S_power is correct here.
    chroma = librosa.feature.chroma_stft(S=S_power, sr=sr)
    features.extend([np.mean(chroma, axis=1), np.std(chroma, axis=1)])

    # Pass pre-computed chroma to avoid CQT/harmonic separation overhead.
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    features.extend([np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)])

    # Spectral contrast (14-d)
    contrast = librosa.feature.spectral_contrast(S=S_mag, sr=sr)
    features.extend([np.mean(contrast, axis=1), np.std(contrast, axis=1)])

    # Temporal dynamics (6-d)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    features.append(np.array([np.mean(zcr), np.std(zcr)]))

    rms = librosa.feature.rms(S=S_mag)
    features.append(np.array([np.mean(rms), np.std(rms)]))

    # Bug fix: onset_strength expects a log-power spectrogram, not linear power.
    # Passing S_power inflates the envelope magnitude ~350x and causes onset_detect
    # to fire ~65% more false positives, corrupting onset_rate.
    # mel_db is already computed and is the correct representation.
    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr)

    # tempo returns shape (1,)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    features.append(np.array([tempo[0]]))

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    onset_rate = len(onset_frames) / max(duration, 0.1)
    features.append(np.array([onset_rate]))

    result = np.concatenate(features).astype(np.float32)
    assert result.shape[0] == CLASSICAL_DIM, f"Expected {CLASSICAL_DIM} dims, got {result.shape[0]}"
    return result


def extract_classical_chunked(
    y: np.ndarray,
    sr: int = 16000,
    chunk_sec: float = 10.0,
    hop_sec: float = 5.0,
    **kwargs,
) -> np.ndarray:
    """Extract classical features per chunk using librosa.util.frame.

    Pads the signal so the final chunk always covers the end of the track.
    Without padding, librosa.util.frame silently drops the tail (up to
    chunk_sec - 1 samples), meaning the last segment of the track is never
    seen by the ChunkEncoder.
    """
    chunk_samples = int(chunk_sec * sr)
    hop_samples = int(hop_sec * sr)

    if len(y) < chunk_samples:
        y = np.pad(y, (0, chunk_samples - len(y)))
    else:
        # Pad so the last window is always included, even if len(y) isn't
        # aligned to hop_samples after the first chunk.
        remainder = (len(y) - chunk_samples) % hop_samples
        if remainder != 0:
            y = np.pad(y, (0, hop_samples - remainder))

    frames = librosa.util.frame(y, frame_length=chunk_samples, hop_length=hop_samples)

    # frames shape is (chunk_samples, n_frames); iterate over columns.
    chunks = []
    for chunk in frames.T:
        chunk = np.ascontiguousarray(chunk)
        chunks.append(extract_classical_features(chunk, sr=sr, **kwargs))

    return np.stack(chunks)