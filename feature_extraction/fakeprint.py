"""
Fakeprint feature extraction.

Isolates generator-specific spectral artifacts in the 1–8 kHz band by
subtracting a sliding-window lower envelope from the time-averaged spectrum.
Based on the approach in Afchar et al., ISMIR 2025.
"""

import numpy as np
import librosa
import scipy.ndimage
from typing import Dict, Any

_DEFAULT_SR = 16000
_DEFAULT_NFFT = 2048
_DEFAULT_HOP = 512
_DEFAULT_ENVELOPE_WIN = 50
_FREQ_LOW = 1000
_FREQ_HIGH = 8000

# At sr=16000, n_fft=2048, 1–8 kHz band: bins 128–1024 = 897 bins.
FAKEPRINT_DIM = 897


def _sliding_min(spectrum: np.ndarray, window: int) -> np.ndarray:
    """Compute lower envelope via sliding-window minimum."""
    return scipy.ndimage.minimum_filter1d(spectrum, size=window, mode='nearest')


def extract_fakeprint_features(
    y: np.ndarray,
    sr: int = _DEFAULT_SR,
    n_fft: int = _DEFAULT_NFFT,
    hop_length: int = _DEFAULT_HOP,
    envelope_window: int = _DEFAULT_ENVELOPE_WIN,
    freq_low: float = _FREQ_LOW,
    freq_high: float = _FREQ_HIGH,
) -> Dict[str, Any]:
    """Extract fakeprint vector and summary statistics for a full-length track.

    Processing the full track (rather than a window) prevents boundary bias
    and smooths transient melodic content, leaving persistent artifact spikes.
    """
    stft_matrix = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    if stft_matrix.shape[1] == 0:
        return {"fakeprint": np.array([]), "n_peaks": 0, "peak_regularity": 0.0}

    # Average over time to isolate persistent artifacts.
    avg_spectrum = np.mean(stft_matrix, axis=1)
    avg_spectrum_db = 20 * np.log10(avg_spectrum + 1e-10)

    # Subtract lower envelope to isolate spikes above the noise floor.
    envelope = _sliding_min(avg_spectrum_db, envelope_window)
    fakeprint_full = avg_spectrum_db - envelope

    # Restrict to the 1–8 kHz band where deconvolution checkerboard artifacts occur.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= freq_low) & (freqs <= freq_high)
    fakeprint = fakeprint_full[band_mask]

    threshold = np.mean(fakeprint) + 2 * np.std(fakeprint)
    peaks = np.where(fakeprint > threshold)[0]

    if len(peaks) > 1:
        spacings = np.diff(peaks)
        # Lower value = more regular spacing = stronger AI artifact signal.
        peak_regularity = float(np.std(spacings) / (np.mean(spacings) + 1e-10))
    else:
        peak_regularity = 0.0

    return {
        "fakeprint": fakeprint.astype(np.float32),
        "n_peaks": len(peaks),
        "peak_regularity": peak_regularity,
        "max_peak_height": float(np.max(fakeprint)) if len(fakeprint) > 0 else 0.0,
        "artifact_energy": float(np.sum(fakeprint[peaks])) if len(peaks) > 0 else 0.0,
    }


def extract_fakeprint(
    y: np.ndarray,
    sr: int = _DEFAULT_SR,
    n_fft: int = _DEFAULT_NFFT,
    hop_length: int = _DEFAULT_HOP,
    **kwargs,
) -> np.ndarray:
    """Extract fakeprint vector (ndarray) for caching and model input."""
    result = extract_fakeprint_features(y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs)
    fp = result["fakeprint"]
    if len(fp) == 0:
        return np.zeros(FAKEPRINT_DIM, dtype=np.float32)
    return fp
