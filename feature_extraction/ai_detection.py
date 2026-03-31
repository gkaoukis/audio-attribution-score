"""
AI-detection features: phase continuity, spectral indicators, Fourier artifact summary.
"""

import numpy as np
import librosa
import scipy.signal
import scipy.ndimage

AI_DETECTION_DIM = 22

def _phase_continuity_features(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Phase continuity features (7 dims)."""
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(D)

    if phase.shape[1] < 3:
        return np.zeros(7, dtype=np.float32)

    # Instantaneous frequency wrapped to [-pi, pi]
    inst_freq = np.angle(np.exp(1j * np.diff(phase, axis=1)))

    pci = float(np.mean(np.abs(np.diff(inst_freq, axis=1))))

    phase_dev_per_band = np.std(phase, axis=1)
    phase_dev_mean = float(np.mean(phase_dev_per_band))
    phase_dev_std = float(np.std(phase_dev_per_band))

    if_stability = np.var(inst_freq, axis=1)
    if_stab_mean = float(np.mean(if_stability))
    if_stab_std = float(np.std(if_stability))

    group_delay = np.angle(np.exp(1j * -np.diff(phase, axis=0)))
    gd_std_per_frame = np.std(group_delay, axis=0)
    gd_dev_mean = float(np.mean(gd_std_per_frame))
    gd_dev_std = float(np.std(gd_std_per_frame))

    return np.array([
        pci, phase_dev_mean, phase_dev_std,
        if_stab_mean, if_stab_std,
        gd_dev_mean, gd_dev_std,
    ], dtype=np.float32)


def _spectral_ai_indicators(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Spectral AI indicators (5 dims): HNR, flatness, rolloff ratio, SSM novelty."""

    # HNR via FFT-based autocorrelation; sample up to 500 frames for speed.
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)
    if frames.shape[1] > 500:
        frames = frames[:, np.linspace(0, frames.shape[1] - 1, 500, dtype=int)]

    fft_frames = np.fft.rfft(frames, axis=0)
    power_frames = np.abs(fft_frames) ** 2
    acf_frames = np.fft.irfft(power_frames, axis=0)
    acf_frames /= (acf_frames[0, :] + 1e-10)

    min_lag = int(sr / 500)
    max_lag = min(int(sr / 50), acf_frames.shape[0] - 1)
    if max_lag > min_lag:
        peak_vals = np.max(acf_frames[min_lag:max_lag + 1, :], axis=0)
        peak_vals = np.clip(peak_vals, 1e-7, 1 - 1e-7)
        hnr_vals = 10 * np.log10(peak_vals / (1 - peak_vals + 1e-12))
        hnr = float(np.nan_to_num(np.mean(hnr_vals), nan=0.0, posinf=100.0, neginf=-100.0))
    else:
        hnr = 0.0

    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    flatness_mean = float(np.mean(flatness))
    flatness_std = float(np.std(flatness))

    rolloff_85 = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    rolloff_95 = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95))
    rolloff_ratio = float(rolloff_85 / (rolloff_95 + 1e-10))

    # SSM novelty via checkerboard-kernel convolution.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    if chroma.shape[1] > 200:
        chroma = chroma[:, np.linspace(0, chroma.shape[1] - 1, 200, dtype=int)]

    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)
    ssm = chroma_norm.T @ chroma_norm

    kernel_size = min(8, ssm.shape[0] // 4)
    if kernel_size >= 2:
        kernel = np.ones((kernel_size, kernel_size))
        half_k = kernel_size // 2
        kernel[:half_k, half_k:] = -1
        kernel[half_k:, :half_k] = -1
        novelty_matrix = scipy.signal.convolve2d(ssm, kernel, mode='valid')
        ssm_novelty = float(np.mean(np.abs(novelty_matrix)))
    else:
        ssm_novelty = 0.0

    return np.array([hnr, flatness_mean, flatness_std, rolloff_ratio, ssm_novelty], dtype=np.float32)


def _fourier_artifact_summary(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Fourier artifact summary statistics (10 dims)."""
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    spectrum = np.mean(stft, axis=1)

    n_bins = len(spectrum)
    freq_resolution = sr / n_fft
    spectrum_db = 20 * np.log10(spectrum + 1e-12)

    background = scipy.ndimage.minimum_filter1d(spectrum_db, size=50, mode='nearest')
    peaks_above_bg = spectrum_db - background

    results = []

    # Up-sampling artifact peaks at 2x, 4x, 8x harmonics
    for factor in [2, 4, 8]:
        peak_spacing_bins = int((sr / factor) / freq_resolution)

        if 2 <= peak_spacing_bins < n_bins:
            expected_positions = np.arange(peak_spacing_bins, n_bins, peak_spacing_bins)
            peak_values = peaks_above_bg[expected_positions]

            threshold = np.mean(peaks_above_bg) + 2 * np.std(peaks_above_bg)
            n_significant = np.sum(peak_values > threshold)

            results.extend([float(np.mean(peak_values)), float(n_significant / len(expected_positions))])
        else:
            results.extend([0.0, 0.0])

    threshold = np.mean(peaks_above_bg) + 2 * np.std(peaks_above_bg)
    all_peaks = np.where(peaks_above_bg > threshold)[0]

    if len(all_peaks) > 2:
        spacings = np.diff(all_peaks)
        peak_regularity = float(np.std(spacings) / (np.mean(spacings) + 1e-10))
    else:
        peak_regularity = 1.0

    max_ptb = float(np.max(peaks_above_bg))

    acf = np.correlate(peaks_above_bg[:min(n_bins, 2000)], peaks_above_bg[:min(n_bins, 2000)], mode='full')
    acf = acf[len(acf) // 2:]
    acf /= (acf[0] + 1e-10)
    spectral_periodicity = float(np.max(acf[10:500])) if len(acf) > 20 else 0.0

    artifact_ratio = (
        float(np.sum(peaks_above_bg[all_peaks]) / (np.sum(np.abs(peaks_above_bg)) + 1e-10))
        if len(all_peaks) > 0 else 0.0
    )

    results.extend([peak_regularity, max_ptb, spectral_periodicity, artifact_ratio])
    return np.array(results[:10], dtype=np.float32)


def extract_ai_detection_features(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract all AI-detection features (22 dims) from a waveform."""
    phase_feats = _phase_continuity_features(y, sr, n_fft, hop_length)
    spectral_feats = _spectral_ai_indicators(y, sr, n_fft, hop_length)
    fourier_feats = _fourier_artifact_summary(y, sr, n_fft, hop_length)

    result = np.concatenate([phase_feats, spectral_feats, fourier_feats])
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    assert result.shape[0] == AI_DETECTION_DIM
    return result.astype(np.float32)
