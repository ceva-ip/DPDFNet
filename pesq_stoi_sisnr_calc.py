#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import (
    resample_poly,
    correlate,
    correlation_lags,
    butter,
    sosfiltfilt,
)

# Metrics
from pystoi.stoi import stoi
from pesq import pesq  # WB-PESQ: pesq(fs, ref, deg, 'wb')

TARGET_SR = 16000
HIGHPASS_CUTOFF_HZ = 70.0
HIGHPASS_ORDER = 4


def si_snr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    """
    Scale-Invariant SNR (in dB). Not symmetric: si_snr(ref, est).
    We remove DC to avoid bias on long windows with small signals.
    """
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.sum(ref ** 2) + eps
    alpha = np.dot(est, ref) / ref_energy  # projection of est on ref
    s_target = alpha * ref
    e_noise = est - s_target
    return 10.0 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps))


def apply_highpass(
    x: np.ndarray,
    sr: int,
    cutoff_hz: float = HIGHPASS_CUTOFF_HZ,
    order: int = HIGHPASS_ORDER,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth high-pass filter.
    Used optionally to remove very low-frequency content below the cutoff.
    """
    if cutoff_hz <= 0:
        return x.astype(np.float32)

    nyquist = 0.5 * sr
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"High-pass cutoff must be below Nyquist ({nyquist} Hz), got {cutoff_hz} Hz."
        )

    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, x, padtype=None)
    return y.astype(np.float32)


# (Legacy) SI-SNR-based sliding alignment (kept for reference, not used)
def align_by_sisnr_valid(a: np.ndarray, b: np.ndarray):
    """
    Slide the *short* signal fully inside the *long* signal and choose the offset
    that maximizes SI-SNR. Returns (a_aligned, b_aligned, lag_a_vs_b, best_sisnr_db),
    where lag>0 means 'a' lags 'b'.
    """
    if len(a) >= len(b):
        long_sig, short_sig = a, b
        long_is_a = True
    else:
        long_sig, short_sig = b, a
        long_is_a = False

    Ls = len(short_sig)
    max_start = len(long_sig) - Ls
    if max_start < 0:
        raise ValueError("The 'short' signal is longer than the 'long' signal.")

    best_sisnr = -np.inf
    best_k = 0

    # Use the short signal as the reference for SI-SNR
    for k in range(max_start + 1):
        seg = long_sig[k:k + Ls]
        score = si_snr(short_sig, seg)
        if score > best_sisnr:
            best_sisnr = score
            best_k = k

    # Build outputs and lag sign with the "a vs b" convention
    if long_is_a:
        a_al = long_sig[best_k:best_k + Ls]
        b_al = short_sig
        # b starts at index best_k within a; positive lag means a lags b -> negative here
        lag_a_vs_b = -int(best_k)
    else:
        a_al = short_sig
        b_al = long_sig[best_k:best_k + Ls]
        # a starts at index best_k within b; positive lag means a lags b -> positive here
        lag_a_vs_b = int(best_k)

    return a_al.astype(np.float32), b_al.astype(np.float32), lag_a_vs_b, float(best_sisnr)


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Average channels if needed."""
    if x.ndim == 1:
        return x.astype(np.float32)
    return np.mean(x, axis=1).astype(np.float32)


def _ensure_float_minus1_1(x: np.ndarray) -> np.ndarray:
    """Convert to float32 in [-1, 1]."""
    if np.issubdtype(x.dtype, np.integer):
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val
    else:
        x = x.astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def load_audio_mono_16k(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio with soundfile, convert to mono, resample to target_sr, normalize to [-1, 1]."""
    data, sr = sf.read(path, always_2d=False)
    data = _to_mono(data)
    data = _ensure_float_minus1_1(data)
    if sr != target_sr:
        data = resample_poly(data, target_sr, sr).astype(np.float32)
    return data


def align_by_xcorr_trim(a: np.ndarray, b: np.ndarray):
    """
    Align two 1D signals by cross-correlation (SciPy) and return trimmed, aligned versions.
    Returns: a_aligned, b_aligned, lag_samples (positive means 'a' lags 'b').
    """
    if len(a) >= len(b):
        long_sig, short_sig = a, b
        long_is_a = True
    else:
        long_sig, short_sig = b, a
        long_is_a = False

    # Use FFT-based cross-correlation for speed on long signals
    corr = correlate(long_sig, short_sig, mode='full', method='fft')
    lags = correlation_lags(len(long_sig), len(short_sig), mode='full')
    best_idx = int(np.argmax(corr))
    best_lag = int(lags[best_idx])

    if best_lag >= 0:
        long_start = best_lag
        short_start = 0
    else:
        long_start = 0
        short_start = -best_lag

    overlap_len = min(len(long_sig) - long_start, len(short_sig) - short_start)
    if overlap_len <= 0:
        # Fallback: ensure non-empty overlap by trimming to min length
        overlap_len = min(len(a), len(b))
        a_al = a[:overlap_len]
        b_al = b[:overlap_len]
        return a_al.astype(np.float32), b_al.astype(np.float32), 0

    long_slice = slice(long_start, long_start + overlap_len)
    short_slice = slice(short_start, short_start + overlap_len)
    long_al = long_sig[long_slice]
    short_al = short_sig[short_slice]

    if long_is_a:
        a_al, b_al = long_al, short_al
        lag_a_vs_b = best_lag
    else:
        a_al, b_al = short_al, long_al
        lag_a_vs_b = -best_lag

    return a_al.astype(np.float32), b_al.astype(np.float32), int(lag_a_vs_b)


def compute_metrics(clean: np.ndarray, noisy: np.ndarray, sr: int = TARGET_SR):
    """Compute STOI and WB-PESQ on aligned, equal-length, mono, 16k signals (float in [-1,1])."""
    s_val = stoi(clean, noisy, sr, extended=False)
    p_val = pesq(sr, clean, noisy, 'wb')
    return float(s_val), float(p_val)


def resolve_path(base_dir: str, p: str) -> str:
    """Return absolute path; if p is relative, join to base_dir."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Align enhanced/clean audio via SciPy cross-correlation, then compute SI-SNR, "
            "STOI, and WB-PESQ on the overlapped region."
        )
    )
    parser.add_argument("csv", help="Path to metadata CSV containing enhanced_path and clean_path")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV filename or path (default: results.csv next to the metadata CSV)"
    )
    parser.add_argument(
        "--enable-highpass",
        action="store_true",
        help=(
            "Apply a 70 Hz high-pass filter to both clean and enhanced signals "
            "before alignment and metric calculation (default: disabled)."
        ),
    )
    args = parser.parse_args()

    meta_path = os.path.abspath(args.csv)
    if not os.path.exists(meta_path):
        print(f"metadata CSV not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    base_dir = os.path.dirname(meta_path)
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(base_dir, "results.csv")
    elif not os.path.isabs(out_path) and os.path.dirname(out_path) == "":
        # simple filename -> write next to CSV
        out_path = os.path.join(base_dir, out_path)

    df = pd.read_csv(meta_path)
    required_cols = {"enhanced_path", "clean_path"}
    if not required_cols.issubset(set(df.columns)):
        print(f"CSV must contain columns: {required_cols}", file=sys.stderr)
        sys.exit(1)

    results = []
    for _, row in df.iterrows():
        mix_rel = str(row["enhanced_path"])  # degraded/enhanced/processed signal
        clean_rel = str(row["clean_path"])    # clean/reference signal
        mix_path = resolve_path(base_dir, mix_rel)
        clean_path = resolve_path(base_dir, clean_rel)

        record = dict(
            enhanced_path=mix_rel,
            clean_path=clean_rel,
            aligned_lag_samples=np.nan,
            overlap_seconds=np.nan,
            si_snr_db=np.nan,   # SI-SNR computed AFTER correlation-based alignment
            stoi=np.nan,
            wb_pesq=np.nan,
            error=""
        )

        try:
            if not os.path.exists(mix_path):
                raise FileNotFoundError(f"Mixture not found: {mix_path}")
            if not os.path.exists(clean_path):
                raise FileNotFoundError(f"Clean not found: {clean_path}")

            x_noisy = load_audio_mono_16k(mix_path, TARGET_SR)
            x_clean = load_audio_mono_16k(clean_path, TARGET_SR)

            if args.enable_highpass:
                x_noisy = apply_highpass(x_noisy, TARGET_SR)
                x_clean = apply_highpass(x_clean, TARGET_SR)

            if len(x_noisy) == 0 or len(x_clean) == 0:
                raise ValueError("Empty audio after load/resample")

            # --- Alignment via SciPy cross-correlation ---
            clean_al, noisy_al, lag_a_vs_b = align_by_xcorr_trim(x_clean, x_noisy)

            overlap_sec = len(clean_al) / float(TARGET_SR)
            record["aligned_lag_samples"] = int(lag_a_vs_b)
            record["overlap_seconds"] = round(overlap_sec, 6)

            if len(clean_al) < TARGET_SR // 2 or len(noisy_al) < TARGET_SR // 2:
                raise ValueError("Aligned overlap too short for metrics (<0.5 s).")

            # Metrics on the aligned overlap
            record["si_snr_db"] = round(float(si_snr(clean_al, noisy_al)), 6)
            s_val, p_val = compute_metrics(clean_al, noisy_al, TARGET_SR)
            record["stoi"] = s_val
            record["wb_pesq"] = p_val

        except Exception as e:
            record["error"] = f"{type(e).__name__}: {str(e)}"

        results.append(record)

    out_df = pd.DataFrame(
        results,
        columns=[
            "enhanced_path", "clean_path",
            "aligned_lag_samples", "overlap_seconds",
            "si_snr_db",
            "stoi", "wb_pesq", "error"
        ],
    )
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to: {out_path}")


if __name__ == "__main__":
    main()
