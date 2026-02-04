import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import queue
import tensorflow as tf

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# =======================
# MODEL → AUDIO PARAMS
# =======================
def get_audio_params_from_model_name(model_name: str):
    """
    Returns (sample_rate, window_size, hop_size) for a given model name.
    """

    # 16 kHz family (exact names)
    family_16k = {
        "baseline",
        "dpdfnet2",
        "dpdfnet4",
        "dpdfnet8",
    }

    # 48 kHz family
    family_48k = {
        "dpdfnet2_48khz_hr",
    }
    
    if model_name in family_16k:
        return 16000, 320, 160
    elif model_name in family_48k:
        return 48000, 960, 480
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# =======================
# CONFIG
# =======================
TFLITE_DIR = Path('./model_zoo/tflite')
MODEL_NAME = 'dpdfnet2' # baseline | dpdfnet2 | dpdfnet4 | dpdfnet8 | dpdfnet2_48khz_hr
SAMPLE_RATE, N_FFT, HOP_SIZE = get_audio_params_from_model_name(MODEL_NAME)
BUFFER_SECONDS = 5.0
HOP_FFT = HOP_SIZE

PLAYBACK_ENABLED = True

SAMPLES_PER_BUFFER = int(BUFFER_SECONDS * SAMPLE_RATE)

# =======================
# SPECTROGRAM (PLOTTING) CONFIG
# =======================
SPEC_N_FFT = 1024                      # larger FFT for better freq resolution (plot only)
SPEC_HOP = HOP_FFT                     # hop for spectrogram (can change if desired)
SPEC_MAX_FREQ = SAMPLE_RATE / 2.0      # show full 0–Nyquist

# =======================
# PLAYBACK MODE
# =======================
# "enhanced" → play model output
# "noisy"    → play raw mic input
PLAYBACK_MODE = "noisy"
btn_noisy = None
btn_enh = None

# =======================
# SPECTROGRAM STATE (FAST LIVE VIEW)
# =======================
SPEC_COLS = int(BUFFER_SECONDS * SAMPLE_RATE / HOP_SIZE)

spec_noisy = np.zeros((SPEC_N_FFT // 2 + 1, SPEC_COLS), dtype=np.float32)
spec_enh   = np.zeros_like(spec_noisy)

# dB versions of the spectrograms (for fast incremental updates)
DB_MIN, DB_MAX = -80.0, 0.0
eps = 1e-10
db_spec_noisy = 20 * np.log10(spec_noisy + eps)
db_spec_enh   = db_spec_noisy.copy()

spec_window = np.hanning(SPEC_N_FFT).astype(np.float32)

# rolling time-domain buffers for proper 1024-sample FFTs
spec_td_noisy = np.zeros(SPEC_N_FFT, dtype=np.float32)
spec_td_enh   = np.zeros(SPEC_N_FFT, dtype=np.float32)

spec_queue = queue.Queue(maxsize=500)  # for (noisy_mag, enh_mag)

# =======================
# DSP HELPERS
# =======================
def vorbis_window(window_len: int):
    window_size_h = window_len / 2
    indices = np.arange(window_len)
    sin = np.sin(0.5 * np.pi * (indices + 0.5) / window_size_h)
    window = np.sin(0.5 * np.pi * sin * sin)
    return window


def get_wnorm(window_len: int, frame_size: int) -> float:
    # window_len - #samples of the window
    # frame_size - hop size
    return 1.0 / (window_len**2 / (2 * frame_size))


class STFTStreamingPreprocess:
    """
    Streaming STFT-style preprocessor (NumPy version).
    """

    def __init__(self, win_len, hop_size, window, wnorm, **kwargs):
        self.win_len = int(win_len)
        self.hop_size = int(hop_size)

        window = np.asarray(window, dtype=np.float32)
        if window.shape[0] != self.win_len:
            raise ValueError(f"window length mismatch: {window.shape[0]} vs {self.win_len}")
        self.window = window  # [WIN_LEN]

        wnorm = np.asarray(wnorm, dtype=np.float32)
        self.wnorm = wnorm  # scalar or [F]

        # Internal time-domain buffer, holds last win_len samples
        self.buffer = np.zeros(self.win_len, dtype=np.float32)

    def reset_state(self):
        """Reset internal time-domain buffer to zeros."""
        self.buffer[...] = 0.0

    def call(self, inputs):
        """
        inputs:
            [HOP_SIZE] or [1, HOP_SIZE]

        returns:
            STFT frame: [1, 1, F, 2]  (B=1, T=1, freq, real/imag)
        """
        x = np.asarray(inputs, dtype=np.float32)

        # Normalize to [HOP_SIZE]
        if x.ndim == 1:
            if x.shape[0] != self.hop_size:
                raise ValueError(
                    f"Expected {self.hop_size} samples, got {x.shape[0]}"
                )
            new_samples = x
        elif x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("Streaming STFT currently supports batch size 1 only")
            if x.shape[1] != self.hop_size:
                raise ValueError(
                    f"Expected last dim {self.hop_size}, got {x.shape[1]}"
                )
            new_samples = x[0]
        else:
            raise ValueError(f"Expected rank 1 or 2, got rank {x.ndim}")

        # ----- Update internal buffer: drop oldest hop, append new hop -----
        buf = self.buffer
        shifted = np.concatenate([buf[self.hop_size:], new_samples], axis=0)
        self.buffer = shifted

        # ----- One STFT frame from the current buffer -----
        frame = shifted * self.window          # [WIN_LEN]
        frame = frame[None, :]                 # [1, WIN_LEN]

        # [1, F] complex
        spec = np.fft.rfft(frame, axis=-1)

        # apply wnorm (scalar or [F])
        wnorm = self.wnorm.astype(spec.real.dtype, copy=False)
        if wnorm.ndim == 0:
            spec = spec * wnorm
        elif wnorm.ndim == 1:
            if wnorm.shape[0] != spec.shape[-1]:
                raise ValueError(
                    f"wnorm length {wnorm.shape[0]} does not match "
                    f"number of freq bins {spec.shape[-1]}"
                )
            spec = spec * wnorm[None, :]
        else:
            raise ValueError("wnorm must be scalar or 1D [F]")

        real = spec.real
        imag = spec.imag
        out = np.stack([real, imag], axis=-1)  # [1, F, 2]
        out = out[:, None, :, :]               # [1, 1, F, 2]
        return out

    __call__ = call


class ISTFTStreamingPostprocess:
    """
    Streaming iSTFT-style postprocessor (NumPy version).
    """

    def __init__(self, win_len, hop_size, window, wnorm, **kwargs):
        self.win_len = int(win_len)
        self.hop_size = int(hop_size)

        window = np.asarray(window, dtype=np.float32)
        if window.shape[0] != self.win_len:
            raise ValueError(f"window length mismatch: {window.shape[0]} vs {self.win_len}")
        self.window = window  # [WIN_LEN]

        wnorm = np.asarray(wnorm, dtype=np.float32)
        if wnorm.shape != ():
            raise ValueError("Streaming ISTFT expects scalar wnorm")
        self.wnorm = float(wnorm)

        # Internal OLA buffer
        self.ola_buffer = np.zeros(self.win_len, dtype=np.float32)

    def reset_state(self):
        """Reset OLA buffer to zeros (for new utterance/stream)."""
        self.ola_buffer[...] = 0.0

    def call(self, inputs):
        """
        inputs: single frame spec, shape:
            - [F, 2] or
            - [1, 1, F, 2] or
            - [1, F, 2]

        returns:
            [hop_size] samples for this step.
        """
        x = np.asarray(inputs, dtype=np.float32)

        # Normalize shapes to [F, 2]
        if x.ndim == 4:
            if x.shape[0] != 1 or x.shape[1] != 1:
                raise ValueError("Expected [1, 1, F, 2] for rank-4 input")
            x = x[0, 0]
        elif x.ndim == 3:
            if x.shape[0] != 1:
                raise ValueError("Expected [1, F, 2] for rank-3 input")
            x = x[0]
        elif x.ndim == 2:
            pass
        else:
            raise ValueError(
                f"Expected [F, 2] or [1, 1, F, 2] or [1, F, 2], got shape {x.shape}"
            )

        if x.ndim != 2 or x.shape[-1] != 2:
            raise ValueError(f"Expected [F, 2] after squeeze, got {x.shape}")

        real = x[..., 0]
        imag = x[..., 1]

        # [F] complex
        spec = np.asarray(real + 1j * imag, dtype=np.complex64)
        spec = spec[None, :]  # [1, F]

        # Time-domain frame, length win_len
        frame = np.fft.irfft(spec, n=self.win_len, axis=-1)[0]
        frame = frame.astype(np.float32, copy=False)
        frame = frame * self.window
        frame = frame / self.wnorm

        # ----- Overlap-add using internal buffer -----
        buf = self.ola_buffer

        shifted = np.concatenate(
            [buf[self.hop_size:], np.zeros(self.hop_size, dtype=buf.dtype)],
            axis=0,
        )

        new_buf = shifted + frame

        out = new_buf[:self.hop_size].copy()  # [hop_size]

        self.ola_buffer = new_buf

        return out

    def flush_tail(self):
        """
        After the last frame, call this once to get the remaining samples
        (the tail that hasn't been emitted yet).
        """
        tail = self.ola_buffer[self.hop_size:].copy()
        self.reset_state()
        return tail

    __call__ = call


# Instantiate streaming STFT/iSTFT for the model
WINDOW = vorbis_window(N_FFT)
WNORM = get_wnorm(N_FFT, HOP_FFT)
stft = STFTStreamingPreprocess(N_FFT, HOP_FFT, WINDOW, WNORM)
istft = ISTFTStreamingPostprocess(N_FFT, HOP_FFT, WINDOW, WNORM)


# =======================
# TFLite MODEL
# =======================
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_DIR / (MODEL_NAME + '.tflite')))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def enhance_frame(noisy_frame: np.ndarray) -> np.ndarray:
    noisy_spec = stft(noisy_frame).astype(np.float32)
    noisy_spec = np.ascontiguousarray(noisy_spec, dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], noisy_spec)

    tic = time.perf_counter_ns()
    interpreter.invoke()
    toc = time.perf_counter_ns()
    elapsed_ms = (toc - tic) / 1e6  # ns → ms
    print(f"{elapsed_ms:.3f} ms per frame")

    enh_spec = interpreter.get_tensor(output_details[0]["index"])
    enh_frame = istft(enh_spec).astype(np.float32)
    return enh_frame


# =======================
# GLOBAL STATE
# =======================
audio_queue = queue.Queue(maxsize=100)

noisy_buffer = np.zeros(SAMPLES_PER_BUFFER, dtype=np.float32)
enhanced_buffer = np.zeros(SAMPLES_PER_BUFFER, dtype=np.float32)

# PyQtGraph / Qt globals
app = None
win = None
plot_timer = None
img_noisy = None
img_enh = None

# Colormap & dB range for spectrogram
try:
    _cmap = pg.colormap.get("magma")
    CMAP_LUT = _cmap.getLookupTable(0.0, 1.0, 256)
except Exception:
    CMAP_LUT = None  # fallback to default if colormap not available


# =======================
# AUDIO CALLBACK
# =======================
def buffer_to_mag(buf: np.ndarray) -> np.ndarray:
    """
    Convert a rolling 1024-sample buffer to a windowed FFT magnitude.
    buf must be length SPEC_N_FFT.
    """
    frame = buf.astype(np.float32, copy=True)
    frame *= spec_window
    spec = np.fft.rfft(frame)
    return np.abs(spec).astype(np.float32)


def audio_callback(indata, outdata, frames, time_info, status):
    global spec_td_noisy, spec_td_enh, PLAYBACK_MODE

    if status:
        print(status)

    noisy = indata[:, 0].astype(np.float32).copy()
    enhanced = enhance_frame(noisy)

    if enhanced.shape != noisy.shape:
        enhanced = np.resize(enhanced, noisy.shape)

    if PLAYBACK_MODE == "enhanced":
        outdata[:, 0] = enhanced
    else:
        outdata[:, 0] = noisy

    # --- update rolling 1024-sample buffers for spectrogram ---
    L = len(noisy)  # should be HOP_SIZE
    if L > SPEC_N_FFT:
        noisy = noisy[-SPEC_N_FFT:]
        enhanced = enhanced[-SPEC_N_FFT:]
        L = SPEC_N_FFT

    # shift left by L, append new samples
    spec_td_noisy = np.roll(spec_td_noisy, -L)
    spec_td_enh   = np.roll(spec_td_enh, -L)
    spec_td_noisy[-L:] = noisy
    spec_td_enh[-L:]   = enhanced

    # now compute magnitudes from the full 1024-sample windows
    noisy_mag = buffer_to_mag(spec_td_noisy)
    enh_mag   = buffer_to_mag(spec_td_enh)

    try:
        spec_queue.put_nowait((noisy_mag, enh_mag))
    except queue.Full:
        pass

    # still push full audio into audio_queue if you need it
    try:
        audio_queue.put_nowait((noisy.copy(), enhanced.copy()))
    except queue.Full:
        pass


# =======================
# SPECTROGRAM UTILS
# =======================
def frame_to_mag(x: np.ndarray) -> np.ndarray:
    """Convert a small block to a zero-padded, windowed FFT magnitude."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    frame = np.zeros(SPEC_N_FFT, dtype=np.float32)
    L = min(len(x), SPEC_N_FFT)
    frame[:L] = x[:L]
    frame *= spec_window
    spec = np.fft.rfft(frame)
    return np.abs(spec).astype(np.float32)


def compute_spectrogram(x: np.ndarray, n_fft: int = SPEC_N_FFT, hop: int = SPEC_HOP):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if len(x) < n_fft:
        S = np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
        times = np.array([0.0], dtype=np.float32)
        return S, times

    n_frames = 1 + (len(x) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)

    S = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + n_fft] * window
        spec = np.fft.rfft(frame)
        S[:, i] = np.abs(spec)

    times = np.arange(n_frames) * hop / SAMPLE_RATE
    return S, times


# =======================
# GUI (PyQtGraph) SETUP
# =======================
def update_button_styles():
    """Update button styles to reflect current PLAYBACK_MODE."""
    if btn_noisy is None or btn_enh is None:
        return

    active_style = (
        "QPushButton {"
        "background-color: #4CAF50; "
        "color: white; "
        "font-weight: bold; "
        "border-radius: 6px; "
        "padding: 6px 12px;"
        "}"
    )
    inactive_style = (
        "QPushButton {"
        "background-color: #DDDDDD; "
        "color: black; "
        "border-radius: 6px; "
        "padding: 6px 12px;"
        "}"
    )

    if PLAYBACK_MODE == "noisy":
        btn_noisy.setStyleSheet(active_style)
        btn_enh.setStyleSheet(inactive_style)
    else:
        btn_noisy.setStyleSheet(inactive_style)
        btn_enh.setStyleSheet(active_style)


def set_noisy():
    global PLAYBACK_MODE
    PLAYBACK_MODE = "noisy"
    print("Playback mode: NOISY")
    update_button_styles()


def set_enhanced():
    global PLAYBACK_MODE
    PLAYBACK_MODE = "enhanced"
    print("Playback mode: ENHANCED")
    update_button_styles()


def init_gui():
    """Create PyQtGraph window with two live spectrograms + buttons."""
    global app, win, img_noisy, img_enh, btn_noisy, btn_enh

    # Create QApplication if needed
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    win = QtWidgets.QWidget()
    win.setWindowTitle("DPDFNet: Real-Time Enhancement Demo")

    main_layout = QtWidgets.QVBoxLayout()
    win.setLayout(main_layout)

    # Graphics layout for plots
    glw = pg.GraphicsLayoutWidget()
    main_layout.addWidget(glw)

    # Initial spectrogram image (current dB buffer)
    db_init = db_spec_noisy

    # Noisy plot
    p1 = glw.addPlot(row=0, col=0)
    p1.setTitle("Noisy")
    p1.setLabel("left", "Frequency", units="Hz")
    p1.setLabel("bottom", "Time", units="s")

    img_noisy = pg.ImageItem()
    p1.addItem(img_noisy)

    # Enhanced plot
    glw.nextRow()
    p2 = glw.addPlot(row=1, col=0)
    p2.setTitle("Enhanced")
    p2.setLabel("left", "Frequency", units="Hz")
    p2.setLabel("bottom", "Time", units="s")

    img_enh = pg.ImageItem()
    p2.addItem(img_enh)

    # Map array indices to physical axes
    # shape: (freq_bins, SPEC_COLS) => width = SPEC_COLS, height = freq_bins
    freq_bins = SPEC_N_FFT // 2 + 1
    img_noisy.setImage(db_init, autoLevels=False)
    img_enh.setImage(db_init, autoLevels=False)

    # scale to [0, BUFFER_SECONDS] and [0, SPEC_MAX_FREQ]
    dx = BUFFER_SECONDS / SPEC_COLS
    dy = SPEC_MAX_FREQ / freq_bins
    img_noisy.resetTransform()
    img_enh.resetTransform()
    transform_noisy = QtGui.QTransform()
    transform_noisy.scale(dx, dy)
    img_noisy.setTransform(transform_noisy)

    transform_enh = QtGui.QTransform()
    transform_enh.scale(dx, dy)
    img_enh.setTransform(transform_enh)

    if CMAP_LUT is not None:
        img_noisy.setLookupTable(CMAP_LUT)
        img_enh.setLookupTable(CMAP_LUT)

    img_noisy.setLevels((DB_MIN, DB_MAX))
    img_enh.setLevels((DB_MIN, DB_MAX))

    p1.setLimits(xMin=0, xMax=BUFFER_SECONDS, yMin=0, yMax=SPEC_MAX_FREQ)
    p2.setLimits(xMin=0, xMax=BUFFER_SECONDS, yMin=0, yMax=SPEC_MAX_FREQ)
    p1.setRange(xRange=(0, BUFFER_SECONDS), yRange=(0, SPEC_MAX_FREQ))
    p2.setRange(xRange=(0, BUFFER_SECONDS), yRange=(0, SPEC_MAX_FREQ))

    # Buttons for playback mode
    btn_layout = QtWidgets.QHBoxLayout()
    main_layout.addLayout(btn_layout)

    btn_noisy_local = QtWidgets.QPushButton("Noisy")
    btn_enh_local = QtWidgets.QPushButton("Enhanced")

    btn_layout.addStretch(1)
    btn_layout.addWidget(btn_noisy_local)
    btn_layout.addWidget(btn_enh_local)
    btn_layout.addStretch(1)

    btn_noisy_local.clicked.connect(set_noisy)
    btn_enh_local.clicked.connect(set_enhanced)

    # assign globals
    btn_noisy = btn_noisy_local
    btn_enh = btn_enh_local

    update_button_styles()

    return win


def update_plot():
    """Drain the spectrogram queue and update the PyQtGraph images."""
    global spec_noisy, spec_enh, db_spec_noisy, db_spec_enh, img_noisy, img_enh

    if img_noisy is None or img_enh is None:
        return

    eps = 1e-10

    try:
        updated = False
        while True:
            noisy_mag, enh_mag = spec_queue.get_nowait()

            # Roll magnitude buffers
            spec_noisy = np.roll(spec_noisy, -1, axis=1)
            spec_enh   = np.roll(spec_enh,   -1, axis=1)

            # Roll dB buffers
            db_spec_noisy = np.roll(db_spec_noisy, -1, axis=1)
            db_spec_enh   = np.roll(db_spec_enh,   -1, axis=1)

            # Insert new magnitudes
            spec_noisy[:, -1] = noisy_mag
            spec_enh[:, -1]   = enh_mag

            # Insert new dB values only for the last column
            db_spec_noisy[:, -1] = 20 * np.log10(noisy_mag + eps)
            db_spec_enh[:, -1]   = 20 * np.log10(enh_mag   + eps)

            updated = True

    except queue.Empty:
        pass

    if not updated:
        return

    # Just push the dB images; no full log10 over the whole buffer
    img_noisy.setImage(db_spec_noisy, autoLevels=False)
    img_enh.setImage(db_spec_enh, autoLevels=False)
    img_noisy.setLevels((DB_MIN, DB_MAX))
    img_enh.setLevels((DB_MIN, DB_MAX))



# =======================
# MAIN
# =======================
def main():
    global plot_timer

    # Ensure spectrogram time runs along +X (freq along +Y)
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    window = init_gui()

    # Create the audio stream (context manager will close it when GUI exits)
    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=HOP_SIZE,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        print("Streaming... speak into the mic. Close the window to stop.")

        # Timer to refresh the plots
        plot_timer = QtCore.QTimer()
        plot_timer.timeout.connect(update_plot)
        plot_timer.start(30)  # ~33 FPS, lighter on CPU

        window.show()
        # Start Qt event loop (blocks until window is closed)
        qt_app = QtWidgets.QApplication.instance()
        if hasattr(qt_app, "exec"):
            qt_app.exec()
        else:
            qt_app.exec_()


if __name__ == "__main__":
    main()
