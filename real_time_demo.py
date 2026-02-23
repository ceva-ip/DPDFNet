import queue
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyqtgraph as pg
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


# =======================
# CONFIG (GLOBAL STYLE)
# =======================
ONNX_DIR = Path("./model_zoo/onnx")
MODEL_NAME = "dpdfnet2"  # baseline | dpdfnet2 | dpdfnet4 | dpdfnet8 | dpdfnet2_48khz_hr
ONNX_PATH = ONNX_DIR / f"{MODEL_NAME}.onnx"

PROVIDERS_PRIORITY = ["CPUExecutionProvider"]
BUFFER_SECONDS = 5.0
PLAYBACK_MIX = 0.0
ONNX_MS_EMA_ALPHA = 0.02

MODEL_AUDIO_PARAMS_BY_NAME = {
    "baseline": (16000, 320, 160),
    "dpdfnet2": (16000, 320, 160),
    "dpdfnet4": (16000, 320, 160),
    "dpdfnet8": (16000, 320, 160),
    "dpdfnet2_48khz_hr": (48000, 960, 480),
}

def infer_audio_params_from_model_name(model_name: str) -> tuple[int, int, int]:
    try:
        return MODEL_AUDIO_PARAMS_BY_NAME[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported MODEL_NAME='{model_name}'. "
            f"Expected one of: {sorted(MODEL_AUDIO_PARAMS_BY_NAME)}"
        ) from exc


def load_initial_state(state_path: Path) -> np.ndarray:
    suffix = state_path.suffix.lower()
    if suffix == ".npz":
        with np.load(state_path) as data:
            if "init_state" not in data:
                raise ValueError(f"Missing 'init_state' key in state file: {state_path}")
            state = data["init_state"]
    elif suffix == ".npy":
        state = np.load(state_path)
    else:
        raise ValueError(f"Unsupported state file: {state_path}. Use .npz or .npy.")
    return np.ascontiguousarray(state.astype(np.float32, copy=False))


def validate_state_shape(session: ort.InferenceSession, state: np.ndarray) -> None:
    expected = session.get_inputs()[1].shape
    if len(expected) != state.ndim:
        raise ValueError(f"Initial state rank mismatch: expected={expected}, actual={state.shape}")
    for exp_dim, act_dim in zip(expected, state.shape):
        if isinstance(exp_dim, int) and exp_dim != act_dim:
            raise ValueError(f"Initial state shape mismatch: expected={expected}, actual={state.shape}")


def vorbis_window(window_len: int) -> np.ndarray:
    window_size_h = window_len / 2
    indices = np.arange(window_len)
    sin = np.sin(0.5 * np.pi * (indices + 0.5) / window_size_h)
    window = np.sin(0.5 * np.pi * sin * sin)
    return window.astype(np.float32)


def get_wnorm(window_len: int, frame_size: int) -> float:
    return 1.0 / (window_len**2 / (2 * frame_size))


class STFTStreamingPreprocess:
    def __init__(self, win_len: int, hop_size: int, window: np.ndarray, wnorm: float):
        self.win_len = int(win_len)
        self.hop_size = int(hop_size)
        self.window = np.asarray(window, dtype=np.float32)
        self.wnorm = np.asarray(wnorm, dtype=np.float32)
        self.buffer = np.zeros(self.win_len, dtype=np.float32)

    def call(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim != 1 or x.shape[0] != self.hop_size:
            raise ValueError(f"Expected [{self.hop_size}] samples, got {x.shape}")

        shifted = np.concatenate([self.buffer[self.hop_size:], x], axis=0)
        self.buffer = shifted
        frame = shifted * self.window
        spec = np.fft.rfft(frame[None, :], axis=-1)
        spec = spec * self.wnorm.astype(spec.real.dtype, copy=False)
        out = np.stack([spec.real, spec.imag], axis=-1)
        return out[:, None, :, :].astype(np.float32, copy=False)

    __call__ = call


class ISTFTStreamingPostprocess:
    def __init__(self, win_len: int, hop_size: int, window: np.ndarray, wnorm: float):
        self.win_len = int(win_len)
        self.hop_size = int(hop_size)
        self.window = np.asarray(window, dtype=np.float32)
        self.wnorm = float(np.asarray(wnorm, dtype=np.float32))
        self.ola_buffer = np.zeros(self.win_len, dtype=np.float32)

    def call(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim == 4:
            x = x[0, 0]
        elif x.ndim == 3:
            x = x[0]
        if x.ndim != 2 or x.shape[-1] != 2:
            raise ValueError(f"Expected [F,2] style tensor, got {x.shape}")

        spec = (x[..., 0] + 1j * x[..., 1]).astype(np.complex64, copy=False)[None, :]
        frame = np.fft.irfft(spec, n=self.win_len, axis=-1)[0].astype(np.float32, copy=False)
        frame = (frame * self.window) / self.wnorm

        shifted = np.concatenate(
            [self.ola_buffer[self.hop_size:], np.zeros(self.hop_size, dtype=np.float32)], axis=0
        )
        new_buf = shifted + frame
        out = new_buf[:self.hop_size].copy()
        self.ola_buffer = new_buf
        return out

    __call__ = call


def align_signal_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:x.shape[0]] = x
    return out


def main() -> None:
    # Must be set before creating ImageItem/plots so spectrogram arrays
    # shaped [freq, time] map to y=freq and x=time.
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    onnx_path = ONNX_PATH.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    state_path = onnx_path.with_name(f"{onnx_path.stem}_state.npz")
    if not state_path.is_file():
        state_path_npy = onnx_path.with_name(f"{onnx_path.stem}_state.npy")
        if state_path_npy.is_file():
            state_path = state_path_npy
    if not state_path.is_file():
        raise FileNotFoundError(f"State file not found: {state_path}")

    available = set(ort.get_available_providers())
    providers = [p for p in PROVIDERS_PRIORITY if p in available]
    if not providers:
        raise RuntimeError(
            f"No requested providers available. requested={PROVIDERS_PRIORITY}, available={sorted(available)}"
        )

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)

    init_state = load_initial_state(state_path)
    validate_state_shape(session, init_state)

    sample_rate, n_fft, hop_size = infer_audio_params_from_model_name(MODEL_NAME)
    buffer_seconds = float(BUFFER_SECONDS)
    playback_mix = float(np.clip(PLAYBACK_MIX, 0.0, 1.0))
    samples_per_buffer = int(buffer_seconds * sample_rate)

    spec_n_fft = 1024
    spec_max_freq = sample_rate / 2.0
    spec_cols = int(buffer_seconds * sample_rate / hop_size)
    spec_window = np.hanning(spec_n_fft).astype(np.float32)
    db_min, db_max = -80.0, 0.0
    eps = 1e-10

    spec_noisy = np.zeros((spec_n_fft // 2 + 1, spec_cols), dtype=np.float32)
    spec_enh = np.zeros_like(spec_noisy)
    db_spec_noisy = 20 * np.log10(spec_noisy + eps)
    db_spec_enh = db_spec_noisy.copy()
    spec_td_noisy = np.zeros(spec_n_fft, dtype=np.float32)
    spec_td_enh = np.zeros(spec_n_fft, dtype=np.float32)
    spec_queue: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(maxsize=500)
    audio_queue: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(maxsize=100)
    noisy_buffer = np.zeros(samples_per_buffer, dtype=np.float32)
    enhanced_buffer = np.zeros(samples_per_buffer, dtype=np.float32)

    agc_enabled = True
    agc_target_rms = 0.12
    agc_rms_floor = 1e-3
    agc_min_gain = 0.25
    agc_max_gain = 8.0
    agc_attack_sec = 0.03
    agc_release_sec = 0.30
    agc_gain = 1.0

    window = vorbis_window(n_fft)
    wnorm = get_wnorm(n_fft, hop_size)
    stft = STFTStreamingPreprocess(n_fft, hop_size, window, wnorm)
    istft = ISTFTStreamingPostprocess(n_fft, hop_size, window, wnorm)
    runtime_state = init_state.copy()
    onnx_ms_ema = float("nan")

    in_spec_name = session.get_inputs()[0].name
    in_state_name = session.get_inputs()[1].name
    out_spec_name = session.get_outputs()[0].name
    out_state_name = session.get_outputs()[1].name

    def buffer_to_mag(buf: np.ndarray) -> np.ndarray:
        frame = buf.astype(np.float32, copy=True) * spec_window
        return np.abs(np.fft.rfft(frame)).astype(np.float32)

    def apply_output_agc(x: np.ndarray, frames: int) -> np.ndarray:
        nonlocal agc_gain
        x = np.asarray(x, dtype=np.float32)
        if (not agc_enabled) or x.size == 0:
            return np.clip(x, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        desired_gain = float(np.clip(agc_target_rms / max(rms, agc_rms_floor), agc_min_gain, agc_max_gain))
        attack_samples = max(1.0, agc_attack_sec * sample_rate)
        release_samples = max(1.0, agc_release_sec * sample_rate)
        attack_coeff = float(np.exp(-frames / attack_samples))
        release_coeff = float(np.exp(-frames / release_samples))
        coeff = attack_coeff if desired_gain < agc_gain else release_coeff
        agc_gain = coeff * agc_gain + (1.0 - coeff) * desired_gain
        return np.clip(x * agc_gain, -1.0, 1.0)

    def enhance_frame(noisy_frame: np.ndarray) -> np.ndarray:
        nonlocal runtime_state, onnx_ms_ema
        noisy_spec = np.ascontiguousarray(stft(noisy_frame), dtype=np.float32)
        t0 = time.perf_counter()
        spec_e, state_out = session.run(
            [out_spec_name, out_state_name],
            {in_spec_name: noisy_spec, in_state_name: runtime_state},
        )
        frame_ms = (time.perf_counter() - t0) * 1000.0
        if np.isnan(onnx_ms_ema):
            onnx_ms_ema = frame_ms
        else:
            onnx_ms_ema = (1.0 - ONNX_MS_EMA_ALPHA) * onnx_ms_ema + ONNX_MS_EMA_ALPHA * frame_ms
        runtime_state = np.ascontiguousarray(state_out.astype(np.float32, copy=False))
        return istft(spec_e).astype(np.float32, copy=False)

    def update_mix_label() -> None:
        if mix_value_label is not None:
            mix_value_label.setText(f"{playback_mix:.1f}")

    def set_playback_mix(value: float) -> None:
        nonlocal playback_mix
        playback_mix = float(np.clip(value, 0.0, 1.0))
        update_mix_label()

    def on_mix_slider_changed(value: int) -> None:
        set_playback_mix(value / 10.0)

    def update_agc_button_text() -> None:
        agc_button.setText(f"AGC: {'ON' if agc_enabled else 'OFF'}")

    def on_agc_button_toggled(checked: bool) -> None:
        nonlocal agc_enabled
        agc_enabled = bool(checked)
        update_agc_button_text()

    try:
        cmap = pg.colormap.get("magma")
        cmap_lut = cmap.getLookupTable(0.0, 1.0, 256)
    except Exception:
        cmap_lut = None

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    win = QtWidgets.QWidget()
    win.setWindowTitle("DPDFNet ONNX: Real-Time Enhancement Demo")
    main_layout = QtWidgets.QVBoxLayout()
    win.setLayout(main_layout)
    glw = pg.GraphicsLayoutWidget()
    main_layout.addWidget(glw)

    p1 = glw.addPlot(row=0, col=0)
    p1.setTitle("Noisy")
    p1.setLabel("left", "Frequency", units="Hz")
    p1.setLabel("bottom", "Time", units="s")
    img_noisy = pg.ImageItem()
    p1.addItem(img_noisy)

    glw.nextRow()
    p2 = glw.addPlot(row=1, col=0)
    p2.setTitle("Enhanced")
    p2.setLabel("left", "Frequency", units="Hz")
    p2.setLabel("bottom", "Time", units="s")
    img_enh = pg.ImageItem()
    p2.addItem(img_enh)

    img_noisy.setImage(db_spec_noisy, autoLevels=False)
    img_enh.setImage(db_spec_noisy, autoLevels=False)

    freq_bins = spec_n_fft // 2 + 1
    dx = buffer_seconds / spec_cols
    dy = spec_max_freq / freq_bins
    t_noisy = QtGui.QTransform()
    t_noisy.scale(dx, dy)
    img_noisy.setTransform(t_noisy)
    t_enh = QtGui.QTransform()
    t_enh.scale(dx, dy)
    img_enh.setTransform(t_enh)
    if cmap_lut is not None:
        img_noisy.setLookupTable(cmap_lut)
        img_enh.setLookupTable(cmap_lut)
    img_noisy.setLevels((db_min, db_max))
    img_enh.setLevels((db_min, db_max))
    p1.setRange(xRange=(0, buffer_seconds), yRange=(0, spec_max_freq))
    p2.setRange(xRange=(0, buffer_seconds), yRange=(0, spec_max_freq))

    mix_layout = QtWidgets.QHBoxLayout()
    main_layout.addLayout(mix_layout)
    mix_title = QtWidgets.QLabel("Playback Mix (Noisy -> Enhanced)")
    lbl_noisy = QtWidgets.QLabel("Noisy")
    lbl_enh = QtWidgets.QLabel("Enhanced")

    mix_slider_local = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    mix_slider_local.setRange(0, 10)  # 0.0 ... 1.0 in 0.1 steps
    mix_slider_local.setSingleStep(1)
    mix_slider_local.setPageStep(1)
    mix_slider_local.setTickInterval(1)
    mix_slider_local.setTickPosition(QtWidgets.QSlider.TicksBelow)
    mix_slider_local.setValue(int(round(playback_mix * 10)))

    mix_value_label = QtWidgets.QLabel(f"{playback_mix:.1f}")
    mix_value_label.setAlignment(QtCore.Qt.AlignCenter)
    mix_value_label.setMinimumWidth(36)
    mix_layout.addWidget(mix_title)
    mix_layout.addWidget(lbl_noisy)
    mix_layout.addWidget(mix_slider_local, 1)
    mix_layout.addWidget(lbl_enh)
    mix_layout.addWidget(mix_value_label)
    mix_slider_local.valueChanged.connect(on_mix_slider_changed)

    agc_layout = QtWidgets.QHBoxLayout()
    main_layout.addLayout(agc_layout)
    agc_title = QtWidgets.QLabel("Output AGC")
    agc_button = QtWidgets.QPushButton()
    agc_button.setCheckable(True)
    agc_button.setChecked(agc_enabled)
    agc_button.toggled.connect(on_agc_button_toggled)
    update_agc_button_text()
    agc_layout.addWidget(agc_title)
    agc_layout.addStretch(1)
    agc_layout.addWidget(agc_button)

    perf_layout = QtWidgets.QHBoxLayout()
    main_layout.addLayout(perf_layout)
    perf_title = QtWidgets.QLabel("ONNX Inference (EMA)")
    perf_value_label = QtWidgets.QLabel("-- ms/frame")
    perf_value_label.setAlignment(QtCore.Qt.AlignCenter)
    perf_value_label.setMinimumWidth(120)
    perf_layout.addWidget(perf_title)
    perf_layout.addStretch(1)
    perf_layout.addWidget(perf_value_label)

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal spec_td_noisy, spec_td_enh, noisy_buffer, enhanced_buffer
        # Suppress sounddevice callback status spam (e.g., underflow/overflow).

        noisy = align_signal_length(indata[:, 0], frames)
        enhanced = align_signal_length(enhance_frame(noisy), frames)
        mixed = (1.0 - playback_mix) * noisy + playback_mix * enhanced
        outdata[:, 0] = apply_output_agc(mixed, frames)

        noisy_buffer = np.roll(noisy_buffer, -frames)
        noisy_buffer[-frames:] = noisy
        enhanced_buffer = np.roll(enhanced_buffer, -frames)
        enhanced_buffer[-frames:] = enhanced

        l = min(len(noisy), spec_n_fft)
        spec_td_noisy = np.roll(spec_td_noisy, -l)
        spec_td_enh = np.roll(spec_td_enh, -l)
        spec_td_noisy[-l:] = noisy[-l:]
        spec_td_enh[-l:] = enhanced[-l:]

        noisy_mag = buffer_to_mag(spec_td_noisy)
        enh_mag = buffer_to_mag(spec_td_enh)
        try:
            spec_queue.put_nowait((noisy_mag, enh_mag))
        except queue.Full:
            pass
        try:
            audio_queue.put_nowait((noisy.copy(), enhanced.copy()))
        except queue.Full:
            pass

    def update_plot():
        nonlocal spec_noisy, spec_enh, db_spec_noisy, db_spec_enh, onnx_ms_ema
        updated = False
        try:
            while True:
                noisy_mag, enh_mag = spec_queue.get_nowait()
                spec_noisy = np.roll(spec_noisy, -1, axis=1)
                spec_enh = np.roll(spec_enh, -1, axis=1)
                db_spec_noisy = np.roll(db_spec_noisy, -1, axis=1)
                db_spec_enh = np.roll(db_spec_enh, -1, axis=1)
                spec_noisy[:, -1] = noisy_mag
                spec_enh[:, -1] = enh_mag
                db_spec_noisy[:, -1] = 20 * np.log10(noisy_mag + eps)
                db_spec_enh[:, -1] = 20 * np.log10(enh_mag + eps)
                updated = True
        except queue.Empty:
            pass

        if updated:
            img_noisy.setImage(db_spec_noisy, autoLevels=False)
            img_enh.setImage(db_spec_enh, autoLevels=False)
            img_noisy.setLevels((db_min, db_max))
            img_enh.setLevels((db_min, db_max))
        if np.isnan(onnx_ms_ema):
            perf_value_label.setText("-- ms/frame")
        else:
            perf_value_label.setText(f"{onnx_ms_ema:.3f} ms/frame")

    print(f"[INFO] ONNX Runtime: {ort.__version__}")
    print(f"[INFO] ONNX model: {onnx_path}")
    print(f"[INFO] State file: {state_path} shape={tuple(init_state.shape)}")
    print(f"[INFO] Providers: {session.get_providers()}")
    print(f"[INFO] Audio params: sr={sample_rate}, n_fft={n_fft}, hop={hop_size}")
    print("Streaming... speak into the mic. Close the window to stop.")

    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(30)

    with sd.Stream(
        samplerate=sample_rate,
        blocksize=hop_size,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        win.show()
        qt_app = QtWidgets.QApplication.instance()
        if hasattr(qt_app, "exec"):
            qt_app.exec()
        else:
            qt_app.exec_()


if __name__ == "__main__":
    main()
