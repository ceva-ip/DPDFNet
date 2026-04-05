from __future__ import annotations

from pathlib import Path

import numpy as np
from urllib.error import HTTPError
from urllib.error import URLError

import pytest


def _fake_download_one(_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"ok")


def test_import_surface() -> None:
    import dpdfnet

    assert hasattr(dpdfnet, "enhance")
    assert hasattr(dpdfnet, "enhance_file")
    assert hasattr(dpdfnet, "StreamEnhancer")
    assert hasattr(dpdfnet, "available_models")
    assert hasattr(dpdfnet, "download")


def test_model_resolution_auto_download(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    monkeypatch.setattr(models, "_download_one", _fake_download_one)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    model_name = models.supported_models()[0]

    resolved = models.resolve_model(
        model=model_name,
        auto_download=True,
        verbose=False,
    )

    assert resolved.info.name == model_name
    assert resolved.onnx_path.is_file()


def test_download_all_matches_registry(tmp_path, monkeypatch) -> None:
    import dpdfnet
    from dpdfnet import models

    monkeypatch.setattr(models, "_download_one", _fake_download_one)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    downloaded = dpdfnet.download(quiet=True)

    assert isinstance(downloaded, dict)
    assert sorted(downloaded) == models.supported_models()

    for model_name in models.supported_models():
        info = models.MODEL_REGISTRY[model_name]
        assert (tmp_path / info.onnx_filename).is_file()


def test_cli_download_all_matches_registry(tmp_path, monkeypatch, capsys) -> None:
    from dpdfnet import cli, models

    monkeypatch.setattr(models, "_download_one", _fake_download_one)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    exit_code = cli.main(["download", "--quiet"])

    assert exit_code == 0
    output = capsys.readouterr().out
    for model_name in models.supported_models():
        assert f"- {model_name}:" in output


def test_enhance_progress_callback_reports_frame_progress(monkeypatch) -> None:
    from dpdfnet import api

    class _FakeSession:
        def run(self, _outputs, _inputs):
            return np.zeros((1, 1, 3, 2), dtype=np.float32), np.zeros((1,), dtype=np.float32)

    class _FakeRuntime:
        session = _FakeSession()
        init_state = np.zeros((1,), dtype=np.float32)
        in_spec_name = "spec"
        in_state_name = "state"
        out_spec_name = "out_spec"
        out_state_name = "out_state"

    class _ResolvedInfo:
        sample_rate = 16000

    class _ResolvedModel:
        info = _ResolvedInfo()
        onnx_path = Path("fake.onnx")

    monkeypatch.setattr(api, "resolve_model", lambda **_kwargs: _ResolvedModel())

    import dpdfnet.audio as audio_mod
    import dpdfnet.onnx_backend as backend_mod

    monkeypatch.setattr(audio_mod, "to_mono", lambda x: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(audio_mod, "ensure_sample_rate", lambda x, _a, _b: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(audio_mod, "make_stft_config", lambda _win: type("Cfg", (), {"win_len": 4})())
    monkeypatch.setattr(audio_mod, "preprocess_waveform", lambda _w, _cfg: np.zeros((1, 4, 3, 2), dtype=np.float32))
    monkeypatch.setattr(audio_mod, "postprocess_spec", lambda _spec, _cfg: np.zeros((8,), dtype=np.float32))
    monkeypatch.setattr(audio_mod, "fit_length", lambda x, _n: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(backend_mod, "build_runtime_model", lambda _onnx: _FakeRuntime())
    monkeypatch.setattr(backend_mod, "infer_win_len", lambda _session, _sr: 4)

    updates = []
    api.enhance(
        audio=np.zeros((8,), dtype=np.float32),
        sample_rate=16000,
        progress_callback=lambda done, total: updates.append((done, total)),
    )

    assert updates[0] == (0, 4)
    assert updates[-1] == (4, 4)
    assert len(updates) == 5


def test_enhance_applies_attn_limit_before_postprocess(monkeypatch) -> None:
    from dpdfnet import api

    spec_r = np.arange(1, 1 + (1 * 4 * 3 * 2), dtype=np.float32).reshape(1, 4, 3, 2)
    spec_e_frames = np.full((4, 1, 3, 2), 77.0, dtype=np.float32)
    captured: dict[str, np.ndarray] = {}

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, _outputs, _inputs):
            frame = spec_e_frames[self.calls : self.calls + 1]
            self.calls += 1
            return frame.copy(), np.zeros((1,), dtype=np.float32)

    class _FakeRuntime:
        session = _FakeSession()
        init_state = np.zeros((1,), dtype=np.float32)
        in_spec_name = "spec"
        in_state_name = "state"
        out_spec_name = "out_spec"
        out_state_name = "out_state"

    class _ResolvedInfo:
        sample_rate = 16000

    class _ResolvedModel:
        info = _ResolvedInfo()
        onnx_path = Path("fake.onnx")

    monkeypatch.setattr(api, "resolve_model", lambda **_kwargs: _ResolvedModel())

    import dpdfnet.audio as audio_mod
    import dpdfnet.onnx_backend as backend_mod

    monkeypatch.setattr(audio_mod, "to_mono", lambda x: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(audio_mod, "ensure_sample_rate", lambda x, _a, _b: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(audio_mod, "make_stft_config", lambda _win: type("Cfg", (), {"win_len": 4})())
    monkeypatch.setattr(audio_mod, "preprocess_waveform", lambda _w, _cfg: spec_r.copy())

    def _capture_postprocess(spec, _cfg):
        captured["spec"] = np.asarray(spec, dtype=np.float32).copy()
        return np.zeros((8,), dtype=np.float32)

    monkeypatch.setattr(audio_mod, "postprocess_spec", _capture_postprocess)
    monkeypatch.setattr(audio_mod, "fit_length", lambda x, _n: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(backend_mod, "build_runtime_model", lambda _onnx: _FakeRuntime())
    monkeypatch.setattr(backend_mod, "infer_win_len", lambda _session, _sr: 4)

    api.enhance(
        audio=np.zeros((8,), dtype=np.float32),
        sample_rate=16000,
        attn_limit_db=0.0,
    )

    expected = np.zeros_like(spec_r)
    expected[:, 4:, :, :] = spec_r[:, :-4, :, :]
    np.testing.assert_array_equal(captured["spec"], expected)


def test_model_download_http_error_reports_status(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    def _raise_http(url: str, _destination: Path) -> None:
        raise HTTPError(url=url, code=403, msg="Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(models, "_download_one", _raise_http)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match=r"HTTP 403"):
        models.resolve_model(
            model=models.supported_models()[0],
            auto_download=True,
            verbose=False,
        )


def test_model_download_url_error_reports_network_issue(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    def _raise_url(_url: str, _destination: Path) -> None:
        raise URLError("unreachable")

    monkeypatch.setattr(models, "_download_one", _raise_url)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match=r"Network error"):
        models.resolve_model(
            model=models.supported_models()[0],
            auto_download=True,
            verbose=False,
        )


def test_model_download_unwritable_dir_reports_path(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    model_dir = tmp_path / "blocked"
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(model_dir))

    def _raise_permission(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(models.tempfile, "mkstemp", _raise_permission)

    with pytest.raises(RuntimeError, match=r"not writable"):
        models.resolve_model(
            model=models.supported_models()[0],
            auto_download=True,
            verbose=False,
        )


def test_model_download_retries_transient_network_errors(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    attempts = {"count": 0}

    def _flaky_download(_url: str, destination: Path) -> None:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise URLError("[Errno 104] Connection reset by peer")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"ok")

    monkeypatch.setattr(models, "_download_one", _flaky_download)
    monkeypatch.setattr(models.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))

    model_name = models.supported_models()[0]
    info = models.MODEL_REGISTRY[model_name]
    resolved = models.resolve_model(
        model=model_name,
        auto_download=True,
        verbose=False,
    )

    assert resolved.onnx_path == (tmp_path / info.onnx_filename).resolve()
    assert resolved.onnx_path.is_file()
    assert attempts["count"] >= 3


def test_missing_model_message_does_not_reference_unsupported_cli_flags(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError) as exc_info:
        models.resolve_model(
            model=models.supported_models()[0],
            auto_download=False,
            verbose=False,
        )

    message = str(exc_info.value)
    assert "--onnx" not in message
    assert "onnx_path" in message


# ---------------------------------------------------------------------------
# StreamEnhancer tests
# ---------------------------------------------------------------------------

def _make_fake_stream_enhancer(monkeypatch, win_len: int = 8, model_sr: int = 16000):
    """Return a StreamEnhancer backed by a fake zero-output ONNX session."""
    from dpdfnet import stream as stream_mod

    freq_bins = win_len // 2 + 1

    class _FakeSession:
        def get_inputs(self):
            class _Spec:
                name = "spec"
                shape = (1, 1, freq_bins, 2)

            class _State:
                name = "state"

            return [_Spec(), _State()]

        def get_outputs(self):
            class _OutSpec:
                name = "out_spec"

            class _OutState:
                name = "out_state"

            return [_OutSpec(), _OutState()]

        def run(self, _outputs, _inputs):
            return (
                np.zeros((1, 1, freq_bins, 2), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
            )

    class _FakeRuntime:
        session = _FakeSession()
        init_state = np.zeros((1,), dtype=np.float32)
        in_spec_name = "spec"
        in_state_name = "state"
        out_spec_name = "out_spec"
        out_state_name = "out_state"

    class _ResolvedInfo:
        sample_rate = model_sr

    class _ResolvedModel:
        info = _ResolvedInfo()
        onnx_path = Path("fake.onnx")

    monkeypatch.setattr(stream_mod, "resolve_model", lambda **_kw: _ResolvedModel())
    monkeypatch.setattr(stream_mod, "build_runtime_model", lambda _p: _FakeRuntime())
    monkeypatch.setattr(stream_mod, "infer_win_len", lambda _s, _sr: win_len)

    from dpdfnet import StreamEnhancer

    return StreamEnhancer(model="dpdfnet2")


def test_stream_enhancer_buffers_small_chunks(monkeypatch) -> None:
    """Small chunks are buffered; output appears only after win_len samples arrive."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)  # hop_size=4
    # 3 samples < win_len=8 → nothing to process yet
    out = e.process(np.zeros(3, dtype=np.float32), sample_rate=16000)
    assert len(out) == 0
    # 5 more → total 8 = win_len → one frame committed → hop_size=4 output samples
    out = e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)
    assert len(out) == 4


def test_stream_enhancer_misaligned_block_size(monkeypatch) -> None:
    """Chunk sizes not aligned to hop_size must not drop or duplicate samples.

    Simulates sounddevice with BLOCK_SIZE=512 at 48 kHz against a 16 kHz model
    by feeding 171-sample chunks (≈ 512 * 16000 / 48000) at model SR directly.
    """
    WIN = 320
    HOP = 160
    e = _make_fake_stream_enhancer(monkeypatch, win_len=WIN)

    total_input = int(16000 * 1.0)  # 1 second at model SR
    CHUNK = 171  # not a multiple of HOP

    outputs = []
    fed = 0
    while fed < total_input:
        n = min(CHUNK, total_input - fed)
        out = e.process(np.zeros(n, dtype=np.float32), sample_rate=16000)
        outputs.append(out)
        fed += n

    total_output = sum(len(o) for o in outputs)
    expected_frames = max(0, (total_input - WIN) // HOP + 1)
    assert total_output == expected_frames * HOP
    for o in outputs:
        assert o.dtype == np.float32


def test_stream_enhancer_reset_clears_state(monkeypatch) -> None:
    """After reset(), the buffer is empty and a full win_len is needed again."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)  # partial buffer
    e.reset()
    out = e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)
    assert len(out) == 0


def test_stream_enhancer_flush_drains_remainder(monkeypatch) -> None:
    """flush() zero-pads the last partial window and returns enhanced samples."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)  # 5 in buf, no output
    out = e.flush()
    assert len(out) > 0
    assert out.dtype == np.float32


def test_stream_enhancer_flush_on_empty_buffer_returns_empty(monkeypatch) -> None:
    """flush() on a fresh (or fully-drained) instance returns an empty array."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    out = e.flush()
    assert len(out) == 0
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# StreamEnhancer correctness: passthrough reconstruction & block-size invariance
# ---------------------------------------------------------------------------

def _make_passthrough_stream_enhancer(monkeypatch, win_len: int = 320, model_sr: int = 16000):
    """StreamEnhancer backed by a passthrough ONNX session (returns input spectrum unchanged).

    Unlike ``_make_fake_stream_enhancer`` which outputs zeros, this helper lets
    the real causal STFT / overlap-add pipeline run, making it suitable for
    testing signal reconstruction properties.
    """
    from dpdfnet import stream as stream_mod

    freq_bins = win_len // 2 + 1

    class _PassthroughSession:
        def get_inputs(self):
            class _Spec:
                name = "spec"
                shape = (1, 1, freq_bins, 2)
            class _State:
                name = "state"
            return [_Spec(), _State()]

        def get_outputs(self):
            class _OutSpec:
                name = "out_spec"
            class _OutState:
                name = "out_state"
            return [_OutSpec(), _OutState()]

        def run(self, _outputs, inputs):
            return inputs["spec"].copy(), inputs["state"].copy()

    class _Runtime:
        session = _PassthroughSession()
        init_state = np.zeros((1,), dtype=np.float32)
        in_spec_name = "spec"
        in_state_name = "state"
        out_spec_name = "out_spec"
        out_state_name = "out_state"

    class _Info:
        sample_rate = model_sr

    class _Resolved:
        info = _Info()
        onnx_path = Path("fake.onnx")

    monkeypatch.setattr(stream_mod, "resolve_model", lambda **_kw: _Resolved())
    monkeypatch.setattr(stream_mod, "build_runtime_model", lambda _p: _Runtime())
    monkeypatch.setattr(stream_mod, "infer_win_len", lambda _s, _sr: win_len)

    from dpdfnet import StreamEnhancer

    return StreamEnhancer(model="dpdfnet2")


def _stream_enhance(enhancer, signal: np.ndarray, block_size: int, sr: int = 16000):
    """Feed *signal* through *enhancer* in chunks of *block_size* and return concatenated output."""
    parts: list[np.ndarray] = []
    for start in range(0, len(signal), block_size):
        out = enhancer.process(signal[start : start + block_size], sample_rate=sr)
        parts.append(out)
    parts.append(enhancer.flush())
    return np.concatenate(parts)


def test_stream_passthrough_reconstructs_signal(monkeypatch) -> None:
    """With a passthrough model, the causal STFT → ISTFT pipeline perfectly
    reconstructs the input (after the one-hop COLA ramp-up)."""
    WIN = 320
    HOP = WIN // 2

    e = _make_passthrough_stream_enhancer(monkeypatch, win_len=WIN)

    np.random.seed(123)
    signal = np.random.randn(8000).astype(np.float32) * 0.5

    out = e.process(signal, sample_rate=16000)
    # Skip the first HOP samples: only one window contributes there (half COLA).
    # From sample HOP onward, two overlapping windows sum to 1.
    np.testing.assert_allclose(
        out[HOP:], signal[HOP : len(out)], atol=1e-5,
        err_msg="Passthrough streaming did not reconstruct input after initial latency",
    )


@pytest.mark.parametrize("block_size", [7, 64, 160, 171, 320, 512, 1000])
def test_stream_block_size_matches_reference(monkeypatch, block_size: int) -> None:
    """Streaming output must be identical regardless of how the input is chunked.

    Uses block_size=1 as the reference (one sample at a time), then parametrises
    over a range of realistic and adversarial block sizes.
    """
    WIN = 320

    np.random.seed(42)
    signal = np.random.randn(4000).astype(np.float32) * 0.5

    # Reference: one-sample-at-a-time (guarantees every buffering edge case)
    ref_enhancer = _make_passthrough_stream_enhancer(monkeypatch, win_len=WIN)
    ref = _stream_enhance(ref_enhancer, signal, block_size=1)

    # Test block size
    enhancer = _make_passthrough_stream_enhancer(monkeypatch, win_len=WIN)
    result = _stream_enhance(enhancer, signal, block_size=block_size)

    assert len(result) == len(ref), (
        f"block_size={block_size}: length {len(result)} != ref length {len(ref)}"
    )
    np.testing.assert_allclose(
        result, ref, atol=1e-5,
        err_msg=f"block_size={block_size} differs from sample-by-sample reference",
    )


def test_stream_and_offline_both_reconstruct_with_passthrough(monkeypatch) -> None:
    """With a passthrough model, both enhance() and StreamEnhancer approximately
    reconstruct the input signal (each in its own alignment convention).

    The offline path uses center=True STFT; the streaming path uses center=False.
    They are not bit-identical, but both should faithfully represent the input.
    """
    WIN = 320
    HOP = WIN // 2
    MODEL_SR = 16000
    freq_bins = WIN // 2 + 1

    class _PassthroughSession:
        def run(self, _outputs, inputs):
            return inputs["spec"].copy(), inputs["state"].copy()

    class _Runtime:
        session = _PassthroughSession()
        init_state = np.zeros((1,), dtype=np.float32)
        in_spec_name = "spec"
        in_state_name = "state"
        out_spec_name = "out_spec"
        out_state_name = "out_state"

    class _Info:
        sample_rate = MODEL_SR

    class _Resolved:
        info = _Info()
        onnx_path = Path("fake.onnx")

    # --- patch offline enhance() ---
    from dpdfnet import api
    import dpdfnet.onnx_backend as backend_mod

    monkeypatch.setattr(api, "resolve_model", lambda **_kw: _Resolved())
    monkeypatch.setattr(backend_mod, "build_runtime_model", lambda _p: _Runtime())
    monkeypatch.setattr(backend_mod, "infer_win_len", lambda _s, _sr: WIN)

    # --- patch streaming ---
    from dpdfnet import stream as stream_mod

    monkeypatch.setattr(stream_mod, "resolve_model", lambda **_kw: _Resolved())
    monkeypatch.setattr(stream_mod, "build_runtime_model", lambda _p: _Runtime())
    monkeypatch.setattr(stream_mod, "infer_win_len", lambda _s, _sr: WIN)

    np.random.seed(99)
    signal = np.random.randn(8000).astype(np.float32) * 0.5

    # Offline
    offline_out = api.enhance(audio=signal, sample_rate=MODEL_SR)

    # Streaming (single large chunk)
    from dpdfnet import StreamEnhancer

    se = StreamEnhancer(model="dpdfnet2")
    stream_out = np.concatenate([
        se.process(signal, sample_rate=MODEL_SR),
        se.flush(),
    ])

    # Both should produce output of expected length
    assert len(offline_out) == len(signal)
    assert len(stream_out) > 0

    # Both should contain non-trivial (non-zero) samples
    assert np.any(offline_out != 0), "Offline output is all zeros"
    assert np.any(stream_out != 0), "Stream output is all zeros"

    # Streaming path: perfect reconstruction after first HOP (verified separately)
    np.testing.assert_allclose(
        stream_out[HOP : len(stream_out)],
        signal[HOP : len(stream_out)],
        atol=1e-5,
        err_msg="Streaming path did not reconstruct input",
    )

    # Offline path: postprocess_spec strips the first 2*win_len samples from the
    # ISTFT output, so offline_out[k] corresponds to signal[k + 2*WIN].
    shift = 2 * WIN
    valid_len = len(signal) - shift  # samples where real signal exists
    np.testing.assert_allclose(
        offline_out[:valid_len],
        signal[shift : shift + valid_len],
        atol=1e-4,
        err_msg="Offline passthrough did not reconstruct the shifted input",
    )


def test_stream_enhancer_sample_rate_mismatch_raises(monkeypatch) -> None:
    """process() raises ValueError if sample_rate changes between calls."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    e.process(np.zeros(3, dtype=np.float32), sample_rate=16000)
    with pytest.raises(ValueError, match="Sample rate changed"):
        e.process(np.zeros(3, dtype=np.float32), sample_rate=8000)


def test_stream_enhancer_empty_chunk_returns_empty(monkeypatch) -> None:
    """process() with an empty array returns an empty float32 array."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    out = e.process(np.zeros(0, dtype=np.float32), sample_rate=16000)
    assert len(out) == 0
    assert out.dtype == np.float32


def test_stream_enhancer_stereo_converted_to_mono(monkeypatch) -> None:
    """Stereo (2D) chunks are averaged to mono before processing."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    stereo = np.zeros((10, 2), dtype=np.float32)
    # Should not raise; stereo is silently averaged to mono
    out = e.process(stereo, sample_rate=16000)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# audio.py unit tests
# ---------------------------------------------------------------------------

def test_to_mono_passthrough_for_1d() -> None:
    from dpdfnet.audio import to_mono
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = to_mono(x)
    np.testing.assert_array_equal(result, x)
    assert result.dtype == np.float32


def test_to_mono_averages_stereo_channels() -> None:
    from dpdfnet.audio import to_mono
    # shape (samples, channels) — axis-1 is channels, axis-0 is time
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([0.0, 1.0], dtype=np.float32)
    stereo = np.stack([left, right], axis=1)  # (2, 2)
    result = to_mono(stereo)
    np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-6)
    assert result.dtype == np.float32


def test_to_mono_invalid_shape_raises() -> None:
    from dpdfnet.audio import to_mono
    bad = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected mono/stereo"):
        to_mono(bad)


def test_fit_length_truncates() -> None:
    from dpdfnet.audio import fit_length
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = fit_length(x, 2)
    assert len(result) == 2
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_fit_length_pads_with_zeros() -> None:
    from dpdfnet.audio import fit_length
    x = np.array([1.0, 2.0], dtype=np.float32)
    result = fit_length(x, 5)
    assert len(result) == 5
    np.testing.assert_array_equal(result, [1.0, 2.0, 0.0, 0.0, 0.0])


def test_fit_length_exact_unchanged() -> None:
    from dpdfnet.audio import fit_length
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = fit_length(x, 3)
    np.testing.assert_array_equal(result, x)


def test_pcm16_safe_clips_and_converts() -> None:
    from dpdfnet.audio import pcm16_safe
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = pcm16_safe(x)
    assert result.dtype == np.int16
    assert result[0] == int(-1.0 * 32767)
    assert result[1] == int(-1.0 * 32767)
    assert result[2] == 0
    assert result[3] == 32767
    assert result[4] == 32767


def test_vorbis_window_dtype_and_length() -> None:
    from dpdfnet.audio import vorbis_window
    win = vorbis_window(16)
    assert win.dtype == np.float32
    assert len(win) == 16


def test_vorbis_window_cola_at_50pct_overlap() -> None:
    """w[n]^2 + w[n + hop]^2 == 1 for all n in [0, hop)."""
    from dpdfnet.audio import vorbis_window
    win_len = 32
    hop = win_len // 2
    w = vorbis_window(win_len).astype(np.float64)
    cola = w[:hop] ** 2 + w[hop:] ** 2
    np.testing.assert_allclose(cola, 1.0, atol=1e-6)


def test_make_stft_config_hop_is_half_win() -> None:
    from dpdfnet.audio import make_stft_config
    cfg = make_stft_config(64)
    assert cfg.hop_size == 32
    assert cfg.win_len == 64
    assert len(cfg.window) == 64


def test_ensure_sample_rate_identity_when_same_sr() -> None:
    from dpdfnet.audio import ensure_sample_rate
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = ensure_sample_rate(x, 16000, 16000)
    np.testing.assert_array_equal(result, x)
    assert result.dtype == np.float32


def test_apply_attn_limit_none_returns_enhanced() -> None:
    from dpdfnet.audio import apply_attn_limit

    spec_noisy = np.ones((1, 4, 2, 2), dtype=np.float32)
    spec_enh = np.full((1, 4, 2, 2), 3.0, dtype=np.float32)

    result = apply_attn_limit(spec_noisy, spec_enh, None)

    np.testing.assert_array_equal(result, spec_enh)
    assert result.dtype == np.float32


def test_apply_attn_limit_zero_db_returns_shifted_noisy_reference() -> None:
    from dpdfnet.audio import apply_attn_limit

    spec_noisy = np.arange(1, 1 + (1 * 7 * 1 * 2), dtype=np.float32).reshape(1, 7, 1, 2)
    spec_enh = np.full_like(spec_noisy, 99.0)

    result = apply_attn_limit(spec_noisy, spec_enh, 0.0)

    expected = np.zeros_like(spec_noisy)
    expected[:, 4:, :, :] = spec_noisy[:, :-4, :, :]
    np.testing.assert_array_equal(result, expected)


def test_apply_attn_limit_finite_db_blends_shifted_noisy_and_enhanced() -> None:
    from dpdfnet.audio import apply_attn_limit

    spec_noisy = np.arange(1, 1 + (1 * 7 * 1 * 2), dtype=np.float32).reshape(1, 7, 1, 2)
    spec_enh = np.full_like(spec_noisy, 8.0)
    attn_limit_db = 6.0

    result = apply_attn_limit(spec_noisy, spec_enh, attn_limit_db)

    alpha = 10.0 ** (-attn_limit_db / 20.0)
    aligned = np.zeros_like(spec_noisy)
    aligned[:, 4:, :, :] = spec_noisy[:, :-4, :, :]
    expected = alpha * aligned + (1.0 - alpha) * spec_enh
    np.testing.assert_allclose(result, expected.astype(np.float32), atol=1e-6)


def test_apply_attn_limit_inf_returns_enhanced() -> None:
    from dpdfnet.audio import apply_attn_limit

    spec_noisy = np.ones((1, 4, 2, 2), dtype=np.float32)
    spec_enh = np.full((1, 4, 2, 2), 7.0, dtype=np.float32)

    result = apply_attn_limit(spec_noisy, spec_enh, np.inf)

    np.testing.assert_array_equal(result, spec_enh)


@pytest.mark.parametrize("attn_limit_db", [-1.0, np.nan])
def test_apply_attn_limit_invalid_values_raise(attn_limit_db: float) -> None:
    from dpdfnet.audio import apply_attn_limit

    spec = np.zeros((1, 4, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="attn_limit_db"):
        apply_attn_limit(spec, spec, attn_limit_db)


# ---------------------------------------------------------------------------
# models.py unit tests
# ---------------------------------------------------------------------------

def test_get_model_info_invalid_raises() -> None:
    from dpdfnet.models import get_model_info
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model_info("nonexistent_model_xyz")


def test_available_model_entries_keys(tmp_path, monkeypatch) -> None:
    from dpdfnet import models
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    entries = models.available_model_entries()
    assert len(entries) == len(models.MODEL_REGISTRY)
    for entry in entries:
        for key in ("name", "sample_rate", "onnx_found", "ready", "cached", "onnx_path"):
            assert key in entry, f"Missing key '{key}' in entry {entry}"


def test_available_model_entries_onnx_found_when_file_present(tmp_path, monkeypatch) -> None:
    from dpdfnet import models
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    first_name = models.supported_models()[0]
    info = models.MODEL_REGISTRY[first_name]
    # Write a non-empty stub file
    (tmp_path / info.onnx_filename).write_bytes(b"stub")
    entries = {e["name"]: e for e in models.available_model_entries()}
    assert entries[first_name]["onnx_found"] is True
    assert entries[first_name]["ready"] is True


def test_available_model_entries_onnx_found_false_when_absent(tmp_path, monkeypatch) -> None:
    from dpdfnet import models
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    first_name = models.supported_models()[0]
    entries = {e["name"]: e for e in models.available_model_entries()}
    assert entries[first_name]["onnx_found"] is False


def test_hf_url_env_var_overrides(monkeypatch) -> None:
    from dpdfnet.models import _hf_url
    monkeypatch.setenv("DPDFNET_HF_REPO", "my-org/my-repo")
    monkeypatch.setenv("DPDFNET_HF_BASE_URL", "https://example.com")
    monkeypatch.setenv("DPDFNET_HF_SUBDIR", "weights")
    url = _hf_url("model.onnx", "v1")
    assert "my-org/my-repo" in url
    assert "example.com" in url
    assert "weights/model.onnx" in url
    assert "v1" in url


def test_download_quiet_and_verbose_raises() -> None:
    import dpdfnet
    with pytest.raises(ValueError, match="mutually exclusive"):
        dpdfnet.download(quiet=True, verbose=True)


# ---------------------------------------------------------------------------
# api.py unit tests
# ---------------------------------------------------------------------------

def test_enhance_file_missing_input_raises(tmp_path) -> None:
    from dpdfnet.api import enhance_file
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        enhance_file(tmp_path / "nonexistent.wav", tmp_path / "out.wav")


def test_enhance_file_default_output_path(tmp_path, monkeypatch) -> None:
    """When output_path is None, the output file is <stem>_enhanced.wav next to input."""
    import soundfile as sf
    from dpdfnet import api

    # Write a minimal valid WAV
    in_path = tmp_path / "speech.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    # Patch enhance() to return a trivial array so we don't need a real model
    monkeypatch.setattr(api, "enhance", lambda audio, sample_rate, **_kw: np.zeros_like(audio))

    result = api.enhance_file(in_path)
    assert result == in_path.with_name("speech_enhanced.wav")
    assert result.is_file()


def test_enhance_file_forwards_attn_limit_db(tmp_path, monkeypatch) -> None:
    """enhance_file() forwards attn_limit_db to enhance()."""
    import soundfile as sf
    from dpdfnet import api

    in_path = tmp_path / "speech.wav"
    out_path = tmp_path / "speech_out.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    captured: dict[str, float] = {}

    def _fake_enhance(audio, sample_rate, **kwargs):
        captured["attn_limit_db"] = kwargs["attn_limit_db"]
        return np.zeros_like(np.asarray(audio, dtype=np.float32))

    monkeypatch.setattr(api, "enhance", _fake_enhance)

    result = api.enhance_file(in_path, out_path, attn_limit_db=12.0)

    assert result == out_path.resolve()
    assert captured["attn_limit_db"] == 12.0


def test__enhance_file_with_runtime_forwards_attn_limit_db(tmp_path, monkeypatch) -> None:
    """_enhance_file_with_runtime() forwards attn_limit_db to _enhance_with_runtime()."""
    import soundfile as sf
    from dpdfnet import api

    in_path = tmp_path / "speech.wav"
    out_path = tmp_path / "speech_out.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    captured: dict[str, float] = {}

    def _fake_enhance_with_runtime(audio, sample_rate, **kwargs):
        captured["attn_limit_db"] = kwargs["attn_limit_db"]
        return np.zeros_like(np.asarray(audio, dtype=np.float32))

    monkeypatch.setattr(api, "_enhance_with_runtime", _fake_enhance_with_runtime)

    result = api._enhance_file_with_runtime(
        input_path=in_path,
        output_path=out_path,
        runtime=object(),
        model_sample_rate=16000,
        attn_limit_db=9.0,
    )

    assert result == out_path.resolve()
    assert captured["attn_limit_db"] == 9.0


def test_read_audio_unsupported_extension_raises(tmp_path) -> None:
    from dpdfnet.api import _read_audio
    bad = tmp_path / "clip.xyz"
    bad.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        _read_audio(bad)


def test_read_audio_pydub_missing_import_error(tmp_path, monkeypatch) -> None:
    """Reading an .mp3 file without pydub installed raises a helpful ImportError."""
    import builtins
    from dpdfnet.api import _read_audio

    mp3 = tmp_path / "clip.mp3"
    mp3.write_bytes(b"fake-mp3")

    real_import = builtins.__import__

    def _block_pydub(name, *args, **kwargs):
        if name == "pydub":
            raise ImportError("No module named 'pydub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pydub)
    with pytest.raises(ImportError, match="pydub"):
        _read_audio(mp3)


# ---------------------------------------------------------------------------
# cli.py unit tests
# ---------------------------------------------------------------------------

def test_cli_no_command_returns_0(capsys) -> None:
    from dpdfnet import cli
    exit_code = cli.main([])
    assert exit_code == 0


def test_cli_models_command_lists_all_models(capsys) -> None:
    from dpdfnet import cli, models
    exit_code = cli.main(["models"])
    assert exit_code == 0
    output = capsys.readouterr().out
    for name in models.supported_models():
        assert name in output


def test_cli_enhance_dir_nonexistent_input_returns_error(tmp_path, capsys) -> None:
    from dpdfnet import cli
    exit_code = cli.main(["enhance-dir", str(tmp_path / "nope"), str(tmp_path / "out")])
    assert exit_code == 2


def test_cli_enhance_dir_empty_dir_returns_error(tmp_path, capsys) -> None:
    from dpdfnet import cli
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    exit_code = cli.main(["enhance-dir", str(in_dir), str(tmp_path / "out")])
    assert exit_code == 2


def test_cli_enhance_single_file(tmp_path, monkeypatch, capsys) -> None:
    """CLI enhance command calls enhance_file and reports the output path."""
    import soundfile as sf
    from dpdfnet import cli, api

    in_path = tmp_path / "input.wav"
    out_path = tmp_path / "output.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    monkeypatch.setattr(api, "enhance_file", lambda *_a, **_kw: out_path)
    # Also patch _run_enhance's import of enhance_file
    import dpdfnet.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_run_enhance",
                        lambda _args: (print(f"Wrote enhanced audio: {out_path}") or 0))

    exit_code = cli.main(["enhance", str(in_path), str(out_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "output.wav" in output


def test_cli_enhance_forwards_attn_limit_db(tmp_path, monkeypatch) -> None:
    import soundfile as sf
    from dpdfnet import api, cli

    in_path = tmp_path / "input.wav"
    out_path = tmp_path / "output.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    captured: dict[str, float] = {}

    def _fake_enhance_file(*_args, **kwargs):
        captured["attn_limit_db"] = kwargs["attn_limit_db"]
        return out_path

    monkeypatch.setattr(cli, "print_banner", lambda **_kwargs: None)
    monkeypatch.setattr(api, "enhance_file", _fake_enhance_file)

    exit_code = cli.main(
        ["enhance", str(in_path), str(out_path), "--attn_limit_db", "12"]
    )

    assert exit_code == 0
    assert captured["attn_limit_db"] == 12.0


def test_cli_enhance_dir_forwards_attn_limit_db(tmp_path, monkeypatch) -> None:
    import soundfile as sf
    from dpdfnet import api, cli
    import dpdfnet.models as models
    import dpdfnet.onnx_backend as onnx_backend

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sf.write(str(in_dir / "input.wav"), np.zeros(160, dtype=np.float32), 16000)

    class _Info:
        sample_rate = 16000

    class _Resolved:
        info = _Info()
        onnx_path = Path("fake.onnx")

    captured: dict[str, float] = {}

    def _fake_enhance_file_with_runtime(*_args, **kwargs):
        captured["attn_limit_db"] = kwargs["attn_limit_db"]
        return Path(kwargs["output_path"])

    monkeypatch.setattr(cli, "print_banner", lambda **_kwargs: None)
    monkeypatch.setattr(models, "resolve_model", lambda **_kwargs: _Resolved())
    monkeypatch.setattr(onnx_backend, "build_runtime_model", lambda _path: object())
    monkeypatch.setattr(api, "_enhance_file_with_runtime", _fake_enhance_file_with_runtime)

    exit_code = cli.main(
        ["enhance-dir", str(in_dir), str(out_dir), "--attn-limit-db", "9"]
    )

    assert exit_code == 0
    assert captured["attn_limit_db"] == 9.0
