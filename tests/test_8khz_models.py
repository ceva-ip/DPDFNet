from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@contextmanager
def _model_import_path():
    previous_path = list(sys.path)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        yield
    finally:
        sys.path[:] = previous_path


def test_dpdfnet8khz_waveform_forward_smoke() -> None:
    torch = pytest.importorskip("torch")
    with _model_import_path():
        from dpdfnet_8khz import DPDFNet8KHz

    model = DPDFNet8KHz(dprnn_num_blocks=2).eval()
    waveform = torch.randn(1, 8000, dtype=torch.float32)

    with torch.no_grad():
        enhanced, lsnr = model(waveform)

    assert model.freq_bins == 81
    assert model.nb_df == 80
    assert enhanced.ndim == 2
    assert enhanced.shape[0] == 1
    assert lsnr.shape[0] == 1


def test_dpdfnet8khz_streaming_frame_forward_smoke() -> None:
    torch = pytest.importorskip("torch")
    from onnx_model.dpdfnet_8khz import DPDFNet8KHz

    model = DPDFNet8KHz(dprnn_num_blocks=2).eval()
    state = model.initial_state(dtype=torch.float32)
    spec = torch.randn(1, 1, 81, 2, dtype=torch.float32)

    with torch.no_grad():
        enhanced, next_state = model(spec, state)

    assert model.freq_bins == 81
    assert model.nb_df == 80
    assert enhanced.shape == spec.shape
    assert next_state.shape == state.shape


def test_dpdfnet8khz_random_weight_export_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from argparse import Namespace

    from onnx_model.export_dpdfnet_8khz_to_onnx import (
        add_meta_data,
        build_meta_data,
        build_model,
        export_onnx,
    )
    from onnx_model.export_utils import check_single_file_onnx, validate_onnx_runtime

    args = Namespace(model_name="dpdfnet2_8khz", dprnn_num_blocks=2, checkpoint=None)
    model = build_model(args)
    output_path = tmp_path / "dpdfnet2_8khz.onnx"

    wrapper = export_onnx(
        model,
        output_path,
        opset=17,
        use_dynamic_axes=False,
        exporter="legacy",
    )
    add_meta_data(output_path, build_meta_data(model, args.model_name))

    check_single_file_onnx(output_path)
    validate_onnx_runtime(wrapper, model, output_path)
