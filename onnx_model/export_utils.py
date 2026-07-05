from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import external_data_helper


def _external_data_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".data")


def model_uses_external_data(model: onnx.ModelProto) -> bool:
    return any(
        tensor.data_location == onnx.TensorProto.EXTERNAL or bool(tensor.external_data)
        for tensor in model.graph.initializer
    )


def load_onnx_model(path: Path) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=True)


def save_single_file_onnx(model: onnx.ModelProto, path: Path) -> None:
    external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model, str(path), save_as_external_data=False)


def remove_unused_external_data_sidecar(path: Path) -> None:
    model = load_onnx_model(path)
    if model_uses_external_data(model):
        return

    sidecar = _external_data_sidecar(path)
    if sidecar.exists():
        sidecar.unlink()
        print(f"[INFO] Removed stale external data sidecar: {sidecar}")


def check_single_file_onnx(path: Path) -> None:
    model = load_onnx_model(path)
    if model_uses_external_data(model):
        raise RuntimeError(
            f"Export still references external ONNX tensor data next to {path}. "
            "This model should be small enough for a single .onnx file."
        )
    onnx.checker.check_model(model)
    remove_unused_external_data_sidecar(path)


def validate_onnx_runtime(
    wrapper: Any,
    model: Any,
    path: Path,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    import onnxruntime as ort
    import torch

    torch.manual_seed(0)
    spec = torch.randn(1, 1, model.freq_bins, 2, dtype=torch.float32)
    state_in = model.initial_state(dtype=torch.float32)

    with torch.no_grad():
        ref_spec, ref_state = wrapper(spec, state_in)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    ort_spec, ort_state = session.run(
        ["spec_e", "state_out"],
        {
            "spec": np.ascontiguousarray(spec.cpu().numpy(), dtype=np.float32),
            "state_in": np.ascontiguousarray(state_in.cpu().numpy(), dtype=np.float32),
        },
    )

    np.testing.assert_allclose(
        ort_spec,
        ref_spec.cpu().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg="ONNX Runtime spec_e output does not match PyTorch export wrapper.",
    )
    np.testing.assert_allclose(
        ort_state,
        ref_state.cpu().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg="ONNX Runtime state_out output does not match PyTorch export wrapper.",
    )
    print("[INFO] ONNX Runtime validation matched PyTorch wrapper.")
