from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class RuntimeModel:
    session: ort.InferenceSession
    init_state: np.ndarray
    in_spec_name: str
    in_state_name: str
    out_spec_name: str
    out_state_name: str


def create_cpu_session(onnx_path: Union[str, Path]) -> ort.InferenceSession:
    path = Path(onnx_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {path}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize ONNX Runtime CPU session. "
            "Install the CPU package 'onnxruntime' and avoid GPU-only providers."
        ) from exc

    providers = session.get_providers()
    if "CPUExecutionProvider" not in providers:
        raise RuntimeError(
            f"CPUExecutionProvider is not active. Active providers: {providers}"
        )

    return session


def load_initial_state_from_metadata(session: ort.InferenceSession) -> np.ndarray:
    if len(session.get_inputs()) < 2:
        raise ValueError(
            "Expected streaming ONNX model with two inputs: (spec, state)."
        )

    meta = session.get_modelmeta().custom_metadata_map
    try:
        state_size = int(meta["state_size"])
        erb_norm_state_size = int(meta["erb_norm_state_size"])
        spec_norm_state_size = int(meta["spec_norm_state_size"])
        erb_norm_init = np.array(
            [float(x) for x in meta["erb_norm_init"].split(",")], dtype=np.float32
        )
        spec_norm_init = np.array(
            [float(x) for x in meta["spec_norm_init"].split(",")], dtype=np.float32
        )
    except KeyError as exc:
        raise ValueError(
            f"ONNX model is missing required metadata key: {exc}. "
            "Re-export the model to embed state initialisation metadata."
        ) from exc

    init_state = np.zeros(state_size, dtype=np.float32)
    init_state[0:erb_norm_state_size] = erb_norm_init
    init_state[erb_norm_state_size:erb_norm_state_size + spec_norm_state_size] = spec_norm_init
    return np.ascontiguousarray(init_state)


def build_runtime_model(
    onnx_path: Union[str, Path],
) -> RuntimeModel:
    session = create_cpu_session(onnx_path)
    init_state = load_initial_state_from_metadata(session)

    if len(session.get_inputs()) < 2 or len(session.get_outputs()) < 2:
        raise ValueError(
            "Expected streaming ONNX signature with 2 inputs and 2 outputs."
        )

    return RuntimeModel(
        session=session,
        init_state=init_state,
        in_spec_name=session.get_inputs()[0].name,
        in_state_name=session.get_inputs()[1].name,
        out_spec_name=session.get_outputs()[0].name,
        out_state_name=session.get_outputs()[1].name,
    )


def infer_win_len(session: ort.InferenceSession, default_sr: int) -> int:
    spec_shape = session.get_inputs()[0].shape
    freq_bins = spec_shape[-2] if len(spec_shape) >= 2 else None
    if isinstance(freq_bins, int) and freq_bins > 1:
        return int((freq_bins - 1) * 2)
    return int(round(default_sr * 0.02))
