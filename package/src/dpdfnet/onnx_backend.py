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


def load_initial_state(
    session: ort.InferenceSession,
    state_path: Union[str, Path],
) -> np.ndarray:
    path = Path(state_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Initial state file not found: {path}")

    with np.load(path) as data:
        if "init_state" not in data:
            raise ValueError(f"Missing 'init_state' key in state file: {path}")
        init_state = np.ascontiguousarray(
            data["init_state"].astype(np.float32, copy=False)
        )

    if len(session.get_inputs()) < 2:
        raise ValueError(
            "Expected streaming ONNX model with two inputs: (spec, state)."
        )

    expected_shape = session.get_inputs()[1].shape
    if len(expected_shape) != init_state.ndim:
        raise ValueError(
            f"Initial state rank mismatch: expected={expected_shape}, actual={init_state.shape}"
        )
    for exp_dim, act_dim in zip(expected_shape, init_state.shape):
        if isinstance(exp_dim, int) and exp_dim != act_dim:
            raise ValueError(
                f"Initial state shape mismatch: expected={expected_shape}, actual={init_state.shape}"
            )
    return init_state


def build_runtime_model(
    onnx_path: Union[str, Path],
    state_path: Union[str, Path],
) -> RuntimeModel:
    session = create_cpu_session(onnx_path)
    init_state = load_initial_state(session, state_path)

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
