import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from streaming.dpdfnet import DPDFNet, correct_state_dict


class DPDFNetOnnxWrapper(nn.Module):
    def __init__(self, model: DPDFNet):
        super().__init__()
        self.model = model

    def forward(self, spec: torch.Tensor, state_in: torch.Tensor):
        spec_e, state_out = self.model(spec, state_in)
        return spec_e, state_out


def build_model(args: argparse.Namespace) -> DPDFNet:
    model = DPDFNet(
        conv_kernel_inp=(3, 3),
        conv_ch=64,
        enc_gru_dim=256,
        erb_dec_gru_dim=256,
        df_dec_gru_dim=256,
        enc_lin_groups=32,
        lin_groups=16,
        upsample_conv_type="subpixel",
        group_linear_type="loop",
        point_wise_type="cnn",
        separable_first_conv=True,
        dprnn_num_blocks=args.dprnn_num_blocks,
    )
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        stream_state_dict = correct_state_dict(state_dict)
        model.load_state_dict(stream_state_dict, strict=True)
    model.eval()
    return model


def export_onnx(model: DPDFNet, output_path: Path, opset: int, use_dynamic_axes: bool) -> None:
    wrapper = DPDFNetOnnxWrapper(model).eval()
    spec = torch.randn(1, 1, model.freq_bins, 2, dtype=torch.float32)
    state_in = model.initial_state(dtype=torch.float32)

    export_kwargs = dict(
        f=str(output_path),
        input_names=["spec", "state_in"],
        output_names=["spec_e", "state_out"],
        opset_version=opset,
        do_constant_folding=True,
    )
    if use_dynamic_axes:
        export_kwargs["dynamic_axes"] = {
            "spec": {0: "batch", 1: "time"},
            "spec_e": {0: "batch", 1: "time"},
        }

    # Prefer the newer exporter and fall back to legacy if needed.
    try:
        torch.onnx.export(wrapper, (spec, state_in), dynamo=True, **export_kwargs)
    except Exception as dynamo_err:
        print(f"[WARN] ONNX dynamo export failed: {dynamo_err}")
        print("[INFO] Falling back to legacy torch.onnx.export path.")
        torch.onnx.export(wrapper, (spec, state_in), dynamo=False, **export_kwargs)


def export_initial_state(model: DPDFNet, state_path: Path) -> tuple[int, ...]:
    init_state = model.initial_state(dtype=torch.float32).cpu().numpy()
    np.savez_compressed(
        state_path,
        init_state=init_state,
        state_shape=np.asarray(init_state.shape, dtype=np.int64),
        state_dtype=np.asarray(str(init_state.dtype)),
    )
    return tuple(init_state.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export streaming DPDFNet model to ONNX.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional .pth checkpoint for DPDFNet weights.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Export with dynamic batch/time axes for spec input/output.",
    )
    parser.add_argument(
        "--dprnn-num-blocks",
        type=int,
        default=2,
        help="Number of DPRNN blocks in encoder branches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    state_output = output.with_name(f"{output.stem}_state.npz")
    state_output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    export_onnx(model, output, opset=args.opset, use_dynamic_axes=args.dynamic_axes)
    state_shape = export_initial_state(model, state_output)
    print(f"[OK] Exported ONNX model to: {output}")
    print(f"[OK] Exported initial state to: {state_output}")
    print(f"[INFO] Initial state shape: {state_shape}")
    print(f"[INFO] State vector size: {model.state_size()}")
    print(f"[INFO] Frequency bins: {model.freq_bins}")


if __name__ == "__main__":
    main()
