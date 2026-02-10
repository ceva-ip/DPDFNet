import argparse
from pathlib import Path

import torch
from torch import nn

from streaming.dpdfnet_48khz_hr import DPDFNet48HR, correct_state_dict


class DPDFNet48HROnnxWrapper(nn.Module):
    def __init__(self, model: DPDFNet48HR):
        super().__init__()
        self.model = model

    def forward(self, spec: torch.Tensor, state_in: torch.Tensor):
        spec_e, state_out = self.model(spec, state_in)
        return spec_e, state_out


def build_model(args: argparse.Namespace) -> DPDFNet48HR:
    model = DPDFNet48HR(
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


def export_onnx(model: DPDFNet48HR, output_path: Path, opset: int, use_dynamic_axes: bool) -> None:
    wrapper = DPDFNet48HROnnxWrapper(model).eval()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export streaming 48 kHz DPDFNet model to ONNX.")
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
        help="Optional .pth checkpoint for DPDFNet48HR weights.",
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

    model = build_model(args)
    export_onnx(model, output, opset=args.opset, use_dynamic_axes=args.dynamic_axes)
    print(f"[OK] Exported ONNX model to: {output}")
    print(f"[INFO] State vector size: {model.state_size()}")
    print(f"[INFO] Frequency bins: {model.freq_bins}")


if __name__ == "__main__":
    main()
