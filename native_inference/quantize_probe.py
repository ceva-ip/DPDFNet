"""Experimental partial dynamic int8 conversion, NOT a production model.

Lower the pinned graph's Gemm nodes to MatMul + Add so ORT's dynamic
quantizer can cover them. Bidirectional GRU nodes remain FP32; ORT 1.27's
IntegerOpsRegistry has no GRU entry. This does not reproduce faster-enhancer.c.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.model.resolve() == args.output.resolve():
        parser.error('output must differ from the source model')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.model))
    constants = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    nodes = []
    lowered = 0
    # Generated names are isolated; fail rather than overwrite source tensors.
    names = {x for n in model.graph.node for x in list(n.input) + list(n.output)} | set(constants)
    for index, node in enumerate(model.graph.node):
        if node.op_type != 'Gemm':
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get('alpha', 1) != 1 or attrs.get('beta', 1) != 1 or attrs.get('transA', 0) != 0:
            raise ValueError(f'Unsupported Gemm attributes: {node.name}: {attrs}')
        weight = node.input[1]
        base = f'probe_gemm_{index}'
        if any(x.startswith(base) for x in names):
            raise ValueError(f'Name collision: {base}')
        if attrs.get('transB', 0):
            if weight not in constants:
                raise ValueError(f'Nonconstant Gemm weight: {weight}')
            transposed = base + '_weight'
            model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(constants[weight].T), transposed))
            weight = transposed
        has_bias = len(node.input) > 2 and bool(node.input[2])
        output = base + '_product' if has_bias else node.output[0]
        nodes.append(helper.make_node('MatMul', [node.input[0], weight], [output], name=base + '_matmul'))
        if has_bias:
            nodes.append(helper.make_node('Add', [output, node.input[2]], list(node.output), name=base + '_bias'))
        lowered += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {x for n in nodes for x in n.input}
    retained = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    onnx.checker.check_model(model)
    lowered_path = args.output.with_suffix('.fp32_lowered.onnx')
    if lowered_path.resolve() == args.model.resolve():
        parser.error('intermediate path would overwrite the source')
    onnx.save(model, str(lowered_path))
    # Batched/grouped MatMul weights are 3D. ORT 1.27's fused dynamic kernel
    # rejects their per-channel zero-point shape, so keep those paths FP32.
    dims = {x.name: len(x.dims) for x in model.graph.initializer}
    selected = [n.name for n in nodes if n.op_type == 'MatMul' and dims.get(n.input[1]) == 2]
    quantize_dynamic(str(lowered_path), str(args.output), op_types_to_quantize=['MatMul'],
                     nodes_to_quantize=selected,
                     per_channel=True, reduce_range=True, weight_type=QuantType.QInt8,
                     extra_options={'MatMulConstBOnly': True})
    onnx.checker.check_model(onnx.load(str(args.output)))
    print(f'Lowered {lowered} Gemm nodes; selected {len(selected)} 2D MatMuls. '
          f'Saved {args.output}; GRU, Conv and grouped MatMul remain FP32.')


if __name__ == '__main__':
    main()
