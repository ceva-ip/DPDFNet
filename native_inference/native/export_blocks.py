"""Extract DPRNN oracles from audited models and replace them with native ops.

Only explicitly audited model hashes are accepted. Each block boundary and
attribute is checked before replacement; no general ONNX pattern matching is
claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

AUDITED_MODELS = {
    '7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631': {
        'profile': 'dpdfnet8_48khz_hr', 'blocks_per_branch': 8, 'state_size': 90228,
    },
    '7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b': {
        'profile': 'dpdfnet2_48khz_hr', 'blocks_per_branch': 2, 'state_size': 56436,
    },
}
WEIGHT_FLOATS = 87552


def attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def export(source, output):
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    audited = AUDITED_MODELS.get(source_sha)
    if audited is None:
        supported = ', '.join(sorted(v['profile'] for v in AUDITED_MODELS.values()))
        raise ValueError(f'Unsupported model SHA-256; audited models: {supported}')
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    if int(metadata.get('state_size', 0)) != audited['state_size']:
        raise ValueError('State size does not match the audited model hash')
    constants = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    initializers = {x.name: x for x in model.graph.initializer}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in model.graph.value_info}
    replacements, removed, manifest = {}, set(), []
    for branch, freq in [('erb', 40), ('df', 48)]:
        for index in range(audited['blocks_per_branch']):
            name = f'{branch}_{index}'
            prefix = f'/model/enc/dprnn_{branch}/blocks.{index}/'
            nodes = [n for n in model.graph.node if n.name.startswith(prefix)]
            by_name = {n.name[len(prefix):]: n for n in nodes}
            gru = by_name['intra_gru/GRU']
            ga = attributes(gru)
            if ga != {'direction': b'bidirectional', 'hidden_size': 64, 'linear_before_reset': 1}:
                raise ValueError(f'Unexpected GRU attributes: {ga}')
            if np.any(constants[gru.input[5]]):
                raise ValueError('Intra-frequency initial state must be zero')
            norms = [by_name[f'ln_{path}/LayerNormalization'] for path in ['intra', 'inter']]
            eps = [attributes(n)['epsilon'] for n in norms]
            if any(attributes(n).get('axis') != -1 for n in norms):
                raise ValueError('Unsupported normalization axis')
            w, r, b = [constants[gru.input[i]] for i in [1, 2, 3]]
            assert w.shape == r.shape == (2, 192, 64) and b.shape == (2, 384)
            order = np.r_[64:128, 0:64, 128:192]  # ONNX z,r,h -> native r,z,n
            chunks = [w[:, order, :].transpose(0, 2, 1), r[:, order, :].transpose(0, 2, 1),
                      b.reshape(2, 2, 192)[:, :, order]]
            fi = by_name['fc_intra/MatMul']
            fib = by_name['fc_intra/Add']
            fib_name = next(x for x in fib.input if x in constants)
            chunks.extend([constants[fi.input[1]], constants[fib_name],
                           constants[norms[0].input[1]], constants[norms[0].input[2]]])
            temporal = [by_name['inter_gru/grucell/Gemm'], by_name['inter_gru/grucell/Gemm_1']]
            fo = by_name['fc_inter/Gemm']
            for n in temporal + [fo]:
                a = attributes(n)
                assert a.get('transB') == 1 and a.get('transA', 0) == 0
                assert a.get('alpha', 1) == a.get('beta', 1) == 1
            chunks.extend([constants[n.input[1]].T for n in temporal])
            chunks.extend([constants[n.input[2]] for n in temporal])
            chunks.extend([constants[fo.input[1]].T, constants[fo.input[2]],
                           constants[norms[1].input[1]], constants[norms[1].input[2]]])
            packed = np.concatenate([c.ravel() for c in chunks]).astype('<f4')
            assert packed.size == WEIGHT_FLOATS and np.isfinite(packed).all()
            packed.tofile(output / f'{name}.f32')
            x_name = by_name['Transpose'].input[0]
            y_name = by_name['Add_1'].output[0]
            state_slice = by_name['inter_gru/Slice']
            s_name = state_slice.output[0]
            so_name = by_name['inter_gru/Reshape_1'].output[0]
            assert shapes[x_name] == shapes[y_name] == [1, 64, 1, freq]
            start, end = [int(constants[x].item()) for x in state_slice.input[1:3]]
            assert end-start == freq*64 and state_slice.input[0] == 'state_in'
            produced = {v for n in nodes for v in n.output}
            outside = {v for n in model.graph.node if not n.name.startswith(prefix) for v in n.input}
            assert produced & outside == {y_name, so_name}, produced & outside
            oracle_nodes = [n for n in nodes if n.name != state_slice.name]
            used = {v for n in oracle_nodes for v in n.input if v}
            oracle_produced = {v for n in oracle_nodes for v in n.output}
            assert used - oracle_produced - constants.keys() == {x_name, s_name}
            graph = helper.make_graph(oracle_nodes, name,
                [helper.make_tensor_value_info(x_name, onnx.TensorProto.FLOAT, [1,64,1,freq]),
                 helper.make_tensor_value_info(s_name, onnx.TensorProto.FLOAT, [freq*64])],
                [helper.make_tensor_value_info(y_name, onnx.TensorProto.FLOAT, [1,64,1,freq]),
                 helper.make_tensor_value_info(so_name, onnx.TensorProto.FLOAT, [freq*64])],
                initializer=[initializers[k] for k in sorted(used & constants.keys())])
            oracle = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
            onnx.checker.check_model(oracle)
            onnx.save(oracle, str(output / f'{name}.onnx'))
            custom = helper.make_node('DpdfDprnn', [x_name, s_name], [y_name, so_name],
                name=prefix+'Native', domain='ceva.dpdfnet.experimental',
                freq=freq, weights=packed.tolist(), intra_epsilon=float(eps[0]), inter_epsilon=float(eps[1]))
            replacements[nodes[0].name] = [state_slice, custom]
            removed.update(n.name for n in nodes)
            manifest.append({'name': name, 'frequency': freq, 'eps': eps, 'state_start': start,
                             'state_end': end, 'input_name': x_name, 'output_name': y_name,
                             'state_input_name': s_name, 'state_output_name': so_name,
                             'weight_floats': packed.size, 'replaced_nodes': len(nodes)})
    updated = []
    for node in model.graph.node:
        if node.name in replacements:
            updated.extend(replacements[node.name])
        elif node.name not in removed:
            updated.append(node)
    del model.graph.node[:]
    model.graph.node.extend(updated)
    used = {x for n in updated for x in n.input}
    live = used | {x for n in updated for x in n.output}
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    kept_shapes = [v for v in model.graph.value_info if v.name in live]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_shapes)
    model.opset_import.append(helper.make_opsetid('ceva.dpdfnet.experimental', 1))
    onnx.checker.check_model(model)
    onnx.save(model, str(output / 'hybrid.onnx'))
    (output / 'manifest.json').write_text(json.dumps({
        'source_sha256': source_sha,
        'profile': audited['profile'],
        'state_size': int(metadata['state_size']),
        'blocks': manifest,
    }, indent=2)+'\n')
    print(f'Exported {len(manifest)} blocks for {audited["profile"]}; hybrid graph has {len(updated)} nodes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
