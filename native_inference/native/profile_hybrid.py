"""Profile the native hybrid separately from latency measurements."""
import collections
import json
from pathlib import Path

import onnxruntime as ort
from probe import initial_state, spectra

options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
options.enable_profiling = True
options.profile_file_prefix = 'results/scratch/hybrid'
options.register_custom_ops_library('/bench/build/baseline/libdpdf_ort.so')
sess = ort.InferenceSession('models/native_blocks/hybrid.onnx', options, providers=['CPUExecutionProvider'])
state = initial_state(sess)
for frame in spectra(300):
    _, state = sess.run(None, {'spec': frame, 'state_in': state})
events = json.loads(Path(sess.end_profiling()).read_text())
ops, nodes = collections.Counter(), collections.Counter()
for event in events:
    if event.get('cat') == 'Node' and event['name'].endswith('_kernel_time'):
        ops[event['args'].get('op_name', 'unknown')] += event['dur']
        nodes[event['name']] += event['dur']
result = {'note': 'Instrumented durations; not unprofiled CPU attribution.',
          'ops_us': ops.most_common(), 'nodes_us': nodes.most_common(30)}
Path('results/hybrid_profile.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
