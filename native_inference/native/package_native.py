"""Reproduce the versioned, Python-free integration artifacts from pinned ONNX.

Only publishers need Python/ONNX. Consumers build the checked-in artifact set.
Run from any directory; output is independent of source path and wall clock.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

from generate_extended import extend

MODELS = ('dpdfnet2_48khz_hr', 'dpdfnet8_48khz_hr')


def package(models, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        with tempfile.TemporaryDirectory(prefix='dpdf-export-') as temporary:
            generated = Path(temporary)
            extend(models / f'{name}.onnx', generated, symbol_prefix=name)
            target = output / name
            target.mkdir(parents=True, exist_ok=True)
            for filename in ('generated_model.c', 'generated_model.h', 'weights.f32', 'manifest.json'):
                shutil.copyfile(generated / filename, target / filename)
    records = []
    for name in MODELS:
        for filename in ('generated_model.c', 'generated_model.h', 'weights.f32', 'manifest.json'):
            relative = f'{name}/{filename}'
            digest = hashlib.sha256((output / relative).read_bytes()).hexdigest()
            records.append(f'{digest}  {relative}\n')
    (output / 'SHA256SUMS').write_text(''.join(records), encoding='ascii', newline='\n')
    print(f'Packaged both models and SHA256SUMS in {output}')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=root / 'models')
    parser.add_argument('--output', type=Path, default=root / 'artifacts' / 'v1')
    args = parser.parse_args()
    package(args.models, args.output)
