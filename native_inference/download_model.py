"""Fetch an exact audited public 48 kHz HR model."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

MODELS = {
    'dpdfnet8_48khz_hr': '7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631',
    'dpdfnet2_48khz_hr': '7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b',
}


def download(name):
    expected = MODELS[name]
    target = Path(__file__).resolve().parent / 'models' / f'{name}.onnx'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        print(f'Already verified: {target}')
        return
    url = f'https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/{name}.onnx?download=true'
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f'Model checksum changed: expected {expected}, received {actual}')
    temporary = target.with_suffix('.download')
    temporary.write_bytes(data)
    temporary.replace(target)
    print(f'Downloaded and verified: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', default='dpdfnet8_48khz_hr',
                        choices=[*MODELS, 'all'])
    args = parser.parse_args()
    for name in MODELS if args.model == 'all' else [args.model]:
        download(name)


if __name__ == '__main__':
    main()
