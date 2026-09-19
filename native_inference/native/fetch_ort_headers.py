"""Download only the pinned official ORT headers needed for the optional adapter."""
import hashlib
from pathlib import Path
import urllib.request

HEADERS = {
    'onnxruntime_c_api.h': 'c4ed26d7bbc7b649c1c38622424d88dfb1a28e3d3594c5f1456cf458814e8381',
    'onnxruntime_cxx_api.h': 'c5f9bb73d7674cf99a197b383a144eb2a5eda71cc060193a86c7d24e6fa36ed7',
    'onnxruntime_cxx_inline.h': '114b4d008351eaa4462560438474cebd104f74259cc8af607acc0b97453200d2',
    'onnxruntime_ep_c_api.h': 'f65ac849718cabe3ff4902698dd2de643c3b3dbe0010242b2b0407548e7ee00c',
    'onnxruntime_float16.h': '88b242845d25981633a0bbd1c148e273cf8bfb016ea3f57c4af41a06530f72b0',
}


def main():
    folder = Path(__file__).resolve().parents[1] / 'vendor' / 'onnxruntime'
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in HEADERS.items():
        target = folder / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(f'Verified {name}')
            continue
        url = f'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.27.0/include/onnxruntime/core/session/{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Header checksum changed: {name}')
        target.write_bytes(data)
        print(f'Downloaded and verified {name}')


if __name__ == '__main__':
    main()
