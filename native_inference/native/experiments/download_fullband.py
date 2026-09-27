"""Fetch official EARS-WHAM test assets only, with ZIP CRC checks and SHA256 manifest."""
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path('/bench/scratch/fullband')
WHAM = 'https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip'


def fetch(url, start=None, end=None):
    headers = {'User-Agent': 'DPDFNet-research'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                if start is not None:
                    assert r.status == 206, 'Server did not honor range request'
                    assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                data = r.read()
                if start is not None:
                    assert len(data) == end-start+1
                return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, template=None):
        self.url = url
        if template is not None:
            self.size, self.etag = template.size, template.etag
            self.pos, self.cache = 0, list(template.cache)
            return
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=120) as r:
            self.size = int(r.headers['Content-Length'])
            self.etag = r.headers.get('ETag')
        self.pos = 0
        self.cache = []

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos+n)
        if end <= self.pos:
            return b''
        for a, b, data in self.cache:
            if a <= self.pos and end <= b:
                result = data[self.pos-a:end-a]
                self.pos = end
                return result
        a = self.pos
        # Cache central-directory reads, plus local headers and adjacent data.
        b = min(self.size, max(end, a+65536))
        data = fetch(self.url, a, b-1)
        self.cache.append((a, b, data))
        self.cache = self.cache[-8:]
        self.pos = end
        return data[:end-a]


def extract_selected(url, destination, predicate):
    remote = RemoteZip(url)
    records = []
    with zipfile.ZipFile(remote) as z:
        selected = [i for i in z.infolist() if not i.is_dir() and predicate(i.filename)]
        print(url, 'selected', len(selected), 'compressed MB', round(sum(i.compress_size for i in selected)/1e6), flush=True)
        def extract(info):
            target = (destination/info.filename).resolve()
            assert target.is_relative_to(destination.resolve())
            if target.exists() and target.stat().st_size == info.file_size:
                import zlib
                data = target.read_bytes()
                assert zlib.crc32(data) == info.CRC
            else:
                with zipfile.ZipFile(RemoteZip(url, remote)) as local:
                    data = local.read(info)  # validates ZIP CRC
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return {'path': str(target.relative_to(ROOT)), 'size': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(), 'crc32': info.CRC}
        with ThreadPoolExecutor(max_workers=4 if url == WHAM else 2) as pool:
            for index, record in enumerate(pool.map(extract, selected)):
                records.append(record)
                if index % 20 == 0:
                    print(url.split('/')[-1], index+1, '/', len(selected), flush=True)
    return {'url': url, 'etag': remote.etag, 'archive_size': remote.size, 'files': records}


def main():
    metadata_path = ROOT/'data/WHAM48kHz/high_res_wham/high_res_metadata.csv'
    extract_selected(WHAM, ROOT/'data/WHAM48kHz', lambda n: n.endswith('high_res_metadata.csv'))
    rows = list(csv.DictReader(metadata_path.read_text().splitlines()))
    names = {row['Filename'] for row in rows if row['WHAM! Split'].lower() == 'test'}
    speech = json.loads((ROOT/'test_files.json').read_text())
    assets = json.loads((ROOT/'speech_assets.json').read_text(encoding='utf-8-sig'))
    jobs = [(WHAM, ROOT/'data/WHAM48kHz', lambda n: n.endswith('high_res_metadata.csv') or Path(n).name in names)]
    for asset in assets:
        speaker = asset['name'][:-4]
        needed = set(speech[speaker])
        jobs.append((asset['browser_download_url'], ROOT/'data/EARS', lambda n, needed=needed: Path(n).stem in needed and n.endswith('.wav')))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda job: extract_selected(*job), jobs))
    (ROOT/'download_manifest.json').write_text(json.dumps(results, indent=2)+'\n')
    print('All test assets downloaded and checked', flush=True)


if __name__ == '__main__':
    main()
