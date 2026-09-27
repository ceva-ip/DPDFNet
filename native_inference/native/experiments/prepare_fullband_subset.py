"""Select 50 official test clips and preserve their original mixtures.

Replay the unselected generator iterations using zero arrays of the exact
original shape. This preserves every random draw and style counter. Process
all cuts of a source recording if any is selected, because upstream reuses
the previous cut's noise residual. Only selected clip IDs are written.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import zipfile
import zlib

import numpy as np
import soundfile as sf
from download_fullband import ROOT, WHAM, RemoteZip, fetch, extract_selected


def wave_header(url, template, info):
    data = fetch(url, info.header_offset, info.header_offset+min(info.compress_size+512, 65536)-1)
    assert data[:4] == b'PK\x03\x04'
    name_len, extra_len = struct.unpack_from('<HH',data,26)
    payload = data[30+name_len+extra_len:]
    header = zlib.decompressobj(-15).decompress(payload,65536) if info.compress_type == 8 else payload
    assert header[:4] == b'RIFF' and header[8:12] == b'WAVE'
    position = 12
    channels = rate = block = None
    while position+8 <= len(header):
        kind = header[position:position+4]
        length = struct.unpack_from('<I',header,position+4)[0]
        if kind == b'fmt ':
            _, channels, rate, _, block = struct.unpack_from('<HHIIH',header,position+8)
        if kind == b'data':
            assert rate == 48000 and block and length % block == 0
            return {'frames':length//block,'channels':channels,'sample_rate':rate}
        position += 8+length+(length%2)
    raise ValueError('WAV header not found: '+info.filename)


def metadata():
    target=ROOT/'wave_metadata.json'
    if target.exists():
        return json.loads(target.read_text())
    cuts=json.loads((ROOT/'test_files.json').read_text())
    import csv
    noise_rows=list(csv.DictReader((ROOT/'data/WHAM48kHz/high_res_wham/high_res_metadata.csv').read_text().splitlines()))
    noise_names={r['Filename'] for r in noise_rows if r['WHAM! Split'].lower()=='test'}
    assets=json.loads((ROOT/'speech_assets.json').read_text(encoding='utf-8-sig'))
    result={}
    for url,destination,names in [(WHAM,ROOT/'data/WHAM48kHz',noise_names)]+[
        (a['browser_download_url'],ROOT/'data/EARS',{n+'.wav' for n in cuts[a['name'][:-4]]}) for a in assets]:
        remote=RemoteZip(url)
        with zipfile.ZipFile(remote) as z:
            infos=[i for i in z.infolist() if Path(i.filename).name in names]
        def get(info):
            path=destination/info.filename
            if path.exists():
                w=sf.info(path)
                record={'frames':w.frames,'channels':w.channels,'sample_rate':w.samplerate}
            else:
                record=wave_header(url,remote,info)
            return str(path),record
        with ThreadPoolExecutor(max_workers=12) as pool:
            result.update(pool.map(get,infos))
        print('Headers ready',url.split('/')[-1],len(infos),flush=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    return result


def select(meta, count=50):
    cuts=json.loads((ROOT/'test_files.json').read_text())
    source_files=[str(ROOT/'data/EARS'/s/(f+'.wav')) for s in sorted(cuts) for f in cuts[s]]
    np.random.seed(42)
    np.random.shuffle(source_files)
    eligible=[]
    for path in source_files:
        p=Path(path)
        for start,end in cuts[p.parent.name][p.stem]:
            length=len(range(*slice(start,end).indices(meta[path]['frames'])))
            if length <= 29*48000:
                style=p.stem.split('_')[1] if p.stem.startswith(('emo_','style_')) else p.stem.split('_')[0]
                eligible.append({'id':f'{len(eligible):05}','source':path,'speaker':p.parent.name,
                                 'style':style,'start':start,'end':end,'frames':length})
    # Allocate the requested clip count proportionally to each style, then use
    # deterministic hashes and speaker balancing within each group.
    styles=sorted({e['style'] for e in eligible})
    groups={s:[e for e in eligible if e['style']==s] for s in styles}
    quotas={s:count*len(groups[s])//len(eligible) for s in styles}
    remaining=count-sum(quotas.values())
    for s in sorted(styles,key=lambda s:(-(count*len(groups[s])%len(eligible)),s))[:remaining]:
        quotas[s]+=1
    counts={s:0 for s in cuts}
    chosen=[]
    for style in styles:
        pool=list(groups[style])
        for _ in range(quotas[style]):
            candidate=min(pool,key=lambda e:(counts[e['speaker']],hashlib.sha256(('20260926:'+e['id']).encode()).hexdigest()))
            chosen.append(candidate)
            counts[candidate['speaker']]+=1
            pool.remove(candidate)
    selection={'seed':20260926,'full_test_clips':len(eligible),'selected_clips':len(chosen),
               'fraction':len(chosen)/len(eligible),'by_speaker':counts,
               'method':'Proportional style quotas, deterministic SHA256 ordering, greedy speaker balancing; selected before model scoring',
               'clips':sorted(chosen,key=lambda e:e['id'])}
    (ROOT/'subset_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('Selected',selection['selected_clips'],'/',selection['full_test_clips'],counts,flush=True)
    return selection


def generate(meta,selection,planning):
    source=(ROOT/'generate_ears_wham.py').read_text()
    assert source.count('for subset in ["train", "valid"]:')==1
    source=source.replace('for subset in ["train", "valid"]:','for subset in []:')
    if planning:
        source=source.replace('"EARS-WHAM_v2"','"EARS-WHAM_v2_plan"')
    namespace={'__name__':'ears_generator','__file__':str(ROOT/'generate_ears_wham.py')}
    exec(compile(source,namespace['__file__'],'exec'),namespace)
    selected_ids={e['id'] for e in selection['clips']}
    active_sources={e['source'] for e in selection['clips']}
    active=False
    needed=set(active_sources)
    original_read=namespace['read']
    original_filter=namespace['highpass_biquad']
    original_save=namespace['save_files']
    original_meter=namespace['pyln'].Meter

    def read(path,*args,**kwargs):
        nonlocal active
        path=str(path)
        if '/EARS/' in path:
            active=path in active_sources
        if active:
            needed.add(path)
        if active and not planning:
            return original_read(path,*args,**kwargs)
        info=meta[path]
        shape=(info['frames'],info['channels']) if kwargs.get('always_2d') or info['channels']>1 else (info['frames'],)
        return np.zeros(shape,dtype=np.float64),info['sample_rate']

    class Meter:
        def __init__(self,*args,**kwargs):
            self.actual=original_meter(*args,**kwargs)
        def integrated_loudness(self,audio):
            return self.actual.integrated_loudness(audio) if active and not planning else 0.0

    def save(*args,**kwargs):
        identifier=args[3]
        if not planning and f'{identifier:05}' in selected_ids:
            return original_save(*args,**kwargs)
        return identifier+1

    namespace['read']=read
    namespace['highpass_biquad']=lambda audio,**kw: original_filter(audio,**kw) if active and not planning else audio
    namespace['pyln'].Meter=Meter
    namespace['save_files']=save
    namespace['exists']=lambda path:False
    namespace['makedirs']=lambda path:os.makedirs(path,exist_ok=True)
    args=SimpleNamespace(data_dir=str(ROOT/'data'),sr=48000,min_snr=-2.5,max_snr=17.5,min_length=4.,cut_length=10.,
                         cutoff_freq=75.,min_dB=-55.,ramp_time_in_ms=10,max_time_test_set_in_s=29)
    setattr(args,'16k',False)
    os.chdir(ROOT)
    try:
        namespace['main'](args)
    finally:
        namespace['pyln'].Meter=original_meter
    return needed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--count',type=int,default=50)
    args=parser.parse_args()
    meta=metadata()
    selection=select(meta,args.count)
    if not args.generate_only:
        needed=generate(meta,selection,True)
        (ROOT/'subset_needed.json').write_text(json.dumps(sorted(needed),indent=2)+'\n')
        assets=json.loads((ROOT/'speech_assets.json').read_text(encoding='utf-8-sig'))
        jobs=[(WHAM,ROOT/'data/WHAM48kHz')]+[(a['browser_download_url'],ROOT/'data/EARS') for a in assets]
        with ThreadPoolExecutor(max_workers=3) as pool:
            records=list(pool.map(lambda job:extract_selected(*job,lambda name:str(job[1]/name) in needed),jobs))
        (ROOT/'subset_download_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    generate(meta,selection,False)
    print('Test subset ready',flush=True)


if __name__=='__main__':
    main()
