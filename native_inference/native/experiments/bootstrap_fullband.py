"""Download pinned official benchmark code/metadata and the SIGMOS model."""
import hashlib
import json
from download_fullband import ROOT, WHAM, fetch, extract_selected

EARS_REV='36dc8a88cb2ebf7cc746b51cf2b876bb570bc3e0'
SIG_REV='bf4525153b6ed998f19d9e79ff1fd00f55dec42b'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    files={name:f'https://raw.githubusercontent.com/sp-uhh/ears_benchmark/{EARS_REV}/{name}'
           for name in ('README.md','generate_ears_wham.py','test_files.json','requirements.txt','download_ears_wham.sh')}
    files.update({name:f'https://raw.githubusercontent.com/microsoft/SIG-Challenge/{SIG_REV}/ICASSP2024/sigmos/{name}'
                  for name in ('sigmos.py','Transparency_FAQ.md')})
    model='model-sigmos_1697718653_41d092e8-epo-200.onnx'
    files[model]=f'https://media.githubusercontent.com/media/microsoft/SIG-Challenge/{SIG_REV}/ICASSP2024/sigmos/{model}'
    files['SIG_LICENSE']=f'https://raw.githubusercontent.com/microsoft/SIG-Challenge/{SIG_REV}/LICENSE'
    records={}
    for name,url in files.items():
        data=fetch(url)
        if name==model:
            assert hashlib.sha256(data).hexdigest()=='f939dcc1945055a435565b4369e27dafd0f87df3cea4e2ff6eb81225e52cc53b'
        (ROOT/name).write_bytes(data)
        records[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
    assets=[{'name':f'p{i}.zip','browser_download_url':f'https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{i}.zip'}
            for i in range(102,108)]
    (ROOT/'speech_assets.json').write_text(json.dumps(assets,indent=2)+'\n')
    (ROOT/'source_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    extract_selected(WHAM, ROOT/'data/WHAM48kHz', lambda n:n.endswith('high_res_metadata.csv'))


if __name__=='__main__':
    main()
