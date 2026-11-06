#!/usr/bin/env python3
"""Reanalyze an immutable comparison journal with the current inference code."""
import argparse
import json
from pathlib import Path
from compare_benchmarks import summarize, sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[json.loads(line) for line in args.journal.read_text().splitlines()]
    protocol=json.loads(args.protocol.read_text())
    family=protocol['protocol']['family_size'];seed=protocol['protocol']['seed']
    result={'schema_version':2,'journal_sha256':sha(args.journal),'protocol_sha256':sha(args.protocol),
            'analysis_code_sha256':sha(Path(__file__).with_name('compare_benchmarks.py')),
            'family_size':family,'models':{}}
    for model in sorted({r['model_id'] for r in rows}):
        summary=summarize([r for r in rows if r['model_id']==model],seed=seed,alpha=.05/family)
        summary['family_adjusted_randomization_p']=min(1,summary['two_sided_sign_randomization_p']*family)
        result['models'][str(model)]=summary
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
