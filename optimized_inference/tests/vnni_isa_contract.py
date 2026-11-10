#!/usr/bin/env python3
"""Reject an accidental AVX-512 requirement in the AVX2/VEX-VNNI backends."""
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: vnni_isa_contract.py objdump object...')
    vnni = 0
    for name in sys.argv[2:]:
        path = Path(name)
        output = subprocess.check_output([sys.argv[1], '-d', '-w', '--', str(path)], text=True)
        for line in output.splitlines():
            # In 64-bit executable sections, opcode prefix 62 denotes EVEX.
            # Native tests on an AVX-512 CPU would otherwise hide this error.
            if re.match(r'^\s*[0-9a-f]+:\s+62\s', line):
                raise SystemExit(f'Unexpected EVEX instruction in {path}: {line.strip()}')
            if re.match(r'^\s*[0-9a-f]+:.*\bvpdpbusd\b', line):
                if not re.match(r'^\s*[0-9a-f]+:\s+c4\s', line):
                    raise SystemExit(f'Expected VEX VPDPBUSD in {path}: {line.strip()}')
                vnni += 1
    if not vnni:
        raise SystemExit('No VEX VPDPBUSD instructions were inspected')
    print(json.dumps({'objects': len(sys.argv) - 2, 'vex_dot_sites': vnni,
                      'evex_instructions': 0, 'avx512_not_required': True}))


if __name__ == '__main__':
    main()
