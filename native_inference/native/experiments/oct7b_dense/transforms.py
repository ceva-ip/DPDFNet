"""Reduce only INT8 dense output padding; real-column arithmetic is unchanged."""
import hashlib
from pathlib import Path


VARIANTS = ('dense_int8_pad8',)
EXPECTED_SHA256 = '5d8e2946796c2d4c72527a79edd277b911ec580ab7a092a455c3c918478c1bd7'


def apply(name, source):
    if name not in VARIANTS:
        raise ValueError(name)
    path = Path(source)/'extended_ops.c'
    code = path.read_text()
    if hashlib.sha256(code.encode()).hexdigest()!=EXPECTED_SHA256:
        raise RuntimeError('Dense padding transform requires preserved Oct7 extended_ops.c')
    old = 'd->np=(n+31)/32*32; d->precision=precision;'
    if code.count(old)!=1:
        raise RuntimeError('Expected exactly one dense output-padding assignment')
    new = ('d->np=precision==8 ? (n+7)/8*8 : (n+31)/32*32; '
           'd->precision=precision; /* INT8 has an exact eight-output tail. */')
    path.write_text(code.replace(old,new))
