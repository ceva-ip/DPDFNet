"""Conservative lifetime reuse for this generator's straight-line C arena.

All references to a root allocation (including views) keep it live for the
whole node. Inputs and outputs of the same node can never share a slot.
Operator implementations must not retain arena pointers after returning.
"""
import re


def reuse_arena(code, tensors, arena_floats):
    prefix, body = code.split('int dpdf_model_process(', 1)
    chunks = re.split(r'/\* \d+: \w+ [^\n]*\*/', body)
    references = [set(map(int, re.findall(r'm->arena\+(\d+)', chunk))) for chunk in chunks]
    used = set().union(*references)
    boundaries = sorted({t['offset'] for t in tensors} | used | {arena_floats})
    sizes = {a: b-a for a, b in zip(boundaries, boundaries[1:])}
    intervals = {offset: (min(i for i, refs in enumerate(references) if offset in refs),
                          max(i for i, refs in enumerate(references) if offset in refs))
                 for offset in used}
    active, free, mapping, records = [], [], {}, []
    high = 0
    for old in sorted(used, key=lambda x: (intervals[x][0], -sizes[x], x)):
        begin, end = intervals[old]
        keep = []
        for stop, start, size in active:
            if stop < begin:
                free.append((start, size))
            else:
                keep.append((stop, start, size))
        active = keep
        merged = []
        for start, size in sorted(free):
            if merged and sum(merged[-1]) == start:
                p, n = merged.pop(); merged.append((p, n+size))
            else:
                merged.append((start, size))
        free = merged
        size = (sizes[old]+15)//16*16
        fits = [(n, p, i) for i, (p, n) in enumerate(free) if n >= size]
        if fits:
            n, start, i = min(fits)
            free.pop(i)
            if n > size:
                free.append((start+size, n-size))
        else:
            start = high; high += size
        mapping[old] = start
        active.append((end, start, size))
        records.append({'old_offset': old, 'offset': start, 'floats': size,
                        'first_node_chunk': begin, 'last_node_chunk': end})
    # Check the plan independently of the allocator before emitting addresses.
    for i, a in enumerate(records):
        for b in records[i+1:]:
            time_overlap = max(a['first_node_chunk'], b['first_node_chunk']) <= min(a['last_node_chunk'], b['last_node_chunk'])
            memory_overlap = max(a['offset'], b['offset']) < min(a['offset']+a['floats'], b['offset']+b['floats'])
            if time_overlap and memory_overlap:
                raise ValueError('Arena plan overlaps live allocations')
    body = re.sub(r'm->arena\+(\d+)', lambda m: f'm->arena+{mapping[int(m[1])]}', body)
    prefix, count = re.subn(r'(size_t dpdf_model_arena_bytes\(void\) \{ return )\d+(\*sizeof\(float\); \})',
                           lambda m: m[1]+str(high)+m[2], prefix)
    if count != 1:
        raise ValueError('Expected exactly one generated arena-size function')
    layout = [dict(t, original_offset=t['offset'], offset=mapping.get(t['offset'])) for t in tensors]
    return prefix+'int dpdf_model_process('+body, layout, high, records
