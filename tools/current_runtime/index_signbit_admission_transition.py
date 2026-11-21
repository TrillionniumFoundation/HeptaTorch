"""Restore each complete original775 observer before the new source admission.

Only the named source-admission function changes. Every numerical comparison,
reference process check, old request, owner predicate and replay stays intact.
"""
import ast
import hashlib
import json
from pathlib import Path

PATHS = ('tests/numeric_predicates/run.py', 'tests/public_f64_arithmetic/run.py',
         'tests/same_f64_arithmetic/run.py')
DEFINITIONS = ('profile', 'source_inverse', 'source_inverse')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def definitions(raw):
    lines = raw.splitlines(keepends=True)
    return {node.name: b''.join(lines[node.lineno-1:node.end_lineno])
            for node in ast.parse(raw).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def restore_runner(raw, path, bind):
    bind(Path(__file__).resolve())
    delta_raw = bind(Path(__file__).with_name('index_signbit_admission_delta.json'))
    delta = json.loads(delta_raw)
    require(delta['schema'] == 'heptatorch.index-signbit778.admission-delta.v1'
            and tuple(row['path'] for row in delta['changes']) == PATHS
            and tuple(row['definition'] for row in delta['changes']) == DEFINITIONS
            and path in PATHS, 'exact three source-admission edits')
    row = delta['changes'][PATHS.index(path)]
    require(sha(raw) == row['source_sha256'] and len(raw) == row['source_bytes'],
            'complete actual observer before source-only reconstruction')
    before, after = row['before'].encode(), row['after'].encode()
    require(before != after and raw.count(after) == 1, 'one complete source-admission definition')
    parent = raw.replace(after, before, 1)
    require(sha(parent) == row['parent_sha256'] and len(parent) == row['parent_bytes']
            and parent.count(before) == 1 and parent.replace(before, after, 1) == raw,
            'all original observer bytes and exact complete forward round trip')
    a, b = definitions(parent), definitions(raw)
    require(a.keys() == b.keys()
            and [name for name in a if a[name] != b[name]] == [row['definition']]
            and a[row['definition']] == before and b[row['definition']] == after,
            'all other full observer definitions and top-level bytes retained')
    return parent, dict(schema='heptatorch.index-signbit778.admission-inverse.v1', path=path,
        source_sha256=sha(raw), parent_sha256=sha(parent), delta_sha256=sha(delta_raw),
        changed_definitions=[row['definition']], whole_observer_inverse_exact=True,
        whole_observer_forward_round_trip_exact=True,
        all_original_numerical_error_lifetime_and_allocation_predicates_retained=True,
        original_reference_and_process_readback_predicates_retained=True,
        source_admission_only=True, runtime_qualification_transferred=False)
