"""Restore the exact arithmetic runners before typed ReLU source admission.

Only the complete source_inverse definition changes in each runner. Every
original recipe, reference observer, numerical check, ownership check, process
gate and physical replay predicate remains byte identical. This source proof
does not execute a native image or transfer any previous qualification.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


DELTA = '866bd11ce2ed420e968a88521877e256b70f3b2d56045fc542b5075226ce5c51'
PATHS = ('tests/same_f64_arithmetic/run.py', 'tests/public_f64_arithmetic/run.py')
PARENTS = ('dfd7c857b35b05b1723f6abeeb14d5545b3c5aaee93afdc144dcdaccb8d33875', 'ecf23defe1b35abeb2d51c2fab144d4d57992a8cdc57bcfb53defa8ee36e1f57')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def definitions(raw):
    lines = raw.splitlines(keepends=True)
    result = {}
    for node in ast.parse(raw).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            require(node.name not in result, 'unique complete observer definitions')
            result[node.name] = b''.join(lines[node.lineno - 1:node.end_lineno])
    return result


def restore_runner(raw, path, bind):
    bind(Path(__file__).resolve())
    current_raw = raw
    extension_path = Path(__file__).with_name('index_signbit_admission_transition.py')
    bind(extension_path)
    spec = importlib.util.spec_from_file_location('index_signbit_observer_admission_inverse', extension_path)
    extension = importlib.util.module_from_spec(spec); spec.loader.exec_module(extension)
    raw, extension_proof = extension.restore_runner(raw, path, bind)
    delta_raw = bind(Path(__file__).with_name('typed_relu_admission_observer_delta.json'))
    require(sha(delta_raw) == DELTA, 'immutable complete typed ReLU admission delta')
    delta = json.loads(delta_raw)
    require(delta['schema'] == 'heptatorch.typed-relu775.admission-observer-delta.v1'
            and tuple(row['path'] for row in delta['changes']) == PATHS
            and tuple(row['parent_sha256'] for row in delta['changes']) == PARENTS
            and path in PATHS,
            'exact same5088 and mixed2541 source admission predecessors')
    record = delta['changes'][PATHS.index(path)]
    require(len(raw) == record['source_bytes'] and sha(raw) == record['source_sha256'],
            'complete actual typed ReLU arithmetic observer identity')
    original, current = record['original_definition'].encode(), record['current_definition'].encode()
    require(original != current and raw.count(current) == 1,
            'one complete actual source admission definition')
    parent = raw.replace(current, original, 1)
    require(len(parent) == record['parent_bytes'] and sha(parent) == record['parent_sha256'],
            'every predecessor arithmetic observer byte recovered')
    require(parent.count(original) == 1 and parent.replace(original, current, 1) == raw,
            'complete source admission forward round trip')
    before, after = definitions(parent), definitions(raw)
    require(before.keys() == after.keys()
            and [name for name in before if before[name] != after[name]] == ['source_inverse']
            and before['source_inverse'] == original and after['source_inverse'] == current,
            'only source admission changes; all numerical and physical predicates retained')
    return parent, dict(schema='heptatorch.typed-relu775.admission-observer-inverse.v1',
        path=path, source_sha256=sha(current_raw), parent_sha256=sha(parent),
        source_bytes=len(current_raw), parent_bytes=len(parent), delta_sha256=DELTA,
        index_signbit_source_admission_inverse=extension_proof,
        changed_definitions=['source_inverse'],
        all_original_numerical_error_lifetime_and_allocation_predicates_retained=True,
        original_reference_and_process_readback_predicates_retained=True,
        whole_observer_inverse_exact=True, whole_observer_forward_round_trip_exact=True,
        source_admission_only=True, runtime_qualification_transferred=False)


def restore_observers(bind):
    root = Path(__file__).resolve().parents[2]
    parents, proofs = {}, {}
    for path in PATHS:
        parents[path], proofs[path] = restore_runner(bind(root / path), path, bind)
    return parents, dict(schema='heptatorch.typed-relu775.arithmetic-admission-inverse.v1',
        files=proofs, original_case_counts=dict(same_f64_arithmetic=5088, public_f64_arithmetic=2541),
        changed_definitions_by_file={path: ['source_inverse'] for path in PATHS},
        every_other_byte_unchanged=True, whole_observer_inverse_exact=True,
        whole_observer_forward_round_trip_exact=True, runtime_qualification_transferred=False)
