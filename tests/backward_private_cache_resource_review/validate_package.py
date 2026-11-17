#!/usr/bin/env python3
"""Pure source, dependency, exact inverse and historical binding checks."""
from pathlib import Path
import argparse, ast, hashlib, json, stat, sys

HERE = Path(__file__).resolve().parent
SOURCE01 = '54f1eb3b9ba7a4ee3f317c068e3c0ccb97b12e33515d018c28d52a1acde8cb5b'
SOURCE02 = '5183c0d6cd0274d29e4fd9e5c7dac1e89d79e8bf35ff0bec6fdcdb1432ee498c'
PARENT_WRAPPER = '00becd65f6897286e82723e0e4042eca6c0dbce18e2719855f57af0262aaca53'
PARENT_MANIFEST = 'f38c0cc3c2105e6180798a1fbccbe7fd84426bea18ef95ef14306cd8f63ce98d'
PARENT_DRIVER = '9925244a326107793255fe8ea2681797e8f7bf00adf6d1b03022a94c6f3a3e72'
PARENT_SOURCE_MANIFEST = 'a499508477bd78e6fa28fd09fab364af6eb38ebdcd56c79672d37d57f0e9f031'
OBLIGATIONS = '6549ff1e0a853340be3706c256eec646005d1f70649df3885a903ea1c73c3af7'
REVIEW = '30296b98346704540a9175ef8f1547c0807af7b95a2444c16ba7f6f5c40d2e49'
ACTUAL = 'f3e69cfd730d7431e4b010b2e5a814fd77444992d5a2a3f0b4c6646aa0c6223c'
SOURCE = 'c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a'
IMAGE = '0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74'
BASIS_BUILD = 'b31a45a850f06196432b2786ad0a39b56088ccf6c7b100d0da3517a44de5ba48'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def file(path, pin=None):
    path = Path(path)
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1, ('regular unique file', str(path))
    raw = path.read_bytes()
    meta = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(st.st_mode))
    assert pin is None or meta['sha256'] == pin, (str(path), meta['sha256'], pin)
    return raw, meta

def names(root):
    assert root.is_dir() and not root.is_symlink()
    result = set()
    for p in root.rglob('*'):
        assert not p.is_symlink(), str(p)
        if p.is_file():
            result.add(p.relative_to(root).as_posix())
        else:
            assert p.is_dir(), str(p)
    return result

def definitions(raw):
    text = raw.decode()
    tree = ast.parse(text)
    return {n.name: ast.get_source_segment(text, n).encode()
            for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

def inverse(before, after, proof):
    assert sha(before) == proof['before_sha256']
    assert sha(after) == proof['after_sha256']
    forward = before
    for edit in proof['edits']:
        old, new = bytes.fromhex(edit['before_hex']), bytes.fromhex(edit['after_hex'])
        assert old and new and forward.count(old) == edit['count']
        forward = forward.replace(old, new, edit['count'])
    assert forward == after, 'complete forward bytes'
    backward = after
    for edit in reversed(proof['edits']):
        old, new = bytes.fromhex(edit['before_hex']), bytes.fromhex(edit['after_hex'])
        assert backward.count(new) == edit['count']
        backward = backward.replace(new, old, edit['count'])
    assert backward == before, 'complete inverse bytes'

def verify_package(repo):
    repo = Path(repo).resolve()
    assert HERE == repo / 'tests/backward_private_cache_resource_review'
    manifest_raw, manifest_meta = file(HERE / 'package-manifest.json')
    manifest = json.loads(manifest_raw)
    assert manifest['schema'] == 'heptatorch.backward-private-cache.portable-source-package.v1'
    assert manifest['installation_mode'] == 0o644
    expected = manifest['files']
    assert expected and names(HERE) == set(expected) | {'package-manifest.json'}
    bound = {}
    for rel, meta in expected.items():
        p = Path(rel)
        assert not p.is_absolute() and '..' not in p.parts and p.as_posix() == rel
        assert meta['mode'] == 0o644
        assert file(HERE / rel)[1] == meta, ('installed source member', rel)
        bound[rel] = meta
    old = file(HERE / 'provenance/source01-driver.py', SOURCE01)[0]
    basis = file(HERE / 'provenance/source02-driver.py', SOURCE02)[0]
    current = file(HERE / 'driver.py')[0]
    reporting = json.loads(file(HERE / 'provenance/source01-to02-inverse.json')[0])
    interface = json.loads(file(HERE / 'provenance/driver-interface-inverse.json')[0])
    inverse(old, basis, reporting)
    inverse(basis, current, interface)
    assert len(interface['edits']) == 5
    a, b = definitions(basis), definitions(current)
    assert set(a) == set(b)
    assert [name for name in a if a[name] != b[name]] == ['main']
    unchanged = ('guard', 'original_module', 'slot_headroom', 'consume_extent_budget',
                 'geometry', 'run_case')
    assert all(a[name] == b[name] for name in unchanged)
    olddefs = definitions(old)
    assert set(olddefs) == set(a)
    assert [name for name in a if olddefs[name] != a[name]] == ['main']
    assert all(olddefs[name] == a[name] for name in unchanged)

    obligations = json.loads(file(HERE / 'obligations.json', OBLIGATIONS)[0])
    assert obligations['source_sha256'] == SOURCE
    assert len(obligations['functions']) == 24
    review = json.loads(file(HERE / 'provenance/independent-source-review.json', REVIEW)[0])
    assert review['status'] == 'SOURCE_ONLY_NO_ADDITIONAL_SUBSTANTIVE_BLOCKER'
    assert review['reviewed_driver02']['sha256'] == SOURCE02
    historical = json.loads(file(HERE / 'provenance/original-actual-report.json', ACTUAL)[0])
    for key, value in {
        'schema': 'heptatorch.scalar07b.backward-private-preserve-resource.v1',
        'source_sha256': SOURCE, 'native_sha256': IMAGE,
        'original_native_build_sha256': BASIS_BUILD,
        'status': 'PASS', 'case_count': 1, 'passed': 1,
        'native_executed': True, 'Torch_executed': False, 'compiler_executed': False,
        'numeric_oracle_computed_here': False, 'original_resource3_reexecuted': False,
        'all_inputs_unchanged': True, 'whole_pytorch_gate': 'HOLD', 'closed_schema_count': 0,
        'frames_before_EOF': 22, 'checks': 46, 'committed_84Q_snapshots': 23,
        'full_owner_rollback_bytes': 4369040, 'full_rng_bytes': 96064}.items():
        assert type(historical[key]) is type(value) and historical[key] == value, ('historical field', key)
    assert len(historical['cases']) == 1
    history_case = historical['cases'][0]
    assert history_case['name'] == 'backward_private_preserve_allocation_oom_recovery'
    assert history_case['status'] == 'PASS' and history_case['EOF_exit_code'] == 0
    assert history_case['target_allocation_qualified'] is True
    assert history_case['original_limit_restored'] is True

    parent = repo / 'tests/wrapped_scalar_resource_review'
    dependencies = {
        'run_resources.py': PARENT_WRAPPER,
        'package-manifest.json': PARENT_MANIFEST,
        'portable07b/test_wrapped_scalar_resources.py': PARENT_DRIVER,
        'portable07b/source-manifest.json': PARENT_SOURCE_MANIFEST}
    parent_bound = {rel: file(parent / rel, pin)[1] for rel, pin in dependencies.items()}
    parent_wrapper = file(parent / 'run_resources.py', PARENT_WRAPPER)[0]
    assert parent_wrapper == file(HERE / 'provenance/parent-run_resources.py', PARENT_WRAPPER)[0]
    parent_defs = definitions(parent_wrapper)
    assert 'source_check' in parent_defs and 'fresh_build' in parent_defs
    for p in HERE.rglob('*.py'):
        ast.parse(file(p)[0], filename=str(p))
    profile = json.loads(file(HERE / 'profile.json')[0])
    assert profile['source_sha256'] == SOURCE and profile['native_sha256'] == IMAGE
    assert profile['source_basis_sha256'] == SOURCE02
    assert profile['portable_execution_status'] == 'NOT_RUN'
    assert profile['historical_source02_report_sha256'] == ACTUAL
    return dict(
        status='PASS_SOURCE_CUSTODY_ONLY', native_executed=False, Torch_executed=False,
        compiler_executed=False, emitter_executed=False,
        source_basis_sha256=SOURCE02, portable_driver_sha256=sha(current),
        changed_definitions=['main'], byte_identical_definitions=list(unchanged),
        source01_to02_whole_byte_inverse=True, interface_whole_byte_inverse=True,
        original_actual_report_sha256=ACTUAL, historical_pass_not_transferred=True,
        parent_wrapper_sha256=PARENT_WRAPPER,
        parent_functions_reused=['source_check', 'fresh_build'],
        parent_dependency_pins=parent_bound,
        parent_full_package_check='Executed separately by the wrapper through the unmodified parent source_check().',
        package_manifest=manifest_meta, bound=bound)

def main():
    if sys.flags.optimize:
        raise RuntimeError('assertions must be enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists(), 'fresh result required'
    result = verify_package(a.repo)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as stream:
        stream.write((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    print(json.dumps({k: result[k] for k in ('status', 'native_executed', 'interface_whole_byte_inverse')}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
