#!/usr/bin/env python3
"""Verify the portable wrapper and both separately frozen resource source trees."""
from pathlib import Path
import argparse, ast, hashlib, json, stat

HERE = Path(__file__).resolve().parent
ORIGINAL_MANIFEST = 'fefb9247a2c01ad14c5b8a5016b3a9c6e3063ecc496637052e48665d9d851bd2'
ORIGINAL_DRIVER = '48185e4657f0c57c0c79c82dd4b3192573fe7c5b8b6a8122bdb02f3f21b99d9c'
OWNER = 'e54509ceba0ef3bf171f5fc1018665dc97e56eeccec0cba1e8c14f9a7ed81501'
BASIS_BUILD = 'b31a45a850f06196432b2786ad0a39b56088ccf6c7b100d0da3517a44de5ba48'
BUILDER = '9f6f02a7d749206fb9b523d0bf7ddb308d695a3215fecd4982c5d11a74c5abf8'
RECIPES = ('entry_guards_and_relative_tokens',
           'saved_scalar_and_temporary_lifetime',
           'result_allocation_oom_recovery')

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

def dump(path, obj):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

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

def verify_tree(root, manifest_pin=None):
    raw, meta = file(root / 'source-manifest.json', manifest_pin)
    manifest = json.loads(raw)
    assert isinstance(manifest['files'], dict) and manifest['files']
    assert names(root) == set(manifest['files']) | {'source-manifest.json'}
    for rel, expected in manifest['files'].items():
        p = Path(rel)
        assert not p.is_absolute() and '..' not in p.parts and p.as_posix() == rel
        assert file(root / rel)[1] == expected, ('frozen source member', rel)
    return manifest, meta

def definitions(raw):
    text = raw.decode()
    tree = ast.parse(text)
    return {n.name: ast.get_source_segment(text, n).encode()
            for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

def verify_package():
    package, package_meta = file(HERE / 'package-manifest.json')
    package = json.loads(package)
    expected = package['files']
    assert expected and names(HERE) == set(expected) | {'package-manifest.json'}
    bound = {}
    for rel, meta in expected.items():
        p = Path(rel)
        assert not p.is_absolute() and '..' not in p.parts and p.as_posix() == rel
        assert file(HERE / rel)[1] == meta, ('package member', rel)
        bound[rel] = meta
    before, _ = verify_tree(HERE / 'original07b', ORIGINAL_MANIFEST)
    after, _ = verify_tree(HERE / 'portable07b')
    assert set(before['files']) == set(after['files']) and len(before['files']) == 59
    old = file(HERE / 'original07b/test_wrapped_scalar_resources.py', ORIGINAL_DRIVER)[0]
    new = file(HERE / 'portable07b/test_wrapped_scalar_resources.py')[0]
    inverse = json.loads(file(HERE / 'provenance/driver-interface-inverse.json')[0])
    assert set(inverse['files']) == {'test_wrapped_scalar_resources.py', 'source-manifest.json'}
    for rel, rec in inverse['files'].items():
        a = file(HERE / 'original07b' / rel, rec['before_sha256'])[0]
        b = file(HERE / 'portable07b' / rel, rec['after_sha256'])[0]
        forward = a
        for edit in rec['edits']:
            x, y = bytes.fromhex(edit['before_hex']), bytes.fromhex(edit['after_hex'])
            assert forward.count(x) == edit['count']
            forward = forward.replace(x, y, edit['count'])
        assert forward == b, ('whole forward', rel)
        backward = b
        for edit in reversed(rec['edits']):
            x, y = bytes.fromhex(edit['before_hex']), bytes.fromhex(edit['after_hex'])
            assert backward.count(y) == edit['count']
            backward = backward.replace(y, x, edit['count'])
        assert backward == a, ('whole inverse', rel)
    changed = []
    for rel in before['files']:
        a = file(HERE / 'original07b' / rel)[0]
        b = file(HERE / 'portable07b' / rel)[0]
        if a != b:
            changed.append(rel)
    assert changed == ['test_wrapped_scalar_resources.py']
    olddefs, newdefs = definitions(old), definitions(new)
    assert set(olddefs) == set(newdefs)
    assert [n for n in olddefs if olddefs[n] != newdefs[n]] == ['load_inputs', 'main']
    assert all(olddefs[n] == newdefs[n] for n in RECIPES + ('Case',))
    assert file(HERE / 'original07b/owner_view05.py', OWNER)[0] == file(HERE / 'portable07b/owner_view05.py', OWNER)[0]
    assert file(HERE / 'provenance/original-build.json', BASIS_BUILD)[0]
    assert file(HERE / 'provenance/portable-builder-source.py', BUILDER)[0]
    for root in ('original07b', 'portable07b'):
        for p in (HERE / root).rglob('*.py'):
            ast.parse(file(p)[0], filename=str(p))
    ast.parse(file(HERE / 'run_resources.py')[0])
    ast.parse(file(HERE / 'validate_package.py')[0])
    return dict(status='PASS_SOURCE_CUSTODY_ONLY', native_executed=False, Torch_executed=False,
                compiler_executed=False, emitter_executed=False,
                original_payloads=59, original_manifest_sha256=ORIGINAL_MANIFEST,
                unchanged_payloads=58, changed_payloads=changed,
                whole_byte_forward_inverse=True, all_three_recipe_bodies_identical=True,
                Case_body_identical=True, all_owner_bytes_identical=True,
                package_manifest=package_meta, bound=bound)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'fresh output file required'
    result = verify_package()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    dump(a.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'native_executed', 'original_payloads',
                                            'unchanged_payloads', 'whole_byte_forward_inverse')}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
