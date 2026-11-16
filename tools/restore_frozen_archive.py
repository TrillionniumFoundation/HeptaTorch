"""Restore one frozen archive from byte-exact, individually hashed Git parts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    manifest_path = (repo / args.manifest).resolve()
    assert manifest_path.is_relative_to(repo)
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest['schema'] == 'heptatorch.frozen-archive-parts.v1'
    target = (repo / manifest['target']).resolve()
    assert target.is_relative_to(repo) and target.parent.is_dir()
    assert manifest['parts'] and len({r['path'] for r in manifest['parts']}) == len(manifest['parts'])
    existed = target.exists()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as sink:
            temporary = Path(sink.name)
            whole, total = hashlib.sha256(), 0
            for row in manifest['parts']:
                path = (repo / row['path']).resolve()
                assert path.is_relative_to(repo) and path != target and path.is_file()
                part, length = hashlib.sha256(), 0
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        sink.write(block)
                        whole.update(block)
                        part.update(block)
                        total += len(block)
                        length += len(block)
                assert length == row['bytes'] and part.hexdigest() == row['sha256'], row['path']
            assert total == manifest['bytes'] and whole.hexdigest() == manifest['sha256']
        assert digest(temporary) == manifest['sha256']
        if existed:
            assert target.stat().st_size == total and digest(target) == manifest['sha256'], 'existing archive differs'
        else:
            temporary.chmod(manifest['mode'])
            os.replace(temporary, target)
        assert target.stat().st_size == manifest['bytes'] and digest(target) == manifest['sha256']
        print(json.dumps({'schema': 'heptatorch.frozen-archive-restoration.v1', 'status': 'PASS',
                          'target': manifest['target'], 'bytes': total, 'sha256': manifest['sha256'],
                          'parts': len(manifest['parts']), 'existing_archive_verified': existed,
                          'manifest_sha256': digest(manifest_path)}))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()
