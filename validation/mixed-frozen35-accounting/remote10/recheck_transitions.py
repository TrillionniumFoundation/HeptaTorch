#!/usr/bin/env python3
"""Pure recomputation of every archived fixed35 owner transition; no target IO."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

OBSERVER_SHA = '8361f28ab6a6548ec0b3d4b97fb0f5b7c96788bcbfc6ba96fe644f5c2834bb3f'

def recheck(archive, manifest, original, observer_file):
    observer_file = Path(observer_file)
    assert hashlib.sha256(observer_file.read_bytes()).hexdigest() == OBSERVER_SHA
    spec = importlib.util.spec_from_file_location('mixed556_fixed35_pure_recheck', observer_file)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    assert len(original['rows']) == 35 and original['frames'] == 6344
    prefix = 'run-frozen35-01/frozen35/journals/owners/'
    cases = {hashlib.sha256(row['case'].encode()).hexdigest()[:16]: row for row in original['rows']}
    assert len(cases) == 35
    expected_names = {prefix + token + '/' + name for token in cases
                      for name in ('events.json', 'snapshots.jsonl.gz')}
    actual_names = {row['path'] for row in manifest['files'] if row['path'].startswith(prefix)}
    assert actual_names == expected_names
    events, results, seen_members = {}, [], set()
    with tarfile.open(archive, 'r|xz') as tar:
        for member in tar:
            if member.name not in expected_names:
                continue
            assert member.isfile() and member.name not in seen_members
            seen_members.add(member.name)
            token, kind = member.name[len(prefix):].split('/')
            if kind == 'events.json':
                events[token] = json.load(tar.extractfile(member))
                assert isinstance(events[token], list) and events[token]
                continue
            assert token in events, 'The exact frozen archive sorts events before snapshots.'
            rows = events.pop(token)
            needed = Counter(v for row in rows for v in (row['before'], row['after']))
            original_needed = set(needed)
            snapshots, seen = {}, set()
            index = 0
            with tar.extractfile(member) as compressed, gzip.GzipFile(fileobj=compressed, mode='rb') as stream:
                for line in stream:
                    value = json.loads(line)
                    digest, snapshot = value['sha256'], value['snapshot']
                    assert digest in needed and digest not in seen
                    assert hashlib.sha256(observer.canonical(snapshot)).hexdigest() == digest
                    assert snapshot['profile'] == 'mixed-integer-556-fixed35-F32-Saved'
                    assert snapshot['source_sha256'] == '5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
                    assert snapshot['native_sha256'] == '69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'
                    snapshots[digest] = snapshot
                    seen.add(digest)
                    while index < len(rows) and all(rows[index][k] in snapshots for k in ('before', 'after')):
                        row = rows[index]
                        assert row['error'] is None and row['result']['status'] == 'PASS'
                        actual = observer.verify_transition(snapshots[row['before']], snapshots[row['after']],
                            row['tickets'], row['commands'], row['expected_status'])
                        assert actual == row['result'], (cases[token]['case'], index, actual, row['result'])
                        for key in ('before', 'after'):
                            digest = row[key]
                            needed[digest] -= 1
                            if needed[digest] == 0:
                                del snapshots[digest]
                        index += 1
            assert seen == original_needed and index == len(rows) and not snapshots
            assert all(count == 0 for count in needed.values())
            row = cases[token]
            declared = row['private_node_observer']
            assert declared['status'] == 'PASS'
            assert declared['eof_exit_assertions_executed'] is True
            assert declared['post_close_parameter_version_assertions_executed'] is True
            results.append({'case': row['case'], 'journal': token, 'events': len(rows),
                            'unique_snapshots': len(seen), 'all_original_results_recomputed_exactly': True})
    assert not events and seen_members == expected_names and len(results) == 35
    assert sum(row['events'] for row in results) == 6344
    return {'status': 'PASS_PURE_ARCHIVED_TRANSITIONS', 'observer_sha256': OBSERVER_SHA,
            'cases': 35, 'transitions': 6344,
            'unique_snapshot_records': sum(row['unique_snapshots'] for row in results),
            'rows': sorted(results, key=lambda r: r['case']),
            'new_native_executions': 0, 'new_Torch_executions': 0,
            'target_memory_read_or_written': False,
            'scope': 'All archived frames recomputed through the exact frozen full80word Saved-T profile; not a new target execution.'}
