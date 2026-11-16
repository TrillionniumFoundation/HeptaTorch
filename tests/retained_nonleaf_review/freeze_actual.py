#!/usr/bin/env python3
"""Freeze only the already executed, pinned real-Torch retained-gradient data."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ACTUAL_SHA = 'ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3'
PROBE_SHA = '7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba'
CHECKS = ('python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def brief(info):
    if info is None:
        return None
    return {key: info[key] for key in ('shape', 'strides', 'version',
        'binary32_words', 'tensor_identity', 'storage_identity', 'retains_grad')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw, probe = a.observations.read_bytes(), a.probe.read_bytes()
    assert sha(raw) == ACTUAL_SHA and len(raw) == 722720
    assert sha(probe) == PROBE_SHA
    r = json.loads(raw)
    assert r['status'] == 'OBSERVED' and r['torch_executed'] is True
    assert r['native_executed'] is False and r['native_values_compared'] == 0
    assert r['cases'] == r['expected_cases'] == len(r['rows']) == 8
    assert r['expected_variants'] == 12 and r['probe_errors'] == 0
    assert r['source_files_unchanged'] is True and r['probe_sha256'] == PROBE_SHA
    checks = r['oracle_provenance']['checks']
    assert set(checks) == set(CHECKS) and all(checks[k] is True for k in CHECKS)
    summaries = []
    for i, row in enumerate(r['rows']):
        assert row['status'] == 'OBSERVED'
        variants = row['observation'].get('variants', [row['observation']])
        assert len(variants) == row['planned_variants']
        for j, variant in enumerate(variants):
            path = f'/rows/{i}/observation'
            if 'variants' in row['observation']:
                path += f'/variants/{j}'
            events = []
            for k, event in enumerate(variant.get('events', [])):
                result = {'json_pointer': f'{path}/events/{k}',
                          'action': event['action'], 'outcome': event['outcome']}
                if 'exception_type' in event:
                    result.update(exception_type=event['exception_type'], message=event['message'])
                for when in ('before', 'after'):
                    if when not in event:
                        continue
                    snapshot = event[when]
                    result[when] = {
                        'tensor_versions': {n: v['version'] for n, v in snapshot['tensors'].items()},
                        'retains_flags': {n: v['retains_grad'] for n, v in snapshot['tensors'].items()},
                        'gradients': {n: brief(v) for n, v in snapshot['gradients'].items()},
                    }
                    if 'optimizer_state' in snapshot:
                        result[when]['optimizer_state_entries'] = snapshot['optimizer_state_entries']
                        result[when]['optimizer_state'] = [
                            {'parameter_identity': s['parameter_identity'],
                             'entry_exists': s['entry_exists'],
                             'fields': {n: brief(v) for n, v in s['fields'].items()}}
                            for s in snapshot['optimizer_state']]
                events.append(result)
            summary = {'case': row['case'], 'variant': variant.get('variant', 'default'),
                       'json_pointer': path, 'events': events}
            for key in ('after_drop', 'after_public_drop', 'after_optimizer_drop',
                        'after_final_gradient_drop', 'q_storage_expired_after_backward'):
                if key in variant:
                    summary[key] = variant[key]
            summaries.append(summary)
    assert len(summaries) == 12
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'torch-retained-nonleaf.json').write_bytes(raw)
    (a.output / 'pytorch_retained_nonleaf_probe.py').write_bytes(probe)
    report = {
        'schema': 'heptatorch.retained-nonleaf-independent-actual-summary.v1',
        'status': 'OBSERVED_REFERENCE_FROZEN',
        'actual_sha256': ACTUAL_SHA, 'actual_bytes': len(raw), 'probe_sha256': PROBE_SHA,
        'families': 8, 'variants': 12, 'probe_errors': 0, 'provenance_checks': checks,
        'source_files_unchanged': True, 'native_executed_by_this_reader': False,
        'torch_executed_by_this_reader': False,
        'identity_policy': 'Compare opaque identity equality relations within a live recipe; never compare process addresses or conflate Tensor with Storage identity.',
        'version_policy': 'Version counters are exact. Equal counters do not prove shared Version identity.',
        'exception_policy': 'Command 58 exposes committed retain prehook effects before a runtime error. Legacy atomic backward is a separate contract.',
        'numeric_policy': 'All binary32 words exact except pre-existing Adam parameter/moment tolerance rtol=4e-5, atol=5e-6; always preserve raw word observations.',
        'diagnostic_only': ['Python wrapper weakref lifetime', 'Python warning objects', 'Python exception message spelling', 'exact Python grad_fn class name'],
        'scope': 'These twelve first-order CPU binary32 retained-gradient variants; not complete PyTorch closure.',
        'rows': summaries,
    }
    (a.output / 'semantic-observations.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    assert sha(a.observations.read_bytes()) == ACTUAL_SHA and sha(a.probe.read_bytes()) == PROBE_SHA
    files = [{'path': str(p.relative_to(a.output)), 'bytes': p.stat().st_size,
              'sha256': sha(p.read_bytes())} for p in sorted(a.output.iterdir()) if p.is_file()]
    manifest = {'schema': 'heptatorch.retained-nonleaf-reference-custody.v1', 'files': files}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'families': 8, 'variants': 12,
                      'manifest_sha256': sha((a.output / 'manifest.json').read_bytes())}))


if __name__ == '__main__':
    main()
