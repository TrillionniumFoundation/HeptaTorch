#!/usr/bin/env python3
"""Execute the frozen 428 extension regressions on the exact retained 446 image.

This is an integration replay, not a new API or semantic-coverage claim.  Four
test programs keep their original bytes.  The cross-contract test changes only
two explicitly checked internal state tags, with a byte-exact inverse proof.
The extreme observer needs its historical executable filename, supplied by an
isolated byte-identical copy; no production file or native code is modified.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import traceback


SOURCE = '666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b'
IMAGE = '7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5'
PROFILE = '8da746dea3493b3d21714eae3d8048a38b073796d519395b96f9dd783ac80758'
PEER = 'tests/adam_distributions_gradient_review/'
EXTREME = 'tests/gradient_composition_review/'
ACTUAL = 'validation/adam-distributions-gradient-review/actual/'
EXTREME_ACTUAL = 'validation/gradient-composition-review/extreme-stride/'
PINS = {
    PEER+'test_zero_merge_layout.py': 'd822117f0941d2dde812b30c35eeffd4070cd2ed712af712e40d3e29a4ed2bf1',
    PEER+'replay_zero_merge_layout.py': 'ce47ebe6d5e6d6399a3e32d8f37bb732fb80a705b6870c3cd56ad692fc711008',
    PEER+'replay_scalar_add_nan.py': '88ab8627d45769ca6f153a0272d5ace0c4e63cac2ab6dce4948e7750dd275b5c',
    PEER+'test_joint_gradients_v2.py': 'c3bf33cbb5220469c6727b61bf612551202088c350551271833d15bd51bfd5fc',
    PEER+'protocol.py': '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9',
    PEER+'owner_client.py': 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
    EXTREME+'observe_extreme_stride_native.py': '8a0e809e8a5294dd6fa82e47158eb94864fdf599bb3bfff879d7f5ed8cc05be4',
    EXTREME+'compare_extreme_stride.py': 'c9200891612c28ac089fa21393599bae1d6a6ae891d71c3248b7829762d4d661',
    ACTUAL+'torch-zero-merge-layout.json': '72150ed1d2a1792f2d16be85b4b3bd323bafa8b20d2a6b7f23218682eda63f43',
    ACTUAL+'torch-scalar-add-nan.json': 'e664c9ff717a104b8c7317dfc55677e0248e58f981cb8315eacb3975049048ff',
    'tools/cpu_distributions/evidence/torch-bernoulli-dropout.json': 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467',
    EXTREME_ACTUAL+'prepared.json': '2ecfa99b998d6f384880bdd3b59bc1e6a81bf8dbbe81848e280ee7b78848d27d',
    EXTREME_ACTUAL+'torch-extreme-stride.json': '2de8fbd82d10fc16b0640684f4b36ae958a5a0455773401f82092210967137c8',
    'tools/retained_gradients/profile.json': PROFILE,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def metadata(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': sha(raw), 'mode': stat.S_IMODE(path.stat().st_mode)}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'build', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    repo, build, out = [getattr(args, name).resolve() for name in ('repo', 'build', 'output')]
    out.mkdir(parents=True, exist_ok=False)
    report = {
        'schema': 'heptatorch.retained-446.frozen-extension-replay.v1',
        'status': 'RUNNING', 'native_source_sha256': SOURCE,
        'native_executable_sha256': IMAGE, 'native_rebuilt_here': False,
        'scope': 'Fresh native execution of the inherited frozen extension tests on the exact retained-446 cold image.',
        'new_semantic_review_claimed': False, 'Torch_executed_here': False,
        'production_host_numerics': False, 'suites': [], 'inputs': {},
        'whole_PyTorch_status': 'HOLD', 'pytorch_api_denominator': 2590,
        'newly_closed_pytorch_api_rows': 0, 'full_rewrite_complete': False,
    }
    (out/'executed-integration.py').write_bytes(Path(__file__).read_bytes())
    tracked = {}

    def bind(path, expected=None, snapshot=None):
        path = path.resolve()
        value = metadata(path)
        if expected is not None:
            assert value['sha256'] == expected, (str(path), value, expected)
        tracked[path] = value
        report['inputs'][str(path)] = value
        if snapshot is not None:
            destination = out/'inputs'/snapshot
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
            destination.chmod(value['mode'])
        return path.read_bytes()

    try:
        bind(Path(__file__), snapshot='integration/run_regressions.py')
        for relative, pin in PINS.items():
            bind(repo/relative, pin, 'repository/'+relative)
        profile = json.loads((repo/'tools/retained_gradients/profile.json').read_bytes())
        assert profile['source_sha256'] == SOURCE and profile['native_sha256'] == IMAGE
        assert profile['functions'] == 446 and profile['module_count'] == len(profile['units']) == 9
        assert profile['state_format'] == 'HTRET001' and profile['backing_format'] == 'HTORET01'
        source = build/'runtime.hepta'
        binary = build/'tensor-retained-session'
        bind(source, SOURCE, 'native/runtime.hepta')
        bind(binary, IMAGE, 'native/tensor-retained-session')
        for unit in profile['units']:
            for suffix, field in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
                bind(build/(unit['name']+suffix), unit[field], 'native/'+unit['name']+suffix)
        build_report = bind(build/'build.json', snapshot='native/build.json')
        report['cold_build_receipt_sha256'] = sha(build_report)
        report['native_units_verified'] = profile['units']
        report['runtime_dimensions'] = {key: profile[key] for key in ('functions', 'blocks', 'instructions', 'descriptor_bytes', 'module_count')}

        originals = (repo/(PEER+'test_joint_gradients_v2.py')).read_bytes()
        replacements = [(b'HTGAD002', b'HTRET001'), (b'HTOGAD02', b'HTORET01')]
        adapted = originals
        for before, after in replacements:
            assert adapted.count(before) == 1 and adapted.count(after) == 0
            adapted = adapted.replace(before, after)
        inverse = adapted
        for before, after in reversed(replacements):
            assert inverse.count(after) == 1
            inverse = inverse.replace(after, before)
        assert inverse == originals
        adapter_dir = out/'adapter'
        adapter_dir.mkdir()
        (adapter_dir/'test_joint_gradients_v2.original.py').write_bytes(originals)
        adapted_path = adapter_dir/'test_joint_gradients_v2.retained.py'
        adapted_path.write_bytes(adapted)
        (adapter_dir/'inverse.py').write_bytes(inverse)
        (adapter_dir/'tags-only.diff').write_text(''.join(difflib.unified_diff(
            originals.decode().splitlines(True), adapted.decode().splitlines(True),
            fromfile='frozen-428', tofile='retained-446-tags')))
        proof = {'schema': 'heptatorch.retained-cross6.tags-only-adapter.v1',
                 'original_sha256': sha(originals), 'adapted_sha256': sha(adapted),
                 'inverse_sha256': sha(inverse), 'inverse_byte_identical': True,
                 'literal_edits': [{'old': a.decode(), 'new': b.decode(), 'count': 1} for a, b in replacements],
                 'numeric_Version_and_other_assertions_unchanged': True,
                 'original_unadapted_executed_on_446_here': False,
                 'reason': 'The new retained owner has different checked internal format tags; no wire protocol, outcome or numerical expectation changes.'}
        dump(adapter_dir/'proof.json', proof)
        report['cross6_adapter'] = proof

        # Keep the unmodified extreme observer's historical filename contract.
        alias = out/'extreme-build-filename-adapter'
        alias.mkdir()
        (alias/'runtime.hepta').write_bytes(source.read_bytes())
        (alias/'tensor-gradient-adam-distributions-session').write_bytes(binary.read_bytes())
        (alias/'tensor-gradient-adam-distributions-session').chmod(stat.S_IMODE(binary.stat().st_mode))
        assert metadata(alias/'runtime.hepta')['sha256'] == SOURCE
        assert metadata(alias/'tensor-gradient-adam-distributions-session')['sha256'] == IMAGE
        report['extreme_filename_adapter'] = {
            'original_filename': binary.name,
            'observer_required_filename': 'tensor-gradient-adam-distributions-session',
            'native_bytes_identical': True, 'source_bytes_identical': True,
            'production_mutation': False, 'native_rebuilt': False,
        }
        protocol = repo/(PEER+'protocol.py')
        common = ['--binary', binary, '--source', source, '--protocol', protocol,
                  '--expected-native-sha256', IMAGE, '--expected-source-sha256', SOURCE]
        children = out/'children'
        children.mkdir()
        logs = out/'logs'
        logs.mkdir()

        def execute(name, script, arguments, expected_status='PASS'):
            destination = children/name
            command = [sys.executable, '-I', '-S', '-B', str(script), *map(str, arguments), '--output', str(destination)]
            started = time.time()
            with (logs/(name+'.stdout')).open('wb') as stdout, (logs/(name+'.stderr')).open('wb') as stderr:
                result = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr, check=False)
            child_path = destination/'report.json'
            child_raw = child_path.read_bytes() if child_path.exists() else None
            child = json.loads(child_raw) if child_raw is not None else None
            row = {'name': name, 'command': command, 'exit_code': result.returncode,
                   'elapsed_seconds': time.time()-started,
                   'stdout': {'path': str(logs/(name+'.stdout')), **metadata(logs/(name+'.stdout'))},
                   'stderr': {'path': str(logs/(name+'.stderr')), **metadata(logs/(name+'.stderr'))},
                   'expected_status': expected_status,
                   'status': 'PASS' if result.returncode == 0 and child is not None and child['status'] == expected_status else 'FAIL'}
            if child_raw is not None:
                row.update(report_path=str(child_path), report_bytes=len(child_raw), report_sha256=sha(child_raw), report=child)
            report['suites'].append(row)
            dump(out/'report.json', report)
            print(json.dumps({'suite': name, 'exit_code': result.returncode, 'status': row['status'],
                              'report_sha256': row.get('report_sha256')}), flush=True)
            return child

        execute('math2', repo/(PEER+'test_zero_merge_layout.py'), common)
        execute('mixed2', repo/(PEER+'replay_zero_merge_layout.py'), [*common,
                '--observations', repo/(ACTUAL+'torch-zero-merge-layout.json'),
                '--expected-observations-sha256', PINS[ACTUAL+'torch-zero-merge-layout.json']])
        execute('scalar-nan-48', repo/(PEER+'replay_scalar_add_nan.py'), [*common,
                '--observations', repo/(ACTUAL+'torch-scalar-add-nan.json'),
                '--expected-observations-sha256', PINS[ACTUAL+'torch-scalar-add-nan.json']])
        execute('cross6-tags-only', adapted_path, [*common,
                '--distribution-observations', repo/'tools/cpu_distributions/evidence/torch-bernoulli-dropout.json'])
        execute('extreme8-observe', repo/(EXTREME+'observe_extreme_stride_native.py'), [
                '--build', alias, '--protocol', protocol, '--prepared', repo/(EXTREME_ACTUAL+'prepared.json'),
                '--expected-source', SOURCE, '--expected-image', IMAGE], 'OBSERVED_NOT_TORCH_QUALIFIED')
        if (children/'extreme8-observe/report.json').exists():
            execute('extreme8-compare', repo/(EXTREME+'compare_extreme_stride.py'), [
                    '--reference', repo/(EXTREME_ACTUAL+'torch-extreme-stride.json'),
                    '--native-observations', children/'extreme8-observe/report.json',
                    '--expected-source', SOURCE, '--expected-image', IMAGE])
        for path, before in tracked.items():
            assert metadata(path) == before, ('input changed', str(path))
        report['input_custody_after'] = True
        by_name = {row['name']: row for row in report['suites']}
        expected = ['math2', 'mixed2', 'scalar-nan-48', 'cross6-tags-only', 'extreme8-observe', 'extreme8-compare']
        assert list(by_name) == expected
        for row in report['suites']:
            child = row.get('report') or {}
            for field in ('native_source_sha256', 'source_sha256'):
                if field in child:
                    assert child[field] == SOURCE
            for field in ('native_executable_sha256', 'executable_sha256'):
                if field in child:
                    assert child[field] == IMAGE
        all_pass = all(row['status'] == 'PASS' for row in report['suites'])
        if all_pass:
            math = by_name['math2']['report']
            mixed = by_name['mixed2']['report']
            nan = by_name['scalar-nan-48']['report']
            cross = by_name['cross6-tags-only']['report']
            extreme = by_name['extreme8-compare']['report']
            assert len(math['cases']) == 2 and math['frames'] == 34
            assert mixed['case_count'] == 2 and mixed['failed_cases'] == 0 and mixed['forward_layout_difference_count'] == 0
            assert nan['case_count'] == 12 and nan['variant_count'] == 48 and nan['failed_variants'] == 0
            assert cross['case_count'] == cross['passed'] == 6 and cross['failed'] == 0
            assert [extreme[k] for k in ('cases', 'passing_cases', 'failing_cases', 'field_comparisons', 'different_fields')] == [8,8,0,344,0]
        report.update(status='PASS' if all_pass else 'FAIL', commands_executed=len(report['suites']),
                      regression_groups=5, successful_commands=sum(row['status']=='PASS' for row in report['suites']),
                      per_group_counting='math2; actual mixed2; actual scalar12x4; native cross6; actual extreme8 with separate observation/comparison. No sum is a PyTorch API denominator.')
    except BaseException as error:
        report.update(status='FAIL_INTEGRATION', error=repr(error), traceback=traceback.format_exc())
        for path, before in tracked.items():
            try:
                report.setdefault('after_failure_input_custody', {})[str(path)] = metadata(path) == before
            except BaseException:
                report.setdefault('after_failure_input_custody', {})[str(path)] = False
    dump(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'report_sha256': sha((out/'report.json').read_bytes()),
                      'commands_executed': len(report['suites'])}), flush=True)
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
