#!/usr/bin/env python3
"""Run the unchanged PR38 comparisons plus two observed Bool dtype extensions."""
import argparse
import importlib.util
import json
import lzma
from pathlib import Path
import struct
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_support import (HERE, ROOT, bind_frozen_reader, current_profile, load,
                            load_reader, require, sha)


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def extension_reference(inputs):
    package = ROOT / 'validation/bool-dtype-extension-reference'
    manifest = json.loads(inputs.bind(package / 'manifest.json', '7469d89c4f70fc7d46a83731103a130a6b0cf4c28934c03520e808000c62d1bb'))
    require(manifest['schema'] == 'heptatorch.bool-dtype-extension.fixed-reference.v1'
            and manifest['cases'] == 2 and manifest['torch_executed'] is True,
            'complete observed two-dtype reference')
    archive = inputs.bind(package / 'report.json.xz', manifest['archive']['sha256'])
    raw = lzma.decompress(archive)
    require(sha(raw) == manifest['report']['sha256'] == '7b38959d3e052a0105c0a5e17fe9688dc8bcda975501b388a805f35599a1537c' and len(raw) == manifest['report']['bytes'] == 20389,
            'exact two-dtype observed report bytes')
    reference = json.loads(raw)
    cases = load(HERE / 'bool_dtype_cases.py', 'fusion_bool_dtype_cases')
    inputs.bind(HERE / 'bool_dtype_cases.py', manifest['case_source']['sha256'])
    inputs.bind(ROOT / 'tests/logical_reductions/reference.py', manifest['reference_source']['sha256'])
    require(reference['status'] == 'OBSERVED' and reference['cases'] == reference['expected_cases'] == 2
            and reference['case_sha256'] == cases.CASE_SHA256
            and [r['spec'] for r in reference['rows']] == list(cases.CASES)
            and reference['inputs_unchanged'] is True and reference['rng_unchanged'] is True
            and reference['torch_executed'] is True and reference['native_executed'] is False,
            'complete independently observed extension recipes')
    provenance = reference['oracle_provenance']
    require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
            and all(v is True for v in provenance['checks'].values()), 'fixed eleven oracle checks')
    execution = json.loads(inputs.bind(package / 'execution.json', manifest['execution']['sha256']))
    require(execution['status']['exit_code'] == 0 and execution['status']['output_complete'] is True
            and execution['logs']['stdout']['eof'] is True and execution['logs']['stderr']['eof'] is True
            and execution['report']['sha256'] == sha(raw), 'actual extension execution with complete EOF')
    for row in reference['rows']:
        require(row['accepted'] is True and row['fresh_tensor'] is True and row['fresh_storage'] is True
                and row['input_unchanged'] is True and row['rng_unchanged'] is True,
                'observed accepted fresh Bool result')
    return reference, dict(manifest_sha256=sha((package / 'manifest.json').read_bytes()),
                           report_sha256=sha(raw), actual_torch_cases=2)


def run_extension(args, native, adapter, support):
    inputs = support.Inputs()
    reference, binding = extension_reference(inputs)
    profile = current_profile()
    profile.admit_build(args.build_report.parent, inputs.bind, args.expected_build_sha256)
    bind_frozen_reader(args.reader_directory, inputs)
    for name in ('fusion.py', 'fusion_support.py'):
        inputs.bind(HERE / name)
    observer, owner, proof = adapter.make_observer(ROOT, inputs, profile.SOURCE, profile.IMAGE)
    rows = []

    def execute(c, row):
        if row['spec']['dtype'] == 'float32':
            backing = c.n.create(0, [2], struct.pack('<2I', 0x3f800000, 0xbf800000), requires=0)
            c.publish('backing', backing)
            c.publish('input', c.view(backing, [2], [1], 0))
        else:
            c.live['leaf'] = c.n.create(0, [2], struct.pack('<2f', 1.0, 2.0), requires=1)
            c.live['graph'] = c.n.one(114, c.live['leaf'], 0x3ff0000000000000)['value']
            c.publish('input', c.n.one(107, c.live['graph'], 1)['value'])
        source = c.tensor(c.live['input'])

        def compare(actual, expected, label):
            for field, wanted in expected.items():
                target = 'storage_offset_elements' if field == 'storage_offset' else field
                c.check(label + '/' + field, actual[target], wanted)

        compare(source, row['input'], 'actual-Torch-input')
        status, record = c.issue((115, c.live['input'], (1 << 64) - 1, 0, 0, 1, 0, 0),
                                 'extended-Bool-dtype')
        c.check('actual-Torch-accepted', status, 0)
        c.publish('result', record['value'])
        result = c.tensor(c.live['result'])
        compare(result, row['result'], 'actual-Torch-result')
        c.check('fresh-Tensor', result['native_incarnation'] != source['native_incarnation'], True)
        c.check('fresh-Storage', result['native_storage_id'] != source['native_storage_id'], True)
        c.check('input-complete-readback-unchanged', c.tensor(c.live['input']), source)
        for name in list(c.live)[::-1]:
            if name != 'result':
                c.drop(name)
        compare(c.tensor(c.live['result']), row['result'], 'result-after-all-input-release')
        c.drop('result')
        status, record = c.issue((43, c.weak['result']), 'released-result-Storage')
        c.check('released-result-Storage-expires', status, 6)
        c.check('released-result-publishes-nothing', record, None)
        c.check('full-native-RNG-unchanged', c.n.one(63, c.gen)['payload'].hex(), c.rng.hex())
        c.finish_controls()

    for index, row in enumerate(reference['rows']):
        rows.append(native.run_one(args, observer, owner, args.output / ('dtype-%02d' % (index + 5)),
                                   row['spec']['name'], lambda c, row=row: execute(c, row)))
    report = dict(schema='heptatorch.bool-dtype-extension.native.v1',
        status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
        cases=rows, children=2, frames=sum(r['frames'] for r in rows), reference=binding,
        source_sha256=profile.SOURCE, native_sha256=profile.IMAGE, actual_build_sha256=args.expected_build_sha256,
        native_executed=True, torch_executed=False, observer_adaptation=proof,
        inputs_unchanged=inputs.unchanged(), inputs=inputs.rows)
    dump(args.output / 'report.json', report)
    return report


def worker(path, expected):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, 'exact internal worker request')
    cfg = json.loads(raw)
    native, support, adapter = load_reader(Path(cfg['reader_directory']))
    argv = [str(Path(cfg['reader_directory']) / 'native.py')]
    for key in ('repository', 'binary', 'source', 'build_report', 'reference', 'output',
                'expected_source_sha256', 'expected_native_sha256', 'expected_build_sha256', 'expected_reference_sha256'):
        argv += ['--' + key.replace('_', '-'), cfg[key]]
    if cfg['prepare_only']:
        argv += ['--prepare-only']
    sys.argv = argv
    result = native.main()
    if not cfg['prepare_only']:
        original = json.loads((Path(cfg['output']) / 'report.json').read_bytes())
        require(original['native_executed'] is True and original.get('children') == 31,
                'all original F32 sessions before extension')
        args = argparse.Namespace(**cfg)
        for name in ('repository', 'binary', 'source', 'build_report', 'reference', 'reader_directory'):
            setattr(args, name, Path(getattr(args, name)))
        args.output = Path(cfg['extension_output'])
        args.output.mkdir(parents=True, exist_ok=False)
        args.protocol = ROOT / 'tests/differentiable_view_native_review/protocol.py'
        run_extension(args, native, adapter, support)
    # The original reader's exit is retained. The outer layer validates its two
    # exact, expected historical failures and both new observed successes.
    return result


def validate_original(report, directory, history):
    require(report['schema'] == 'heptatorch.sum-integral-f32.native17.v1'
            and report['status'] == 'FAIL' and report['error'] is None
            and report['native_executed'] is True and report['inputs_unchanged'] is True,
            'retain actual unchanged original reader FAIL')
    require([len(report[k]) for k in ('cases','axis_controls','dtype_controls','entry_controls','allocation_controls')]
            == [17,4,7,1,2], 'all original F32 comparisons and controls executed')
    for key in ('cases', 'axis_controls', 'entry_controls', 'allocation_controls'):
        for row in report[key]:
            history.passing_case(row)
    retained = []
    for index, row in enumerate(report['dtype_controls']):
        if index < 5:
            history.passing_case(row)
            continue
        failure = dict(label='documented-current-dtype-boundary', actual=0, expected=7, passed=False)
        require(row['status'] == 'FAIL' and row['exit_code'] == 0 and row['error']['type'] == 'AssertionError'
                and 'EOF_failure' not in row['error']
                and [c for c in row['checks'] if c['passed'] is not True] == [failure],
                'only old F32/F64 Bool refusal assertion fails')
        require(row['error']['message'] == str(dict(rule='native-comparison',
                    label='documented-current-dtype-boundary', actual=0, expected=7))
                and len(row['operations']) == 1, 'specific original F32 control failure')
        operation = row['operations'][0]
        require(operation['frame_status'] == operation['normalized_status'] == 0,
                'original Bool dtype request succeeded')
        wire = history.raw_observation(directory / ('dtype-%02d' % index),
            operation['request_hex'], operation['response_hex'], 0, 1)
        require(wire['command'][0] == 115 and wire['command'][5] == 1, 'newly added explicit Bool target')
        retained.append(dict(index=index, name=row['name'], original_status='FAIL',
                             original_expected_status=7, **wire))
    return retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository','binary','source','build-report','reference','reader-directory','output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('source','native','build','reference'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    require(not sys.flags.optimize, 'assertions enabled')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    require(args.repository == ROOT, 'selected current repository')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='heptatorch.current-fusion.f32.v1', status='RUNNING', error=None,
                  native_executed=False, Torch_executed=False, full_project_status='HOLD', schema_closure=False)
    code = 1
    try:
        cfg = {key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
        cfg['output'] = str(args.output / 'original')
        cfg['extension_output'] = str(args.output / 'extension')
        dump(args.output / 'worker.json', cfg)
        process = load(ROOT / 'tests/current_runtime_process.py', 'fusion_process')
        command = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--internal-worker',
                   str(args.output / 'worker.json'), sha((args.output / 'worker.json').read_bytes())]
        receipt = process.execute(command, args.output, 1800, cwd=ROOT)
        report['original_runner'] = receipt
        require(receipt['started'] is True and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
                and receipt['forced_cleanup'] is False and receipt['stderr_bytes'] == 0
                and receipt['exit_code'] == (0 if args.prepare_only else 1), 'complete original reader exit and EOF')
        path = args.output / 'original/report.json'
        original_raw = path.read_bytes()
        original = json.loads(original_raw)
        if args.prepare_only:
            require(original['status'] == 'PREPARED_NOT_RUN' and original['native_executed'] is False,
                    'preparation never transfers a previous pass')
            report.update(status='PREPARED_NOT_RUN', source_transition=original['source_transition'])
            code = 0
        else:
            history = load(ROOT / 'tests/current_runtime_historical/run.py', 'fusion_historical_validator')
            retained = validate_original(original, args.output / 'original', history)
            extension_path = args.output / 'extension/report.json'
            extension_raw = extension_path.read_bytes()
            extension = json.loads(extension_raw)
            require(extension['status'] == 'PASS' and extension['children'] == len(extension['cases']) == 2
                    and extension['inputs_unchanged'] is True and extension['native_executed'] is True
                    and extension['source_sha256'] == args.expected_source_sha256
                    and extension['native_sha256'] == args.expected_native_sha256,
                    'both observed new Bool dtype successes on same image')
            for row in extension['cases']:
                history.passing_case(row)
            transport = history.native_transports(args.output, 33)
            require(transport['frames'] == original['frames'] + extension['frames'], 'all 33 native frame streams')
            report.update(original)
            report.update(schema='heptatorch.current-fusion.f32.v1', status='PASS', error=None,
                dtype_controls=original['dtype_controls'][:5] + extension['cases'], children=33,
                frames=transport['frames'], transport=transport,
                original_runner=receipt, original_report_status='FAIL_RETAINED',
                original_report=dict(path='original/report.json', sha256=sha(original_raw), bytes=len(original_raw)),
                extension_report=dict(path='extension/report.json', sha256=sha(extension_raw), bytes=len(extension_raw)),
                original_dtype_controls=original['dtype_controls'], original_dtype_failed_indices=[5,6],
                original_dtype_failures_retained=retained,
                current_dtype_accepted_indices=[0,1,2,3,5,6], current_dtype_refused_indices=[4],
                new_Bool_dtype_reference=extension['reference'],
                historical_Bool7_changed_indices=[2,3,5,6], all_original_comparison_and_control_bytes_unchanged=True,
                all_original_FAIL_reports_unchanged=True, actual_build_sha256=args.expected_build_sha256)
            require(path.read_bytes() == original_raw and extension_path.read_bytes() == extension_raw,
                    'both original reports remain unchanged after qualification')
            code = 0
    except BaseException as failure:
        report.update(status='FAIL', error=dict(type=type(failure).__name__, message=str(failure),
                                                traceback=traceback.format_exc()))
    dump(args.output / 'report.json', report)
    print(json.dumps({key:report.get(key) for key in ('status','passed','children','frames','error')}))
    return code


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--internal-worker':
        raise SystemExit(worker(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
