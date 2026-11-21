#!/usr/bin/env python3
"""Launch the admitted public F64 observer and read back its complete evidence."""
import json
import lzma
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common
import native
import run736 as run

ROOT = run.ROOT
sha = run.sha
require = run.require
dump = run.dump


def run_native(command,out):
    receipt=dict(argv=command,timeout_seconds=900,started=False,stdout_EOF=False,stderr_EOF=False)
    child=None;stdout=stderr=b'';start=time.monotonic()
    try:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            cwd=ROOT,start_new_session=True,env={'PATH':os.defpath,'LANG':'C.UTF-8','LC_ALL':'C.UTF-8'})
        receipt['started']=True
        try:stdout,stderr=child.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            receipt['timed_out']=True;os.killpg(child.pid,signal.SIGKILL)
            stdout,stderr=child.communicate(timeout=10)
        receipt.update(exit_code=child.returncode,stdout_EOF=True,stderr_EOF=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);stdout,stderr=child.communicate(timeout=10)
            receipt.update(forced_cleanup=True,exit_code=child.returncode)
        receipt.update(elapsed_seconds=time.monotonic()-start,stdout_bytes=len(stdout),stderr_bytes=len(stderr),
                       stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
        (out/'runner.stdout').write_bytes(stdout);(out/'runner.stderr').write_bytes(stderr)
        dump(out/'runner.json',receipt)
    return receipt


def split_frames(raw, magic):
    frames, offset = [], 0
    while offset < len(raw):
        require(offset + 16 <= len(raw), 'truncated owner frame header')
        actual_magic, size = struct.unpack_from('<8sQ', raw, offset)
        require(actual_magic == magic and 32 <= size <= 1 << 20,
                'owner frame magic or bounded extent changed')
        end = offset + 16 + size
        require(end <= len(raw), 'truncated owner frame body')
        frames.append(raw[offset + 16:end])
        offset = end
    require(offset == len(raw), 'owner frame trailing bytes')
    return frames


def request_commands(body):
    magic, count, extent, flags = struct.unpack_from('<8s3Q', body)
    require(magic == b'HTORCH03' and extent == len(body) and flags == 0
            and 1 <= count <= 16383 and 32 + 64 * count <= len(body),
            'complete request command header changed')
    return [struct.unpack_from('<8Q', body, 32 + 64 * index) for index in range(count)]


def read_child(folder, inputs, wire, image_sha, expected_journals):
    transport = json.loads(inputs.bind(folder / 'raw/transport.json'))
    observed = json.loads(inputs.bind(folder / 'observations.json'))
    require(transport['exit_code'] == observed['exit_code'] == 0
            and observed['stdout_EOF'] is True and observed['stderr_EOF'] is True
            and transport['executable_sha256'] == image_sha
            and transport['responses_before_stdin_eof'] is True,
            'native child actual image, final exit or full EOF changed: ' + folder.name)
    requests = split_frames(inputs.bind(folder / 'raw/requests.frames.bin'), b'HTOWNR01')
    responses = split_frames(inputs.bind(folder / 'raw/responses.frames.bin'), b'HTOWNS01')
    require(len(requests) == len(responses) == transport['frames'] == observed['frames']
            == len(transport['rows']) and len(requests) > 0,
            'complete raw request/response/transport denominator changed: ' + folder.name)
    pairs = []
    for ordinal, (request, response, event) in enumerate(zip(requests, responses, transport['rows']), 1):
        commands = request_commands(request)
        status, records = wire.parse(response)
        require(event['frame'] == ordinal and event['pid'] == transport['pid']
                and event['response_bytes'] == len(response) and event['stdin_open'] is True,
                'native transport frame identity changed: ' + folder.name)
        if status == 0:
            require(len(records) == len(commands)
                    and [record['opcode'] for record in records] == [command[0] for command in commands],
                    'complete successful response command mapping changed')
        else:
            require(records == [] and len(response) == 32, 'atomic native failure response changed')
        pairs.append(dict(request_sha256=sha(request), response_sha256=sha(response), status=status))
    journals = observed['journal']
    require(len(journals) == expected_journals, 'observer journal denominator changed: ' + folder.name)
    cursor, owner_cache = 0, {}
    for row in journals:
        for index in range(cursor, len(pairs)):
            if all(pairs[index][key] == row[key] for key in ('request_sha256', 'response_sha256', 'status')):
                cursor = index + 1
                break
        else:
            raise ValueError('observer journal is absent from original ordered frames: ' + folder.name)
        for key in ('owner_before_sha256', 'owner_after_sha256'):
            digest = run.pin(row[key])
            if digest not in owner_cache:
                compressed = inputs.bind(folder / (digest + '.owner.xz'))
                decoder = lzma.LZMADecompressor()
                owner = decoder.decompress(compressed, max_length=(64 << 20) + 1)
                require(decoder.eof and not decoder.unused_data and len(owner) <= 64 << 20 and sha(owner) == digest,
                        'full owner archive digest or complete XZ extent changed')
                require(owner[64:72] == b'HTSINT02'
                        and owner[native.BASE:native.BASE + 8] == b'HTOSIN02', 'owner format changed')
                slots, pool = struct.unpack_from('<2Q', owner, native.BASE + 16)
                require(native.BASE + 144 + 552 * slots + pool == len(owner),
                        'full owner extent changed')
                owner_cache[digest] = owner
        if row['status']:
            require(row['complete_owner_rollback'] is True
                    and row['owner_before_sha256'] == row['owner_after_sha256'],
                    'actual failure lost complete owner rollback')
    return dict(pid=transport['pid'], frames=len(pairs), checks=observed['checks'],
                owner_archives=len(owner_cache), journal=journals, stdout_EOF=True, stderr_EOF=True)


def read_back(args, inputs, built, reference, cold_inputs, directory):
    raw = inputs.bind(directory / 'report.json')
    result = json.loads(raw)
    require(result['schema'] == 'heptatorch.public-f64-dense-logical.native.v1'
            and result['status'] == 'PASS_PUBLIC_F64_LOGICAL192_ONLY'
            and result['cases'] == result['expected_cases'] == len(result['rows']) == 192
            and [row['name'] for row in result['rows']] == [row['name'] for row in run.cases.CASES]
            and all(row['status'] == 'PASS' for row in result['rows']),
            'all 192 actual independent native comparisons must pass')
    require(result['native_executed'] is True and result['native_attempted'] is True
            and result['torch_executed'] is False and result['compiler_executed'] is False
            and result['all_inputs_unchanged'] is True and result['input_recheck_errors'] == []
            and result['original147_executed'] is False and result['full_pytorch_rewrite_complete'] is False
            and result['full_schema_closure'] is False and result['prepare_only'] is False,
            'native execution scope changed')
    require(result['source_sha256'] == built['source_sha256']
            and result['native_sha256'] == built['native_sha256']
            and result['actual_build_sha256'] == args.expected_build_sha256
            and result['reference_sha256'] == args.expected_reference_sha256
            and result['reference_execution_sha256'] == args.expected_execution_sha256
            and result['case_sha256'] == run.cases.CASE_SHA256
            and result['allocation_sha256'] == run.cases.ALLOCATION_SHA256
            and result['boundary_sha256'] == run.cases.BOUNDARY_SHA256
            and result['boundary_observations'] == reference['boundary_rows']
            and result['boundary_observations_not_counted_as_native_pass'] is True,
            'native source/build/reference or scope binding changed')
    for path, identity in result['inputs'].items():
        bound = inputs.bind(Path(path), identity['sha256'])
        require(len(bound) == identity['bytes'] and inputs.records[str(Path(path).absolute())] == identity,
                'native bound physical input changed: ' + path)
    require(result['cold_admission_inputs'] == cold_inputs and len(result['cold_admission_inputs']) == 73
            and all(result['inputs'][path] == identity for path, identity in result['cold_admission_inputs'].items()),
            'native observer omitted one of the 73 physical cold-admission inputs')
    for path, expected in [(args.build_directory / 'build.json', args.expected_build_sha256),
                           (args.reference, args.expected_reference_sha256),
                           (args.reference_execution, args.expected_execution_sha256),
                           (Path(run.__file__).resolve(), sha(inputs.bind(Path(run.__file__).resolve())))]:
        require(result['inputs'][str(path)]['sha256'] == expected,
                'native observer omitted a required exact input')
    control = result['packet_controls']
    expected_controls = [name + str(op) for op in (118, 119) for name in
                         ('reserved48', 'reserved56', 'invalid-dtype', 'invalid-keepdim', 'too-many-axes',
                          'invalid-source', 'missing-axis-payload', 'second-command-rollback')]
    expected_controls += ['stale-source118', 'stale-source119']
    require(control['status'] == 'PASS' and control['controls'] == 18 and control['names'] == expected_controls,
            'all original eighteen packet controls must pass')
    allocations = result['allocation_controls']
    require(len(allocations) == 2 and [row['opcode'] for row in allocations] == [118, 119]
            and all(row['status'] == 'PASS' and row['full_owner_rollback'] is True
                    and row['same_request_after_restore'] is True and row['attempts'][-1]['status'] == 3
                    and all(attempt['status'] == 0 for attempt in row['attempts'][:-1]) for row in allocations),
            'both actual owner-growth failures and identical retries are required')
    boundary = result['scope_controls']
    from f64_view.legacy_successors import replay_scope
    require(args.f64_view_successor.proof['strict_reference_qualified'] is True,
            'strict shared original188 reference is mandatory for the view successor')
    replay_scope(args, native.load_wire(ROOT), directory / 'scope-controls', native,
                 args.f64_view_successor, inputs, boundary)
    expected_folders = ['case-%03d' % index for index in range(192)]
    expected_folders += ['packet-controls', 'allocation-118', 'allocation-119', 'scope-controls']
    require(sorted(path.parent.parent.name for path in directory.glob('*/raw/transport.json')) == sorted(expected_folders),
            'the exact 196-child evidence set is required')
    wire = native.load_wire(ROOT)
    children = []
    for index, folder in enumerate(expected_folders):
        expected_journals = 1 if index < 192 else {'packet-controls': 18, 'scope-controls': 2,
            'allocation-118': len(allocations[0]['attempts']) + 1,
            'allocation-119': len(allocations[1]['attempts']) + 1}[folder]
        observed = read_child(directory / folder, inputs, wire, built['native_sha256'], expected_journals)
        if index < 192:
            case = json.loads(inputs.bind(directory / folder / 'report.json'))
            require(case == result['rows'][index] and case['checks'] == observed['checks']
                    and case['frames'] == observed['frames']
                    and case['reference_error_status'] == native.expected_status(reference['rows'][index])
                    and case['accepted'] == reference['rows'][index]['accepted']
                    and observed['journal'][0]['status'] == case['reference_error_status'],
                    'native case report differs from actual reference/journal/frame evidence')
        else:
            row = control if folder == 'packet-controls' else boundary if folder == 'scope-controls' else allocations[0 if folder.endswith('118') else 1]
            require(row['checks'] == observed['checks'] and row['frames'] == observed['frames'],
                    'native control frame/check denominator changed')
        children.append(dict(folder=folder, **observed))
    require(sum(child['frames'] for child in children) == result['frames']
            and sum(child['checks'] for child in children[:192]) == result['checks'],
            'complete native frame/check denominator changed')
    return dict(status='PASS_PUBLIC_F64_LOGICAL192_ONLY', cases=192, packet_controls=18,
                allocation_controls=2, native_boundary_controls=2, native_children=196,
                frames=result['frames'], case_checks=result['checks'],
                total_checks=sum(child['checks'] for child in children),
                native_report_sha256=sha(raw), view_successor=args.f64_view_successor.proof, all_native_EOF=True,
                complete_raw_frames_reparsed=True, full_owner_archives_rechecked=True,
                children=children)


def main():
    args = run.arguments()
    require(not sys.flags.optimize, 'invariant assertions must remain enabled')
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = common.Inputs()
    report = dict(schema='heptatorch.public-f64-dense-logical.qualification.v1', status='RUNNING',
                  native_executed=False, native_attempted=False, torch_executed=False, compiler_executed=False,
                  original147_executed=False, full_pytorch_rewrite_complete=False, full_schema_closure=False,
                  prepare_only=args.prepare_only)
    code = 0
    try:
        inputs.bind(Path(__file__).resolve())
        inputs.bind(Path(sys.executable).resolve())
        built, reference, execution, cold_inputs = run.admit(args, inputs)
        report.update(source_sha256=built['source_sha256'], native_sha256=built['native_sha256'],
                      actual_build_sha256=args.expected_build_sha256,
                      reference_sha256=args.expected_reference_sha256,
                      reference_execution_sha256=args.expected_execution_sha256,
                      cold_admission_inputs=cold_inputs)
        if args.prepare_only:
            report['status'] = 'PREPARED_NOT_EXECUTED'
        else:
            command = [sys.executable, '-E', '-S', '-B', str(HERE / 'run736.py'),
                       '--repository', str(ROOT), '--build-directory', str(args.build_directory),
                       '--expected-build-sha256', args.expected_build_sha256,
                       '--reference', str(args.reference), '--expected-reference-sha256', args.expected_reference_sha256,
                       '--reference-execution', str(args.reference_execution),
                       '--expected-execution-sha256', args.expected_execution_sha256,
                       '--f64-view-reference', str(args.f64_view_reference),
                       '--f64-view-reference-execution', str(args.f64_view_reference_execution),
                       '--output', str(args.output / 'native')]
            report['native_attempted'] = True
            receipt = run_native(command, args.output)
            report['runner'] = receipt
            for stream in ('stdout', 'stderr'):
                stream_raw = inputs.bind(args.output / ('runner.' + stream), receipt[stream + '_sha256'])
                require(len(stream_raw) == receipt[stream + '_bytes'], 'native runner stream extent changed')
            require(receipt['started'] is True and receipt.get('exit_code') == 0
                    and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
                    and receipt['stderr_bytes'] == 0 and not receipt.get('timed_out')
                    and not receipt.get('forced_cleanup'), 'native observer failed or streams were incomplete')
            report.update(read_back(args, inputs, built, reference, cold_inputs, args.output / 'native'))
            report['native_executed'] = True
    except BaseException as error:
        report.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error),
                                               traceback=traceback.format_exc()))
        code = 1
    finally:
        changed = inputs.recheck()
        report.update(inputs=inputs.records, all_inputs_unchanged=not changed, input_recheck_errors=changed)
        report['native_executed'] = any(path.stat().st_size > 0 for path in args.output.glob('native/*/raw/responses.frames.bin'))
        if changed:
            report['status'] = 'FAIL'
            code = 1
        dump(args.output / 'report.json', report)
    print(json.dumps({key: report.get(key) for key in ('status', 'cases', 'packet_controls',
                                                     'allocation_controls', 'native_children', 'frames', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
