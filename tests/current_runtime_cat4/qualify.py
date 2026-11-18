#!/usr/bin/env python3
"""Pure readback of the four original cat/stack packets and physical owner bytes.

No native image, compiler, linker or Torch reference is executed here. The
original runner and transport remain unchanged; their complete files and the
saved physical evidence are independently bound and checked again.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
import tarfile
import traceback
import zlib

sys.dont_write_bytecode = True

SOURCE_FILES = {
    'run.py': '97fc3c409bb64228c8202b6f4a4d840d2467c5b70b268a898a741d1aed5a54fc',
    'packet.py': '874c40452fdfe9b1827ef3352b79b0d49f4842b7ee608aa2290582509e2a3e36',
    'transport.py': '1619952a094057ebd4b837b03e65c287b90bd0283b44482d32585f2bdfbaed0f',
    'cases.json': 'f8f635548b1e9fd421c9b1af44dc57d82b33f8d3e8a476c19fa906ee4b095d5e',
}
DEPENDENCIES = {
    'tests/current_runtime270.py': '2e54cfb121ca56d641c0feca30ede4992d1fad505f7a4ea5bd399e966a4392d7',
    'tests/current_runtime_original35/binding.py': '29d25567c08e309c25d96d56f4476e3842ceda164c9647eb5c9397a04fad181a',
    'tests/current_runtime_fixtures.py': 'a8c00f1958bf9166d1e36c012bec95b29333a6cd434a19b2296c5ce4434c175b',
}
CASE_NAMES = ('cat-saved-list-arena-boundary', 'cat-saved-list-arena-overflow',
              'cat-256-repeated-inputs', 'stack-scalars-axis-0')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def load_bound(path, name, inputs, expected=None):
    raw = inputs.bind(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


class CapturedInputs:
    """Repeat exactly the fixed runner's input census through the outer binder."""
    def __init__(self, outer):
        self.outer = outer
        self.records = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        raw = self.outer.bind(path, expected)
        row = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(path.stat().st_mode))
        require(str(path) not in self.records or self.records[str(path)] == row,
                'a fixed runner input changed during independent readback')
        self.records[str(path)] = row
        return raw


def single_xz(compressed):
    decoder = lzma.LZMADecompressor()
    raw = decoder.decompress(compressed)
    require(decoder.eof and not decoder.unused_data, 'one complete XZ stream without trailing bytes')
    return raw


def single_gzip(compressed):
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = decoder.decompress(compressed) + decoder.flush()
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            'one complete gzip stream without trailing bytes')
    return raw


def check_original_package(repository, directory, fixtures, inputs):
    """Check the existing extracted package without writing or restoring files."""
    folder = repository / 'validation/current-runtime270-source'
    compressed = inputs.bind(folder / 'complete.tar.xz', fixtures.CURRENT270_ARCHIVE_SHA)
    manifest_raw = inputs.bind(folder / 'manifest.json', fixtures.CURRENT270_MANIFEST_SHA)
    manifest = json.loads(manifest_raw)
    require(manifest['schema'] == 'heptatorch.current270.source-package.v1'
            and manifest['payload_files'] == len(manifest['files']) == 12
            and manifest['archive_regular_files'] == 13
            and manifest['payload_bytes'] == sum(pin['bytes'] for pin in manifest['files'].values()),
            'complete fixed original owner package inventory')
    pins = dict(manifest['files'])
    pins['package-manifest.json'] = dict(bytes=len(manifest_raw), sha256=sha(manifest_raw), mode=0o644)
    seen, directories = set(), set()
    with tarfile.open(fileobj=io.BytesIO(single_xz(compressed)), mode='r:') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            require(relative.parts and not relative.is_absolute() and '..' not in relative.parts
                    and str(relative) == member.name and '\\' not in member.name,
                    'safe original package member')
            if member.isdir():
                require(member.name in ('current270', 'current270/frozen')
                        and member.name not in directories and member.mode == 0o755,
                        'exact original package directories')
                directories.add(member.name)
                local = directory.joinpath(*relative.parts[1:])
                require(local.is_dir() and not local.is_symlink()
                        and stat.S_IMODE(local.stat().st_mode) == member.mode,
                        'original package directory mode')
                continue
            require(member.isfile() and relative.parts[0] == 'current270',
                    'original package members must be regular files')
            name = '/'.join(relative.parts[1:])
            require(name in pins and name not in seen, 'complete unique original package member')
            seen.add(name)
            raw = archive.extractfile(member).read()
            pin = pins[name]
            path = directory / name
            require(len(raw) == member.size == pin['bytes'] and sha(raw) == pin['sha256']
                    and member.mode == pin['mode']
                    and inputs.bind(path, pin['sha256']) == raw
                    and stat.S_IMODE(path.stat().st_mode) == pin['mode'],
                    'physical original owner source bytes or modes changed')
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()}
    require(seen == actual == set(pins) and directories == {'current270', 'current270/frozen'},
            'all thirteen original owner package files must be present')
    return dict(files=len(pins), archive_sha256=sha(compressed), manifest_sha256=sha(manifest_raw))


def read_owner(directory, observation, label, owner, inputs):
    require(observation['label'] == label and observation['raw_file'] == label + '.owner.xz'
            and observation['decoded_file'] == label + '.json.gz', 'exact owner observation paths')
    compressed = inputs.bind(directory / observation['raw_file'], observation['compressed_sha256'])
    raw = single_xz(compressed)
    decoded_raw = single_gzip(inputs.bind(directory / observation['decoded_file']))
    decoded = json.loads(decoded_raw)
    require(len(raw) == observation['raw_bytes'] == decoded['length']
            and sha(raw) == observation['raw_sha256'] == decoded['full_sha256']
            and sha(decoded_raw) == observation['decoded_sha256']
            and decoded_raw == canonical(decoded), 'complete original owner byte/JSON identities')
    require(type(decoded['pointer']) is int and 0 <= decoded['pointer'] < 1 << 64,
            'original native owner pointer extent')
    repeated = owner.decode_owner(raw, decoded['pointer'])
    require(canonical(repeated) == decoded_raw, 'physical owner bytes must reproduce the complete snapshot')
    checked = owner.verify_snapshot(repeated)
    require(checked == observation['verified']
            and checked['all_original_84Q_predicates_retained'] is True,
            'all original owner predicates must be recomputed from physical bytes')
    if label == 'before':
        require(raw == b'' and repeated['pointer'] == repeated['length'] == 0,
                'each fixed packet starts in a fresh original owner')
    else:
        require(raw and repeated['pointer'] > 0 and repeated['length'] > 0,
                'complete settled owner after actual response')
    return repeated, len(raw)


def read_transport(folder, row, body, image, owner, inputs):
    directory = folder / 'native'
    actual_raw = inputs.bind(directory / 'transport.json')
    actual = json.loads(actual_raw)
    require(actual == row['transport'], 'original case and physical transport ledger must agree')
    require(actual['status'] == 'PASS_ACTUAL_TRANSPORT'
            and actual['native_started'] is True and actual['native_executed'] is True
            and type(actual['process_id']) is int and actual['process_id'] > 0
            and actual['executed_image_sha256'] == image
            and actual['exit_code'] == 0 and actual['stdout_EOF'] is True
            and actual['stderr_EOF'] is True and actual['forced_cleanup'] is False
            and actual['error'] is None and actual['frames'] == 1
            and actual['response_before_stdin_EOF'] is True
            and actual['unexpected_stdout_bytes'] == actual['stderr_bytes'] == 0,
            'actual native process, response before input EOF, exit and both empty output EOFs')
    require(isinstance(actual['wall_seconds'], (int, float)) and not isinstance(actual['wall_seconds'], bool)
            and math.isfinite(actual['wall_seconds']) and actual['wall_seconds'] > 0,
            'actual transport duration')
    request = inputs.bind(directory / 'actual.stdin', actual['stdin_file']['sha256'])
    response = inputs.bind(directory / 'actual.stdout', actual['stdout_file']['sha256'])
    require(len(request) == actual['stdin_file']['bytes']
            and request == b'HTOWNR01' + struct.pack('<Q', len(body)) + body
            and sha(body) == actual['request_sha256'], 'entire original converted request frame')
    require(len(response) == actual['stdout_file']['bytes'] and len(response) >= 16,
            'entire actual response stream extent')
    magic, extent = struct.unpack_from('<8sQ', response)
    require(magic == b'HTOWNS01' and 32 <= extent <= 1 << 20 and len(response) == extent + 16,
            'exactly one complete native response frame with no tail')
    reply = inputs.bind(directory / 'actual-response-body.bin', actual['response_sha256'])
    require(reply == response[16:] and row['response_sha256'] == sha(reply),
            'full response body must be the actual stdout payload')
    require(inputs.bind(directory / 'unexpected.stdout') == inputs.bind(directory / 'native.stderr') == b''
            and not (directory / 'failure-tail.stdout').exists(), 'both saved native EOF tails must be empty')
    observations = actual['owner_observations']
    require(len(observations) == 2 and [item['label'] for item in observations] == ['before', 'after'],
            'both complete physical owner observations')
    snapshots, total = [], 0
    for item, label in zip(observations, ('before', 'after')):
        snapshot, count = read_owner(directory, item, label, owner, inputs)
        snapshots.append(snapshot)
        total += count
    return reply, snapshots[-1], actual, total


def qualify_cat4(report, repository, build, build_sha, directory, inputs):
    """Independently admit saved evidence; never run the native test again."""
    require(not sys.flags.optimize, 'all fixed original assertions must remain enabled')
    repository, build, directory = map(lambda value: Path(value).resolve(), (repository, build, directory))
    inputs.bind(Path(__file__).resolve())
    require(json.loads(inputs.bind(directory / 'report.json')) == report, 'original report bytes must be bound')
    require(report['schema'] == 'heptatorch.current-runtime.cat4.v1'
            and report['status'] == 'PASS_CAT4_ONLY' and report['prepare_only'] is False
            and report['native_executed'] is True and report['Torch_executed'] is False
            and report['compiler_executed'] is False and report['all_inputs_unchanged'] is True
            and report['input_recheck_errors'] == [] and not report.get('error'),
            'complete actual cat4 execution and original input custody')
    require(report['original_expected_bytes_modified'] is False
            and report['complete_legacy461_qualified'] is False
            and report['schema_closure'] is False and report['full_pytorch_rewrite_complete'] is False
            and report['full_project_status'] == 'HOLD', 'four fixed packets are finite evidence')
    original_inputs = CapturedInputs(inputs)
    folder = repository / 'tests/current_runtime_cat4'
    for name, pin in SOURCE_FILES.items():
        original_inputs.bind(folder / name, pin)
    definitions = json.loads(original_inputs.bind(folder / 'cases.json', SOURCE_FILES['cases.json']))
    require(definitions['schema'] == 'heptatorch.cat4.fixed-fixtures.v1'
            and [item['name'] for item in definitions['cases']] == list(CASE_NAMES)
            and definitions['packet_sha256'] == SOURCE_FILES['packet.py']
            and definitions['transport_sha256'] == SOURCE_FILES['transport.py'], 'the exact four fixed fixtures')
    for path, pin in DEPENDENCIES.items():
        original_inputs.bind(repository / path, pin)
    previous_path = sys.path[:]
    names = ('binding', 'prepare', 'current_runtime_fixtures', 'cat4_readback_packet', 'cat4_readback_adapter')
    previous_modules = {name: sys.modules.get(name) for name in names}
    try:
        load_bound(repository / 'tests/current_runtime_original35/binding.py', 'binding', original_inputs,
                   DEPENDENCIES['tests/current_runtime_original35/binding.py'])
        fixtures = load_bound(repository / 'tests/current_runtime_fixtures.py',
                              'current_runtime_fixtures', original_inputs,
                              DEPENDENCIES['tests/current_runtime_fixtures.py'])
        packet = load_bound(folder / 'packet.py', 'cat4_readback_packet', original_inputs, SOURCE_FILES['packet.py'])
        adapter = load_bound(repository / 'tests/current_runtime270.py', 'cat4_readback_adapter',
                             original_inputs, DEPENDENCIES['tests/current_runtime270.py'])
        package = directory / 'original-owner-package'
        package_proof = check_original_package(repository, package, fixtures, original_inputs)
        built, source_proof = adapter.make_bindings(package, build, build_sha, original_inputs)
        require(report['source_sha256'] == built['source_sha256']
                and report['native_sha256'] == built['native_sha256']
                and report['actual_build_sha256'] == build_sha
                and report['functions'] == built['functions']
                and report['units'] == built['units_planned'],
                'cat4 must use the same complete strictly admitted actual cold build')
        owner, owner_proof, owner_raw = sys.modules['prepare'].owner_module(
            built['source_sha256'], built['native_sha256'])
        require(original_inputs.bind(directory / 'owner-current.py') == owner_raw,
                'all original owner decoder and snapshot predicates must remain unchanged')
        require(json.loads(inputs.bind(directory / 'source-binding.json')) == dict(
            current=source_proof, owner=owner_proof, complete_original_owner_predicates_unchanged=True),
            'full original owner and actual source binding proof must recompute exactly')
        require(len(report['cases']) == 4 and [row['name'] for row in report['cases']] == list(CASE_NAMES),
                'all four actual case reports in original order')
        rows, children, owner_bytes, record_count = [], set(), 0, 0
        successes = controls = 0
        for ordinal, (item, row) in enumerate(zip(definitions['cases'], report['cases'])):
            case = directory / ('%02d-' % ordinal + item['name'])
            require(json.loads(inputs.bind(case / 'case.json')) == row,
                    'complete original case object must agree with aggregate report')
            require(row['name'] == item['name'] and row['mode'] == item['mode']
                    and row['original_status'] == item['legacy_status']
                    and row['native_executed'] is True and row['original_failure_reference_reclassified'] is False
                    and row['exact_request_inverse'] is True and row['exact_actual_response_inverse'] is True
                    and row['actual_status'] == 0 and not row.get('error'), 'actual complete case identity')
            blobs = {}
            for suffix in ('input', 'output'):
                pin = item[suffix]
                path = repository / 'tests/fixtures' / (item['name'] + '.' + suffix + '.hex')
                raw = original_inputs.bind(path, pin['file_sha256'])
                require(len(raw) == pin['file_bytes'] and stat.S_IMODE(path.stat().st_mode) == pin['mode'],
                        'original fixture file bytes and mode')
                blobs[suffix] = bytes.fromhex(raw.decode('ascii'))
                require(sha(blobs[suffix]) == pin['decoded_sha256'], 'entire original fixture payload')
            require(inputs.bind(case / 'original-request.bin') == blobs['input']
                    and inputs.bind(case / 'original-expected.bin') == blobs['output'],
                    'actual case must retain the exact original request and expected response')
            body, inverse, commands, ledger = packet.convert_request(blobs['input'])
            require(inverse['whole_original_packet_inverse_exact'] is True
                    and not inverse['unpublished_composite_inputs'] and all(command[0] != 13 for command in commands)
                    and inputs.bind(case / 'converted-request.bin') == body
                    and json.loads(inputs.bind(case / 'request-inverse.json')) == inverse,
                    'entire actual request and every identity substitution must invert exactly')
            require(packet.response(blobs['output'], 1)[0] == item['legacy_status'],
                    'complete original successful or refused expected response')
            reply, snapshot, actual, raw_count = read_transport(case, row, body, built['native_sha256'], owner, inputs)
            status, records = packet.response(reply, 3)
            require(status == actual['actual_status'] == 0 and len(records) == actual['actual_records'] == len(commands),
                    'every actual response record must correspond to the original command')
            projected, projection = packet.project_success(reply, commands, ledger, snapshot)
            require(projection['complete_actual_response_inverse_exact'] is True
                    and inputs.bind(case / 'projected-response.bin') == projected
                    and json.loads(inputs.bind(case / 'response-inverse.json')) == projection,
                    'complete native response inverse and all numerical payload bytes')
            if item['mode'] == 'original-success-exact':
                require(row['status'] == 'PASS_ORIGINAL_SUCCESS_BYTES' and row['legacy_success_exact'] is True
                        and item['legacy_status'] == 0 and projected == blobs['output'],
                        'complete original successful response bytes must match exactly')
                successes += 1
            else:
                require(item['mode'] == 'expanded-capacity-control' and item['legacy_status'] == 3
                        and row['status'] == 'PASS_EXPANDED_CAPACITY_CONTROL' and row['legacy_failure_qualified'] is False
                        and [command[0] for command in commands] == [1, 34] and len(records) == 2,
                        'explicit independently named expanded-capacity control')
                generation, slot = packet.typed(records[-1]['value'], 1)
                require(slot <= len(snapshot['tables']['1']), 'actual cat TensorID must be physically observed')
                tensor = snapshot['tables']['1'][slot - 1]
                require(tensor[0] == generation and tensor[1] == 1, 'actual expanded-capacity cat Tensor is live')
                node = snapshot['nodes'][str(tensor[6])]
                require(len(node) == 84 and node[0] == 1 and node[1] == 0 and node[4] == 1
                        and node[5] == 37 and node[12] == 0 and snapshot['header']['pool'] > 1 << 20
                        and row['actual_owner_pool_bytes'] == snapshot['header']['pool']
                        and row['actual_tensor_descriptor'] == node and projected != blobs['output'],
                        'actual complete empty differentiable cat and native arena growth')
                controls += 1
            require(actual['process_id'] not in children, 'four distinct original native children')
            children.add(actual['process_id'])
            owner_bytes += raw_count
            record_count += len(records)
            rows.append(dict(name=item['name'], mode=item['mode'], pid=actual['process_id'], frames=1,
                response_records=len(records), response_sha256=sha(reply), original_request_sha256=sha(blobs['input']),
                original_expected_sha256=sha(blobs['output']), projected_response_sha256=sha(projected),
                owner_observations=2, owner_raw_bytes=raw_count, exit_code=0, stdout_EOF=True, stderr_EOF=True,
                original_transport_sha256=sha(inputs.bind(case / 'native/transport.json'))))
        require(successes == report['original_success_cases_qualified'] == 3
                and controls == report['capacity_controls_passed'] == 1 and len(children) == 4,
                'three original byte-equivalent successes and one distinct expansion control')
        require(original_inputs.records == report['inputs'],
                'the complete fixed runner input census must physically reproduce every declared input')
        return dict(cases=4, frames=4, original_success_cases=3, expanded_capacity_controls=1,
            native_sessions=4, native_EOF_observations=8, owner_observations=8, owner_raw_bytes=owner_bytes,
            response_records=record_count, original_inputs_rechecked=len(original_inputs.records),
            exact_source_binding_recomputed=True, owner_source_inverse=owner_proof,
            original_owner_package=package_proof, case_readbacks=rows,
            new_native_execution_by_qualification=False, Torch_executed_by_qualification=False,
            original_failure_reference_reclassified=False, full_legacy461_qualified=False)
    finally:
        sys.path[:] = previous_path
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--result-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'a new independent readback report is required')
    binding_path = args.repository.resolve() / 'tests/current_runtime_original35/binding.py'
    raw = binding_path.read_bytes()
    require(sha(raw) == DEPENDENCIES['tests/current_runtime_original35/binding.py'], 'fixed input binder identity')
    spec = importlib.util.spec_from_file_location('cat4_readback_input_binding', binding_path)
    binding = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(binding_path), 'exec'), binding.__dict__)
    inputs = binding.Inputs()
    result = dict(schema='heptatorch.current-runtime.cat4-independent-readback.v1', status='FAIL',
                  native_executed_by_readback=False, compiler_executed_by_readback=False,
                  Torch_executed_by_readback=False, observed_prior_native_executed=None,
                  full_687_executed_by_readback=False, full_project_status='HOLD')
    try:
        report = json.loads(inputs.bind(args.result_directory.resolve() / 'report.json'))
        result['observed_prior_native_executed'] = report.get('native_executed')
        result['counts'] = qualify_cat4(report, args.repository, args.build_directory,
                                       args.expected_build_sha256, args.result_directory, inputs)
        result['status'] = 'PASS_CAT4_INDEPENDENT_READBACK'
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        changed = inputs.recheck()
        result.update(inputs=inputs.records, all_inputs_unchanged=not changed, input_recheck_errors=changed)
        if changed:
            result['status'] = 'FAIL'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=result['status'], cases=result.get('counts', {}).get('cases'),
                          native_executed_by_readback=False)))
    return 0 if result['status'] == 'PASS_CAT4_INDEPENDENT_READBACK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
