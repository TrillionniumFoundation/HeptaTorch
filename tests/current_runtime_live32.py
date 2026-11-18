"""Read the complete original live32 result and raw streams without executing it."""
import gzip
import json
from pathlib import Path
import stat
import struct
import types
import sys
import current_runtime_cold as generator_cold

from current_runtime_admission import SOURCE, IMAGE, require, sha

RUNNER_SHA = 'a3038375f2558bb5f2571d3d7ea0b2f294b315d9c1f6939ff677f98f8236dfe2'
ORACLE_SHA = '554671168c7a4a408dc1e1df89a19db15c3efcefcf4ae1a300af1b6dcae40160'
ADMISSION_SHA = '5b7b9a538877c06bce9daff0631657ae3b9ce79bcd7336d42839a01ef9cbcfde'
OWNER_SHA = 'c39600e7c37dea390f88cf9e74d86dfa924301456e0f32e127d82c98ad6778b0'
OWNER_EDITS = [['HTSINT01', 'HTSINT02', 2], ['HTOSIN01', 'HTOSIN02', 2],
    ['functions == 699', 'functions == 712', 1],
    ['sum-integral-source01-UNBOUND-699', 'live32-current712-UNBOUND', 1],
    ['sum-integral-source01-%d-preserved47-48', 'live32-current%d-full-owner', 1],
    ['heptatorch.sum-integral-owner-source01.v1', 'heptatorch.live32-current712-owner.v1', 1]]


def load_source(path, raw, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def inverse_source(directory, package, inputs, name, frozen_name, frozen_sha, proof, edits):
    frozen = inputs.bind(package / frozen_name, frozen_sha)
    raw = inputs.bind(directory / name, proof['executed_sha256'])
    inverse = raw
    for before, after, count in reversed(edits):
        require(inverse.count(after.encode()) == count, 'complete live32 identity inverse edit count')
        inverse = inverse.replace(after.encode(), before.encode())
    require(inverse == frozen and proof['frozen_sha256'] == frozen_sha
            and proof['whole_file_inverse_exact'] is True, 'whole original live32 support source inverse')
    return load_source(directory / name, raw, 'current_runtime_live32_' + name.replace('.', '_'))


def read_frame(request, response, frame):
    require(request[:8] == b'HTOWNR01' and len(request) >= 48
            and struct.unpack_from('<Q', request, 8)[0] == len(request) - 16
            and request[16:24] == b'HTORCH03', 'complete actual original live32 request framing')
    count, body_size, reserved = struct.unpack_from('<3Q', request, 24)
    require(count == frame['command_count'] and 1 <= count <= 16383
            and body_size == len(request) - 16 and reserved == 0
            and len(request) >= 48 + 64 * count, 'original live32 command and payload denominator')
    require(response[:8] == b'HTOWNS01' and len(response) >= 48
            and struct.unpack_from('<Q', response, 8)[0] == len(response) - 16,
            'complete actual original live32 response framing')
    magic, version, returned, status = struct.unpack_from('<8s3Q', response, 16)
    require(magic == b'HTORSP03' and version == 3 and status == frame['expected_transport_status'],
            'original live32 wire response status')
    if status:
        require(status == 7 and returned == 0 and len(response) == 48,
                'one actual count0 uniform refusal, without a synthesized record')
        return
    expected = count if frame['returned_records_override'] is None else frame['returned_records_override']
    require(returned == expected, 'complete successful request or original two-record backward prefix')
    at = 48
    for index in range(returned):
        require(at + 32 <= len(response), 'complete original live32 result header')
        opcode, error, value, size = struct.unpack_from('<4Q', response, at)
        at += 32
        require(opcode == struct.unpack_from('<Q', request, 48 + 64 * index)[0]
                and error == 0 and at + size <= len(response), 'original live32 result identity and extent')
        at += size
    require(at == len(response), 'no extra bytes after original live32 result records')


def owner_observation(directory, frame_directory, label, observation, owner, inputs):
    expected = frame_directory + '/' + label
    original = json.loads(inputs.bind(directory / (expected + '.owner-observation.json')))
    require(original == observation and observation['passed'] is True and observation['error'] is None
            and observation['label'] == label, 'original complete per-frame owner observation')
    raw_pin, decoded_pin = observation['raw'], observation['decoded']
    require(raw_pin['path'] == expected + '.owner.bin.gz'
            and decoded_pin['path'] == expected + '.owner.json.gz', 'exact per-frame owner paths')
    raw = gzip.decompress(inputs.bind(directory / raw_pin['path']))
    encoded = gzip.decompress(inputs.bind(directory / decoded_pin['path']))
    require(len(raw) == raw_pin['bytes'] and sha(raw) == raw_pin['sha256']
            and len(encoded) == decoded_pin['bytes'] and sha(encoded) == decoded_pin['sha256'],
            'all actual owner memory and decoded bytes retained')
    decoded = owner.decode_owner(raw, raw_pin['pointer'])
    require(json.dumps(decoded, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() == encoded,
            'original complete decoder reproduces the recorded actual owner state')
    checked = json.loads(json.dumps(owner.verify_snapshot(decoded), allow_nan=False))
    require(checked == observation['verification']
            and checked['all_original_84Q_predicates_retained'] is True,
            'repeat all original structural owner predicates on the retained memory')


def qualify_live32(report, build, build_sha, directory, package, inputs):
    require(report['schema'] == 'heptatorch.current712-original-live32.v1'
            and report['status'] == 'PASS_CURRENT712_ORIGINAL_LIVE32' and report['error'] is None,
            'the complete original live32 execution must actually pass')
    require(report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
            and report['build_sha256'] == build_sha and report['expected_cases'] == report['cases'] == 32
            and report['native_children'] == 32 and report['original_entry_exit'] == 0,
            'same complete runtime and all original32 trajectories')
    require(report['native_executed'] is True and report['Torch_executed'] is True
            and report['Torch_execution_attempted'] is True and report['all_inputs_unchanged'] is True
            and report['original_oracle_sha256'] == ORACLE_SHA,
            'actual same-image native and independent live Torch execution')
    for key in ('historical_runtime_executed', 'request_or_response_translation',
                'original_comparisons_changed', 'full_private_allocation_ledger_claimed',
                'full_pytorch_rewrite_complete'):
        require(report[key] is False, 'original live32 semantics and finite qualification scope')
    require(report['full_owner_structural_checks_required'] is True
            and report['owner_raw_memory_captured_before_decode'] is True,
            'actual full original owner observation required for every frame')
    require(report['frames'] == 2132 and report['exact_binary32_words'] == 17455
            and report['serialized_state_bytes'] == 1137600, 'all original live32 frame/numeric/state denominators')
    runner_raw = inputs.bind(package / 'run.py', RUNNER_SHA)
    generator_raw = inputs.bind(package / 'generator_admission.py',
        '26ff5b4ac13da51ef18912f120094a4545d21907d1f7cd3b7010b1dac74242b4')
    inputs.bind(Path(__file__).with_name('current_runtime_cold.py'), sha(generator_raw))
    original_generator_module = sys.modules.get('generator_admission')
    sys.modules['generator_admission'] = generator_cold
    try:
        reader = load_source(package / 'run.py', runner_raw, 'current_runtime_original_live32_reader')
    finally:
        if original_generator_module is None:
            del sys.modules['generator_admission']
        else:
            sys.modules['generator_admission'] = original_generator_module
    require(report['case_plan'] == reader.PLAN, 'original32 ordered plan is unchanged')
    pin = report['original_report']
    raw = inputs.bind(directory / 'original-report.json', pin['sha256'])
    require(len(raw) == pin['bytes'] and stat.S_IMODE((directory / 'original-report.json').stat().st_mode) == pin['mode'],
            'complete unchanged original live32 report bytes and mode')
    original = json.loads(raw)
    children = report['native_children_actual']
    require(len(children) == 32, 'exact native child denominator')
    reader.check_actual(original, children)
    for key in ('frames', 'exact_binary32_words', 'serialized_state_bytes',
                'absolute_version_comparisons', 'gradient_presence_comparisons', 'oracle_provenance'):
        require(report[key] == original[key], 'original live32 aggregate binding: ' + key)
    admission_proof = report['admission_source_inverse']
    frozen = inputs.bind(package / 'frozen/admission701.py', ADMISSION_SHA)
    expected_source, expected_proof = generator_cold.derive(frozen, SOURCE, IMAGE)
    expected_proof = json.loads(json.dumps(expected_proof))
    require(admission_proof == expected_proof,
            'only the exact explicit 712 source/tool/support/count/four-argument contract changed')
    admission = inverse_source(directory, package, inputs, 'executed-admission.py', 'frozen/admission701.py',
        ADMISSION_SHA, admission_proof, expected_proof['edits'])
    require(inputs.bind(directory / 'executed-admission.py') == expected_source,
            'exact fully inverse-proved new cold-admission source was actually used')
    cold = admission.admit_build(build, build_sha, build / 'heptatorch', build / 'runtime.hepta')
    require(cold == report['cold_build_admission']
            == json.loads(inputs.bind(directory / 'cold-build-admission.json')),
            'repeat the entire new fifteen-unit/sixteen-process cold-build admission')
    owner_proof = report['owner_source_inverse']
    require(owner_proof['edits'] == OWNER_EDITS and owner_proof['all_original_84Q_predicates_retained'] is True
            and owner_proof['full_private_allocation_transition_ledger_claimed'] is False,
            'complete original owner with only fixed identity/profile changes')
    owner = inverse_source(directory, package, inputs, 'executed-owner.py', 'frozen/owner_integral.py',
                           OWNER_SHA, owner_proof, OWNER_EDITS)
    owner.bind_profile(SOURCE, IMAGE, 712)
    require(report['owner_profile'] == owner.pin_profile(build / 'heptatorch', build / 'runtime.hepta'),
            'original owner profile is pinned to the same actual source and executable')
    import hashlib
    frames = 0
    for index, child in enumerate(children):
        child_directory = 'native/' + f'{index:02d}-' + reader.PLAN[index]
        require(child['directory'] == child_directory
                and json.loads(inputs.bind(directory / child_directory / 'transport.json')) == child,
                'exact actual child transport receipt')
        request_hash, response_hash = hashlib.sha256(), hashlib.sha256()
        for ordinal, frame in enumerate(child['frames']):
            frame_directory = child_directory + f'/frame-{ordinal:05d}'
            require(frame['index'] == ordinal and frame['directory'] == frame_directory
                    and frame['original_assertions_passed'] is True and frame['early_stdout_EOF'] is False
                    and json.loads(inputs.bind(directory / frame_directory / 'transport.json')) == frame,
                    'complete original per-frame assertion and transport receipt')
            request = inputs.bind(directory / frame_directory / 'request.bin', frame['request_sha256'])
            response = inputs.bind(directory / frame_directory / 'response.bin', frame['response_sha256'])
            require(len(request) == frame['request_bytes'] and len(response) == frame['response_bytes']
                    and inputs.bind(directory / frame_directory / 'attempted-request.bin') == request,
                    'actual original writes and reads retained without translation')
            read_frame(request, response, frame)
            request_hash.update(request); response_hash.update(response)
            for label in ('before', 'after'):
                owner_observation(directory, frame_directory, label, frame['owner_' + label], owner, inputs)
            frames += 1
        require(request_hash.hexdigest() == child['request_stream_sha256']
                and response_hash.hexdigest() == child['response_stream_sha256'], 'complete actual child byte streams')
        close = child['communications'][0]
        for stream in ('stdout', 'stderr'):
            raw = inputs.bind(directory / child_directory / ('close-00.' + stream), close[stream + '_sha256'])
            require(len(raw) == close[stream + '_bytes'] == 0, 'original normal close with no trailing native output')
    require(frames == report['frames'], 'all original live32 frames independently accounted for')
    for stream in ('stdout', 'stderr'):
        inputs.bind(directory / ('original-entry.' + stream))
    require(set(report['inputs']) == set(report['input_checks_after_execution'])
            and all(value is True for value in report['input_checks_after_execution'].values()),
            'complete post-execution original source/build/package input census')
    for path, row in report['inputs'].items():
        raw = inputs.bind(Path(path), row['sha256'])
        require(len(raw) == row['bytes'] and stat.S_IMODE(Path(path).stat().st_mode) == row['mode'],
                'original live32 input bytes and mode remain unchanged')
    return dict(cases=32, native_sessions=32, frames=2132, exact_binary32_words=17455,
                serialized_state_bytes=1137600, original_partial_backward_prefixes=4,
                original_uniform_refusals=1, oracle_provenance_checks=11,
                owner_snapshots=4264, full_private_allocation_transition_ledger_claimed=False)
