"""Read original Generator131 reports and their complete actual raw evidence."""
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys
import types

from current_runtime_fixtures import require, safe_name, sha
from current_runtime_generator_fixtures import verify_qualifier, QUALIFIER_MANIFEST

SOURCE = 'abbc84b8eca16670e72b8872d1d7ab479f8a1d44f7b2b65deea1649f7682b2c7'
IMAGE = '606f69aaecedbedc7ed6ad8f033275e0b64697f70632036d180106d315243c32'
OWNER = 'c39600e7c37dea390f88cf9e74d86dfa924301456e0f32e127d82c98ad6778b0'
GROUPS = ('oracle24', 'native11', 'entry59', 'dispatch')


def loaded(path, name, inputs):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(inputs.bind(path), str(path), 'exec'), module.__dict__)
    return module


@contextmanager
def readers(package, inputs):
    names = ('admission', 'observe', 'processes', 'controls')
    previous = {name: sys.modules.get(name) for name in names}
    modules = {}
    try:
        for name in names:
            module = loaded(package / (name + '.py'), name, inputs)
            modules[name] = sys.modules[name] = module
        modules['run'] = loaded(package / 'run.py', 'current_generator131_original_reader', inputs)
        yield modules
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def payload(root, row, inputs, compressed=False):
    relative = safe_name(row['path'])
    value = inputs.bind(root.joinpath(*relative.parts))
    raw = gzip.decompress(value) if compressed else value
    require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'complete original raw evidence')
    return raw


def json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


def original_report(path, row, inputs):
    raw = inputs.bind(path, row['sha256'])
    require(len(raw) == row['bytes'] and (path.stat().st_mode & 0o7777) == row['mode'],
            'entire original report bytes and mode')
    return json.loads(raw)


def tensor_owner(package, directory, proof, inputs):
    original = inputs.bind(package / 'frozen/owner_integral.py', OWNER)
    raw = original
    changes = [('HTSINT01', 'HTSINT02', 2), ('HTOSIN01', 'HTOSIN02', 2),
               ('functions == 699', 'functions == 742', 1),
               ('sum-integral-source01-UNBOUND-699', 'generator742-dispatch-UNBOUND', 1),
               ('sum-integral-source01-%d-preserved47-48', 'generator%d-dispatch-full-tensor-owner', 1),
               ('heptatorch.sum-integral-owner-source01.v1', 'heptatorch.generator742-dispatch-tensor-owner.v1', 1)]
    for before, after, count in changes:
        require(raw.count(before.encode()) == count and after.encode() not in raw, 'exact owner identity edits')
        raw = raw.replace(before.encode(), after.encode())
    expected = dict(original_sha256=OWNER, executed_sha256=sha(raw), whole_file_inverse_exact=True,
                    only_profile_identity_tag_and_function_count_changed=True,
                    all_original_84Q_predicates_retained=True)
    require(proof == expected and json.loads(inputs.bind(directory / 'tensor-owner-inverse.json')) == expected,
            'complete original Tensor owner identity proof')
    path = directory / 'executed-tensor-owner.py'
    require(inputs.bind(path, sha(raw)) == raw, 'actual executed owner source')
    module = loaded(path, 'current_generator131_actual_tensor_owner', inputs)
    module.bind_profile(SOURCE, IMAGE, 742)
    return module


def owner_snapshot(root, row, observe, owner, inputs):
    require(row['passed'] is True and row['error'] is None, 'complete actual owner observation')
    control = payload(root, row['control_raw'], inputs)
    require(len(control) == 264 and struct.unpack_from('<2Q', control, 32) == (row['pointer'], row['length'])
            and struct.unpack_from('<Q', control, 192)[0] == 1
            and control[128:192] == bytes(64) and control[256:264] == bytes(8),
            'actual startup owner and cleared trusted entropy/state slots')
    raw = payload(root, row['raw'], inputs, compressed=True)
    require(len(raw) == row['length'] and bool(row['pointer']) == bool(raw), 'entire actual owner extent')
    if not raw or raw[:8] == b'HTCGST01':
        decoded = observe.decode_generator(raw, row['pointer'])
        verification = dict(native_compact_owner_header_and_unique_records=True,
                            every_live_state_5056_bytes_captured=True)
    else:
        decoded = owner.decode_owner(raw, row['pointer'])
        verification = owner.verify_snapshot(decoded)
        require(verification['all_original_84Q_predicates_retained'] is True,
                'full original Tensor owner predicates rechecked')
        decoded = dict(decoded, family='tensor')
    saved = payload(root, row['decoded'], inputs, compressed=True)
    require(json.loads(saved) == json_value(decoded) and row['family'] == decoded['family']
            and row['verification'] == json_value(verification), 'decoded owner must equal original raw bytes')
    return raw, decoded


def children(root, rows, observe, owner, expected_images, inputs):
    observe.require_children(rows)
    frames = 0
    for child, image in zip(rows, expected_images):
        require(child['proc_exe_sha256'] == child['expected_image_sha256'] == image,
                'same qualified actual persistent native image')
        child_directory = root.joinpath(*safe_name(child['directory']).parts)
        require(json.loads(inputs.bind(child_directory / 'transport.json')) == child,
                'complete original child transport report')
        request_digest, response_digest = hashlib.sha256(), hashlib.sha256()
        for index, frame in enumerate(child['frames']):
            require(frame['index'] == index and frame['original_call_returned'] is True
                    and frame['early_stdout_EOF'] is False and frame['error'] is None,
                    'every original frame completed in order')
            directory = root.joinpath(*safe_name(frame['directory']).parts)
            require(directory.parent == child_directory
                    and json.loads(inputs.bind(directory / 'transport.json')) == frame,
                    'exact original frame report and child ownership')
            request = inputs.bind(directory / 'request.bin', frame['request_sha256'])
            response = inputs.bind(directory / 'response.bin', frame['response_sha256'])
            require(len(request) == frame['request_bytes'] and len(response) == frame['response_bytes']
                    and inputs.bind(directory / 'attempted-request.bin') == request,
                    'original attempted and actual wire bytes')
            require(request[:8] == b'HTOWNR01' and response[:8] == b'HTOWNS01'
                    and len(request) >= 16 and len(response) >= 16
                    and struct.unpack_from('<Q', request, 8)[0] == len(request) - 16
                    and struct.unpack_from('<Q', response, 8)[0] == len(response) - 16,
                    'complete actual request and response envelopes')
            before, old = owner_snapshot(root, frame['owner_before'], observe, owner, inputs)
            after, new = owner_snapshot(root, frame['owner_after'], observe, owner, inputs)
            body, value = request[16:], response[16:]
            if value[:8] == b'HTCGRS01':
                require(len(value) >= 64 and new['family'] in ('empty', 'generator'), 'actual CG response family')
                code, size, live, allocation, last, epoch, reserved = struct.unpack_from('<7Q', value, 8)
                require(reserved == 0 and size == len(value) - 64
                        and (live, allocation, last, epoch) == (new['live'], new['allocation'], new['last'], new['epoch'])
                        and frame['generator_reply_bound_to_full_physical_owner'] is True,
                        'full CG response is bound to actual physical owner')
                if code:
                    require(size == 0 and before == after
                            and frame['owner_before']['pointer'] == frame['owner_after']['pointer'],
                            'actual CG refusal preserves entire owner and pointer')
                elif len(body) >= 64 and body[:8] == b'HTCGEN01':
                    op, handle, count, start = struct.unpack_from('<4Q', body, 8)
                    if op in (4, 5):
                        offset = new['records'][str(handle)]['offset'] + 32
                        require(value[64:] == after[offset:offset + (8 if op == 4 else 5056)],
                                'actual complete state or seed response bytes')
                    if op == 18:
                        require(value[64:] == after[start:start + count], 'actual complete physical dump bytes')
            else:
                require(value[:8] == b'HTORSP03' and len(value) >= 32, 'actual Tensor response family')
                version, count, code = struct.unpack_from('<3Q', value, 8)
                require(version == 3, 'actual Tensor response version')
                if code:
                    require(count == 0 and len(value) == 32 and before == after
                            and frame['owner_before']['pointer'] == frame['owner_after']['pointer'],
                            'actual Tensor refusal preserves entire owner and pointer')
            require(frame['actual_status'] == code, 'original status matches actual raw response')
            request_digest.update(request); response_digest.update(response)
            frames += 1
        require(child['frames_completed'] == len(child['frames'])
                and child['request_stream_sha256'] == request_digest.hexdigest()
                and child['response_stream_sha256'] == response_digest.hexdigest(),
                'full original child byte-stream digests')
        end = child['communications'][0]
        for name in ('stdout', 'stderr'):
            raw = inputs.bind(child_directory / ('close-00.' + name), end[name + '_sha256'])
            require(raw == b'' and end[name + '_bytes'] == 0, 'actual empty trailing stream and complete EOF')
    require(len(rows) == len(expected_images), 'complete expected persistent native census')
    return frames


def process_row(row, inputs, expected_kind=None):
    require(row['stdout_EOF'] is True and row['stderr_EOF'] is True and not row.get('timed_out')
            and row['error'] is None and not row.get('outer_cleanup_required')
            and not row.get('cleanup_error'), 'complete actual tool/reference process')
    if expected_kind is not None:
        require(row['kind'] == expected_kind, 'actual tool/probe kind')
    directory = Path(row['directory'])
    require(directory.is_absolute() and json.loads(inputs.bind(directory / 'process.json')) == row,
            'original complete process receipt')
    values = {}
    for name in ('input', 'stdout', 'stderr'):
        if name + '_sha256' not in row:
            require(name == 'input', 'actual process output identity required')
            continue
        raw = inputs.bind(directory / (name + '.bin'), row[name + '_sha256'])
        require(len(raw) == row[name + '_bytes'], 'entire actual process stream bytes')
        values[name] = raw
    require(values['stderr'] == b'', 'native/reference stderr must remain empty')
    return values


def audit_generator_reports(report, directory, package, inputs):
    """Pure raw audit, also usable on preserved source03 actual reports.

    This function alone does not qualify the production coordinator. The public
    qualifier below additionally requires the current portable reader and the
    complete actual cold/tool and source-custody admission.
    """
    directory, package = Path(directory), Path(package)
    require(report['schema'] == 'heptatorch.generator742-current-native-qualification.v1'
            and report['status'] == 'PASS_CURRENT742_GENERATOR131' and report['selected_group'] == 'all'
            and report['cases'] == 131 and report['original_cases'] == 94 and report['new_dispatch_cases'] == 37,
            'all original94 and new37 actually qualified together')
    require(report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
            and report['native_executed'] is True and report['Torch_executed'] is True
            and report['all_inputs_unchanged'] is True and report['error'] is None
            and report['historical_runtime_PASS_transferred'] is False
            and report['source_or_expected_numerical_values_modified'] is False
            and report['production_rng_or_owner_in_python'] is False
            and report['full_private_tensor_allocation_transition_ledger_claimed'] is False
            and report['generator_tensor_handle_interoperability_claimed'] is False
            and report['full_pytorch_rewrite_complete'] is False,
            'actual bounded qualification with original expectations and no broader transfer')
    require(set(report['groups']) == set(GROUPS), 'all four original Generator groups')
    with readers(package, inputs) as modules:
        observe, original = modules['observe'], modules['run']
        owner = tensor_owner(package, directory, report['tensor_observer_source_inverse'], inputs)
        groups = report['groups']
        for name in GROUPS:
            require(groups[name]['status'] == 'PASS'
                    and json.loads(inputs.bind(directory / name / 'report.json')) == groups[name],
                    'unaltered complete original subgroup report: ' + name)
        oracle, native, entry, dispatch = (groups[name] for name in GROUPS)
        require(oracle['cases'] == 24 and oracle['original_entry_exit'] == 0
                and oracle['native_executed'] is True and oracle['Torch_executed'] is True
                and oracle['original_oracle_whole_source_unchanged'] is True,
                'actual original live24')
        provenance = oracle['oracle_provenance']
        require(provenance['status'] == 'PASS' and set(provenance['checks']) == modules['admission'].CHECKS
                and all(v is True for v in provenance['checks'].values()), 'all eleven real Torch provenance checks')
        source = original_report(directory / 'oracle24/original-report.json', oracle['original_report'], inputs)
        require(source['schema'] == 'heptatorch.live-pytorch-owned-cpu-generator.v1'
                and source['status'] == 'PASS' and source['cases'] == source['expected_cases'] == 24
                and source['failed_cases'] == 0 and source['native_executable_sha256'] == IMAGE
                and source['oracle_provenance'] == provenance
                and [r['case'] for r in source['rows']] == [r['case'] for r in oracle['native_children']] == original.PLAN
                and all(r['status'] == 'PASS' for r in source['rows']), 'complete original24 ordered observations')
        require(set(source['source_files_unchanged_during_run']) == {'native', 'oracle', 'reference_support', 'lock'}
                and all(v is True for v in source['source_files_unchanged_during_run'].values()),
                'complete original24 actual source custody')
        for row, child in zip(source['rows'], oracle['native_children']):
            require(row['frames'] == child['frames_completed']
                    and row['request_stream_sha256'] == child['request_stream_sha256']
                    and row['response_stream_sha256'] == child['response_stream_sha256'], 'original24 exact wire custody')
        for key in ('frames', 'exact_values', 'serialized_state_bytes_compared'):
            require(source[key] == oracle[key] == sum(row[key] for row in source['rows']) > 0,
                    'complete original24 aggregate')
        require(len(source['rows'][19]['nondeterministic_seed_observations']) == 4,
                'four independent system-seed observations')
        frame_count = children(directory / 'oracle24/native', oracle['native_children'], observe, owner, [IMAGE] * 24, inputs)
        require(frame_count == oracle['frames'], 'original24 actual full frame count')
        require(native['cases'] == 11 and native['native_executed'] is True and native['Torch_executed'] is False
                and native['frames'] == 1715 and native['exact_output_values'] == 121611
                and native['full_rng_state_checks'] == 471 and native['full_pool_failure_rollbacks'] == 72,
                'all original native11 comparisons and resource controls')
        old11 = original_report(directory / 'native11/original/report.json', native['original_report'], inputs)
        require(old11['schema'] == 'heptatorch.cpu-generator-native-qualification.v1'
                and old11['status'] == 'PASS' and len(old11['cases']) == len({r['name'] for r in old11['cases']}) == 11
                and all(r['status'] == 'PASS' for r in old11['cases']) and old11['image_sha256'] == IMAGE
                and old11['build_receipt_sha256'] == report['build_sha256'], 'original native11 report intact')
        for key in ('frames', 'exact_output_values', 'full_rng_state_checks', 'full_pool_failure_rollbacks'):
            require(old11[key] == native[key], 'original native11 count binding')
        nframes = children(directory / 'native11/native', native['native_children'], observe, owner, [IMAGE] * 5, inputs)
        require(nframes == 1715, 'all1715 original actual native11 frames')
        frame_count += nframes
        require(native['independent_reference_processes'], 'actual independent host reference execution')
        for row in native['independent_reference_processes']:
            process_row(row, inputs)
            require(row['exit_code'] == 0 and row['production_runtime'] is False,
                    'successful independent reference process')
            executable = Path(row['argv'][0]).resolve()
            raw = inputs.bind(executable, row['executable']['sha256'])
            require(len(raw) == row['executable']['bytes']
                    and (executable.stat().st_mode & 0o7777) == row['executable']['mode'],
                    'actual independent reference executable bytes/mode')
        require(entry['cases'] == 59 and entry['native_executed'] is True and entry['Torch_executed'] is False,
                'actual complete entry59')
        old59 = original_report(directory / 'entry59/original/report.json', entry['original_report'], inputs)
        require(old59['schema'] == 'heptatorch.cpu-generator-native-entry-qualification.v1'
                and old59['status'] == 'PASS' and old59['case_count'] == len(old59['cases']) == 59
                and len({row['case'] for row in old59['cases']}) == 59
                and all(row['status'] == 'PASS' for row in old59['cases'])
                and old59['production_image_sha256'] == IMAGE
                and old59['probe_image_sha256'] == entry['actual_current14_probe_sha256']
                and old59['caller_created_owners_in_qualification_only'] is True
                and old59['production_accepts_native_pointers_from_packets'] is False,
                'all59 original ABI assertions retained')
        eframes = children(directory / 'entry59/native', entry['native_children'], observe, owner,
                           [IMAGE, entry['actual_current14_reordered_image_sha256']], inputs)
        require(eframes == 5, 'original current and reordered image actual frames')
        frame_count += eframes
        tool_rows = entry['tool_and_probe_processes']
        require(len(tool_rows) == 71 and [r['index'] for r in tool_rows] == list(range(71))
                and sum(r['kind'] == 'qualification-owned-step-probe' for r in tool_rows) == 47,
                'all71 original tool/probe processes')
        require(dispatch['schema'] == 'heptatorch.generator712-dispatch-controls.v1'
                and dispatch['case_count'] == len(dispatch['cases']) == 37
                and dispatch['production_cases'] == 6 and dispatch['qualification_abi_cases'] == 31
                and all(row['status'] == 'PASS' for row in dispatch['cases'])
                and len({r['case'] for r in dispatch['cases']}) == 37
                and [r['case'] for r in dispatch['cases'][:6]] == modules['controls'].PRODUCTION_PLAN
                and [r['case'] for r in dispatch['native_children']] == modules['controls'].PRODUCTION_PLAN
                and dispatch['actual_image_sha256'] == IMAGE
                and dispatch['actual_probe_sha256'] == entry['actual_current14_probe_sha256']
                and dispatch['production_accepts_foreign_owners_or_entropy_pointers'] is False
                and dispatch['os_entropy_consumption_before_application_validation_prohibited'] is False
                and dispatch['raw_entropy_cleared_between_every_observed_frame'] is True,
                'all37 new controls and their explicit production/probe boundary')
        dframes = children(directory / 'dispatch/native', dispatch['native_children'], observe, owner, [IMAGE] * 6, inputs)
        require(dframes == 18 and all(c['frames_completed'] == 3 for c in dispatch['native_children']),
                'all six actual three-frame production controls')
        frame_count += dframes
        probes = dispatch['probe_processes']
        require(len(probes) == 31 and [r['index'] for r in probes] == list(range(71, 102)),
                'all31 new actual ABI probe processes')
        values = []
        for row in tool_rows + probes:
            streams = process_row(row, inputs)
            require(row['started'] is True and row['proc_exe_sha256'] == row['binary_sha256']
                    and row['exit_code'] in (0, 2)
                    and (row['exit_code'] == 0 or streams['stdout'] == b''),
                    'actual tool success or original empty refusal with complete streams')
            inputs.bind(Path(row['argv'][0]).resolve(), row['binary_sha256'])
            values.append(streams)
        require(sum(r['binary_sha256'] == modules['admission'].COMPILER for r in tool_rows) == 11,
                'all eleven actual entry compilation requests')
        for row, process, streams in zip(dispatch['cases'][6:], probes, values[71:]):
            require(process['kind'] == 'qualification-owned-step-probe' and process['exit_code'] == 0
                    and process['binary_sha256'] == dispatch['actual_probe_sha256']
                    and row['qualification_owned_step_probe'] is True
                    and row['caller_created_owner_only_in_qualification'] is True
                    and row['entire_actual_owner_retained'] is True
                    and row['request_sha256'] == sha(streams['input'])
                    and row['response_sha256'] == sha(streams['stdout']), 'actual31 ABI request/response custody')
            request, response = streams['input'], streams['stdout']
            size, body, capacity, entropy = struct.unpack_from('<4Q', request)
            require(len(request) == 32 + size + body + entropy and len(response) == 24 + capacity + size,
                    'complete actual ABI test buffers')
            length, returned_size, returned_capacity = struct.unpack_from('<3Q', response)
            before, after = request[32:32 + size], response[24 + capacity:]
            require(returned_size == size and returned_capacity == capacity == row['actual_capacity']
                    and length == row['actual_returned_length'] and before == after
                    and sha(before) == row['original_owner_sha256'] == row['returned_owner_sha256'],
                    'byte-for-byte full test-owned ABI owner preservation')
        return dict(cases=131, original_cases=94, new_dispatch_cases=37,
                    persistent_children=37, persistent_frames=frame_count,
                    entry_tool_and_probe_processes=71, dispatcher_probe_processes=31,
                    actual_source03_or_later_raw_reports_audited=True)


def verify_current_object_links(report, build, stack_build, directory, package, inputs):
    """Bind the actual probe/reordered links to every freshly built current object."""
    with readers(package, inputs) as modules:
        admission, process = modules['admission'], modules['processes']
        manifest = json.loads(inputs.bind(build / 'manifest.json', admission.MANIFEST))
        objects = [inputs.bind(build / (row['name'] + '.o')) for row in manifest['units']]
        require(len(objects) == 14, 'all fourteen actual cold objects')
        entry = report['groups']['entry59']
        rows = entry['tool_and_probe_processes']
        probe_source = inputs.bind(package / 'probe.hia')
        compiles = [row for row in rows if row['binary_sha256'] == admission.COMPILER
                    and row['input_sha256'] == sha(probe_source) and row['exit_code'] == 0]
        require(len(compiles) == 1, 'one actual original probe compilation')
        probe_object = inputs.bind(Path(compiles[0]['directory']) / 'stdout.bin', compiles[0]['stdout_sha256'])
        expected_probe = process.link_request(b'HIASTKR1', objects + [probe_object], b'probe')
        expected_reordered = process.link_request(b'HIACGS01', objects[::-1], b'step')
        probe_links = [row for row in rows if row['binary_sha256'] == admission.RUNTIME_LINKER
                       and row['input_sha256'] == sha(expected_probe)]
        reordered_links = [row for row in rows if row['binary_sha256'] == admission.LINKER
                           and row['input_sha256'] == sha(expected_reordered)]
        require(len(probe_links) == len(reordered_links) == 1, 'one actual complete probe and reordered link')
        for row, request, pin in ((probe_links[0], expected_probe, entry['actual_current14_probe_sha256']),
                                  (reordered_links[0], expected_reordered, entry['actual_current14_reordered_image_sha256'])):
            require(row['exit_code'] == 0
                    and inputs.bind(Path(row['directory']) / 'input.bin') == request
                    and row['stdout_sha256'] == pin, 'actual link request and resulting image identity')
            inputs.bind(Path(row['directory']) / 'stdout.bin', pin)
        tools = {str((stack_build / 'native' / name).resolve()): pin for name, pin in (
            ('stack-module-compiler', admission.COMPILER), ('stack-linker-runtime', admission.RUNTIME_LINKER),
            ('stack-linker-session', admission.OLD_LINKER))}
        tools[str((build / 'toolchain/stack-linker-session').resolve())] = admission.LINKER
        probe = directory / 'entry59/original/step-probe'
        reordered = directory / 'entry59/original/reordered-service'
        inputs.bind(probe, entry['actual_current14_probe_sha256'])
        inputs.bind(reordered, entry['actual_current14_reordered_image_sha256'])
        tools[str(probe.resolve())] = entry['actual_current14_probe_sha256']
        for row in rows + report['groups']['dispatch']['probe_processes']:
            require(type(row['argv']) is list and len(row['argv']) == 1
                    and tools.get(str(Path(row['argv'][0]).resolve())) == row['binary_sha256'],
                    'only fixed tools or actual current-object probe executed')
            require(row['current14_object_probe_link'] is (row is probe_links[0])
                    and row['current14_object_reordered_link'] is (row is reordered_links[0]),
                    'exact actual current-object link classification')


def qualify_generator131(report, repository, build, build_sha, directory, package,
                         candidate, original26, tool_build, stack_build, inputs):
    package, directory, build = Path(package), Path(directory), Path(build)
    qualifier_proof = verify_qualifier(package, inputs)
    require(report['build_sha256'] == build_sha, 'same actual current cold build')
    tool_raw = inputs.bind(Path(tool_build) / 'build.json')
    require(report['tool_build_sha256'] == sha(tool_raw), 'same actual reproduced entropy linker build')
    # This production entry requires the actual portable reader. A separate raw
    # audit above may inspect old source03; it never grants this current gate.
    for leaf in ('run.py', 'admission.py', 'package-manifest.json'):
        path = package / leaf
        actual = inputs.bind(path)
        bound = report['inputs'].get(str(path.absolute()))
        require(bound is not None and bound['bytes'] == len(actual) and bound['sha256'] == sha(actual)
                and bound['mode'] == (path.stat().st_mode & 0o7777), 'actual current portable reader source')
    require(sha(inputs.bind(package / 'package-manifest.json')) == qualifier_proof['manifest_sha256'],
            'exact current portable qualification package')
    with readers(package, inputs) as modules:
        args = types.SimpleNamespace(repository=Path(repository), candidate=Path(candidate),
            build_directory=build, tool_build=Path(tool_build), stack_build=Path(stack_build),
            original_package_root=Path(original26), expected_native_sha256=IMAGE,
            expected_build_sha256=build_sha, expected_tool_build_sha256=sha(tool_raw))
        actual = modules['admission'].admit(args, inputs)
        require(actual == report['cold_build_admission'], 'entire actual cold/tool/source admission repeated')
        require(modules['run'].package(inputs) == report['source_adaptations'], 'whole original94 source inverses')
    require(set(report['inputs']) == set(report['input_checks_after_execution'])
            and report['inputs'] and all(v is True for v in report['input_checks_after_execution'].values()),
            'all original source and tool inputs unchanged after execution')
    for name, row in report['inputs'].items():
        raw = inputs.bind(Path(name), row['sha256'])
        require(len(raw) == row['bytes'] and (Path(name).stat().st_mode & 0o7777) == row['mode'],
                'current actual reader complete input custody')
    verify_current_object_links(report, build, Path(stack_build), directory, package, inputs)
    return audit_generator_reports(report, directory, package, inputs)
