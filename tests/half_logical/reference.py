"""Read the complete immutable Torch264 execution; never calculate its outputs."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile

PACKAGE = 'validation/half-logical264-reference'
PREFIX = 'heptatorch-half-reference'
ARCHIVE = '54d7c68a6879328eb3e66f7c82c265bc39df831c812228f75c045b5749c349fb'
MANIFEST = 'a0336c08a1fc729b207c09c196ea263f742eeccba900f0abd6301881e5667d40'
CUSTODY = '6bb8d0fc36d8ba349f71dac12f20bcffc64c327a362780ce42a89b7e4db29d30'
REPORT = '891a2081ffe8df31eb016f51d9a57ffd1d7c588fb61173e2922f080aa33bfc33'
PRODUCER = '7383598cf7f7452dcf08b5e4bfa24f5fd5b128960c9a1254bf538fe1d15f26e8'
COMMIT = 'ac2713b2333a71e4dd9796b38e3f648d76326714'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(raw, mode):
    return dict(bytes=len(raw), mode=mode, sha256=sha(raw))


def package(repository, bind):
    root = repository / PACKAGE
    archive = bind(root / (PREFIX + '.tar.xz'), ARCHIVE)
    manifest = json.loads(bind(root / (PREFIX + '.manifest.json'), MANIFEST))
    custody = json.loads(bind(root / (PREFIX + '.custody.json'), CUSTODY))
    publication = json.loads(bind(root / 'publication.json'))
    require(len(archive) == 6448068 and manifest['member_count'] == len(manifest['files']) == 375,
            'complete immutable Half264 archive census')
    require(manifest['source_commit'] == custody['source_commit'] == publication['source_commit'] == COMMIT
            and manifest['producer_manifest_sha256'] == custody['producer_manifest_sha256'] == PRODUCER,
            'fixed original reference source and producer')
    require(custody['status'] == 'SEALED_OBSERVED_HALF_LOGICAL264_REFERENCE'
            and custody['missing_or_changed'] == [] and custody['original_reference_observed'] is True
            and custody['reference_or_runtime_results_rewritten'] is False
            and custody['runtime_qualification_transferred'] is False,
            'successful complete original reference custody')
    require(manifest['archive']['sha256'] == ARCHIVE and manifest['archive']['bytes'] == len(archive)
            and custody['archive']['sha256'] == ARCHIVE and custody['manifest']['sha256'] == MANIFEST,
            'original archive and manifest bindings')
    require(publication['workflow_run'] == 44353303549 and publication['artifact_id'] == 14450180717
            and publication['artifact_zip_sha256'] == '66378146181477cd82a472394ead76de831b105904004a5b74831743c14718c7',
            'actual GitHub publication identity')
    for name, expected in publication['original_files'].items():
        require(PurePosixPath(name).name == name, 'canonical original artifact filename')
        raw = bind(root / name, expected['sha256'])
        require(len(raw) == expected['bytes'], 'complete original artifact file')
    return archive, manifest


def restore(repository, output, bind):
    archive, manifest = package(repository, bind)
    require(not output.exists() and not output.is_symlink(), 'new reference destination')
    output.mkdir(parents=True)
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:xz') as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and name.parts and '..' not in name.parts
                    and '\\' not in member.name and str(name) == member.name
                    and member.isfile() and member.name not in seen and member.name in manifest['files'],
                    'ordinary canonical unique Half264 reference member')
            raw = stream.extractfile(member).read()
            require(identity(raw, member.mode) == {k:manifest['files'][member.name][k] for k in ('bytes', 'mode', 'sha256')},
                    'reference member bytes/mode match original seal')
            path = output.joinpath(*name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as target:
                target.write(raw)
            path.chmod(member.mode)
            seen.add(member.name)
    require(seen == set(manifest['files']), 'every original reference member retained')
    return read(repository, output, bind)


def read(repository, root, bind):
    _, manifest = package(repository, bind)
    payloads = {}
    for name, expected in manifest['files'].items():
        path = root / name
        raw = bind(path, expected['sha256'])
        require(identity(raw, stat.S_IMODE(path.stat().st_mode)) == {k:expected[k] for k in ('bytes', 'mode', 'sha256')},
                'restored original reference bytes and modes')
        payloads[name] = raw
    require(sum(len(raw) for raw in payloads.values()) == manifest['payload_bytes'],
            'complete original payload extent')
    mapping = manifest['original_input_path_mapping']
    for row in manifest['declared_input_bindings']:
        name = mapping[row['source_path']]
        require({k:manifest['files'][name][k] for k in ('bytes', 'mode', 'sha256')} == row['expected'],
                'every original physical input mapped exactly')
    report_name = 'evidence/production/reference/report.json'
    report = json.loads(payloads[report_name])
    production = json.loads(payloads['evidence/production/production.json'])
    ci = json.loads(payloads['evidence/ci.json'])
    require(sha(payloads[report_name]) == REPORT and production['report']['sha256'] == REPORT
            and report['status'] == 'OBSERVED_HALF_LOGICAL_REFERENCE'
            and production['status'] == 'OBSERVED_COMPLETE_HALF_LOGICAL_REFERENCE_EXECUTION'
            and ci['status'] == 'OBSERVED_HALF_LOGICAL264_REFERENCE_ONLY'
            and ci['source_commit'] == COMMIT, 'actual complete Torch reference execution')
    for record in (report, production, ci):
        require(record['cases'] == 264 and record['accepted'] == 222 and record['observed_errors'] == 42,
                'original 264/222/42 observed denominator')
        require(record['native_executed'] is False and record['compiler_executed'] is False,
                'reference did not execute native or compiler code')
        provenance = record['oracle_provenance']
        require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
                and all(v is True for v in provenance['checks'].values()), 'all eleven original oracle predicates')
    require(report['Torch_executed'] is True and report['host_expected_values_computed'] is False
            and report['inputs_unchanged'] is True and report['all_observed_blobs_unchanged'] is True
            and report['rng_unchanged'] is True and report['failures'] == []
            and report['forbidden_child_process_attempts'] == []
            and production['all_physical_inputs_unchanged'] is True,
            'independent actual Torch observations and unchanged inputs')
    process = production['process']
    intent = json.loads(payloads['evidence/production/execution/intent.json'])
    observed = report['actual_process']
    require(process['started'] is True and process['exit_code'] == 0 and process['stdout_EOF'] is True
            and process['stderr_EOF'] is True and not process['forced_cleanup']
            and not process['cleanup_errors'] and not process['running_descendants_after_cleanup'],
            'actual reference process completed with both EOFs')
    require(process['argv'] == intent['argv'] == observed['argv']
            and observed['pid'] == process['pid'] and observed['python'] == production['interpreter'] == intent['observer_python'],
            'original argv, PID and prebound interpreter agree')
    entity = payloads['evidence/production/execution/observer-python.bin']
    require(identity(entity, 0o755) == {k:production['interpreter'][k] for k in ('bytes', 'mode', 'sha256')},
            'actual original interpreter entity retained but never executed here')
    for name in ('stdout', 'stderr'):
        raw = payloads['evidence/production/execution/runner.' + name]
        require(len(raw) == process[name + '_bytes'] and sha(raw) == process[name + '_sha256']
                and production['streams'][name]['EOF'] is True,
                'both complete original producer streams retained, including its warning')
    cases = json.loads(payloads['repository/tests/half_logical_reference264/producer/cases.json'])['cases']
    require(len(cases) == len(report['rows']) == 264, 'complete literal case matrix')
    for index, (spec, row) in enumerate(zip(cases, report['rows'])):
        require(row['index'] == index and row['spec'] == spec
                and json.loads(payloads['evidence/production/reference/case-%03d.json' % index]) == row,
                'literal spec, aggregate and standalone observation agree')
        require(row['input'] == row['input_after'] and row['input_unchanged'] is True
                and row['rng_unchanged'] is True, 'original input and RNG unchanged')
        for value in [row['input'], row['input_after']] + ([row['result']] if row['accepted'] else []):
            for field in ('logical', 'storage'):
                blob = value[field]
                raw = payloads['evidence/production/reference/' + blob['path']]
                require(identity(raw, 0o644) == {k:blob[k] for k in ('bytes', 'mode', 'sha256')},
                        'complete actual observed tensor bytes')
        original = payloads['repository/tests/half_logical_reference264/producer/' + spec['input']['path']]
        require(len(original) == spec['input']['bytes'] and sha(original) == spec['input']['sha256']
                == row['input']['logical']['sha256'], 'literal input reached actual Torch unchanged')
    return dict(root=root, report=report, payloads=payloads, reference_sha256=REPORT,
                source_commit=COMMIT, original_execution_unchanged=True)


def blob(reference, value):
    return reference['payloads']['evidence/production/reference/' + value['path']]


def input_bytes(reference, spec):
    return reference['payloads']['repository/tests/half_logical_reference264/producer/' + spec['input']['path']]
