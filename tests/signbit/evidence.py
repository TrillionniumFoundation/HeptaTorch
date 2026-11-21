"""Read immutable actual signbit references and real dual-EOF process receipts."""

import json
from pathlib import Path
import stat
from numeric_predicates.evidence import require, sha, identity, SUPPORT, LOCK
from .cases import CASES, CASE_SHA256
import pinned_reference_custody as custody


def read(repository, directory, execution, bind, local_observed=False):
    directory, execution = Path(directory).resolve(), Path(execution).resolve()
    raw = bind(directory / 'report.json')
    report = json.loads(raw)
    custody_proof=custody.read(repository,directory,execution,report,bind,'tests/signbit/reference.py',local_observed)
    process = json.loads(bind(execution / 'process.json'))
    require(report['schema'] == 'heptatorch.signbit.reference.v1'
        and report['status'] == 'OBSERVED_SIGNBIT'
        and report['cases'] == report['expected_cases'] == len(CASES) == 97
        and report['case_sha256'] == CASE_SHA256 and len(report['rows']) == len(CASES)
        and report['failures'] == [] and report['Torch_executed'] is True
        and report['native_executed'] is False and report['compiler_executed'] is False
        and report['host_expected_values_computed'] is False
        and report['inputs_unchanged'] is True and report['rng_unchanged'] is True
        and report['all_observed_blobs_unchanged'] is True
        and report['local_observed_oracle'] is local_observed,
        'complete independent actual signbit reference')
    provenance = report['oracle_provenance']
    if local_observed:
        require(provenance['status'] == 'OBSERVED_INSTALLED_WHEEL_IDENTITY_ONLY'
            and len(provenance['checks']) == 5
            and provenance['full_installation_provenance_verified'] is False
            and provenance['original_pip_report_available'] is False
            and report['full_installation_provenance_verified'] is False,
            'local observation cannot confer complete CI provenance')
        for key in ('package', 'extension'):
            saved = provenance[key]
            path = Path(saved['path'])
            require(saved == dict(path=str(path), **identity(bind(path, saved['sha256']),
                    stat.S_IMODE(path.stat().st_mode))), 'original installed Torch file unchanged')
    else:
        require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
            and report['full_installation_provenance_verified'] is True,
            'complete CI requires all eleven original provenance checks')
    require(all(value is True for value in provenance['checks'].values())
            and provenance['lock_sha256'] == LOCK, 'fixed installed-wheel provenance checks')
    bind(repository / 'tests/pytorch_oracle.py', SUPPORT)
    lock = json.loads(bind(repository / 'compatibility/pytorch-oracle-lock.json', LOCK))
    require(provenance['expected'] == lock['oracle'], 'exact original official wheel lock retained')
    require(process['started'] is True and process['exit_code'] == 0
        and process['stdout_EOF'] is True and process['stderr_EOF'] is True
        and process['forced_cleanup'] is False and not process.get('timed_out')
        and not process.get('error') and not process.get('execution_error')
        and not process['cleanup_errors'] and not process['running_descendants_after_cleanup']
        and not process['terminated_process_groups'], 'actual reference exit, both EOFs and normal completion')
    expected_args = ['-B', str(repository / 'tests/signbit/reference.py'), '--repository',
                     str(repository), '--output', str(directory)]
    if local_observed:
        expected_args.append('--local-observed-oracle')
    argv = process['argv']
    require(argv[1:] == expected_args and report['actual_process']['argv'] == argv
        and report['actual_process']['pid'] == process['pid'], 'exact supervised reference argv and PID')
    interpreter = Path(argv[0]).resolve(strict=True)
    python = report['actual_process']['python']
    require(python == dict(path=str(interpreter), **identity(bind(interpreter),
        stat.S_IMODE(interpreter.stat().st_mode))), 'actual exact reference interpreter')
    for channel in ('stdout', 'stderr'):
        stream = bind(execution / ('runner.' + channel), process[channel + '_sha256'])
        require(len(stream) == process[channel + '_bytes'], 'complete actual reference stream')
    require(bind(execution / 'runner.stderr') == b'', 'clean actual Torch reference stderr')
    require(json.loads(bind(execution / 'runner.stdout')) == dict(
        status='OBSERVED_SIGNBIT', cases=97, expected_cases=97, error=None), 'actual reference stdout')
    files = [repository / 'tests/signbit' / name for name in ('reference.py', '__init__.py', 'cases.py')]
    files += [repository / 'tests/numeric_predicates' / name for name in ('__init__.py', 'cases.py', 'reference.py')]
    files += [repository / 'tests/pytorch_oracle.py', repository / 'compatibility/pytorch-oracle-lock.json',
              repository / 'compatibility/upstream-lock.json', repository / 'tests/pinned_reference_custody.py', interpreter]
    files += custody.source_files(repository)
    if not local_observed:
        files.append(repository / 'build/pytorch-install.json')
    require(set(report['inputs']) == {str(path.resolve()) for path in files}, 'complete actual reference input census')
    for path, saved in report['inputs'].items():
        file = Path(path)
        require(saved == dict(path=path, **identity(bind(file, saved['sha256']),
                stat.S_IMODE(file.stat().st_mode))), 'original reference input bytes and modes unchanged')

    def blob(item):
        require(set(item) == {'path', 'bytes', 'sha256'}
            and item['path'] == 'blobs/' + item['sha256'] + '.bin'
            and len(item['sha256']) == 64, 'canonical observed blob binding')
        data = bind(directory / item['path'], item['sha256'])
        require(len(data) == item['bytes'], 'complete independently observed blob extent')
        return data

    for index, (row, spec) in enumerate(zip(report['rows'], CASES)):
        require(row == json.loads(bind(directory / ('case-%03d.json' % index)))
            and row['index'] == index and row['spec'] == spec and row['accepted'] is True
            and all(row[field] is True for field in ('fresh_storage', 'fresh_tensor', 'input_unchanged', 'rng_unchanged')),
            'unchanged actual reference row and original literal recipe')
        for value in (row['input'], row['result']):
            blob(value['logical'])
            blob(value['storage'])
    return dict(root=directory, execution=execution, report=report, report_sha256=sha(raw), blob=blob, custody=custody_proof)
