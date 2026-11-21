#!/usr/bin/env python3
"""Pure physical readback of every original F64 request, reference and process."""
from pathlib import Path
import argparse
import importlib.util
import json
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import dump, identity, require, sha
from support import admit, accuracy, comparable_rows, comparisons, diagnostic, read_process


def qualify(repo, current, consumer, directory):
    built = admit(repo, current, consumer)
    out = directory/'execution'
    result = json.loads((out/'report.json').read_bytes())
    require(result['schema'] == 'heptatorch.current-object-f64-run.v1'
            and result['cases'] == 55, 'complete original55 result')
    process = json.loads((directory/'reference-process.json').read_bytes())
    intent = json.loads((directory/'reference-process.intent.json').read_bytes())
    require(all(process.get(k) == v for k, v in intent.items() if k != 'started'), 'original reference intent')
    require(process['started'] is True and intent['started'] is False
            and process['pid'] == result['reference_inputs']['reference_pid']
            and process['exit_code'] == (0 if result['status'] == 'PASS' else 1)
            and process['timed_out'] is False and process['stdout_EOF'] is True and process['stderr_EOF'] is True,
            'actual reference process, outcome and EOFs')
    argv = process['argv']
    wanted = ['--worker', '--repository', str(repo), '--current-build', str(current),
                            '--consumer-build', str(consumer), '--reference', result['reference'],
                            '--output', str(out)]
    if result['cpython_directory'] is not None:
        wanted += ['--cpython-directory', result['cpython_directory']]
    require(argv[1] == '-B' and Path(argv[2]).name == 'run.py'
            and argv[3:] == wanted, 'complete reference execution command')
    require(process['cwd'] == str(repo) and process['timeout_seconds'] == 900
            and process['environment'] == {'PATH': '/bin:/usr/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
                                          'TORCHINDUCTOR_CACHE_DIR': str(out/'reference-custody/torchinductor-cache')}
            and process['interpreter_realpath'] == str(Path(argv[0]).resolve(strict=True))
            and process['interpreter'] == identity(process['interpreter_realpath']) == result['reference_inputs']['python'],
            'actual reference interpreter and exact prebound cache environment')
    expected_scripts = {name: identity(Path(__file__).resolve().parent/name)
                        for name in ('build.py', 'support.py', 'run.py', 'qualify.py')}
    require(process['implementation'] == result['implementation_inputs'] == expected_scripts, 'complete implementation closure')
    for stream in ('stdout', 'stderr'):
        raw = (directory/('reference-process.'+stream)).read_bytes()
        require(len(raw) == process[stream+'_bytes'] and sha(raw) == process[stream+'_sha256'], 'complete actual reference stream')
        if stream == 'stderr':
            require(raw == b'', 'strict reference stderr must be empty')
    expected_stdout = json.dumps({k: result[k] for k in ('status', 'reference', 'cases', 'failed_cases', 'exact_non_nan', 'nan_classification')})+'\n'
    require((directory/'reference-process.stdout').read_text() == expected_stdout, 'exact numerical reference output')
    require(json.loads((out/'reference-inputs.json').read_bytes()) == result['reference_inputs'], 'actual reference input receipt')
    require(result['reference_inputs']['original_oracle'] == identity(repo/'tests/f64_oracle.py')
            and result['reference_inputs']['reference_support'] == identity(repo/'tests/pytorch_oracle.py'), 'original reference script bytes')
    if result['reference'] == 'pytorch':
        provenance = result['oracle_provenance']
        require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
                and all(v is True for v in provenance['checks'].values()), 'all original strict PyTorch checks')
        helper = repo/'tests/pinned_reference_custody.py'
        require(result['reference_inputs']['custody_helper'] == identity(helper), 'original complete fixed-wheel custody helper')
        spec = importlib.util.spec_from_file_location('f64_reader_pinned_custody', helper)
        custody = importlib.util.module_from_spec(spec)
        exec(compile(helper.read_bytes(), str(helper), 'exec'), custody.__dict__)
        capsule = out/'reference-custody'
        receipt_raw = (capsule/'reference-inputs.json').read_bytes()
        receipt = json.loads(receipt_raw); actual = receipt['actual_process']
        require(sha(receipt_raw) == result['reference_custody'] and actual['pid'] == process['pid']
                and actual['ppid'] == process['supervisor_pid'] and actual['argv'] == argv
                and actual['cwd'] == process['cwd'] and actual['environment'] == process['environment']
                and actual['python'] == dict(path=process['interpreter_realpath'], **process['interpreter'])
                and actual['script'] == dict(path=argv[2], **identity(argv[2]))
                and actual['flags'] == dict(isolated=0, no_site=0, no_user_site=0, ignore_environment=0, dont_write_bytecode=1, optimize=0),
                'exact actual raw reference process')

        def bind(path, expected=None):
            require(not Path(path).is_symlink(), 'regular original reference evidence')
            raw = Path(path).read_bytes()
            require(expected is None or sha(raw) == expected, 'reference evidence byte identity')
            return raw

        def read(relative, expected=None):
            if relative.startswith('pytorch76/'):
                path = capsule/relative.removeprefix('pytorch76/')
            elif relative == 'inputs/runtime/python.bin':
                path = capsule/'reference-inputs/python.bin'
            elif relative.startswith('inputs/repository/'):
                role = relative.removeprefix('inputs/repository/')
                aliases = {'build/pytorch-install.json': 'pip-install.json',
                           'build/pytorch-reference-support-install.json': 'numpy-pip-install.json'}
                path = capsule/'reference-inputs'/aliases[role] if role in aliases else repo/role
            else:
                raise ValueError('unknown original reference evidence role')
            return bind(path, expected)

        custody.read_reference(read, dict(oracle_provenance=provenance, allow_unverified_local=False),
                               dict(local_observed_oracle=False, paths=dict(repository=str(repo))), actual)
        custody.read_wheel_entities(repo, capsule, bind, actual)
    else:
        require(result['reference'] == 'cpython' and result['oracle_provenance'] is None, 'honest reference scope')
    index, reference_index = 0, 0
    stage = 'strict55'
    actual_processes = []

    def invoke(raw):
        nonlocal index
        require(index < len(result['native_processes']), 'no missing native invocation')
        name = 'native-'+str(index).zfill(4)
        response = (out/(name+'.stdout.bin')).read_bytes()
        observed = read_process(out, name, consumer/'f64-native', raw, response, timeout=30)
        require(observed == result['native_processes'][index]
                and observed['supervisor_pid'] == process['pid'], 'original numerical process index')
        actual_processes.append(observed); index += 1
        return response

    def reference(mode, pairs):
        nonlocal reference_index
        require(reference_index < len(result['references']), 'no missing reference batch')
        row = result['references'][reference_index]
        name = 'reference-'+str(reference_index).zfill(4)
        raw = (out/(name+'.bin')).read_bytes()
        request = struct.pack('<II', mode, len(pairs))+b''.join(
            struct.pack('<QQ', x, y) if mode < 4 else struct.pack('<Q', x) for x, y in pairs)
        native_name = actual_processes[-1]['name'] if actual_processes[-1]['request_sha256'] == sha(request) else None
        require(native_name is not None or stage == 'sqrt_diagnostic', 'reference occurs after the same actual native input')
        require(row == {'name': name, 'mode': mode, 'values': len(pairs), 'stage': stage,
                        'native_name': native_name, 'request_sha256': sha(request),
                        'bytes': len(raw), 'sha256': sha(raw), 'actual_reference_pid': process['pid']}
                and len(raw) == len(pairs)*8, 'exact original reference input/batch order')
        reference_index += 1
        return list(struct.unpack('<'+'Q'*len(pairs), raw))

    rows, proof = comparisons(repo, invoke, reference)
    stage = 'sqrt_diagnostic'
    diagnostic_result = diagnostic(repo, consumer/'f64-native', invoke, reference)
    require(diagnostic_result == result['sqrt_diagnostic'] and diagnostic_result['status'] != 'FAIL', 'all original rational and shape diagnostic predicates')
    cp_rows = rows
    if result['reference'] == 'pytorch':
        cp_directory = Path(result['cpython_directory'])
        cp_raw = (cp_directory/'execution/report.json').read_bytes(); cp = json.loads(cp_raw)
        require(cp['reference'] == 'cpython' and cp['status'] == 'PASS'
                and cp['cpython_directory'] is None and sha(cp_raw) == result['cpython_report_sha256'], 'actual CPython predecessor')
        require(qualify(repo, current, consumer, cp_directory)['status'] == 'PASS', 'complete CPython physical readback')
        cp_rows = cp['rows']
    else:
        require(result['cpython_directory'] is None and result['cpython_report_sha256'] is None and result['reference_custody'] is None, 'CPython scope')
    stage = 'rne55'
    accuracy_result = accuracy(repo, rows, cp_rows, invoke, reference)
    require(accuracy_result == result['rne_accuracy'], 'all original55 RNE/accuracy predicates and18 controls')
    require(index == len(result['native_processes']) and reference_index == len(result['references']), 'complete process and reference denominator')
    require(len({r['pid'] for r in actual_processes}) == len(actual_processes), 'all original child identities')
    require(comparable_rows(rows) == comparable_rows(result['rows'])
            and proof == result['original_comparison'], 'every original comparison re-executed on physical bytes')
    require(all(('error' in r) == ('error' in original) for r, original in zip(rows, result['rows'])), 'original exception outcome retained')
    passed = all(r['status'] == 'PASS' for r in rows)
    require(result['status'] == ('PASS' if passed else 'FAIL')
            and result['failed_cases'] == sum(r['status'] != 'PASS' for r in rows)
            and result['exact_non_nan'] == sum(r.get('exact_non_nan', 0) for r in rows)
            and result['nan_classification'] == sum(r.get('nan_classification', 0) for r in rows), 'exact original numeric outcome')
    require(result['current_source_sha256'] == built['current_source_sha256']
            and result['native_sha256'] == built['native_sha256']
            and result['consumer_build_sha256'] == sha((consumer/'build.json').read_bytes())
            and result['strict_failures_reclassified'] is False and result['legacy143_restored'] is False
            and result['sqrt_public_tensor_support'] is False and result['whole_pytorch_rewrite_complete'] is False,
            'source identity and bounded original scope')
    # No current or historical executable, compiler or reference library is called here.
    return {'schema': 'heptatorch.current-object-f64-physical-readback.v1',
            'status': 'PASS' if passed else 'FAIL', 'reference': result['reference'],
            'cases': 55, 'native_processes': index, 'reference_batches': reference_index,
            'sqrt_diagnostic_cases': len(diagnostic_result['rows']), 'rne_repeat_cases': accuracy_result['cases'],
            'integer_boundary_controls': accuracy_result['integer_checker_boundary_controls'],
            'component_qualification_status': 'PARTIAL',
            'exact_non_nan': result['exact_non_nan'], 'nan_classification': result['nan_classification'],
            'failed_cases': result['failed_cases'], 'strict_failures_reclassified': False,
            'current_objects_verified': len(built['current_units']), 'current_source_sha256': built['current_source_sha256'],
            'native_sha256': built['native_sha256'], 'actual_reference_pid': process['pid'],
            'native_or_reference_execution_by_reader': False, 'whole_pytorch_rewrite_complete': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--current-build', type=Path, required=True)
    parser.add_argument('--consumer-build', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(__debug__, 'original assertions required')
    result = qualify(*(p.resolve() for p in (args.repository, args.current_build, args.consumer_build, args.directory)))
    dump(args.output, result); print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
