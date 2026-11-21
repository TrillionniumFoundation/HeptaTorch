#!/usr/bin/env python3
"""Observe signbit with actual Torch; never compute expected signs in the host."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import stat
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from signbit.cases import CASES, CASE_SHA256
from numeric_predicates.reference import setup, tensor

SUPPORT = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'


def identity(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), mode=stat.S_IMODE(path.stat().st_mode),
                sha256=hashlib.sha256(raw).hexdigest())


def source_files(repo):
    return [HERE / name for name in ('reference.py', '__init__.py', 'cases.py')] + [
        repo / 'tests/numeric_predicates' / name for name in ('__init__.py', 'cases.py', 'reference.py')
    ] + [repo / 'tests/pytorch_oracle.py', repo / 'compatibility/pytorch-oracle-lock.json',
         repo / 'compatibility/upstream-lock.json', Path(sys.executable).resolve()]


def local_torch_identity(repo):
    """Validate visible installed-wheel identity without inventing a pip report."""
    import torch
    lock = json.loads((repo / 'compatibility/pytorch-oracle-lock.json').read_text())
    expected = lock['oracle']
    distribution = importlib.metadata.distribution(expected['distribution'])
    direct_text = distribution.read_text('direct_url.json')
    direct = None if direct_text is None else json.loads(direct_text)
    checks = dict(
        python_matches=sys.implementation.name == expected['python']['implementation']
            and list(sys.version_info[:2]) == expected['python']['version'],
        platform_matches=dict(sys_platform=sys.platform, machine=platform.machine()) == expected['platform'],
        version_backend_matches=str(torch.__version__) == expected['version']
            and {key: getattr(torch.version, key, 'MISSING') for key in expected['backend_versions']}
                == expected['backend_versions'],
        git_matches_oracle_build_pin=torch.version.git_version == expected['git_commit'],
        distribution_version_matches=distribution.version == expected['version'],
    )
    if not all(checks.values()):
        raise ValueError('installed Torch identity mismatch: ' + str(checks))
    return torch, dict(status='OBSERVED_INSTALLED_WHEEL_IDENTITY_ONLY', checks=checks,
        expected=expected, lock_sha256=LOCK, version=str(torch.__version__),
        git_commit=torch.version.git_version, package=identity(torch.__file__),
        extension=identity(torch._C.__file__), direct_url=direct,
        direct_url_record_available=direct is not None,
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        full_installation_provenance_verified=False, original_pip_report_available=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-observed-oracle', action='store_true',
                        help='Run local differential work without conferring complete CI provenance.')
    args = parser.parse_args()
    repo, out = args.repository.resolve(), args.output.resolve()
    assert Path.cwd().resolve() == repo and not sys.flags.optimize
    out.mkdir(parents=True, exist_ok=False)
    (out / 'blobs').mkdir()
    files = source_files(repo)
    if not args.local_observed_oracle:
        files.append(repo / 'build/pytorch-install.json')
    inputs = {str(path): identity(path) for path in files}
    assert inputs[str(repo / 'tests/pytorch_oracle.py')]['sha256'] == SUPPORT
    assert inputs[str(repo / 'compatibility/pytorch-oracle-lock.json')]['sha256'] == LOCK
    report = dict(schema='heptatorch.signbit.reference.v1', status='RUNNING', case_sha256=CASE_SHA256,
        expected_cases=len(CASES), rows=[], inputs=inputs, Torch_executed=False,
        native_executed=False, compiler_executed=False, host_expected_values_computed=False,
        failures=[], local_observed_oracle=args.local_observed_oracle,
        full_installation_provenance_verified=False,
        actual_process=dict(pid=os.getpid(), argv=list(sys.orig_argv), python=identity(sys.executable)))
    try:
        if args.local_observed_oracle:
            torch, provenance = local_torch_identity(repo)
        else:
            original = sys.argv
            try:
                sys.argv = [str(repo / 'tests/pytorch_oracle.py'), str(out / 'NO_NATIVE_EXECUTION'),
                            str(out / 'oracle-provenance.json'), str(repo / 'compatibility/pytorch-oracle-lock.json')]
                ctx = runpy.run_path(str(repo / 'tests/pytorch_oracle.py'), run_name='signbit_reference_only')
            finally:
                sys.argv = original
            torch, provenance = ctx['torch'], ctx['PROVENANCE']
            assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
            assert all(provenance['checks'].values())
            report['full_installation_provenance_verified'] = True
        torch.set_num_threads(1)
        torch.manual_seed(1729)
        torch.backends.mkldnn.enabled = False
        torch.use_deterministic_algorithms(True)
        report.update(Torch_executed=True, oracle_provenance=provenance,
                      actual_cpu_capability=torch.backends.cpu.get_cpu_capability())
        rng = bytes(torch.get_rng_state().tolist())
        for index, spec in enumerate(CASES):
            source, retained = setup(torch, spec)
            before = tensor(torch, source, out)
            result = torch.signbit(source)
            actual = tensor(torch, result, out)
            assert source._cdata != result._cdata
            assert source.untyped_storage()._cdata != result.untyped_storage()._cdata
            assert tensor(torch, source, out) == before and bytes(torch.get_rng_state().tolist()) == rng
            row = dict(index=index, spec=spec, input=before, result=actual, accepted=True,
                       input_unchanged=True, rng_unchanged=True, fresh_tensor=True, fresh_storage=True)
            report['rows'].append(row)
            (out / ('case-%03d.json' % index)).write_text(json.dumps(row, sort_keys=True, indent=2) + '\n')
        assert all(identity(path) == inputs[str(path)] for path in files)
        report.update(status='OBSERVED_SIGNBIT', cases=len(report['rows']), inputs_unchanged=True,
                      rng_unchanged=True, rng_sha256=hashlib.sha256(rng).hexdigest(),
                      all_observed_blobs_unchanged=True)
        for row in report['rows']:
            for value in (row['input'], row['result']):
                for field in ('logical', 'storage'):
                    blob = value[field]
                    raw = (out / blob['path']).read_bytes()
                    assert len(raw) == blob['bytes'] and hashlib.sha256(raw).hexdigest() == blob['sha256']
    except BaseException as error:
        report.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error),
                                                traceback=traceback.format_exc()))
    (out / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'cases', 'expected_cases', 'error')}), flush=True)
    return 0 if report['status'] == 'OBSERVED_SIGNBIT' else 1


if __name__ == '__main__':
    raise SystemExit(main())
