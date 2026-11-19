#!/usr/bin/env python3
"""Execute and independently reread current-image Half logical qualification."""

import argparse
import importlib.util
import json
from pathlib import Path
import stat
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from half_logical import native, reference

require, sha = reference.require, reference.sha
GROUP_CASES = {'half_logical264':264, 'half_transactions12':12}


class Inputs:
    def __init__(self):
        self.records = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), 'ordinary physical input: ' + str(path))
        raw = path.read_bytes()
        after = path.stat()
        fields = lambda row: (row.st_dev,row.st_ino,row.st_mode,row.st_size,row.st_mtime_ns,row.st_ctime_ns)
        require(fields(before) == fields(after), 'physical input changed while reading')
        row = reference.identity(raw, stat.S_IMODE(after.st_mode))
        require(expected is None or sha(raw) == expected, 'fixed input SHA: ' + str(path))
        require(str(path) not in self.records or self.records[str(path)] == row, 'previous input changed')
        self.records[str(path)] = row
        return raw

    def recheck(self):
        for name, row in list(self.records.items()):
            self.bind(Path(name), row['sha256'])


def profile(repository, build, build_sha, inputs):
    path = repository / 'tools/current_runtime/qualification.py'
    inputs.bind(path)
    spec = importlib.util.spec_from_file_location('half_logical_current_profile', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.admit_build(build, inputs.bind, build_sha)
    for file in sorted((repository / 'tests/half_logical').glob('*.py')):
        inputs.bind(file)
    return result


def qualify(report, repository, build, build_sha, output, inputs):
    """Reconstruct every original operation from its physical frames and owners."""
    built = profile(repository, build, build_sha, inputs)
    actual = json.loads(inputs.bind(output / 'report.json'))
    require(report == actual, 'unchanged exact native observer report')
    group = report['group']
    require(group in GROUP_CASES and report['cases'] == GROUP_CASES[group]
            and report['source_sha256'] == built['source_sha256']
            and report['native_sha256'] == built['native_sha256']
            and report['actual_build_sha256'] == build_sha and report['native_executed'] is True
            and report['Torch_executed'] is False and report['compiler_executed'] is False
            and report['original_reference_modified'] is False
            and report['historical_refusal12_rewritten'] is False, 'current native observer identity and scope')
    fixed = reference.read(repository, Path(report['reference_directory']), inputs.bind)
    wire = native.load_wire(repository, inputs.bind)
    recipes = native.recipes(fixed, group)
    require(report['status'] == 'PASS' and not report['failures'] and report['all_inputs_unchanged'] is True
            and len(report['results']) == len(recipes), 'complete successful current observation')
    frames = checks = owner_reads = mapping_reads = 0
    for saved, (name, recipe, parameters) in zip(report['results'], recipes):
        directory = output / name
        require(saved == json.loads(inputs.bind(directory / 'observation.json'))
                and saved['name'] == name and saved['status'] == 'PASS', 'one original completed observation per recipe')
        session = native.Replay(build / 'heptatorch', wire, directory, inputs.bind)
        observer = native.Observer(session, wire, fixed)
        result = recipe(observer, *parameters)
        finish = session.finish()
        require(result == saved['result'] and observer.checks == saved['checks'] and finish == saved['finish'],
                'complete physical replay reapplied every original numerical and ownership predicate')
        frames += finish['frames']; checks += observer.checks; owner_reads += finish['owner_reads']
        mapping_reads += finish.get('mapping_reads', 0)
    require(frames == report['frames'] and checks == report['checks'] and owner_reads == report['owner_reads'],
            'all original assertions, raw frames and complete owner snapshots retained')
    require(mapping_reads == (12 if group == 'half_transactions12' else 0),
            'all six failed FE transactions preserve the original pair of complete mapping observations')
    return dict(cases=GROUP_CASES[group], frames=frames, assertions=checks, native_sessions=len(recipes),
        owner_snapshots=owner_reads, mapping_snapshots=mapping_reads,
        physical_replay=True, native_executed_by_qualifier=False,
        Torch_executed_by_qualifier=False, reference_sha256=reference.REPORT,
        additional_packet_controls=52 if group == 'half_logical264' else 0,
        additional_allocation_recoveries=6 if group == 'half_logical264' else 0,
        historical_refusal12_rewritten=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=tuple(GROUP_CASES), required=True)
    parser.add_argument('--reference-directory', type=Path)
    args = parser.parse_args()
    require(not sys.flags.optimize, 'all independent predicates are enabled')
    output, build = args.output.resolve(), args.build_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    started = time.monotonic()
    report = dict(schema='heptatorch.current-half-logical.native-observation.v1', status='FAIL',
        group=args.group, cases=GROUP_CASES[args.group], results=[], failures=[],
        native_executed=False, Torch_executed=False, compiler_executed=False,
        original_reference_modified=False, historical_refusal12_rewritten=False,
        runtime_qualification_transferred=False, full_project_status='HOLD',
        frames=0, checks=0, owner_reads=0, actual_build_sha256=args.expected_build_sha256)
    try:
        inputs.bind(Path(sys.executable).resolve())
        built = profile(ROOT, build, args.expected_build_sha256, inputs)
        report.update(source_sha256=built['source_sha256'], native_sha256=built['native_sha256'])
        if args.reference_directory is None:
            fixed = reference.restore(ROOT, output / 'reference', inputs.bind)
        else:
            fixed = reference.read(ROOT, args.reference_directory.resolve(), inputs.bind)
        report['reference_directory'] = str(fixed['root'])
        report['reference_sha256'] = reference.REPORT
        wire = native.load_wire(ROOT, inputs.bind)
        recipes = native.recipes(fixed, args.group)
        for ordinal, (name, recipe, parameters) in enumerate(recipes):
            row = dict(name=name, status='FAIL')
            session = observer = None
            try:
                session = native.Live(build / 'heptatorch', wire, output / name, inputs.bind)
                report['native_executed'] = True
                observer = native.Observer(session, wire, fixed)
                row['result'] = recipe(observer, *parameters)
                row['status'] = 'PASS'
            except BaseException as error:
                row['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            finally:
                if session is not None:
                    try:
                        row['finish'] = session.finish()
                    except BaseException as error:
                        row['status'] = 'FAIL'
                        row['finish_error'] = dict(type=type(error).__name__, message=str(error))
                row['checks'] = observer.checks if observer is not None else 0
                if (output / name).exists():
                    native.dump(output / name / 'observation.json', row)
            report['results'].append(row)
            report['frames'] += row.get('finish', {}).get('frames', 0)
            report['owner_reads'] += row.get('finish', {}).get('owner_reads', 0)
            report['checks'] += row['checks']
            if row['status'] != 'PASS':
                report['failures'].append(name)
            if ordinal % 32 == 0 or row['status'] != 'PASS':
                print(json.dumps(dict(group=args.group, recipe=name, status=row['status'], error=row.get('error'))), flush=True)
        inputs.recheck()
        report['all_inputs_unchanged'] = True
        report['status'] = 'PASS' if not report['failures'] and len(report['results']) == len(recipes) else 'FAIL'
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    report['inputs'] = inputs.records
    report['elapsed_seconds'] = time.monotonic() - started
    native.dump(output / 'report.json', report)
    if report['status'] == 'PASS':
        try:
            proof = qualify(report, ROOT, build, args.expected_build_sha256, output, Inputs())
            native.dump(output / 'physical-qualification.json', proof)
        except BaseException as error:
            native.dump(output / 'physical-qualification.json', dict(status='FAIL', error=str(error), traceback=traceback.format_exc()))
            print(json.dumps(dict(status='FAIL_PHYSICAL_REPLAY', error=str(error))), flush=True)
            return 1
    print(json.dumps({key:report.get(key) for key in ('group','status','cases','frames','checks','failures','error')}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
