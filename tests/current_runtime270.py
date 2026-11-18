#!/usr/bin/env python3
"""Run all original270 cases using the fixed transport and the actual current build.

The archive remains immutable. Only its two explicitly selected dependencies,
owner profile identity and cold-build admission, are rebound in memory. The
original runner, four inverse-proved comparators, wire codec, numerical inputs,
Torch setup and all 270 cases execute unchanged.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'tests/current_runtime_original35'))
from binding import Inputs, bind_build, dump, require, sha
from current_runtime_fixtures import restore_current270


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_tag(bodies, producer, checker):
    tags = []
    for number in re.findall(rb'const\(\) \{"value":([0-9]+)\}', bodies[producer]):
        value = int(number)
        if not 0 <= value < 1 << 64:
            continue
        tag = value.to_bytes(8, 'little')
        if tag.startswith(b'HT') and all(32 <= byte < 127 for byte in tag):
            tags.append(tag)
    require(len(tags) == 1, 'one actual native owner format tag')
    token = b'{"value":' + str(int.from_bytes(tags[0], 'little')).encode() + b'}'
    require(token in bodies[checker], 'actual owner producer/checker tags agree')
    return tags[0].decode('ascii')


def make_bindings(package, directory, expected, inputs):
    build, bodies = bind_build(directory, expected, inputs)
    prepare = load(package / 'prepare.py', 'prepare')
    source, image, count = build['source_sha256'], build['native_sha256'], build['functions']
    state = source_tag(bodies, 'init_state', 'state_header')
    backing = source_tag(bodies, 'so_state_init', 'so_state_header')
    proof = dict(schema='heptatorch.current-runtime.original270-binding.v1',
        source_sha256=source, native_sha256=image, actual_build_sha256=expected,
        functions=count, units=build['units_planned'],
        original_runner_sha256=sha(inputs.bind(package / 'run.py')),
        original_prepare_sha256=sha(inputs.bind(package / 'prepare.py')),
        original_transport_sha256=sha(inputs.bind(package / 'transport.py')),
        original_wire_sha256=sha(inputs.bind(package / 'wire.py')),
        original_runner_and_comparator_source_unchanged=True,
        original_numeric_inputs_assertions_tolerances_and_Torch_setup_unchanged=True,
        runtime_rebindings=['prepare.owner_module', 'prepare.admission_module'],
        runtime_qualification_transferred=False)

    def owner_module(source_sha256, native_sha256):
        require((source_sha256, native_sha256) == (source, image), 'actual current270 owner binding')
        raw = inputs.bind(package / 'frozen/owner_integral.py', prepare.OWNER)
        changes = [
            ('HTSINT01', state, 2), ('HTOSIN01', backing, 2),
            ('functions == 699', 'functions == ' + str(count), 1),
            ('sum-integral-source01-UNBOUND-699', 'original270-current-runtime-UNBOUND', 1),
            ('sum-integral-source01-%d-preserved47-48', 'original270-current%d-full-owner', 1),
            ('heptatorch.sum-integral-owner-source01.v1', 'heptatorch.original270-current-owner.v1', 1),
        ]
        # A no-op identity change is excluded: whole-source inverse checks
        # remain exact when a later valid current source has an original tag.
        changes = [row for row in changes if row[0] != row[1]]
        derived, owner_proof = prepare.edits(raw, changes)
        module = types.ModuleType('current270_complete_owner')
        module.__file__ = str(package / 'frozen/owner_integral.py') + '#current-runtime-profile'
        exec(compile(derived, module.__file__, 'exec'), module.__dict__)
        module.bind_profile(source, image, count)
        owner_proof.update(actual_function_count=count,
                           original_snapshot_and_transition_predicates_unchanged=True)
        return module, owner_proof, derived

    def admission_module(source_sha256, native_sha256):
        require((source_sha256, native_sha256) == (source, image), 'actual current270 build binding')
        original = inputs.bind(package / 'frozen/admission701.py', prepare.ADMISSION)
        admission_proof = dict(original_frozen_admission_sha256=sha(original),
            original_frozen_admission_executed=False,
            current_admission_path=str(ROOT / 'tools/current_runtime/qualification.py'),
            current_admission_sha256=sha(inputs.bind(ROOT / 'tools/current_runtime/qualification.py')),
            source_sha256=source, native_sha256=image, functions=count, units=build['units_planned'],
            selected_source_image_manifest_objects_link_input_and_EOF_verified=True,
            original_comparators_and_reference_values_changed=False)

        def admit(directory_arg, expected_arg, binary, source_file):
            require(Path(directory_arg).resolve() == directory
                    and expected_arg == expected
                    and Path(binary).resolve() == directory / 'heptatorch'
                    and Path(source_file).resolve() == directory / 'runtime.hepta',
                    'original runner selected the admitted actual current build')
            again, _ = bind_build(directory, expected, inputs)
            require(again == build, 'same entire current cold build receipt')
            return dict(schema='heptatorch.current-runtime.cold-build-admission.v1',
                status='ADMITTED_CURRENT_COLD_BUILD', build_sha256=expected,
                source_sha256=source, native_sha256=image, functions=count,
                units=build['units_planned'], compiler_processes=build['units_planned'], linker_processes=1,
                native_executed_by_admission=False, Torch_executed_by_admission=False,
                runtime_qualification_inferred=False, files=dict(inputs.records))

        def recheck(admitted):
            results = {}
            for name, row in admitted['files'].items():
                try:
                    inputs.bind(Path(name), row['sha256'])
                    results[name] = inputs.records[name] == row
                except (OSError, AssertionError, ValueError):
                    results[name] = False
            return results

        return types.SimpleNamespace(admit_build=admit, recheck_build=recheck), admission_proof

    prepare.owner_module, prepare.admission_module = owner_module, admission_module
    return build, proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    require(not sys.flags.optimize, 'all original270 assertions must remain enabled')
    directory, output = args.build_directory.resolve(), args.output.resolve()
    require(not output.exists(), 'fresh original270 execution directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = Inputs()
    result, proof, error = 1, None, None
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        inputs.bind(Path(__file__).resolve())
        inputs.bind(ROOT / 'tests/current_runtime_original35/binding.py')
        inputs.bind(ROOT / 'tests/current_runtime_fixtures.py')
        package = restore_current270(ROOT, output.with_name(output.name + '-source-package'), inputs)
        build, proof = make_bindings(package, directory, args.expected_build_sha256, inputs)
        sys.path.insert(0, str(package))
        runner = load(package / 'run.py', 'current270_original_unchanged_runner')
        sys.argv = [str(package / 'run.py'), '--repository', str(ROOT),
                    '--binary', str(directory / 'heptatorch'), '--source', str(directory / 'runtime.hepta'),
                    '--build-directory', str(directory), '--output', str(output),
                    '--expected-source-sha256', build['source_sha256'],
                    '--expected-native-sha256', build['native_sha256'],
                    '--expected-build-sha256', args.expected_build_sha256]
        if args.prepare_only:
            sys.argv.append('--prepare-only')
        result = runner.main()
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught), traceback=traceback.format_exc())
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.argv, sys.path[:] = previous_argv, previous_path
        changed = inputs.recheck()
        output.mkdir(parents=True, exist_ok=True)
        receipt = dict(schema='heptatorch.current-runtime.original270-adapter.v1',
            status='PASS' if result == 0 and not changed and error is None else 'FAIL',
            prepare_only=args.prepare_only, original_runner_exit=result, binding=proof,
            error=error, inputs=inputs.records, changed_inputs=changed)
        dump(output / 'current-adapter.json', receipt)
        if changed or error is not None:
            result = 1
        report_path = output / 'report.json'
        if report_path.exists():
            original_raw = report_path.read_bytes()
            with (output / 'report.before-current-binding.json').open('xb') as stream:
                stream.write(original_raw)
            report = json.loads(original_raw)
            report['current_adapter'] = dict(path='current-adapter.json',
                sha256=sha((output / 'current-adapter.json').read_bytes()),
                original_runner_report_sha256=sha(original_raw), status=receipt['status'])
            if result != 0:
                report['status'] = 'FAIL'
                report['current_adapter_error'] = error or changed
            dump(report_path, report)
        else:
            dump(report_path, dict(schema='heptatorch.current701-original270.v1', status='FAIL',
                                   error=error, native_executed=False, Torch_executed=False))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
