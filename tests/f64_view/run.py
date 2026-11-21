#!/usr/bin/env python3
"""Qualify the bounded public F64 view domain on the complete current build."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from f64_view import native, family_native
from f64_view.cases import CASE_COUNT, CASE_SHA256, CASE_INPUT_SHA256, ORIGINAL_CASE_SHA256
from f64_view.evidence import require
from f64_view.family_evidence import read_families
from numeric_predicates.run import Inputs
from numeric_predicates.native import dump
import pinned_reference_custody as custody

sha = lambda raw: hashlib.sha256(raw).hexdigest()
SOURCE = 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
IMAGE = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
COUNTS = dict(original_cases=188, empty_cases=841, wide_cases=20, offset_cases=75,
              cases=1124, native_sessions=1175, alias_chains=3, protocol_controls=19,
              allocation_families=3, owner_controls=44)


def module(path, name, bind):
    path = Path(path).resolve(strict=True)
    raw = bind(path)
    value = types.ModuleType(name)
    value.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), value.__dict__)
    return value


def profile(repository, build, build_sha, inputs):
    repository, build = Path(repository).resolve(strict=True), Path(build).resolve(strict=True)
    require(repository == ROOT and not sys.flags.optimize, 'actual source repository and enabled predicates')
    inputs.bind(Path(sys.executable).resolve(strict=True))
    for folder in ('f64_view', 'numeric_predicates', 'half_logical'):
        for path in sorted((ROOT / 'tests' / folder).iterdir()):
            if path.is_file() and path.suffix in ('.py', '.json'):
                inputs.bind(path)
    qualification = module(ROOT / 'tools/current_runtime/qualification.py', 'f64_views_actual_build', inputs.bind)
    built = qualification.admit_build(build, inputs.bind, build_sha)
    require(built['source_sha256'] == SOURCE and built['native_sha256'] == IMAGE
            and built['functions'] == 800 and len(built['units']) == 15,
            'one exact complete800 native build for every new and retained view request')
    transition = module(ROOT / 'tools/current_runtime/public_f64_view_source_transition.py',
                        'f64_views_complete_source_inverse', inputs.bind)
    source = inputs.bind(ROOT / 'tools/current_runtime/source/runtime.hepta', SOURCE)
    parent, proof = transition.restore_source(source, inputs.bind)
    route = transition.prove_original_domain(source, inputs.bind)
    require(proof['source_sha256'] == SOURCE and proof['functions'] == 800
            and proof['parent_functions'] == 797 and proof['unchanged_parent_functions'] == 792
            and proof['changed_functions'] == ['bind_alias', 'so_tensor_live', 'to_command', 'to_strided_view', 'f64_dtype_audit']
            and proof['appended_functions'] == ['public_f64_empty_strided_view', 'public_f64_empty_descriptor', 'so_f64_empty_offset_live']
            and proof['whole_source_inverse_exact'] is True and proof['whole_round_trip_exact'] is True
            and proof['original_backward_functions_unchanged'] is True
            and sha(parent) == proof['parent_source_sha256'] == transition.PARENT_SOURCE,
            'every changed body, every retained backward and the whole published797 predecessor are bound')
    require(route['original_domain_dispatcher_guard_unchanged'] is True
            and route['original_non_f64_constructor_and_all_arguments_retained'] is True
            and route['original_legal_owner_offsets_never_enter_new_tag_helper'] is True
            and route['ordinary_owner_byte_limit_span_and_successor_arguments_retained'] is True,
            'new tagged-empty representation retains the original admitted owner domain')
    return built, dict(complete_source_inverse=proof, original_domain_source_paths=route,
                       physical_observer_source_relations=family_native.source_relations())


def references(directory, execution, inputs):
    families = read_families(directory, execution, inputs)
    original = families['original_reference']
    require(original['strict_reference_qualified'] is families['strict_reference_qualified'] is True,
            'original188 and all added families require complete strict original wheel custody')
    plans = family_native.recipes(original, families)
    require(len(plans) == COUNTS['native_sessions'] and len({row[0] for row in plans}) == len(plans),
            'exact full independent reference and control session census')
    require(CASE_COUNT == 188 and len(original['report']['rows']) == CASE_COUNT,
            'all original188 observations retain their original denominator')
    tensor_plans = [row for row in plans if re.fullmatch(r'(?:case-\d{3}|empty-family-\d{4}|wide-\d{2}|offset-\d{3})', row[0])]
    require(len(tensor_plans) == 1124
            and sum(row[2][0]['accepted'] is True for row in tensor_plans) == 852
            and sum(row[2][0]['accepted'] is False for row in tensor_plans) == 272,
            'all case outcomes derive from the complete independent Torch observations')
    return original, families, plans


def execute_plan(build, output, plan, wire, inputs, replay):
    name, compare, arguments, reference = plan
    saved = dict(name=name, status='FAIL', original_native_process_observed=False); session = observer = None
    try:
        session = (native.Replay if replay else native.Live)(build / 'heptatorch', wire, output / name, inputs.bind)
        require(type(session.pid) is int and session.pid > 0, 'actual original child process identity')
        saved['original_native_process_observed'] = True
        observer = family_native.Observer(session, wire, reference)
        saved.update(result=compare(observer, *arguments), status='PASS')
    except BaseException as error:
        saved['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        if session is not None:
            try:
                saved['finish'] = session.finish()
            except BaseException as error:
                saved.update(status='FAIL', finish_error=dict(type=type(error).__name__, message=str(error)))
        saved['checks'] = observer.checks if observer is not None else 0
    if not replay and (output / name).is_dir():
        dump(output / name / 'observation.json', saved)
    return saved


def qualify(report, repository, build, build_sha, output, inputs):
    """Reapply every numerical, metadata, lifetime and EOF predicate to raw bytes."""
    build, output = Path(build).resolve(strict=True), Path(output).resolve(strict=True)
    built, proof = profile(repository, build, build_sha, inputs)
    require(report['schema'] == 'heptatorch.public-f64-view.native.v1'
            and report['status'] == 'PASS_NATIVE_F64_VIEW' and report['failures'] == []
            and all(report.get(key) == value for key, value in COUNTS.items())
            and report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
            and report['actual_build_sha256'] == build_sha and report['source_proof'] == proof
            and report['original_case_sha256'] == ORIGINAL_CASE_SHA256
            and report['original_case_input_sha256'] == CASE_INPUT_SHA256
            and report['original188_case_sha256'] == CASE_SHA256
            and report['native_executed'] is True and report['Torch_executed'] is True
            and report['strict_reference_qualified'] is True and report['host_expected_values_computed'] is False
            and report['all_inputs_unchanged'] is True and report['full_project_status'] == 'HOLD'
            and report['full_pytorch_rewrite_complete'] is False,
            'complete bounded observations, exact actual source/build and strict reference scope')
    for spelling, record in report['inputs'].items():
        path = Path(spelling); raw = inputs.bind(path, record['sha256'])
        require(record == dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(path.stat().st_mode)),
                'every complete original observer input and mode remains identical')
    reference_directory = Path(report['reference_directory'])
    execution = Path(report['reference_process_directory'])
    require(reference_directory == output / 'reference' and execution == output / 'reference-execution',
            'one same-run complete original reference and actual process')
    original, families, plans = references(reference_directory, execution, inputs)
    require(report['reference_sha256'] == original['report_sha256']
            and report['families_reference_sha256'] == families['report_sha256']
            and len(report['results']) == len(plans), 'exact same reference and complete original result census')
    wire = native.load_wire(ROOT, inputs.bind)
    totals = dict(frames=0, checks=0, owner_reads=0, mapping_reads=0)
    for plan, saved in zip(plans, report['results']):
        require(saved['name'] == plan[0] and saved['status'] == 'PASS'
                and saved == json.loads(inputs.bind(output / plan[0] / 'observation.json')),
                'one complete original child result for every mandatory session')
        fresh = execute_plan(build, output, plan, wire, inputs, True)
        require(fresh == saved, 'every original predicate reapplied to complete raw frames and owners: ' + plan[0])
        totals['checks'] += fresh['checks']
        for field in ('frames', 'owner_reads', 'mapping_reads'):
            totals[field] += fresh['finish'][field]
    require(all(report[key] == value for key, value in totals.items()), 'complete physical frame and owner census')
    inputs.recheck()
    return dict(schema='heptatorch.public-f64-view.physical.v1', status='PASS_PHYSICAL_F64_VIEW',
        **COUNTS, source_sha256=SOURCE, native_sha256=IMAGE, actual_build_sha256=build_sha,
        original188_case_sha256=CASE_SHA256, original_case_input_sha256=CASE_INPUT_SHA256,
        original_case_sha256=ORIGINAL_CASE_SHA256, reference_sha256=original['report_sha256'],
        families_reference_sha256=families['report_sha256'], **totals,
        compatible_tensor_cases=1124, actual_Torch_acceptances=852, actual_Torch_geometry_errors=272,
        known_native_capacity_gaps=[], original_case187_explicit_successor=True,
        actual_allocation_refusals=3, identical_request_recoveries=3, recovered_aliases_per_family=256,
        all84_live_descriptor_words_checked=True, all8_live_owner_table_inventories_checked=True,
        raw_metadata13_preserved=True, original_root_Storage_Version_lifetime_checked=True,
        public_root_release_and_nonempty_child_recreation_checked=True,
        strict_reference_qualified=True, original_eleven_provenance_checks_recomputed=True,
        complete_raw_physical_replay=True, native_executed_by_qualifier=False, Torch_executed_by_qualifier=False,
        host_expected_values_computed=False, all_inputs_unchanged=True, original35_groups_executed_here=False,
        rank_capacity=4, no_AD_only=True, full_project_status='HOLD', full_pytorch_rewrite_complete=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-directory', type=Path, required=True)
    p.add_argument('--expected-build-sha256', required=True)
    p.add_argument('--torch-python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); build = args.build_directory.resolve(strict=True); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = Inputs(); started = time.monotonic()
    report = dict(schema='heptatorch.public-f64-view.native.v1', status='FAIL', **COUNTS,
        original188_case_sha256=CASE_SHA256, original_case_sha256=ORIGINAL_CASE_SHA256,
        original_case_input_sha256=CASE_INPUT_SHA256, actual_build_sha256=args.expected_build_sha256,
        results=[], failures=[], frames=0, checks=0, owner_reads=0, mapping_reads=0,
        native_executed=False, Torch_executed=False, strict_reference_qualified=False,
        host_expected_values_computed=False, all_inputs_unchanged=False,
        full_project_status='HOLD', full_pytorch_rewrite_complete=False)
    try:
        built, proof = profile(ROOT, build, args.expected_build_sha256, inputs)
        report.update(source_sha256=built['source_sha256'], native_sha256=built['native_sha256'], source_proof=proof)
        reference_directory, execution = output / 'reference', output / 'reference-execution'
        execution.mkdir()
        supervisor = module(ROOT / 'tests/current_runtime_process.py', 'public_f64_view_reference_supervisor', inputs.bind)
        argv = [str(args.torch_python.absolute()), '-B', str(ROOT / 'tests/f64_view/reference.py'),
                '--output', str(reference_directory)]
        process = custody.supervise(supervisor, argv, execution, 300, ROOT, inputs.bind)
        dump(execution / 'process.json', process)
        original, families, plans = references(reference_directory, execution, inputs)
        require(args.torch_python.resolve(strict=True) == Path(original['report']['actual_process']['python']['path']),
                'actual configured reference interpreter')
        report.update(reference_directory=str(reference_directory), reference_process_directory=str(execution),
            reference_sha256=original['report_sha256'], families_reference_sha256=families['report_sha256'],
            Torch_executed=True, strict_reference_qualified=True)
        wire = native.load_wire(ROOT, inputs.bind)
        for at, plan in enumerate(plans):
            saved = execute_plan(build, output, plan, wire, inputs, False)
            report['native_executed'] |= saved['original_native_process_observed']
            report['results'].append(saved)
            report['checks'] += saved['checks']
            for field in ('frames', 'owner_reads', 'mapping_reads'):
                report[field] += saved.get('finish', {}).get(field, 0)
            if saved['status'] != 'PASS':
                report['failures'].append(plan[0])
            if (at+1) % 128 == 0 or saved['status'] != 'PASS':
                print(json.dumps({key: saved.get(key) for key in ('name', 'status', 'checks', 'error', 'finish_error')}), flush=True)
        inputs.recheck(); report['all_inputs_unchanged'] = True
        if len(report['results']) == COUNTS['native_sessions'] and not report['failures']:
            report['status'] = 'PASS_NATIVE_F64_VIEW'
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    report.update(inputs=inputs.records, elapsed_seconds=time.monotonic()-started)
    dump(output / 'report.json', report)
    print(json.dumps({key: report.get(key) for key in ('status', 'cases', 'native_sessions', 'frames', 'checks', 'failures', 'error')}), flush=True)
    return 0 if report['status'] == 'PASS_NATIVE_F64_VIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
