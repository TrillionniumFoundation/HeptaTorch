"""Exact reversible source edits for the three legacy view successors.

These edits are independent of the current native source/image/count refresh.
Frozen input catalogs, original reader modules and the original87 entry code
remain unchanged. Every edit records its complete before/after bytes.
"""
import ast
import hashlib

sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def bodies(raw):
    lines = raw.splitlines(keepends=True)
    return {node.name: b''.join(lines[node.lineno-1:node.end_lineno]) for node in ast.parse(raw).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class Edits:
    def __init__(self, raw):
        self.original, self.raw, self.rows = raw, raw, []
    def one(self, before, after, label):
        before = before.encode() if isinstance(before, str) else before
        after = after.encode() if isinstance(after, str) else after
        require(before != after and self.raw.count(before) == 1, 'unique complete source edit: ' + label)
        self.raw = self.raw.replace(before, after, 1)
        self.rows.append(dict(label=label, before_hex=before.hex(), after_hex=after.hex()))
    def finish(self, allowed, added=()):
        inverse = self.raw
        for row in reversed(self.rows):
            old, new = bytes.fromhex(row['before_hex']), bytes.fromhex(row['after_hex'])
            require(inverse.count(new) == 1, 'unique inverse source edit: ' + row['label'])
            inverse = inverse.replace(new, old, 1)
        require(inverse == self.original, 'complete original observer/reader source inverse')
        old, new = bodies(self.original), bodies(self.raw)
        changed = {name for name in old.keys() & new.keys() if old[name] != new[name]}
        require(changed == set(allowed) and new.keys()-old.keys() == set(added) and not old.keys()-new.keys(),
                'exact bounded function/class inventory changes')
        return self.raw, dict(original_sha256=sha(self.original), derived_sha256=sha(self.raw),
            whole_source_inverse_exact=True, changed_definitions=sorted(changed), added_definitions=sorted(added),
            unchanged_definitions={name: sha(old[name]) for name in sorted(old.keys()-changed)}, edits=self.rows)


def public_run(raw):
    edit = Edits(raw)
    edit.one("sys.path.insert(0, str(HERE))\n", "sys.path.insert(0, str(HERE))\nsys.path.insert(1, str(ROOT / 'tests'))\n", 'explicit shared view package import path')
    original = bodies(raw)['scope_controls']
    replacement = b"def scope_controls(args, wire, folder):\n    from f64_view.legacy_successors import scope_controls as actual_scope\n    return actual_scope(args, wire, folder, native, args.f64_view_successor)\n"
    edit.one(original, replacement, 'one original view control gets the separately observed success contract')
    edit.one("    reference, execution = admit_reference(args, inputs)\n    return built, reference, execution, cold_inputs.records\n",
        "    reference, execution = admit_reference(args, inputs)\n    from f64_view.legacy_binding import from_arguments\n    args.f64_view_successor = from_arguments(args, inputs)\n    return built, reference, execution, cold_inputs.records\n", 'same complete actual build plus strict shared view reference')
    edit.one("    parser.add_argument('--output', type=Path, required=True)\n",
        "    parser.add_argument('--f64-view-reference', type=Path, required=True)\n    parser.add_argument('--f64-view-reference-execution', type=Path, required=True)\n    parser.add_argument('--output', type=Path, required=True)\n", 'explicit shared view reference CLI')
    edit.one("for key in ('repository', 'build_directory', 'reference', 'reference_execution', 'output'):",
        "for key in ('repository', 'build_directory', 'reference', 'reference_execution', 'f64_view_reference', 'f64_view_reference_execution', 'output'):", 'canonical shared view paths')
    return edit.finish({'scope_controls', 'admit', 'arguments'})


def public_qualify(raw):
    edit = Edits(raw)
    start = raw.index(b"    boundary = result['scope_controls']\n")
    stop = raw.index(b"    expected_folders = ", start)
    replacement = b"    boundary = result['scope_controls']\n    from f64_view.legacy_successors import replay_scope\n    require(args.f64_view_successor.proof['strict_reference_qualified'] is True,\n            'strict shared original188 reference is mandatory for the view successor')\n    replay_scope(args, native.load_wire(ROOT), directory / 'scope-controls', native,\n                 args.f64_view_successor, inputs, boundary)\n"
    edit.one(raw[start:stop], replacement, 'same two scope inputs; complete pure native/owner success replay replaces the old7 refusal predicate')
    edit.one("                       '--output', str(args.output / 'native')]",
        "                       '--f64-view-reference', str(args.f64_view_reference),\n                       '--f64-view-reference-execution', str(args.f64_view_reference_execution),\n                       '--output', str(args.output / 'native')]", 'actual inner process uses the same shared reference')
    # Insert beside the original returned native report identity, leaving all
    # original dense/packet/allocation/child predicates and counts untouched.
    marker = "native_report_sha256=sha(raw),"
    require(edit.raw.decode().count(marker) == 1, 'one original native report return binding')
    edit.one(marker, marker + " view_successor=args.f64_view_successor.proof,", 'bind successor in outer qualification report')
    return edit.finish({'read_back', 'main'})


def f64_fixtures(raw):
    edit = Edits(raw)
    edit.one('def reader_cli(repository, build, build_sha, reference, wide_neg, public_f64_arithmetic):',
        'def reader_cli(repository, build, build_sha, reference, wide_neg, public_f64_arithmetic, public_f64_view):', 'explicit view reference to original48 worker')
    edit.one("'--public-f64-arithmetic-reference-execution',str(public_f64_arithmetic['execution'])]",
        "'--public-f64-arithmetic-reference-execution',str(public_f64_arithmetic['execution']),\n            '--f64-view-reference',str(public_f64_view['root']),\n            '--f64-view-reference-execution',str(public_f64_view['execution'])]", 'actual original48 reference argv')
    edit.one('def current_reader(repository, reader_directory, wide_neg=None, public_f64_arithmetic=None):',
        'def current_reader(repository, reader_directory, wide_neg=None, public_f64_arithmetic=None, public_f64_view=None):', 'explicit view binding in actual reader')
    edit.one('    adapter = arithmetic.adapter_type(adapter, repository, directory, public_f64_arithmetic)\n',
        "    adapter = arithmetic.adapter_type(adapter, repository, directory, public_f64_arithmetic)\n    require(public_f64_view is not None, 'the two original views require their shared strict188 reference and actual build')\n    from f64_view.legacy_binding import adapter_type as view_adapter_type\n    adapter = view_adapter_type(adapter, repository, directory, public_f64_view)\n", 'compose exact two views with unchanged24 numeric successor methods')
    edit.one("    parser.add_argument('--candidate', type=Path, required=True)\n",
        "    parser.add_argument('--candidate', type=Path, required=True)\n    parser.add_argument('--build-directory', type=Path, required=True)\n    parser.add_argument('--expected-build-sha256', required=True)\n    parser.add_argument('--f64-view-reference', type=Path, required=True)\n    parser.add_argument('--f64-view-reference-execution', type=Path, required=True)\n", 'actual shared build/reference CLI')
    edit.one("                 '--public-f64-arithmetic-reference-directory', '--public-f64-arithmetic-reference-execution'):",
        "                 '--public-f64-arithmetic-reference-directory', '--public-f64-arithmetic-reference-execution',\n                 '--f64-view-reference', '--f64-view-reference-execution'):", 'strip only new wrapper flags before original entry argv')
    edit.one('        with current_reader(args.candidate, args.reader_directory, reference, arithmetic) as adapter:',
        "        view = dict(root=args.f64_view_reference, execution=args.f64_view_reference_execution,\n                    build=args.build_directory, build_sha=args.expected_build_sha256)\n        with current_reader(args.candidate, args.reader_directory, reference, arithmetic, view) as adapter:", 'same concrete view/build binding for original48 execution')
    return edit.finish({'reader_cli', 'current_reader', 'main'})


def f64_readback(raw):
    edit = Edits(raw)
    edit.one("        '--output', str(directory)]",
        "        '--f64-view-reference', report['source_admission']['public_f64_view_successor']['reference_directory'],\n        '--f64-view-reference-execution', report['source_admission']['public_f64_view_successor']['reference_execution'],\n        '--output', str(directory)]", 'exact same actual original48 view argv')
    edit.one("    with reader.current_reader(repository, reader_directory, shared, mixed) as adapter:",
        "    view_proof = report['source_admission']['public_f64_view_successor']\n    view = dict(root=view_proof['reference_directory'], execution=view_proof['reference_execution'],\n                build=build, build_sha=build_sha)\n    with reader.current_reader(repository, reader_directory, shared, mixed, view) as adapter:", 'reconstruct same strict shared reference and actual native build')
    edit.one("and details['actual_refusals'] == 63 and details['actual_successful_negations'] == 1",
        "and details['actual_refusals'] == 61 and details['actual_successful_views'] == 2\n                        and details['actual_successful_negations'] == 1", 'all87 requests with61 refusals and2 explicit views')
    edit.one("'all original87 requests,63 retained refusals,one checked negation and23 checked arithmetic results'",
        "'all original87 requests,61 retained refusals,two checked views,one checked negation and23 checked arithmetic results'", 'truthful successor denominator description')
    edit.one('unchanged_other_refusal_expectations=61, retained_refusals=63,',
        'unchanged_other_refusal_expectations=59, retained_refusals=61,', 'preserve two changed factory rejection statuses and59 other refusals')
    edit.one('actual_numeric_successors=24, wide_neg_successor=negation, original_OOM_controls=2,',
        'actual_numeric_successors=24, actual_view_successors=2, public_f64_view_successor=view_proof,\n        wide_neg_successor=negation, original_OOM_controls=2,', 'distinct unchanged numeric24 and view2 result categories')
    edit.one("    bindings = {name: method_binding(function, bound) for name, function in methods.items()}",
        "    view_successor = adapter.public_f64_view_successor\n    methods.update({'view.success': view_successor.success, 'view.observe': view_successor.observe,\n                    'view.prepare_entry': view_successor.prepare_entry,\n                    'view.wrap_entry': adapter.public_f64_arithmetic_successor.wrap_entry})\n    bindings = {name: method_binding(function, bound) for name, function in methods.items()}", 'bind exact executed new comparator method code objects')
    return edit.finish({'outer_execution', 'prepare_originals', 'qualify_f64_48'})


def public_adapter(raw):
    edit = Edits(raw)
    edit.one('def reader_arguments(repository, build, build_sha, output, reference):',
        'def reader_arguments(repository, build, build_sha, output, reference, public_f64_view):', 'pure reader explicit shared reference')
    edit.one('        binary=build / \'heptatorch\', prepare_only=False)',
        "        binary=build / 'heptatorch', prepare_only=False,\n        f64_view_reference=Path(public_f64_view['root']),\n        f64_view_reference_execution=Path(public_f64_view['execution']))", 'same public scope reference paths')
    edit.one('def reader_cli(repository, build, build_sha, reference):',
        'def reader_cli(repository, build, build_sha, reference, public_f64_view):', 'actual public192 explicit view argv')
    marker = "'--expected-execution-sha256', EXECUTION_SHA]"
    edit.one(marker, "'--expected-execution-sha256', EXECUTION_SHA,\n            '--f64-view-reference', str(public_f64_view['root']),\n            '--f64-view-reference-execution', str(public_f64_view['execution'])]", 'public192 exact inner/outer view CLI')
    edit.one('    cli = reader_cli(repository, build, build_sha, reference)',
        "    view = dict(root=report['view_successor']['reference_directory'],\n                execution=report['view_successor']['reference_execution'])\n    cli = reader_cli(repository, build, build_sha, reference, view)", 'qualification receipt binds the same shared reference')
    edit.one('    args = reader_arguments(repository, build, build_sha, directory, reference)',
        '    args = reader_arguments(repository, build, build_sha, directory, reference, view)', 'pure scope reader reconstructs actual args')
    edit.one('native_sessions=196, packet_controls=18, allocation_controls=2, native_boundary_controls=2,',
        'native_sessions=196, packet_controls=18, allocation_controls=2, native_boundary_controls=2,\n        retained_native_scope_refusals=1, explicit_view_successors=1, view_successor=report[\'view_successor\'],', 'retain196 children and distinguish one refusal/one view')
    return edit.finish({'reader_arguments', 'reader_cli', 'qualify_public192'})


def extend_public_outputs(outputs, previous):
    """Preserve the original729-to-current identity proof as predecessor scope."""
    result = dict(outputs)
    old_qualify = sha(outputs['qualify736.py'])
    result['qualify736.py'], qualify_proof = public_qualify(outputs['qualify736.py'])
    pin = Edits(outputs['run736.py'])
    pin.one(old_qualify, sha(result['qualify736.py']), 'same actual successor qualification reader source binding')
    refreshed, pin_proof = pin.finish(set())
    result['run736.py'], run_proof = public_run(refreshed)
    return result, dict(schema='heptatorch.public-f64-logical192.explicit-view-successor-derivation.v1',
        predecessor_identity_only_proof=previous, predecessor_proof_is_scoped_to_its_own_outputs=True,
        source_sha256=previous['source_sha256'], native_sha256=previous['native_sha256'],
        functions=previous['functions'], units=previous['units'],
        original_dense_cases=192, original_packet_controls=18, original_allocation_controls=2,
        original_scope_controls=2, retained_scope_refusals=1, explicit_view_successors=1,
        original_native_main_unchanged=True, original_six_frozen_sources_unchanged=True,
        original_read_child_and_wire_predicates_unchanged=True,
        view_successor_reference_index=176, run_inverse=run_proof,
        qualification_inverse=qualify_proof, qualification_source_pin_inverse=pin_proof,
        runtime_PASS_transferred=False, native_executed=False, Torch_executed=False)


def public_derivation(raw):
    edit = Edits(raw)
    old = '    return outputs, proof\n'
    new = "    import importlib.util\n    from pathlib import Path\n    helper_path = Path(__file__).resolve().parent.parent / 'f64_view/legacy_integration.py'\n    helper_spec = importlib.util.spec_from_file_location('public192_exact_view_successor_derivation', helper_path)\n    helper = importlib.util.module_from_spec(helper_spec)\n    helper_spec.loader.exec_module(helper)\n    derived, successor_proof = helper.extend_public_outputs(outputs, proof)\n    successor_proof['integration_source_sha256'] = sha(helper_path.read_bytes())\n    return derived, successor_proof\n"
    edit.one(old, new, 'explicit reversible view stage after complete original identity-only derivation')
    return edit.finish({'derive'})


TRANSFORMS = {
    'tests/public_f64_logical/run736.py': public_run,
    'tests/public_f64_logical/qualify736.py': public_qualify,
    'tests/current_runtime_f64_fixtures.py': f64_fixtures,
    'tests/current_runtime_f64_readback.py': f64_readback,
    'tests/current_runtime_public_f64_logical192.py': public_adapter,
    'tests/public_f64_logical/derive736.py': public_derivation,
}
