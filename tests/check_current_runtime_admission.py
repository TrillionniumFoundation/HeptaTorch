#!/usr/bin/env python3
"""Check current source and reader admission before the complete native gate.

This preparation gate starts no process, imports no Torch, and grants no native
qualification. The complete numerical gate and its independent readers remain
mandatory after this check succeeds.
"""

import argparse
import ast
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys
import textwrap
import traceback


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Inputs:
    def __init__(self):
        self.records = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        require(not path.is_symlink(), 'symlinked source input: ' + str(path))
        before = path.stat()
        require(stat.S_ISREG(before.st_mode), 'regular source input required')
        raw = path.read_bytes()
        after = path.stat()
        fields = lambda value: (value.st_dev, value.st_ino, value.st_mode,
                                value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        require(fields(before) == fields(after), 'input changed while reading: ' + str(path))
        row = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row['sha256'] == expected, 'source hash differs: ' + str(path))
        require(str(path) not in self.records or self.records[str(path)] == row,
                'bound source changed: ' + str(path))
        self.records[str(path)] = row
        return raw

    def recheck(self):
        for path, row in list(self.records.items()):
            self.bind(path, row['sha256'])
        return []


def prohibit_execution(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn',
                 'os.spawn', 'os.fork', 'os.forkpty'):
        raise RuntimeError('source preparation must not start a process: ' + event)
    if event == 'import' and args and str(args[0]).split('.')[0] == 'torch':
        raise RuntimeError('source preparation must not import Torch')


def load(path, inputs):
    path = Path(path).resolve()
    raw = inputs.bind(path)
    name = 'admission_precheck_' + sha(str(path).encode())[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Compile the bytes that were bound, even if a valid stale .pyc exists.
    exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    require(inputs.bind(path) == raw, 'module changed while loading')
    return module


def identity_bindings(repository, inputs, source, image, manifest_sha):
    bindings = {
        'tests/current_runtime_f64_fixtures.py':
            {'CURRENT_SOURCE': source, 'CURRENT_IMAGE': image, 'CURRENT_PARTITION': manifest_sha},
        'tests/current_runtime_half_fixtures.py':
            {'CURRENT_SOURCE': source, 'CURRENT_IMAGE': image},
        'tests/current_runtime_generator_fixtures.py':
            {'CURRENT_SOURCE': source, 'CURRENT_IMAGE': image, 'CURRENT_MANIFEST': manifest_sha},
        'tests/current_runtime_generator_binding.py':
            {'SOURCE': source, 'IMAGE': image, 'MANIFEST': manifest_sha},
        'tools/current_runtime/fusion_source_transition.py': {'CURRENT_SOURCE': source},
        'tools/current_runtime/index_signbit_source_transition.py': {'CURRENT_SOURCE': source},
        'tests/public_f64_logical/derive736.py': {'SOURCE': source, 'IMAGE': image},
    }
    for name in ('current_runtime_generator131', 'current_runtime_f64_readback',
                 'current_runtime_half_readback', 'current_runtime_f64_arithmetic_fe_owner',
                 'current_runtime_half_fe_owner'):
        bindings['tests/' + name + '.py'] = {'SOURCE': source, 'IMAGE': image}
    for path in ('tests/current_runtime_asstrided/common.py',
                 'tests/current_runtime_resources/adapter.py',
                 'tests/current_runtime_resources/qualify.py',
                 'tools/current_runtime/fusion_resource_transition.py'):
        bindings[path] = {'SOURCE': source, 'IMAGE': image}
    for path, fields in bindings.items():
        module = load(repository / path, inputs)
        for field, expected in fields.items():
            require(getattr(module, field) == expected,
                    'stale current identity: ' + path + ':' + field)
    return {'modules': len(bindings), 'fields': sum(map(len, bindings.values()))}


def successor_entry_bindings(repository, inputs, functions):
    """Check the actual derived entry statements missed by profile-only checks."""
    generator = repository / 'tests/current_runtime_generator_fixtures.py'
    tree = ast.parse(inputs.bind(generator))
    derive = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == 'successor_qualifier']
    require(len(derive) == 1, 'one actual Generator qualifier derivation')
    schemas = [keyword.value for node in ast.walk(derive[0])
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'dict'
               for keyword in node.keywords if keyword.arg == 'schema']
    require(len(schemas) == 1 and ast.literal_eval(schemas[0]) ==
            'heptatorch.generator%d-original-qualification-source.v1' % functions,
            'derived Generator manifest must match the actual current runner schema')

    reader = ast.parse(inputs.bind(repository / 'tests/current_runtime_generator131.py'))
    owner = [node for node in reader.body if isinstance(node, ast.FunctionDef)
             and node.name == 'tensor_owner']
    require(len(owner) == 1, 'one actual Generator TensorOwner reader')
    changes = [node.value for node in owner[0].body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == 'changes']
    require(len(changes) == 1, 'one complete Generator TensorOwner derivation')
    labels = {before: after for before, after, count in ast.literal_eval(changes[0])}
    require(labels['sum-integral-source01-UNBOUND-699'] ==
            'generator%d-dispatch-UNBOUND' % functions
            and labels['heptatorch.sum-integral-owner-source01.v1'] ==
            'heptatorch.generator%d-dispatch-tensor-owner.v1' % functions,
            'Generator TensorOwner reader identities match the actual generated observer')
    audits = [node for node in reader.body if isinstance(node, ast.FunctionDef)
              and node.name == 'audit_generator_reports']
    require(len(audits) == 1, 'one actual Generator report reader')
    expected = {'schema': 'heptatorch.generator%d-current-native-qualification.v1' % functions,
                'status': 'PASS_CURRENT%d_GENERATOR131' % functions}
    observed = {key: [] for key in expected}
    for node in ast.walk(audits[0]):
        if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript)
                and isinstance(node.left.value, ast.Name) and node.left.value.id == 'report'
                and isinstance(node.left.slice, ast.Constant) and node.left.slice.value in expected):
            continue
        require(len(node.ops) == len(node.comparators) == 1 and isinstance(node.ops[0], ast.Eq),
                'Generator report identities require exact equality')
        observed[node.left.slice.value].append(ast.literal_eval(node.comparators[0]))
    require(all(observed[key] == [value] for key, value in expected.items()),
            'Generator report reader identities match the actual generated runner')

    # Execute the actual Half successor's second current-source binding. The
    # preceding complete inverse is checked separately. This statement must
    # admit the current input, never an intermediate historical source digest.
    path = repository / 'tests/current_runtime_half_addsub_successor.py'
    tree = ast.parse(inputs.bind(path))
    owners = [node for node in tree.body if isinstance(node, ast.ClassDef)
              and node.name == 'Successor']
    require(len(owners) == 1, 'one actual Half successor class')
    initializers = [node for node in owners[0].body if isinstance(node, ast.FunctionDef)
                    and node.name == '__init__']
    require(len(initializers) == 1, 'one actual Half successor initializer')
    statements = []
    for node in initializers[0].body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'source'
                and isinstance(node.value, ast.Call) and len(node.value.args) == 2):
            continue
        digest = node.value.args[1]
        if (isinstance(digest, ast.Attribute) and isinstance(digest.value, ast.Name)
                and digest.value.id == 'indexing'):
            statements.append(node)
    require(len(statements) == 1, 'one actual current Half source rebind')
    indexing = load(repository / 'tools/current_runtime/index_signbit_source_transition.py', inputs)
    from types import SimpleNamespace
    context = dict(self=SimpleNamespace(repository=repository), inputs=inputs, indexing=indexing)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec',
                 dont_inherit=True, optimize=0), context)
    require(sha(context['source']) == indexing.CURRENT_SOURCE,
            'Half successor rebind retains the complete current source')
    return {'generator_manifest_matches_current_runner': True,
            'half_actual_current_source_binding_executed': True,
            'runtime_executed': False, 'Torch_executed': False}


def contracts(repository, inputs):
    source = inputs.bind(repository / 'tools/current_runtime/source/runtime.hepta')
    bodies = {m[1].decode(): m[0] for m in re.finditer(
        rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', source, re.M | re.S)}
    checked = 0
    owner = json.loads(inputs.bind(
        repository / 'tools/current_runtime/half_addsub_fe_owner_contract.json'))
    tombstone = json.loads(inputs.bind(
        repository / 'tools/current_runtime/half_addsub_fe_tombstone_contract.json'))
    for rows in (owner['function_identities'], tombstone['functions']):
        for name, row in rows.items():
            raw = bodies[name]
            require(sha(raw) == row['sha256'] and len(raw) == row['bytes'],
                    'full FE owner function identity: ' + name)
            checked += 1
    for row in (owner['support_unit'], tombstone['predicate']):
        raw = inputs.bind(repository / row['path'], row['sha256'])
        require(len(raw) == row['bytes'], 'complete FE source extent: ' + row['path'])
    bound_files = 0
    for path, field in (
            ('tests/current_runtime_resources/sources.json', 'files'),
            ('tests/current_runtime_asstrided/sources.json', 'repository_files'),
            ('tests/current_runtime_metadata10/source.json', 'dependencies')):
        rows = json.loads(inputs.bind(repository / path))[field]
        for name, row in rows.items():
            raw = inputs.bind(repository / name, row['sha256'])
            require(len(raw) == row['bytes'], 'bound source extent: ' + name)
            if 'mode' in row:
                require(stat.S_IMODE((repository / name).stat().st_mode) == row['mode'],
                        'bound source mode: ' + name)
            bound_files += 1
    return {'complete_HIA_functions': checked, 'bound_source_files': bound_files}


def extension_case_bindings(repository, inputs):
    """Bind every actual extension catalog before running the numerical gate."""
    seal = load(repository / 'tools/seal_current_ci_evidence.py', inputs)
    groups = tuple(seal.EXTENSION_GROUPS)
    require(groups and len(set(groups)) == len(groups)
            and set(groups).issubset(seal.GROUPS)
            and set(seal.EXTENSION_CASE_HASHES) == set(groups),
            'complete required extension case-hash map is required')
    observed = {}
    for group in groups:
        catalog = load(repository / 'tests' / group / 'cases.py', inputs)
        require(type(catalog.CASES) is list
                and len(catalog.CASES) == seal.REQUIRED_CASES[group],
                'actual extension catalog count differs: ' + group)
        require(isinstance(catalog.CASE_SHA256, str)
                and re.fullmatch(r'[0-9a-f]{64}', catalog.CASE_SHA256) is not None
                and catalog.CASE_SHA256 == seal.EXTENSION_CASE_HASHES[group],
                'actual extension catalog digest differs: ' + group)
        observed[group] = dict(cases=len(catalog.CASES), case_sha256=catalog.CASE_SHA256)
    return {'groups': len(groups), 'catalogs': observed}


def inventory(repository, inputs, coordinator):
    seal = load(repository / 'tools/seal_current_ci_evidence.py', inputs)
    manifest_raw = inputs.bind(repository / 'tools/current_runtime/source/manifest.json')
    manifest = json.loads(manifest_raw)
    groups, cases = tuple(seal.GROUPS), seal.REQUIRED_CASES
    total = sum(cases.values())
    require(groups == tuple(coordinator.GROUPS) == tuple(cases)
            and total == seal.REQUIRED_CASES_TOTAL, 'coordinator/sealer inventory differs')
    coordinator_tree = ast.parse(inputs.bind(repository / 'tests/qualify_current_runtime.py'))
    main_functions = [node for node in coordinator_tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'main']
    require(len(main_functions) == 1, 'one complete coordinator main is required')
    initializers = [node.value for node in main_functions[0].body
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'result']
    require(len(initializers) == 1 and isinstance(initializers[0], ast.Dict),
            'the actual coordinator result initializer is required')
    fields = {ast.literal_eval(key): value
              for key, value in zip(initializers[0].keys, initializers[0].values)}
    require(ast.literal_eval(fields['required_cases']) == cases
            and ast.literal_eval(fields['required_cases_total']) == total
            and ast.dump(fields['required_suites']) == ast.dump(ast.parse('list(GROUPS)', mode='eval').body),
            'the actual coordinator case denominator or suite list differs')
    source, image = manifest['runtime']['sha256'], manifest['expected_native_sha256']
    count, units = manifest['runtime']['functions'], len(manifest['units'])
    require((seal.SOURCE, seal.IMAGE, seal.SOURCE_MANIFEST) == (source, image, sha(manifest_raw)),
            'sealer current source/image/manifest identity differs')
    capabilities = json.loads(inputs.bind(repository / 'compatibility/capabilities.json'))
    capability = capabilities['current_runtime_profile']['current_image_qualification']
    acceptance = json.loads(inputs.bind(repository / 'compatibility/full-acceptance.json'))
    active = acceptance['current_runtime_integration']
    plan = json.loads(inputs.bind(repository / active['qualification_plan']))
    identity = dict(source_sha256=source, native_sha256=image,
                    manifest_sha256=sha(manifest_raw), functions=count, units=units)
    require(capability['candidate_identity'] == active['candidate_identity'] == identity
            and capability['required_groups'] == active['required_cases'] == plan['required_suites'] == cases
            and tuple(active['required_suites']) == groups
            and active['required_current_image_cases'] == plan['required_cases'] == total
            and active['required_current_image_groups'] == plan['required_groups'] == len(groups)
            and plan['required_functions'] == count and plan['required_units'] == units
            and plan['required_source_sha256'] == source and plan['required_native_sha256'] == image
            and plan['required_manifest_sha256'] == sha(manifest_raw),
            'active capability, acceptance or qualification plan differs')
    workflow = inputs.bind(repository / '.github/workflows/current-runtime.yml').decode()
    match = re.search(r"^ +python3 -I -S -B - <<'SUMMARY'\n(.*?)^ +SUMMARY$", workflow, re.M | re.S)
    require(match is not None, 'complete current workflow summary Python is required')
    summary = ast.parse(textwrap.dedent(match[1]))
    expected = {'required_cases_total': total, 'executed_cases_total': total,
                'required_suites': list(groups), 'functions': count, 'units': units,
                'source_sha256': source, 'native_sha256': image}
    observed = {key: [] for key in expected}
    for block in summary.body:
        if not isinstance(block, ast.Try):
            continue
        for branch in block.body:
            if not isinstance(branch, ast.If):
                continue
            comparisons = branch.test.values if isinstance(branch.test, ast.BoolOp) else [branch.test]
            relevant = [node for node in comparisons
                        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript)
                        and isinstance(node.left.value, ast.Name) and node.left.value.id == 'qualification'
                        and isinstance(node.left.slice, ast.Constant) and node.left.slice.value in expected]
            if not relevant:
                continue
            require(isinstance(branch.test, ast.BoolOp) and isinstance(branch.test.op, ast.Or)
                    and len(relevant) == len(comparisons) and not branch.orelse,
                    'workflow identity/count checks must be direct complete OR failure guards')
            require(len(branch.body) == 1 and isinstance(branch.body[0], ast.Expr),
                    'workflow identity/count checks must directly record failure')
            failure = branch.body[0].value
            require(isinstance(failure, ast.Call) and isinstance(failure.func, ast.Attribute)
                    and isinstance(failure.func.value, ast.Name) and failure.func.value.id == 'failures'
                    and failure.func.attr == 'append' and len(failure.args) == 1
                    and isinstance(failure.args[0], ast.Constant) and isinstance(failure.args[0].value, str)
                    and not failure.keywords, 'workflow failure predicate is disconnected')
            for node in relevant:
                require(len(node.ops) == len(node.comparators) == 1 and isinstance(node.ops[0], ast.NotEq),
                        'workflow qualification comparisons must remain exact inequalities')
                observed[node.left.slice.value].append(ast.literal_eval(node.comparators[0]))
    require(all(observed[key] == [value] for key, value in expected.items()),
            'workflow exact qualification inventory/identity predicates differ')
    require(ast.dump(summary.body[-1]) == ast.dump(ast.parse('sys.exit(1 if failures else 0)').body[0]),
            'workflow must reject every recorded failure in its final exit')
    return {'groups': len(groups), 'cases': total, 'functions': count, 'units': units}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not sys.flags.optimize, 'all original source assertions must remain enabled')
    repository, build = args.repository.resolve(), args.build_directory.resolve()
    require(repository == ROOT, 'run the checker belonging to the actual checkout')
    require(re.fullmatch(r'[0-9a-f]{64}', args.expected_build_sha256) is not None,
            'exact cold-build receipt SHA is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), 'preserve every original preparation result')
    sys.addaudithook(prohibit_execution)
    sys.dont_write_bytecode = True
    inputs = Inputs()
    inputs.bind(Path(__file__).resolve())
    original_get_code = importlib.machinery.SourceFileLoader.get_code

    def bound_source_code(loader, fullname):
        path = Path(loader.path).absolute()
        if path.is_relative_to(repository) and path.suffix == '.py':
            return compile(inputs.bind(path), str(path), 'exec', dont_inherit=True, optimize=0)
        return original_get_code(loader, fullname)

    # This also covers ordinary imports and spec.loader.exec_module inside the
    # original helpers. Every checkout module executes its bound source bytes.
    importlib.machinery.SourceFileLoader.get_code = bound_source_code
    sys.path.insert(0, str(repository / 'tests'))
    import qualify_current_runtime as coordinator
    inputs.bind(repository / 'tests/qualify_current_runtime.py')
    report = dict(schema='heptatorch.current-source-admission.preparation.v1', status='FAIL',
                  runtime_executed=False, Torch_executed=False,
                  runtime_qualification_granted=False, full_pytorch_rewrite_complete=False,
                  checks=[], actual_build_sha256=args.expected_build_sha256)
    try:
        built = coordinator.current_profile.admit_build(build, inputs.bind, args.expected_build_sha256)
        report.update(source_sha256=built['source_sha256'], native_sha256=built['native_sha256'])
        checks = [('complete-cold-build', lambda: built),
                  ('actual-extension-case-bindings', lambda: extension_case_bindings(repository, inputs)),
                  ('complete-cat-source-chain', lambda: coordinator.cat_source_binding(build, inputs)),
                  ('generator-source-chain', lambda: load(repository / 'tests/current_runtime_generator_binding.py', inputs).parent_source_files(repository, inputs)),
                  ('actual-successor-entry-bindings', lambda: successor_entry_bindings(repository, inputs, coordinator.current_profile.FUNCTIONS)),
                  ('metadata10-source-admission', lambda: load(repository / 'tests/current_runtime_metadata10/common.py', inputs).admit(repository, build, args.expected_build_sha256, inputs))]
        for name in ('public_f64_arithmetic', 'same_f64_arithmetic', 'typed_relu', 'half_mul', 'half_addsub'):
            function = getattr(coordinator, name + '_observer_binding')
            checks.append((name + '-complete-observer-inverse', lambda function=function: function(inputs)))
        for name in ('numeric_predicates', 'signbit', 'typed_index_select', 'indexed_f32', 'forward_nograd', 'join_forward', 'spatial_forward'):
            module = load(repository / 'tests' / name / 'run.py', inputs)
            checks.append((name + '-source-profile', lambda module=module:
                           module.profile(repository, build, args.expected_build_sha256, inputs)))
        for name in ('public_f64_arithmetic', 'same_f64_arithmetic'):
            module = load(repository / 'tests' / name / 'run.py', inputs)
            checks.append((name + '-source-inverse', lambda module=module: module.source_inverse(built, inputs)))
        checks.append(('resource-source-chain', lambda:
                       load(repository / 'tools/current_runtime/fusion_resource_transition.py', inputs).prove(
                           inputs.bind(build / 'runtime.hepta'), inputs.bind)))
        for name, check in checks:
            check()
            report['checks'].append({'name': name, 'status': 'PASS'})
        with coordinator.public_f64_logical192.frozen_reader(repository, inputs) as reader:
            require(reader.run.SOURCE_SHA == built['source_sha256']
                    and reader.run.IMAGE_SHA == built['native_sha256'],
                    'actual entered public192 reader has a stale identity')
        report['checks'].append({'name': 'entered-public192-complete-reader-derivation', 'status': 'PASS'})
        report['identities'] = identity_bindings(repository, inputs, built['source_sha256'],
                                                 built['native_sha256'], built['manifest_sha256'])
        report['contracts'] = contracts(repository, inputs)
        report['inventory'] = inventory(repository, inputs, coordinator)
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if path.suffix == '.py' and path.is_relative_to(repository):
                    inputs.bind(path)
        require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
                'source preparation imported Torch')
        require(not inputs.recheck(), 'bound source inputs changed during preparation')
        report.update(status='PASS_SOURCE_ADMISSION_ONLY', all_inputs_unchanged=True)
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    report['inputs'] = inputs.records
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('inputs', 'traceback')}, sort_keys=True))
    return 0 if report['status'] == 'PASS_SOURCE_ADMISSION_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
