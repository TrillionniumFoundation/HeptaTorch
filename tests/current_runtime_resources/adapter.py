#!/usr/bin/env python3
"""Bind seven frozen resource recipes to the current HIA runtime.

The original packages remain untouched. Only profile admission is derived in
memory; every edit has an exact whole-file inverse. Actual resource recipes,
owner equations, references, allocation thresholds and recovery expectations
remain the original code. A separate parent supervises this worker.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys
import traceback
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = '24011e3c861e3b0e77c6a25c07116375cded2225c1f05f42cd84aa1e4a139215'
IMAGE = '52e89b1cb6f4b859d3c9c2f5b528c8c22ddd983e21990a4a4dd5357625c3aba2'
SOURCES = '0cc4429fb953e1e430db63465e16b62ed5baa43f6659d706331a54667058a86d'
FUSION_RESOURCE_SHA = '5c127c42cc8a5d27107715d4d47c1a445c6b6c055dfc2fba9367cd078560df15'
R4_SOURCE = 'c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a'
R4_IMAGE = '0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74'
R3_SOURCE = '3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6'
R3_IMAGE = '7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221'
R4 = 'tests/wrapped_scalar_resource_review/portable07b'
BACKWARD = 'tests/backward_private_cache_resource_review'
R3 = 'tests/differentiable_view_preserve_resource_review'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def definitions(raw):
    text = raw.decode()
    return {node.name: ast.get_source_segment(text, node).encode()
            for node in ast.parse(text).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def functions(raw):
    rows = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', raw, re.M | re.S))
    result = {row[1].decode(): row[0] for row in rows}
    require(len(result) == len(rows), 'duplicate HIA function')
    return result


class Adapter:
    def __init__(self, repository, build, evidence, expected_build_sha):
        self.repo, self.build, self.out = repository.resolve(), build.resolve(), evidence.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        require(re.fullmatch(r'[0-9a-f]{64}',expected_build_sha) is not None,
                'explicit actual cold-build receipt SHA is required')
        self.expected_build_sha = expected_build_sha
        self.inputs, self.edits, self.obligations = {}, [], []
        self.current = None
        self.pins = json.loads(self.bind(HERE / 'sources.json', SOURCES))
        require(self.pins['target'] == dict(source_sha256=SOURCE, native_sha256=IMAGE,
                functions=740, units=14), 'target identity changed')
        self.bind(Path(__file__).resolve())
        for relative, row in self.pins['files'].items():
            path = self.repo / relative
            raw = self.bind(path, row['sha256'])
            require(len(raw) == row['bytes'] and stat.S_IMODE(path.stat().st_mode) == row['mode'],
                    'frozen original size or mode changed: ' + relative)
        self.current = self.plain(self.repo / 'tools/current_runtime/qualification.py',
                                  'resource7_current_qualification')
        require((self.current.SOURCE, self.current.IMAGE, self.current.FUNCTIONS, self.current.UNITS)
                == (SOURCE, IMAGE, 740, 14), 'current qualification profile changed')
        self.actual = self.current.admit_build(self.build, self.bind, self.expected_build_sha)
        self.hia = functions(self.bind(self.build / 'runtime.hepta', SOURCE))
        require(len(self.hia) == 740, 'actual HIA function denominator changed')
        helper_path = self.repo / 'tools/current_runtime/fusion_resource_transition.py'
        self.bind(helper_path, FUSION_RESOURCE_SHA)
        source_helper = self.plain(helper_path, 'resource740_actual_source_paths')
        self.fusion_source_path_proof = source_helper.prove(
            self.bind(self.build / 'runtime.hepta', SOURCE), self.bind)

    def bind(self, path, pin=None):
        path = Path(path).absolute()
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                'unique regular input required: ' + str(path))
        raw = path.read_bytes()
        after = path.stat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_mtime_ns', 'st_ctime_ns')
        require(all(getattr(before, key) == getattr(after, key) for key in fields),
                'input changed while reading: ' + str(path))
        row = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(after.st_mode))
        require(pin is None or row['sha256'] == pin, 'input SHA mismatch: ' + str(path))
        require(str(path) not in self.inputs or self.inputs[str(path)] == row,
                'bound input changed: ' + str(path))
        self.inputs[str(path)] = row
        return raw

    def plain(self, path, name):
        raw = self.bind(path)
        mod = types.ModuleType(name)
        mod.__file__ = str(path)
        sys.modules[name] = mod
        exec(compile(raw, str(path), 'exec'), mod.__dict__)
        return mod

    def derive(self, path, name, edits=(), allowed_definitions=(), inject=None, existing=None):
        before = self.bind(path)
        after = before
        rows = []
        for label, old, new, count in edits:
            old = old.encode() if isinstance(old, str) else old
            new = new.encode() if isinstance(new, str) else new
            require(old and new and old != new and after.count(old) == count,
                    'non-exact source edit: ' + label)
            after = after.replace(old, new, count)
            rows.append(dict(label=label, before_hex=old.hex(), after_hex=new.hex(), count=count))
        inverse = after
        for row in reversed(rows):
            old, new = bytes.fromhex(row['before_hex']), bytes.fromhex(row['after_hex'])
            require(inverse.count(new) == row['count'], 'inverse is not unique: ' + row['label'])
            inverse = inverse.replace(new, old, row['count'])
        require(inverse == before, 'whole source inverse differs')
        old_defs, new_defs = definitions(before), definitions(after)
        require(old_defs.keys() == new_defs.keys(), 'definitions were added or removed')
        changed = [key for key in old_defs if old_defs[key] != new_defs[key]]
        require(set(changed) == set(allowed_definitions), 'unexpected changed definitions: ' + repr(changed))
        # All owner predicates are retained apart from literal identity checks.
        output = self.out / (name + '.py')
        output.write_bytes(after)
        proof = dict(original_path=str(path), original_sha256=sha(before), derived_path=output.name,
                     derived_sha256=sha(after), edits=rows, whole_byte_inverse_exact=True,
                     changed_definitions=changed,
                     unchanged_definitions={key:sha(value) for key,value in old_defs.items() if key not in changed})
        self.edits.append(proof)
        mod = existing or types.ModuleType(name)
        mod.__file__ = str(path)
        sys.modules[name] = mod
        if inject:
            mod.__dict__.update(inject)
        exec(compile(after, str(output), 'exec'), mod.__dict__)
        return mod

    def package_checks(self, group):
        checks = []
        if group in ('r4', 'backward'):
            root = self.repo / 'tests/wrapped_scalar_resource_review'
            validator = self.plain(root / 'validate_package.py', 'resource7_r4_validator')
            checks.append(validator.verify_package())
        if group == 'backward':
            validator = self.plain(self.repo / BACKWARD / 'validate_package.py', 'resource7_backward_validator')
            checks.append(validator.verify_package(self.repo))
        if group == 'r3':
            root = self.repo / R3
            manifest = json.loads(self.bind(root / 'source-manifest.json'))
            package = json.loads(self.bind(root / 'package-manifest.json'))
            require({str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
                    == set(package['files']) | {'package-manifest.json'}, 'R3 member set changed')
            require(set(package['files']) == set(manifest['files']) | {'source-manifest.json'}
                    and package['source02_manifest_sha256']==sha((root/'source-manifest.json').read_bytes()),
                    'R3 outer package and inner source manifest disagree')
            for relative, row in manifest['files'].items():
                raw = self.bind(root / relative, row['sha256'])
                require(len(raw) == row['bytes'] and (root / relative).stat().st_mode & 511 == row['mode'],
                        'R3 member mode or size changed')
            checks.append(dict(status='PASS_SOURCE_CUSTODY_ONLY', original_payloads=len(manifest['files']),
                               native_executed=False, all_members_bound=True))
        dump(self.out / 'original-package-checks.json', checks)

    def owners(self, group):
        root = self.repo / (R3 if group == 'r3' else R4)
        old_source, old_image = (R3_SOURCE, R3_IMAGE) if group == 'r3' else (R4_SOURCE, R4_IMAGE)
        state_tag, backing_tag = ('HTVAD003', 'HTOVAD03') if group == 'r3' else ('HTVSHP02', 'HTOVSHP2')
        old_profile = 'differentiable-view-r3-627' if group == 'r3' else 'wrapped-scalar07b-658'
        raw = self.bind(root / 'owner_view05.py')
        edits = [(label, old, new, raw.count(old.encode())) for label, old, new in (
            ('source identity', old_source, SOURCE), ('native identity', old_image, IMAGE),
            ('profile name', old_profile, 'current740-resource7-' + group),
            ('state tag', state_tag, 'HTSINT02'), ('backing tag', backing_tag, 'HTOSIN02'))]
        self.derive(root / 'owner_view05.py', 'owner_view05', edits,
                    ('pin_profile', 'decode_owner'))
        self.derive(root / 'owner_state.py', 'owner_state', (
            ('state tag', state_tag, 'HTSINT02', 1), ('backing tag', backing_tag, 'HTOSIN02', 1)), ('State',))
        self.plain(root / 'support.py', 'support')
        self.plain(root / 'case_transport.py', 'case_transport')

    def admit(self, args, custody):
        require(args.native_build.resolve() == self.build, 'worker build changed')
        if hasattr(args,'expected_build_sha256'):
            require(args.expected_build_sha256 == self.expected_build_sha,
                    'driver and current admission must bind the same actual receipt')
        build = self.current.admit_build(self.build, custody.bind, self.expected_build_sha)
        for path, row in self.inputs.items():
            custody.bind(Path(path), row['sha256'])
        raw = custody.bind(self.build / 'runtime.hepta', SOURCE)
        parsed = functions(raw)
        require(len(parsed) == 740, 'current function count')
        binary = self.build / 'heptatorch'
        custody.bind(binary, IMAGE)
        return parsed, build, binary

    def r4_obligation(self, parsed, name, old_pin):
        reviewed = self.fusion_source_path_proof['obligations']
        if name in reviewed:
            row = reviewed[name]
            require(row['original_sha256'] == old_pin
                    and row['actual_sha256'] == sha(parsed[name])
                    and row['conditional_dtype_domain'] == [0,1,2,3],
                    'exact actual740 body and unchanged original owner domain: ' + name)
            if name == 'to_command':
                route = row['predicate_entry_route']
                delegation = route['old_path_delegation']
                require(route['actual_entry_sha256'] == sha(parsed['entry'])
                    and route['actual_predicate_command_sha256'] == sha(parsed['predicate_command'])
                    and route['actual_to_command_sha256'] == row['actual_sha256']
                    and delegation['actual_body_sha256'] == row['actual_sha256']
                    and delegation['target'] == 'to_command'
                    and delegation['parameter_count'] == 6
                    and delegation['argument_order_preserved'] is True
                    and delegation['complete_parent_body_unchanged'] is True
                    and route['entry_original_delegation_opcodes'] == list(range(108, 120))
                    and route['new_predicate_branch_reachable_for_original_opcodes'] is False
                    and route['historical_body_substituted_for_actual_body'] is False,
                    'actual740 entry and predicate dispatcher preserve the physical original R4 route')
            self.obligations.append(row)
            return
        if name != 'to_command':
            require(sha(parsed[name]) == old_pin, 'original HIA obligation changed: ' + name)
            self.obligations.append(dict(name=name, actual_sha256=sha(parsed[name]), original_sha256=old_pin,
                                         relation='byte-identical'))
            return
        body = parsed['sd_previous_to_command']
        normalized = body.replace(b'fn sd_previous_to_command(', b'fn to_command(', 1)
        require(sha(normalized) == old_pin, 'complete old dispatcher is not retained')
        front = parsed['to_command'].split(b'block b6(', 1)[0]
        require(sha(front) == '56bbfa7109fe9774deb3ab7b3485fb970ed1433157f0690695197dc370ea74a6', 'new dispatcher prefix changed')
        # The pinned prefix reads only the opcode, handles 115..119, and delegates
        # every other value with all six arguments unchanged. It performs no write.
        require(re.findall(rb'\{"function":"([^"]+)"\}', front) ==
                [b'getr', b'sd_permute_command', b'sd_accumulator_command', b'sd_previous_to_command'],
                'dispatcher front call set changed')
        self.obligations.append(dict(name=name, actual_sha256=sha(parsed[name]), original_sha256=old_pin,
            preserved_function='sd_previous_to_command', preserved_body_sha256=sha(body),
            exact_function_name_inverse=True, prefix_sha256=sha(front),
            special_opcodes=[115,116,117,118,119], relation='unchanged delegation for all original commands'))

    def r4_driver(self, existing=None):
        path = self.repo / R4 / 'test_wrapped_scalar_resources.py'
        raw = self.bind(path).decode()
        begin = raw.index("    raw = custody.bind(a.native_build / 'runtime.hepta', SOURCE)\n")
        end = raw.index("    evidence = json.loads(custody.bind(HERE / 'source-obligations.json'))\n", begin)
        edits = [('source identity', R4_SOURCE, SOURCE, 1), ('native identity', R4_IMAGE, IMAGE, 1),
            ('actual current cold-build admission', raw[begin:end],
             '    functions, build, binary = resource7_admit(a, custody)\n', 1),
            ('actual HIA obligations plus unchanged dispatcher delegation',
             '        assert sha(functions[name]) == pin, name\n',
             '        resource7_r4_obligation(functions, name, pin)\n', 1)]
        return self.derive(path, 'resource7_r4_driver', edits, ('load_inputs',),
            dict(resource7_admit=self.admit, resource7_r4_obligation=self.r4_obligation), existing)

    def backward_driver(self):
        path = self.repo / BACKWARD / 'driver.py'
        holder = {}
        def original(directory):
            require(directory.resolve() == (self.repo / R4).resolve(), 'original parent path changed')
            loaded = holder['module'].original_module(directory)
            return self.r4_driver(loaded)
        edits = [('source identity', R4_SOURCE, SOURCE, 1), ('native identity', R4_IMAGE, IMAGE, 1),
            ('bind unchanged parent then derive current admission',
             'mod=original_module(a.resource_source);custody=mod.Custody()',
             'mod=resource7_original(a.resource_source);custody=mod.Custody()', 1),
            ('retain original obligation source identity separately',
             "guard(obligations['source_sha256']==SOURCE,'source-obligations-pin')",
             "guard(obligations['source_sha256']==resource7_original_source,'source-obligations-pin')", 1)]
        mod = self.derive(path, 'resource7_backward_driver', edits, ('main',),
                          dict(resource7_original=original, resource7_original_source=R4_SOURCE))
        holder['module'] = mod
        return mod

    def r3_driver(self):
        path = self.repo / R3 / 'test_preserve_resources.py'
        raw = self.bind(path).decode()
        begin = raw.index("    native=a.native_build.resolve();image=native/'tensor-differentiable-view-session'\n")
        end = raw.index('    assert len(functions)==627\n', begin) + len('    assert len(functions)==627\n')
        edits = [('source identity', R3_SOURCE, SOURCE, 1), ('native identity', R3_IMAGE, IMAGE, 1),
            ('retain original historical profile identity separately',
             "assert profile['source_sha256']==SOURCE and profile['native_sha256']==IMAGE",
             "assert profile['source_sha256']==resource7_original_source and profile['native_sha256']==resource7_original_image", 1),
            ('actual current cold-build admission', raw[begin:end],
             '    functions,info,image=resource7_admit(a,custody)\n', 1),
            ('explicit actual allocation-path proof for two preserved aliases',
             "    for name,pin in budget['source_sha256'].items():assert sha(functions[name])==pin\n",
             '    resource7_budget_proof(functions,budget,c,state,nodes)\n', 1)]
        mod = self.derive(path, 'resource7_r3_driver', edits,
            ('load_inputs','preserve_capacity_oom_recovery'), dict(resource7_admit=self.admit,
                resource7_original_source=R3_SOURCE, resource7_original_image=R3_IMAGE,
                resource7_budget_proof=self.r3_budget_proof))
        # The sole recipe edit above changes only HIA source admission. Reverse it
        # independently and require the complete original recipe bytes, including
        # every numeric bound, comparison, rollback and recovery assertion.
        old = definitions(self.bind(path))['preserve_capacity_oom_recovery']
        derived = (self.out / 'resource7_r3_driver.py').read_bytes()
        new = definitions(derived)['preserve_capacity_oom_recovery']
        restored = new.replace(b'    resource7_budget_proof(functions,budget,c,state,nodes)\n',
            b"    for name,pin in budget['source_sha256'].items():assert sha(functions[name])==pin\n",1)
        require(restored == old, 'R3 resource recipe changed beyond source-binding admission')
        return mod

    def r3_budget_proof(self, parsed, budget, case, state, nodes):
        require(set(budget['source_sha256']) == {
            'gradient_workspace_size','history_prepare_leaf','gradient_new','descriptor',
            'typed_descriptor','descriptor_page','alloc','storage_capacity',
            'dview_previous_mark_preserve_targets','incoming_mark_preserve_targets'},
            'original allocation source denominator changed')
        wrappers = {
            'typed_descriptor': ('scalar_previous_typed_descriptor',
                '04cfe7f40cf76c7370c5cbab816341b354bf955195affcd56bbebd517c1a25c5',
                '41358513e833d2299683cdfe003221a15671f3562c65989a88059f74151710e8'),
            'incoming_mark_preserve_targets': ('scalar_previous_incoming_mark_preserve_targets',
                '8c4c0237279b24a1ce676535fe8f81884b7951d15dc1df7760731f904ae55fa7',
                'c45e68298da075eb7d81533230b54e7f0b0340a007d5a35cbc57d0f05476d619'),
        }
        facts = []
        for node in range(1,state.next):
            words = state.node(node)
            if words[38]:
                facts.append(dict(node=node,op=words[5],dtype=words[72],epoch=words[21],
                                  numel=words[1],reachable=node in nodes))
        require(nodes and facts and all(row['dtype']==0 and row['op'] in (0,44) for row in facts),
                'the reviewed allocation path requires actual F32 op0/op44 descriptors')
        require({row['node'] for row in facts if row['reachable']}==set(nodes),
                'every reachable node must be a live descriptor')
        target = case.out / 'actual-allocation-source-proof'
        target.mkdir()
        rows = []
        for name,pin in budget['source_sha256'].items():
            body = parsed[name]
            (target/(name+'.hepta')).write_bytes(body)
            if name not in wrappers:
                require(sha(body)==pin, 'original R3 allocation function changed: '+name)
                rows.append(dict(name=name,actual_sha256=sha(body),original_sha256=pin,relation='byte-identical'))
                continue
            alias,wrapper_pin,alias_pin = wrappers[name]
            require(sha(body)==wrapper_pin and sha(parsed[alias])==alias_pin,
                    'reviewed wrapper or retained alias changed: '+name)
            original = parsed[alias].replace(('fn '+alias+'(').encode(),('fn '+name+'(').encode(),1)
            require(sha(original)==pin, 'complete original allocation body inverse failed: '+name)
            (target/(alias+'.hepta')).write_bytes(parsed[alias])
            row = dict(name=name,actual_sha256=sha(body),original_sha256=pin,
                alias=alias,alias_sha256=alias_pin,exact_function_name_inverse=True)
            if name=='typed_descriptor':
                require(re.findall(rb'\{"function":"([^"]+)"\}',body)==
                    [b'terror',b'scalar_previous_typed_descriptor'], 'typed wrapper call set changed')
                row.update(relation='dtype0 makes the sole new dtype3 rejection false; direct unchanged delegation',
                    wrapper_writes_or_allocations_on_actual_path=False)
            else:
                require(b'{"function":"alloc"}' not in body,
                        'new marker scan must not allocate')
                row.update(relation='original marker executes first; op3-only new marker is unreachable for actual op0/op44 nodes',
                    extra_scan_writes_or_allocations_on_actual_path=False)
            rows.append(row)
        proof = dict(original_budget_sha256=sha(self.bind(self.repo/R3/'oom-budget.json')),
            actual_source_sha256=SOURCE,actual_native_sha256=IMAGE,
            original_numeric_budget_unchanged=True,actual_live_descriptors=facts,
            actual_reachable_nodes=sorted(nodes),function_count=10,functions=rows,
            no_old_function_bytes_substituted=True)
        dump(target/'proof.json',proof)
        case.row['current_allocation_path_proof']=proof
        self.obligations.extend(rows)

    def finish(self):
        for path, row in list(self.inputs.items()):
            self.bind(Path(path), row['sha256'])
        dump(self.out / 'source-bindings.json', dict(
            schema='heptatorch.resource7.current-740-adaptation.v1',
            source_sha256=SOURCE, native_sha256=IMAGE, actual_build_sha256=self.expected_build_sha,
            state_format='HTSINT02', backing_format='HTOSIN02', functions=740, units=14,
            edits=self.edits, hia_obligations=self.obligations, inputs=self.inputs,
            fusion_source_path_proof=self.fusion_source_path_proof,
            all_original_inputs_unchanged=True, whole_file_inverse_exact=True,
            original_pass_transferred=False, numeric_expectations_modified=False,
            resource_thresholds_modified=False, owner_equations_modified=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository', type=Path, required=True)
    ap.add_argument('--build-directory', type=Path, required=True)
    ap.add_argument('--expected-current-build-sha256', required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--suite', choices=('r4','backward','r3'), required=True)
    a, rest = ap.parse_known_args()
    require(not sys.flags.optimize, 'assertions must be enabled')
    adapter = None
    code = 1
    try:
        adapter = Adapter(a.repository, a.build_directory, a.evidence, a.expected_current_build_sha256)
        adapter.package_checks(a.suite)
        adapter.owners(a.suite)
        if a.suite == 'r4':
            driver = adapter.r4_driver()
        elif a.suite == 'backward':
            driver = adapter.backward_driver()
        else:
            driver = adapter.r3_driver()
        sys.argv = [str(driver.__file__)] + rest
        code = driver.main()
    except BaseException as error:
        a.evidence.mkdir(parents=True, exist_ok=True)
        dump(a.evidence / 'adapter-error.json', dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
        code = 1
    finally:
        if adapter is not None:
            try:
                adapter.finish()
            except BaseException as error:
                dump(a.evidence / 'finish-error.json', dict(error=repr(error), traceback=traceback.format_exc()))
                traceback.print_exc()
                code = 1
    return code


if __name__ == '__main__':
    raise SystemExit(main())
