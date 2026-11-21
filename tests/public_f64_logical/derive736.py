"""Derive current771 identities while preserving the complete original192 suite.

Only source bytes are transformed here. This module never runs a native image,
compiler, linker, or Torch producer, and supplies no runtime qualification.
"""
import ast
import hashlib

SOURCE = 'f534c77069abc15ba15879d44492aacbc5f3709f6a861a0f0b9e0e365232fca1'
IMAGE = 'db935c74bf0a2b359038c7be56f5a5721af8f0c2d757a2330527bd751861ae86'
ORIGINAL_FILES = {
    'cases.py': 'e689699b519e57eec44edac5d4c319d3ea718a3777056d1514d1f0e43a08bef3',
    'common.py': '29d25567c08e309c25d96d56f4476e3842ceda164c9647eb5c9397a04fad181a',
    'native.py': 'cc48760e9dad13dfdd834dca5086bc5fb96803dc29914d43b20f29cfb239320e',
    'qualify.py': '98b150d9295451c3be9ff4209d9f42350430e46ec0c3cd42ab3e84eab91b84f2',
    'reference.py': '2531920b4b13c991b33a3608ea2c168b184e4e7d6891b870ae874e95b1c9e296',
    'run.py': 'a155cea27556e7c8647835f11641b4732557dd882d3ec024820a06cd96aeab5f',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def definition(raw, name):
    lines = raw.decode().splitlines(keepends=True)
    nodes = [node for node in ast.parse(raw).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    require(len(nodes) == 1, 'one complete source definition: ' + name)
    node = nodes[0]
    return ''.join(lines[node.lineno - 1:node.end_lineno]).encode()


def change(before, edits):
    after = before
    for old, new in edits:
        require(after.count(old.encode()) == 1, 'one exact forward source span: ' + old)
        after = after.replace(old.encode(), new.encode(), 1)
    restored = after
    for old, new in reversed(edits):
        require(restored.count(new.encode()) == 1, 'one exact inverse source span: ' + new)
        restored = restored.replace(new.encode(), old.encode(), 1)
    require(restored == before, 'complete original file inverse')
    ast.parse(after)
    return after, dict(before_bytes=len(before), before_sha256=sha(before),
        after_bytes=len(after), after_sha256=sha(after), edits=edits,
        whole_file_inverse_exact=True)


def derive(original, author_raw):
    require(set(original) == set(ORIGINAL_FILES), 'all six original source entities are required')
    for name, pin in ORIGINAL_FILES.items():
        require(sha(original[name]) == pin, 'complete original file changed: ' + name)
    ast.parse(author_raw)
    outputs, proof = {}, {}
    outputs['qualify736.py'], proof['qualify'] = change(original['qualify.py'], [
        ('import run\n', 'import run736 as run\n'),
        ("str(HERE / 'run.py')", "str(HERE / 'run736.py')"),
        ("len(result['cold_admission_inputs']) == 69", "len(result['cold_admission_inputs']) == 73"),
        ('native observer omitted one of the 69 physical cold-admission inputs',
         'native observer omitted one of the 73 physical cold-admission inputs'),
    ])
    anchor = "    'native.py': 'cc48760e9dad13dfdd834dca5086bc5fb96803dc29914d43b20f29cfb239320e',\n"
    extra = ''.join('    ' + repr(name) + ': ' + repr(pin) + ',\n' for name, pin in (
        ('run.py', ORIGINAL_FILES['run.py']), ('qualify.py', ORIGINAL_FILES['qualify.py']),
        ('qualify736.py', sha(outputs['qualify736.py'])), ('derive736.py', sha(author_raw))))
    outputs['run736.py'], proof['run'] = change(original['run.py'], [
        ('04ae430e3e851831fcb97a3fab700b16b6318361d52e90fcea90d2183d1d564b', SOURCE),
        ('4749413ea952bfd9d1fe41770f94ce1b9b9e3efabb407aa90f725cf33d59c4db', IMAGE),
        ("built['functions'] == len(bodies) == 729", "built['functions'] == len(bodies) == 771"),
        ("built['units_planned'] == 14", "built['units_planned'] == 15"),
        ("len(cold_inputs.records) == 69", "len(cold_inputs.records) == 73"),
        ('all 69 physical cold-build inputs are required', 'all 73 physical cold-build inputs are required'),
        ('exact integrated 729-function source and image', 'exact integrated 771-function source and image'),
        (anchor, anchor + extra),
    ])
    preserved = {}
    for original_name, derived_name, changed_bodies in (
            ('run.py', 'run736.py', {'admit'}), ('qualify.py', 'qualify736.py', {'main', 'read_back'})):
        nodes = [node for node in ast.parse(original[original_name]).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        rows = {}
        for node in nodes:
            if node.name in changed_bodies:
                continue
            before = definition(original[original_name], node.name)
            require(before == definition(outputs[derived_name], node.name),
                    'entire original predicate/loop definition: ' + node.name)
            rows[node.name] = dict(bytes=len(before), sha256=sha(before))
        preserved[original_name] = rows
    # The native orchestration body has no edits, including all192 cases,
    # 18 packet controls, both actual OOM retries, and the two scope controls.
    require(definition(original['run.py'], 'main') == definition(outputs['run736.py'], 'main'),
            'complete original native192 main is byte-identical')
    for name in ('admit_reference', 'scope_controls', 'arguments'):
        require(definition(original['run.py'], name) == definition(outputs['run736.py'], name),
                'complete original reference/scope/CLI implementation: ' + name)
    for name in ('run_native', 'split_frames', 'request_commands', 'read_child'):
        require(definition(original['qualify.py'], name) == definition(outputs['qualify736.py'], name),
                'complete original process/readback implementation: ' + name)
    original_readback = definition(original['qualify.py'], 'read_back')
    current_readback = definition(outputs['qualify736.py'], 'read_back')
    expected_readback, readback_proof = change(original_readback, [
        ("len(result['cold_admission_inputs']) == 69", "len(result['cold_admission_inputs']) == 73"),
        ('native observer omitted one of the 69 physical cold-admission inputs',
         'native observer omitted one of the 73 physical cold-admission inputs'),
    ])
    require(current_readback == expected_readback,
            'complete original readback retains every comparison; only cold input count changes')
    proof.update(schema='heptatorch.public-f64-logical192.current736.source-derivation.v1',
        source_sha256=SOURCE, native_sha256=IMAGE, functions=771, units=15,
        original_files=ORIGINAL_FILES, derivation_source_sha256=sha(author_raw),
        original_complete_definitions=preserved, original_six_files_modified=False,
        original_reference_Python_and_producer_identities_changed=False,
        added_worker_source_bindings=['run.py', 'qualify.py', 'qualify736.py', 'derive736.py'],
        run_wrapper_also_bound_by_original_self_input=True,
        original_cold_input_count=69, current_cold_input_count=73,
        cold_input_count_only_readback_inverse=readback_proof,
        cold_input_identity_and_extent_predicates_changed=False,
        numeric_or_case_or_control_predicates_changed=False,
        native_boundary_controls_counted_as_Torch_compatibility=False,
        runtime_PASS_transferred=False, native_executed=False, Torch_executed=False, compiler_executed=False)
    return outputs, proof
