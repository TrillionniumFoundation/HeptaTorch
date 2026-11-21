"""Admit exactly42 add/sub successors in the unchanged239-request Half recipe.

Both previous adapters remain byte identical. Their actual rejection closures
and the multiplication owner hook are reused; the terminal partition explicitly
becomes173 refusals,22 multiplications,2 negations and42 additions/subtractions.
The42 include40 single-command requests and two original factory/FE-add frames.
Live transport and independent recorded transport execute this same recipe.
"""
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import types


MUL_HELPER = '47557e7730d00c374864d2e2948a511e86b47ae653034adaa0f776445fc556ad'
NEG_HELPER = '1d5a464ad7ff3ac56656cd361de2024ca77db3ead2e9da6687a043039a3a5b41'
FE_OWNER_CONTRACT = '4c3ce9d59569c398dedc7b7627227c6468548051ce6562a25b863625f0e268c2'
FE_TOMBSTONE_CONTRACT = '9f539928dc5b6f99fedbfc7faad0ef2b88db1b5fb1e2c50ebb0eff16a189f956'
FE_OWNER_READER = '8dac020a73f0c2f65ea85c6477299e2054a3af5243309608dbcaf13058173bcb'
WORDS = {0: (0x3f800000, 0x80000000), 1: (1, 0), 2: (1, 0),
         4: (0x3ff0000000000000, 0x8000000000000000),
         5: (0x3c00, 0x8000), 6: (0x3f80, 0x8000)}
PUBLIC = tuple(row for half in (5, 6) for other in (0, 1, 2, 4, 5, 6)
               for operation in (1, 2)
               for row in ((('half-binary-left-%d-%d-%d' % (half, other, operation),
                             operation, half, other),) if half == other else
                           (('half-binary-left-%d-%d-%d' % (half, other, operation),
                             operation, half, other),
                            ('half-binary-right-%d-%d-%d' % (half, other, operation),
                             operation, other, half))))
SUCCESSORS = tuple(row for row in PUBLIC if not (row[1] == 2 and 1 in row[2:]))
BOOL_REFUSALS = tuple(row for row in PUBLIC if row not in SUCCESSORS)
PRIVATE3_LABELS = tuple(label for half in (5, 6) for operation in (1, 2)
                        for label in ('half-binary-left-%d-3-%d' % (half, operation),
                                      'half-binary-right-%d-3-%d' % (half, operation)))
FE_SUCCESSORS = (('same-frame-half-FE-binary-5', 5), ('same-frame-half-FE-binary-6', 6))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def closure(function):
    return dict(zip(function.__code__.co_freevars,
                    (cell.cell_contents for cell in function.__closure__ or ())))


class Successor:
    def __init__(self, repository, reader_directory, reference_directory,
                 reference_execution, inputs, negation):
        self.repository = Path(repository).resolve(strict=True)
        self.reader = Path(reader_directory).resolve(strict=True)
        self.reference_directory = Path(reference_directory).resolve(strict=True)
        self.reference_execution = Path(reference_execution).resolve(strict=True)
        self.negation, self.multiplication = negation, negation.multiplication
        tests = self.repository / 'tests'
        inputs.bind(Path(__file__).resolve())
        for helper, digest, instance in (
                ('current_runtime_half_mul_successor.py', MUL_HELPER, self.multiplication),
                ('current_runtime_half_neg_successor.py', NEG_HELPER, self.negation)):
            path = tests / helper
            inputs.bind(path, digest)
            require(Path(instance.wrap_entry.__code__.co_filename).resolve() == path
                    and instance.repository == self.repository and instance.reader == self.reader,
                    'unchanged complete predecessor helper from this actual repository')
        for path in sorted((tests / 'half_addsub').glob('*.py')):
            inputs.bind(path)
        contract_path = self.repository / 'tools/current_runtime/half_addsub_fe_owner_contract.json'
        contract = json.loads(inputs.bind(contract_path, FE_OWNER_CONTRACT))
        source = inputs.bind(self.repository / 'tools/current_runtime/source/runtime.hepta',
                             contract['source_sha256'])
        index_path = self.repository / 'tools/current_runtime/index_signbit_source_transition.py'
        inputs.bind(index_path)
        index_spec = importlib.util.spec_from_file_location('index_signbit_current_source', index_path)
        indexing = importlib.util.module_from_spec(index_spec); index_spec.loader.exec_module(indexing)
        source, index_inverse = indexing.restore_source(source, inputs.bind)
        require(index_inverse['functions'] == 781 and index_inverse['parent_functions'] == 775
                and index_inverse['whole_source_inverse_exact'] is True
                and index_inverse['whole_round_trip_exact'] is True,
                'complete781 source restores every original775 byte before earlier source proofs')
        relu_path = self.repository / 'tools/current_runtime/typed_relu_source_transition.py'
        inputs.bind(relu_path)
        relu_spec = importlib.util.spec_from_file_location('half_FE_typed_relu_source_inverse', relu_path)
        relu = importlib.util.module_from_spec(relu_spec); relu_spec.loader.exec_module(relu)
        parent775 = source
        source = inputs.bind(self.repository / 'tools/current_runtime/source/runtime.hepta', indexing.SOURCE)
        same773, relu_source_proof = relu.restore_source(parent775, inputs.bind)
        require(relu_source_proof['functions'] == 775 and relu_source_proof['parent_functions'] == 773
                and relu_source_proof['changed_functions'] == ['unary']
                and relu_source_proof['whole_source_inverse_exact'] is True
                and relu_source_proof['whole_round_trip_exact'] is True,
                'complete typed ReLU775 inverse before the frozen same-F64 source')
        same_path = self.repository / 'tools/current_runtime/same_f64_arithmetic_source_transition.py'
        inputs.bind(same_path)
        same_spec = importlib.util.spec_from_file_location('half_FE_same_f64_source_inverse', same_path)
        same = importlib.util.module_from_spec(same_spec); same_spec.loader.exec_module(same)
        previous773, same_source_proof = same.restore_source(same773, inputs.bind)
        require(same.SOURCE == relu.PARENT_SOURCE
                and same_source_proof['functions'] == same_source_proof['parent_functions'] == 773
                and same_source_proof['changed_functions'] == ['half_mul_priority','public_f64_binary']
                and same_source_proof['Half_routes_unchanged'] is True
                and same_source_proof['whole_source_inverse_exact'] is True
                and same_source_proof['whole_round_trip_exact'] is True,
                'the same-F64 extension preserves the complete original Half FE route')
        arithmetic_path = self.repository / 'tools/current_runtime/public_f64_arithmetic_source_transition.py'
        inputs.bind(arithmetic_path)
        arithmetic_spec = importlib.util.spec_from_file_location('half_FE_public_f64_source_inverse', arithmetic_path)
        arithmetic = importlib.util.module_from_spec(arithmetic_spec); arithmetic_spec.loader.exec_module(arithmetic)
        previous771, arithmetic_source_proof = arithmetic.restore_source(previous773, inputs.bind)
        require(arithmetic_source_proof['functions'] == 773 and arithmetic_source_proof['parent_functions'] == 771
                and arithmetic_source_proof['changed_functions'] == ['binary','half_mul_cast']
                and arithmetic_source_proof['unchanged_parent_functions'] == 769
                and arithmetic_source_proof['preexisting_half_binary_routes_unchanged'] is True
                and arithmetic_source_proof['whole_source_inverse_exact'] is True
                and arithmetic_source_proof['whole_round_trip_exact'] is True,
                'exact mixed773 predecessor source relation preserves the complete Half binary route')
        inverse_path = self.repository / 'tools/current_runtime/half_div_source_transition.py'
        inputs.bind(inverse_path)
        inverse_spec = importlib.util.spec_from_file_location('half_FE_division_source_inverse', inverse_path)
        inverse = importlib.util.module_from_spec(inverse_spec); inverse_spec.loader.exec_module(inverse)
        predecessor_source, division_source_proof = inverse.restore_source(previous771, inputs.bind)
        require(same.PARENT_SOURCE == arithmetic.SOURCE and inverse.SOURCE == arithmetic.PARENT_SOURCE
                and division_source_proof['non_division_routes_unchanged'] is True
                and division_source_proof['whole_round_trip_exact'] is True,
                'complete native767 inverse preserves the original operation1 FE allocation path')
        require(contract['schema'] == 'heptatorch.half-addsub.exact-fe-allocation-path.v2'
                and contract['successful_descriptor_allocations'] == 3
                and contract['live_new_tensors'] == 2
                and contract['unpublished_released_temporaries'] == 1
                and contract['exact_descriptor_order']
                    == ['factory-Half', 'promotion-F32-temporary', 'result-F32']
                and contract['inference_from_fixed_source'] is True
                and contract['native_qualification_transferred'] is False,
                'three allocations derive from the fixed native source, not observed-count adjustment')
        matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', source, re.M | re.S))
        functions = {item[1].decode('ascii'): item[0] for item in matches}
        require(len(functions) == len(matches) == 781, 'complete actual775 function census')
        cast = functions['half_mul_cast']
        require(sha(cast) == arithmetic_source_proof['actual_cast_sha256']
                and b'  integerw:u64 = const() {"value":2}\n  doublew:u64 = const() {"value":4}\n' in cast
                and b'  from_integerw:bool = compare.eq(sw,integerw)\n'
                    b'  to_doublew:bool = compare.eq(tw,doublew)\n'
                    b'  directw:bool = bool.and(from_integerw,to_doublew)\n'
                    b'  branch directw direct_integer(xw) convert(xw,sw,tw,nw)\n' in cast
                and contract['exact_input'] == dict(half_dtypes=[5,6], left_shape=[2], left_words=[1,2],
                    opcodes=[97,2], operation=1, requires_grad=False, right_dtype=0, right_shape=[2],
                    right_words=[1065353216,2147483648]),
                'the actual direct I64-to-F64 guard is false for both fixed original Half/F32 FE inputs')
        calls = {name: [item.decode('ascii') for item in re.findall(
                    rb'call\([^\n]*\)\s+\{"function":"([^"]+)"\}', raw)]
                 for name, raw in functions.items()}
        for name, identity in contract['function_identities'].items():
            require(identity == dict(bytes=len(functions[name]), sha256=sha(functions[name]),
                                     calls=calls[name]),
                    'complete native allocation/publication/cleanup function and ordered call identity')
        unit = contract['support_unit']
        require(unit['path'] == 'tools/current_runtime/source/units/rng03.hepta'
                and set(contract['support_function_identities']) == {'round_f16', 'round_bf16'},
                'the two linked HIA rounding leaves have an explicit source-unit boundary')
        support = inputs.bind(self.repository / unit['path'], unit['sha256'])
        require(len(support) == unit['bytes'], 'complete actual linked HIA support unit')
        support_matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',
                                          support, re.M | re.S))
        support_functions = {item[1].decode('ascii'): item[0] for item in support_matches}
        require(len(support_functions) == len(support_matches) == 18,
                'complete current rng03 HIA function census')
        for name, expected in contract['support_function_identities'].items():
            raw = support_functions[name]
            operations = sorted({item.decode('ascii') for item in re.findall(
                rb'^\s+\w+:\w+\s*=\s*([A-Za-z0-9_.]+)\(', raw, re.M)})
            leaf_calls = [item.decode('ascii') for item in re.findall(
                rb'call\([^\n]*\)\s+\{"function":"([^"]+)"\}', raw)]
            require(name not in functions and not leaf_calls
                    and expected == dict(bytes=len(raw), sha256=sha(raw), calls=leaf_calls,
                                         signature=raw.splitlines()[0].decode('ascii'),
                                         operations=operations)
                    and set(operations) <= {'const', 'bit.and', 'bit.or', 'bool.and', 'bool.or',
                        'compare.eq', 'compare.ne', 'compare.ge', 'compare.gt', 'compare.lt',
                        'int.add.wrap', 'int.sub.wrap', 'int.shl.checked', 'int.shr.checked', 'select'},
                    'ordinary linked scalar HIA leaves have no buffer, owner or allocation operation')
            functions[name], calls[name] = raw, leaf_calls
        for name, expected in contract['pure_subtrees_without_descriptor_constructors'].items():
            visited, pending = set(), [name]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                require(current in calls, 'complete native subtree body is present')
                visited.add(current)
                pending.extend(calls[current])
            require(sorted(visited) == expected
                    and not (visited & {'typed_descriptor', 'scalar_previous_typed_descriptor',
                                        'typed_from_wire', 'scalar_previous_typed_from_wire'}),
                    'every layout/copy/element subtree is independently free of descriptor constructors')
        tombstone_path = self.repository / 'tools/current_runtime/half_addsub_fe_tombstone_contract.json'
        tombstone = json.loads(inputs.bind(tombstone_path, FE_TOMBSTONE_CONTRACT))
        owner_path = tests / 'current_runtime_half_fe_owner.py'
        owner_raw = inputs.bind(owner_path, FE_OWNER_READER)
        require(tombstone['schema'] == 'heptatorch.current781-half-FE-owner-source-contract.v1'
                and tombstone['source'] == dict(functions=781,
                    path='tools/current_runtime/source/runtime.hepta', sha256=sha(source))
                and tombstone['functions_verified'] == len(tombstone['functions']) == 44
                and tombstone['predicate'] == dict(bytes=len(owner_raw),
                    function='verify_fe_promotion_tombstone',
                    path='tests/current_runtime_half_fe_owner.py', sha256=FE_OWNER_READER),
                'fixed source-derived full descriptor, allocator and retirement contract')
        for name, identity in tombstone['functions'].items():
            require(identity['bytes'] == len(functions[name])
                    and identity['sha256'] == sha(functions[name])
                    and identity['calls'] == sorted(set(calls[name])),
                    'complete source body and call identity for every constructor and cleanup predicate')
        spec = importlib.util.spec_from_file_location('current_half_fe_owner', owner_path)
        self.fe_owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fe_owner)
        require(self.fe_owner.SOURCE == sha(source)
                and self.fe_owner.IMAGE == tombstone['native_sha256'],
                'the exact bound FE owner module names the actual source and native image')
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(tests))
            run = importlib.import_module('half_addsub.run')
            cases = importlib.import_module('half_addsub.cases')
        finally:
            sys.path[:] = old_path
        for module in (run, cases):
            require(Path(module.__file__).resolve().parent == tests / 'half_addsub',
                    'new reference modules belong to this actual repository')
        legacy = self.multiplication.legacy
        raw = inputs.bind(self.reader / legacy.ORIGINAL_MEMBER, cases.LEGACY_SOURCE_SHA256)
        functions = [node for node in ast.parse(raw).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'entry_controls']
        require(len(functions) == 1, 'one complete frozen239 original entry recipe')
        node = functions[0]
        entry = b''.join(raw.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
        require(cases.LEGACY_SOURCE_SHA256 == legacy.ORIGINAL_SOURCE_SHA256
                and cases.LEGACY_ENTRY_SHA256 == legacy.ORIGINAL_ENTRY_SHA256 == sha(entry)
                and len(entry) == legacy.ORIGINAL_ENTRY_BYTES == 7497
                and cases.LEGACY_CONTROLS == PUBLIC and cases.SUCCESSORS == SUCCESSORS
                and cases.BOOL_REFUSALS == BOOL_REFUSALS
                and len(PUBLIC) == 44 and len(SUCCESSORS) == 40 and len(BOOL_REFUSALS) == 4,
                'exact original bytes and complete40/4 public add/sub request mapping')
        self.reference = run.read_reference(self.reference_directory,
                                            self.reference_execution, inputs)
        rows = [row for row in self.reference['report']['rows']
                if 'legacy_refusal_label' in row['spec']]
        require(len(rows) == 44 and [row['spec']['legacy_refusal_label'] for row in rows]
                == [item[0] for item in PUBLIC],
                'all44 original public requests have fresh actual Torch observations')
        self.rows, self.dtypes, self.operations, self.bool_rows = {}, {}, {}, {}
        for row, (label, operation, left, right) in zip(rows, PUBLIC):
            accepted = (label, operation, left, right) in SUCCESSORS
            require(row['accepted'] is accepted and row['spec']['name'] == 'historical-' + label
                    and row['spec']['operation'] == operation
                    and row['spec']['same_handle'] is (left == right),
                    'actual reference repeats the exact operation and same-handle relation')
            for key, dtype, observed in zip(('left', 'right'), (left, right), row['inputs']):
                literal = b''.join(word.to_bytes(cases.WIDTHS[dtype], 'little')
                                   for word in WORDS[dtype])
                require(row['spec'][key] == cases.input_spec(dtype, [2], words=WORDS[dtype])
                        and cases.raw_input(row['spec'][key]) == literal
                        and self.reference['blob'](observed['logical']) == literal
                        and observed['shape'] == [2] and observed['strides'] == [1],
                        'the complete original literal operand is observed by actual Torch')
            if accepted:
                self.rows[label], self.dtypes[label] = row, (left, right)
                self.operations[label] = operation
            else:
                require(operation == 2 and 1 in (left, right)
                        and row['spec']['expected_error'] == 'bool_subtraction'
                        and row['error']['type'] == 'NotImplementedError',
                        'four actual Torch Bool subtraction errors retain original rejection')
                self.bool_rows[label] = row
        fe_rows = [row for row in self.reference['report']['rows']
                   if 'legacy_fe_successor_label' in row['spec']]
        require(len(fe_rows) == 2
                and [row['spec']['legacy_fe_successor_label'] for row in fe_rows]
                    == [item[0] for item in FE_SUCCESSORS],
                'both original factory/FE-add requests have fresh actual Torch observations')
        self.fe_rows, self.fe_dtypes = {}, {}
        for row, (label, dtype) in zip(fe_rows, FE_SUCCESSORS):
            require(row['accepted'] is True and row['spec']['operation'] == 1
                    and row['spec'].get('same_handle', False) is False
                    and row['spec']['left'] == cases.input_spec(dtype, [2], words=[1, 2])
                    and row['spec']['right'] == cases.input_spec(0, [2], words=WORDS[0])
                    and self.reference['blob'](row['inputs'][0]['logical']) == struct.pack('<2H', 1, 2)
                    and self.reference['blob'](row['inputs'][1]['logical']) == struct.pack('<2I', *WORDS[0])
                    and row['result']['dtype'] == 'torch.float32',
                    'exact original raw factory words and existing F32 operand, independently observed')
            self.fe_rows[label], self.fe_dtypes[label] = row, dtype
        ordered_labels = [label for half in (5, 6)
                          for label in ([item[0] for item in SUCCESSORS
                                         if item[0].split('-')[3] == str(half)]
                                        + ['same-frame-half-FE-binary-' + str(half)])]
        self.proof = dict(
            schema='heptatorch.half47.addsub42-successor.v1',
            reference_directory=str(self.reference_directory),
            reference_execution=str(self.reference_execution),
            reference_sha256=self.reference['report_sha256'],
            reference_process_sha256=sha(inputs.bind(self.reference_execution / 'process.json')),
            case_sha256=cases.CASE_SHA256,
            original_controls_sha256=cases.LEGACY_SOURCE_SHA256,
            original_entry_sha256=cases.LEGACY_ENTRY_SHA256, original_entry_bytes=7497,
            original_request_count=239, predecessor_refusals=215, retained_refusals=173,
            existing_multiplication_successors=22, existing_negation_successors=2,
            accepted_successors=42, accepted_single_command_successors=40,
            accepted_same_frame_FE_successors=2, accepted_additions=24, accepted_subtractions=18,
            labels=ordered_labels, single_command_labels=[item[0] for item in SUCCESSORS],
            same_frame_FE_labels=[item[0] for item in FE_SUCCESSORS],
            retained_actual_Bool_error_labels=[item[0] for item in BOOL_REFUSALS],
            retained_private3_protocol_refusal_labels=list(PRIVATE3_LABELS),
            multiplication_helper_sha256=MUL_HELPER, negation_helper_sha256=NEG_HELPER,
            both_previous_reject_closures_and_mul_owner_hook_reused_unchanged=True,
            predecessor_terminal_partition_replaced_explicitly=True,
            original_entry_code_object_preserved=True,
            actual_Torch_reference_verified=True, host_expected_values_computed=False,
            original_frozen_recipe_modified=False, every_original_snapshot_field_checked=True,
            scoped_owner_extension='exact40-single-command-and2-original-FE-add',
            original_FE_packets_preserved=True, original_FE_packet_bytes=228,
            FE_allocation_source_contract_sha256=FE_OWNER_CONTRACT,
            FE_tombstone_source_contract_sha256=FE_TOMBSTONE_CONTRACT,
            FE_complete_owner_reader_sha256=FE_OWNER_READER,
            half_div_source_inverse=division_source_proof,
            public_f64_arithmetic_source_inverse=arithmetic_source_proof,
            typed_relu_source_inverse=relu_source_proof,
            same_f64_arithmetic_source_inverse=same_source_proof,
            FE_new_public_F64_fallback_reachable=False,
            FE_new_I64_to_F64_cast_guard_reachable=False,
            FE_original_promotion_and_result_dtype=0,
            FE_exact_source_functions_verified=len(contract['function_identities']),
            FE_exact_tombstone_source_functions_verified=len(tombstone['functions']),
            FE_exact_no_allocation_subtrees_verified=len(contract['pure_subtrees_without_descriptor_constructors']),
            FE_exact_linked_HIA_support_leaves_verified=len(contract['support_function_identities']),
            FE_linked_HIA_support_unit_sha256=contract['support_unit']['sha256'],
            FE_all_three_complete_descriptors_and_all_owner_tables_checked=True,
            FE_source_proved_allocations=3, FE_final_live_new_descriptors=2,
            FE_released_promotion_descriptors=1,
            successful_FE_requires_two_fresh_Tensor_Storage_Version_owners=True,
            complete_original_pre_F64_frame_predicates_reused=True,
            original_single_command_guards_replaced_only_for_the_two_exact_FE_packets=True,
            intermediate_owner_states_inferred=False,
            input_all84words_unchanged_checked=True, original_same_handle_relations_checked=True)

    def observe(self, driver, token, expected, label):
        return type(self.multiplication).observe(self, driver, token, expected, label)

    def exchange(self, driver, label, body, left, right, row):
        """Scope the full existing promoted-owner predicates to this exact request."""
        owner = driver.owner
        original = owner.f64_command_metadata
        operation = self.operations[label]
        expected_command = [2, operation, left, right, 0, 0, 0, 0]
        result_dtype = {'torch.float32': 0, 'torch.float64': 4,
                        'torch.float16': 5, 'torch.bfloat16': 6}[row['result']['dtype']]
        left_dtype, right_dtype = self.dtypes[label]

        def metadata(before, after, command, record):
            require(operation in (1, 2) and command == expected_command and record['opcode'] == 2,
                    'only the exact current add/sub successor enters the owner extension')
            sources = [owner.public_node(before, token) for token in (left, right)]
            observed = [owner.w(before['nodes'][str(index)], 576) for index in sources]
            require(observed == [left_dtype, right_dtype]
                    and any(dtype in (5, 6) for dtype in observed)
                    and all(dtype in (0, 1, 2, 4, 5, 6) for dtype in observed),
                    'exact two public noAD add/sub successor operand types')
            result = owner.public_node(after, record['value'])
            node = owner._live_node(after, result, 'half-addsub-successor-result')
            require(owner.w(node, 576) == result_dtype,
                    'promoted result dtype equals actual fixed Torch')
            for source in sources:
                require(owner.w(before['nodes'][str(source)], 32) == 0,
                        'successor inputs have no gradients')
                owner._f64_source_unchanged(before, after, source)
            if result_dtype in (5, 6):
                role = owner.half_role(after, result)
                owner._half_fresh_result(before, after, result, sources)
            else:
                if result_dtype == 4:
                    role = owner.f64_role(after, result)
                else:
                    require(not any(owner.w(node, field) for field in owner.F64_ZERO_FIELDS)
                            and owner.w(node, 240) == 1,
                            'F32 promoted result has a live Tensor and no AD Saved or view state')
                    role = dict(node=result, dtype=0, width=4,
                                storage=owner.w(node, 216), version=owner.w(node, 224))
                owner._f64_fresh_result(before, after, result, sources)
            return dict(operation=2, binary_operation=operation, role=role,
                        input_dtypes=observed, result_dtype=result_dtype,
                        input_all84words_unchanged=True, exact_successor_label=label,
                        actual_Torch_result_dtype_used=True,
                        numerical_expected_values_computed_here=False)

        owner.f64_command_metadata = metadata
        try:
            return driver.n.exchange(body, label)
        finally:
            owner.f64_command_metadata = original

    def exchange_fe(self, driver, label, body, commands, dtype, right):
        """Verify the joint committed result of precisely one original FE frame.

The original complete snapshot and pre-F64 frame predicates still execute.
The two later single-command guards have no multi-command contract: replace
them only here with explicit joint ownership/payload predicates. No imaginary
intermediate owner is constructed, and no packet is split or rewritten.
"""
        owner = driver.owner
        original = owner.verify_frame_boundary
        original_base = owner._f64_original_verify_frame_boundary
        observed = []

        def boundary(before, after, actual_commands, status, records, actual_body, **kwargs):
            require(actual_body == body and len(body) == 228 and actual_commands == commands
                    and status == 0 and [item['opcode'] for item in records] == [97, 2],
                    'only the exact original two-command FE-add success enters this joint contract')
            baseline = original_base(before, after, actual_commands, status, records, actual_body, **kwargs)
            before_raw, after_raw = kwargs['before_raw'], kwargs['after_raw']
            require(isinstance(before_raw, bytes) and isinstance(after_raw, bytes)
                    and sha(before_raw) == before['full_sha256']
                    and sha(after_raw) == after['full_sha256'],
                    'both complete physical owners are present without intermediate inference')
            left = owner.public_node(after, records[0]['value'])
            result = owner.public_node(after, records[1]['value'])
            existing = owner.public_node(before, right)
            prior_live = {int(key) for key, node in before['nodes'].items() if owner.w(node, 304)}
            final_live = {int(key) for key, node in after['nodes'].items() if owner.w(node, 304)}
            require(left != result and left not in prior_live and result not in prior_live
                    and final_live == prior_live | {left, result},
                    'exactly two new live descriptors and all original descriptors remain live')
            for index in prior_live:
                owner._f64_source_unchanged(before, after, index)
            old_rows = {}
            for kind, rows in before['tables'].items():
                old_rows[kind] = 0
                for index, row in enumerate(rows):
                    if row[1]:
                        require(after['tables'][kind][index] == row,
                                'every existing live typed owner row remains byte-identical')
                        old_rows[kind] += 1
            half = owner.half_role(after, left)
            require(half['dtype'] == dtype, 'the FE factory has its exact public Half dtype')
            owner._half_fresh_result(before, after, left, [existing])
            owner._half_fresh_result(before, after, result, [existing])
            output = owner._live_node(after, result, 'FE-add-fresh-F32-result')
            require(owner.w(output, 576) == 0 and owner.w(output, 240) == 1
                    and not any(owner.w(output, field) for field in owner.F64_ZERO_FIELDS),
                    'the actual Torch F32 result has no AD, Saved, view or retained input')
            left_node = after['nodes'][str(left)]
            for offset in (192, 216, 224, 320):
                require(owner.w(left_node, offset) != owner.w(output, offset),
                        'the two new Tensor/Storage/Version/incarnation identities are distinct')
            # The pinned native path creates the public Half factory, one
            # temporary F32 left promotion, then the public F32 output. The
            # already-F32 right operand is reused. Commit releases promotion.
            first = before['header']['next']
            allocated = {first, first + 1, first + 2}
            require(after['header']['next'] == first + 3
                    and after['header']['incarnation'] == before['header']['incarnation'] + 3
                    and left == first and result == first + 2
                    and owner.w(left_node, 320) == before['header']['incarnation'] + 1
                    and owner.w(output, 320) == before['header']['incarnation'] + 3,
                    'exact factory, native left-promotion and result descriptor allocation order')
            canonical = self.fe_owner.verify_fe_promotion_tombstone(owner, before, after, left, result)
            temporary = after['nodes'][str(first + 1)]
            require(not owner.w(temporary, 304) and not owner.w(temporary, 240)
                    and str(first + 1) not in after['edges'],
                    'the one source-proved promotion descriptor is no longer live or retained')
            for kind, offset in (('1', 192), ('3', 224)):
                before_live = {index for index, row in enumerate(before['tables'][kind]) if row[1]}
                after_live = {index for index, row in enumerate(after['tables'][kind]) if row[1]}
                added = {(owner.w(node, offset) & 0xffffff) - 1 for node in (left_node, output)}
                require(after_live == before_live | added and len(added) == 2
                        and not (before_live & added),
                        'exactly two new public Tensor rows and two new Version rows')
            page_storages = {}
            for page in before['pages']:
                hold = owner._live_row(before, 6, page['hold'], 'original descriptor-page Hold')
                page_storages[hold[2]] = page['page']
            previous_pages = {page['page']: page for page in before['pages']}
            current_pages = {page['page']: page for page in after['pages']}
            require(all(current_pages.get(index) == page for index, page in previous_pages.items()),
                    'every original descriptor-page owner remains identical')
            new_pages = [page for index, page in current_pages.items() if index not in previous_pages]
            new_page_holds = {(page['hold'] & 0xffffff) - 1 for page in new_pages}
            new_page_storages = {(owner._live_row(after, 6, page['hold'], 'new descriptor-page Hold')[2]
                                  & 0xffffff) - 1 for page in new_pages}
            new_data_storages = {(owner.w(node, 216) & 0xffffff) - 1 for node in (left_node, output)}
            require(len(new_data_storages) == 2 and not (new_data_storages & new_page_storages),
                    'two distinct output Storages separate from internal descriptor pages')
            for kind in ('2', '4', '5', '6', '7', '8'):
                before_live = {index for index, row in enumerate(before['tables'][kind]) if row[1]}
                after_live = {index for index, row in enumerate(after['tables'][kind]) if row[1]}
                added = (new_data_storages | new_page_storages) if kind == '2' else (
                    new_page_holds if kind == '6' else set())
                require(not (before_live & added) and after_live == before_live | added,
                        'no retained promotion Storage, Weak, Saved, Hold, Generator or optimizer owner')
            before_start = owner.BASE + 144 + 552 * before['header']['slots']
            after_start = owner.BASE + 144 + 552 * after['header']['slots']
            preserved_storages = 0
            for index, row in enumerate(before['tables']['2'], 1):
                if not row[1]:
                    continue
                token = owner.token(2, index, row)
                start, extent = row[2], row[3]
                old_bytes = bytearray(before_raw[before_start + start:before_start + start + extent])
                new_bytes = bytearray(after_raw[after_start + start:after_start + start + extent])
                require(len(old_bytes) == len(new_bytes) == extent, 'complete original Storage extent')
                if token in page_storages:
                    page = page_storages[token]
                    for added in allocated:
                        if added >> 8 == page:
                            at = (added & 255) * owner.DESCRIPTOR
                            old_bytes[at:at + owner.DESCRIPTOR] = bytes(owner.DESCRIPTOR)
                            new_bytes[at:at + owner.DESCRIPTOR] = bytes(owner.DESCRIPTOR)
                require(old_bytes == new_bytes,
                        'all prior Storage bytes remain exact except the three source-proved allocated descriptors')
                preserved_storages += 1
            require(before['generator_states'] == after['generator_states'],
                    'all actual native Generator state bytes remain unchanged')
            detail = dict(label=label, existing_right=existing, factory=left, result=result,
                          exact_original_packet_bytes=228, actual_record_opcodes=[97, 2],
                          factory_dtype=dtype, result_dtype=0, allocated_descriptors=3,
                          final_live_new_descriptors=2, released_promotion_descriptors=1,
                          new_Tensor_rows=2, new_Storage_rows=2, new_Version_rows=2,
                          new_incarnations=3, new_internal_descriptor_pages=len(new_pages),
                          old_live_descriptors_unchanged=len(prior_live),
                          old_live_owner_rows_unchanged=old_rows,
                          prior_Storage_extents_checked=preserved_storages,
                          canonical_descriptor_allocator_and_retirement=canonical,
                          complete_original_snapshot_and_pre_F64_frame_predicates=True,
                          intermediate_owner_states_inferred=False,
                          numerical_expected_values_computed_here=False)
            observed.append(dict(before=before, after=after, factory=left, result=result))
            return dict(baseline, f64_command_details=None, half_command_details=None,
                        frame_admission='exact-original2-Half-FE-add-success',
                        explicit_FE_add_successor=detail, numeric_expected_values_computed_here=False,
                        full_transition_recomputed=False)

        owner.verify_frame_boundary = boundary
        try:
            status, records = driver.n.exchange(body, label)
            require(len(observed) == 1, 'one complete actual joint FE transition was verified')
            return status, records, observed[0]
        finally:
            owner.verify_frame_boundary = original

    def fe_success(self, driver, label, body, expected, wire):
        row, dtype = self.fe_rows[label], self.fe_dtypes[label]
        right = driver.live['F32']
        commands = [[97, dtype, 160, 0, 0, 0, 0, 0],
                    [2, 1, wire.reference(1), right, 0, 0, 0, 0]]
        exact = b'HTORCH03' + struct.pack('<3Q', 2, 228, 0)
        exact += b''.join(struct.pack('<8Q', *command) for command in commands)
        exact += struct.pack('<8Q', 1, 0, 2, 0, 0, 0, 224, 4) + struct.pack('<2H', 1, 2)
        require(expected == 7 and body == exact, 'the entire original228-byte factory/FE-add request')
        old_right = self.observe(driver, right, row['inputs'][1], label + '/original-right')
        status, records, transition = self.exchange_fe(driver, label, body, commands, dtype, right)
        driver.operations.append(dict(label=label, request_sha256=sha(body), actual_status=status,
            expected_status=0, record_count=len(records), predecessor_expected_status=7,
            successor_reference_index=row['index'], original_same_frame_FE_packet=True))
        driver.check(label + '/actual-frame-status', status, 0)
        driver.check(label + '/factory-then-add-result', [item['opcode'] for item in records], [97, 2])
        factory, result = [item['value'] for item in records]
        roles = ['half-FE-add-factory/' + label, 'half-FE-add-result/' + label]
        require(all(role not in driver.live for role in roles), 'two fresh temporary FE result roles')
        driver.live[roles[0]], driver.live[roles[1]] = factory, result
        got_factory = self.observe(driver, factory, row['inputs'][0], label + '/actual-factory')
        got_result = self.observe(driver, result, row['result'], label + '/actual-result')
        for key in ('tensor_identity', 'storage_identity', 'native_version_identity'):
            driver.check(label + '/two-fresh-' + key,
                         len({old_right[key], got_factory[key], got_result[key]}), 3)
        after_right = driver.n.tensor(right)
        require(set(after_right) == set(old_right), 'all original input observation fields')
        for key in old_right:
            if key != 'owner':
                driver.check(label + '/right-unchanged/' + key, after_right[key], old_right[key])
        driver.rng_unchanged(label + '/successful-original-FE-add')
        driver.drop(roles[0])
        middle, _, _ = driver.n.capture(label + '/after-factory-release')
        require(not driver.owner.w(middle['nodes'][str(transition['factory'])], 304),
                'the result does not retain its newly created factory input')
        self.observe(driver, result, row['result'], label + '/result-after-factory-release')
        driver.drop(roles[1])
        final, _, _ = driver.n.capture(label + '/after-both-releases')
        for index in (transition['factory'], transition['result']):
            require(not driver.owner.w(final['nodes'][str(index)], 304),
                    'both temporary FE descriptors are fully released')
            node = transition['after']['nodes'][str(index)]
            for kind, offset in ((1, 192), (2, 216), (3, 224)):
                handle = driver.owner.w(node, offset)
                require(final['tables'][str(kind)][(handle & 0xffffff) - 1][1] == 0,
                        'each fresh FE Tensor/Storage/Version row is released without retention')
        return status

    def wrap_entry(self, original_entry):
        old_execute = self.negation.wrap_entry(original_entry)
        old_closure = closure(old_execute)
        require(set(old_closure) == {'accepted_mul', 'accepted_neg', 'cloned', 'self'}
                and old_closure['self'] is self.negation,
                'exact unchanged predecessor terminal wrapper closure')
        neg_entry = old_closure['cloned']
        accepted_mul, accepted_neg = old_closure['accepted_mul'], old_closure['accepted_neg']
        require(neg_entry.__code__ is original_entry.__code__
                and accepted_mul == accepted_neg == [],
                'both unchanged predecessors preserve the complete original entry code')
        neg_reject = neg_entry.__globals__['reject']
        reject_code = [code for code in self.negation.wrap_entry.__code__.co_consts
                       if isinstance(code, types.CodeType) and code.co_name == 'reject']
        require(len(reject_code) == 1 and neg_reject.__code__ is reject_code[0]
                and closure(neg_reject)['accepted_neg'] is accepted_neg
                and closure(closure(neg_reject)['mul_reject'])['accepted'] is accepted_mul,
                'the exact previous22/2 rejection dispatchers and their predicates are reused')
        accepted_addsub, accepted_fe = [], []
        wire = original_entry.__globals__['WIRE']
        names = {0: 'F32', 1: 'Bool', 2: 'I64', 4: 'F64', 5: 'F16', 6: 'BF16'}

        def reject(driver, label, body, expected, **kwargs):
            if label in self.fe_rows:
                require(not kwargs and len(accepted_fe) < 2
                        and label == self.proof['same_frame_FE_labels'][len(accepted_fe)],
                        'only the two exact ordered original FE-add requests change')
                status = self.fe_success(driver, label, body, expected, wire)
                accepted_fe.append(label)
                return status
            if label not in self.rows:
                return neg_reject(driver, label, body, expected, **kwargs)
            require(not kwargs and expected == 7 and len(accepted_addsub) < 40
                    and label == self.proof['single_command_labels'][len(accepted_addsub)],
                    'only the exact ordered original40 add/sub requests change')
            row, operation = self.rows[label], self.operations[label]
            left_dtype, right_dtype = self.dtypes[label]
            left, right = driver.live[names[left_dtype]], driver.live[names[right_dtype]]
            exact = b'HTORCH03' + struct.pack('<3Q', 1, 96, 0)
            exact += struct.pack('<8Q', 2, operation, left, right, 0, 0, 0, 0)
            require(body == exact, 'complete original opcode, operands, padding and envelope')
            driver.check(label + '/same-handle-inputs', left == right, row['spec']['same_handle'])
            before = [self.observe(driver, token, wanted, label + '/input-' + side)
                      for token, wanted, side in zip((left, right), row['inputs'], ('left', 'right'))]
            status, records = self.exchange(driver, label, body, left, right, row)
            driver.operations.append(dict(label=label, request_sha256=sha(body), actual_status=status,
                expected_status=0, record_count=len(records), predecessor_expected_status=7,
                successor_reference_index=row['index']))
            driver.check(label + '/actual-frame-status', status, 0)
            driver.check(label + '/one-addsub-result', [item['opcode'] for item in records], [2])
            result = records[0]['value']
            role = 'half-addsub-successor-result/' + label
            require(role not in driver.live, 'fresh temporary add/sub result role')
            driver.live[role] = result
            got = self.observe(driver, result, row['result'], label + '/result')
            for key in ('tensor_identity', 'storage_identity', 'native_version_identity'):
                driver.check(label + '/fresh-' + key, got[key] not in [item[key] for item in before], True)
            for token, saved, side in zip((left, right), before, ('left', 'right')):
                after = driver.n.tensor(token)
                require(set(after) == set(saved), 'complete original input observation fields')
                for key in saved:
                    if key != 'owner':
                        driver.check(label + '/' + side + '-input-unchanged/' + key,
                                     after[key], saved[key])
            driver.rng_unchanged(label + '/successful-addsub')
            driver.drop(role)
            accepted_addsub.append(label)
            return status

        namespace = dict(neg_entry.__globals__)
        namespace['reject'] = reject
        cloned = types.FunctionType(original_entry.__code__, namespace,
            original_entry.__name__, original_entry.__defaults__, original_entry.__closure__)
        cloned.__kwdefaults__ = original_entry.__kwdefaults__
        require(cloned.__code__ is original_entry.__code__, 'complete original239 entry code object')

        def execute(driver):
            require(not driver.operations and not accepted_mul and not accepted_neg
                    and not accepted_addsub and not accepted_fe,
                    'fresh complete239 entry-only driver')
            details = cloned(driver)
            require(details['actual_refusals'] == len(driver.operations) == 239
                    and accepted_mul == self.multiplication.proof['labels']
                    and accepted_neg == self.negation.proof['labels']
                    and accepted_addsub == self.proof['single_command_labels']
                    and accepted_fe == self.proof['same_frame_FE_labels'],
                    'all239 original requests and each exact successor execute in original order')
            successful = [row for row in driver.operations if row['expected_status'] == 0]
            retained = [row for row in driver.operations if row['expected_status'] != 0]
            require(len(successful) == 66 and len(retained) == 173
                    and [row['label'] for row in successful if row['label'] in self.multiplication.rows] == accepted_mul
                    and [row['label'] for row in successful if row['label'] in self.negation.rows] == accepted_neg
                    and [row['label'] for row in successful if row['label'] in self.rows] == accepted_addsub
                    and [row['label'] for row in successful if row['label'] in self.fe_rows] == accepted_fe
                    and [row['label'] for row in successful
                         if row['label'] in self.rows or row['label'] in self.fe_rows] == self.proof['labels']
                    and all(row['actual_status'] == row['expected_status'] for row in driver.operations),
                    'exact173 refusals,22 multiplication,2 negation,40 single-command and2 FE-add successes')
            for labels in ([item[0] for item in BOOL_REFUSALS], list(PRIVATE3_LABELS)):
                actual = [row for row in retained if row['label'] in labels]
                require([row['label'] for row in actual] == labels
                        and all(row['expected_status'] == row['actual_status'] == 7 for row in actual),
                        'the four actual Bool errors and eight private3 protocol refusals remain intact')
            require(sum(self.operations[label] == 1 for label in accepted_addsub) == 22
                    and sum(self.operations[label] == 2 for label in accepted_addsub) == 18,
                    'exact addition and subtraction successor partition')
            return dict(details, actual_refusals=173, actual_successful_multiplications=22,
                actual_successful_negations=2, actual_successful_additions=24,
                actual_successful_subtractions=18, actual_successful_addsub=42,
                actual_successful_single_command_addsub=40, actual_successful_FE_addsub=2,
                original_entry_requests=239, multiplication_successor=self.multiplication.proof,
                negation_successor=self.negation.proof, addsub_successor=self.proof)
        return execute


def adapter_type(original_type, multiplication_module, negation_module, repository,
                 reader_directory, multiplication_reference, negation_reference, addsub_reference):
    """Bind all three actual references and scope the unchanged original entry."""
    class Adapter(original_type):
        def bind_package(self, inputs):
            digest = super().bind_package(inputs)
            self.half_mul_successor = multiplication_module.Successor(repository, reader_directory,
                multiplication_reference['root'], multiplication_reference['execution'], inputs)
            self.half_neg_successor = negation_module.Successor(repository, reader_directory,
                negation_reference['root'], negation_reference['execution'], inputs,
                self.half_mul_successor)
            self.half_addsub_successor = Successor(repository, reader_directory,
                addsub_reference['root'], addsub_reference['execution'], inputs,
                self.half_neg_successor)
            return digest

        def current_main(self):
            main = super().current_main()
            controls = self.old['controls']
            original = controls.entry_controls
            def entry(driver):
                return self.half_addsub_successor.wrap_entry(original)(driver)
            def execute():
                require(controls.entry_controls is original, 'unmodified original controls before successor scope')
                controls.entry_controls = entry
                try:
                    return main()
                finally:
                    controls.entry_controls = original
            return execute
    return Adapter
