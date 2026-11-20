"""Execute the exact22 supported successors inside the unchanged Half47 recipe.

The original239-request function remains byte-for-byte frozen. Its22 public
Half multiplication requests now require real success and fixed-Torch values;
all other217 refusal requests retain their complete original rejection path.
This same adapter runs on live transport and independently recorded transport.
"""
import ast
import hashlib
import importlib
from pathlib import Path
import struct
import sys
import types


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Successor:
    def __init__(self, repository, reader_directory, reference_directory,
                 reference_execution, inputs):
        self.repository = Path(repository).resolve(strict=True)
        self.reader = Path(reader_directory).resolve(strict=True)
        self.reference_directory = Path(reference_directory).resolve(strict=True)
        self.reference_execution = Path(reference_execution).resolve(strict=True)
        tests = self.repository / 'tests'
        inputs.bind(Path(__file__).resolve())
        # Bind the complete source modules used by the new strict reference
        # reader. Importing them performs no native or Torch execution.
        for package in ('half_mul', 'numeric_predicates'):
            for path in sorted((tests / package).glob('*.py')):
                inputs.bind(path)
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(tests))
            legacy = importlib.import_module('half_mul.legacy_controls')
            run = importlib.import_module('half_mul.run')
            cases = importlib.import_module('half_mul.cases')
        finally:
            sys.path[:] = old_path
        for module in (legacy, run, cases):
            require(Path(module.__file__).resolve().parent == tests / 'half_mul',
                    'new reference modules belong to this actual repository')
        self.legacy = legacy
        raw = inputs.bind(self.reader / legacy.ORIGINAL_MEMBER,
                          legacy.ORIGINAL_SOURCE_SHA256)
        functions = [node for node in ast.parse(raw).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'entry_controls']
        require(len(functions) == 1, 'exact one frozen original entry recipe')
        node = functions[0]
        entry = b''.join(raw.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
        require(len(entry) == legacy.ORIGINAL_ENTRY_BYTES
                and sha(entry) == legacy.ORIGINAL_ENTRY_SHA256,
                'complete immutable original239 entry recipe bytes')
        self.reference = run.read_reference(self.reference_directory,
                                            self.reference_execution, inputs)
        rows = [row for row in self.reference['report']['rows']
                if 'legacy_refusal_label' in row['spec']]
        require(len(rows) == len(legacy.SUCCESSORS) == 22
                and [row['spec']['legacy_refusal_label'] for row in rows]
                    == [label for label, _, _ in legacy.SUCCESSORS],
                'all and only22 original-label successors have actual Torch observations')
        self.rows = {}
        self.dtypes = {}
        for row, (label, left, right) in zip(rows, legacy.SUCCESSORS):
            require(row['accepted'] is True and row['spec']['name'] == 'successor-' + label,
                    'every mapped request has a successful actual Torch result')
            for key, dtype, observed in zip(('left', 'right'), (left, right), row['inputs']):
                spec = row['spec'][key]
                literal = b''.join(word.to_bytes(cases.WIDTHS[dtype], 'little')
                                   for word in legacy.WORDS[dtype])
                require(spec == cases.input_spec(dtype, [2], words=legacy.WORDS[dtype])
                        and cases.raw_input(spec) == literal
                        and self.reference['blob'](observed['logical']) == literal
                        and observed['shape'] == [2] and observed['strides'] == [1],
                        'new reference repeats the exact original literal operand')
            self.rows[label], self.dtypes[label] = row, (left, right)
        self.proof = dict(
            schema='heptatorch.half47.mul22-successor.v1',
            reference_directory=str(self.reference_directory),
            reference_execution=str(self.reference_execution),
            reference_sha256=self.reference['report_sha256'],
            reference_process_sha256=sha(inputs.bind(self.reference_execution / 'process.json')),
            case_sha256=cases.CASE_SHA256,
            original_controls_sha256=legacy.ORIGINAL_SOURCE_SHA256,
            original_entry_sha256=legacy.ORIGINAL_ENTRY_SHA256,
            original_entry_bytes=legacy.ORIGINAL_ENTRY_BYTES,
            original_request_count=239, retained_refusals=217,
            accepted_successors=22, labels=[row[0] for row in legacy.SUCCESSORS],
            actual_Torch_reference_verified=True, host_expected_values_computed=False,
            original_frozen_recipe_modified=False,
            scoped_owner_extension='exact22-Half-mul-public-promotion',
            every_original_snapshot_field_checked=True,
            original_F64_same_dtype_binary_domain_extended_only_for_these_requests=True)

    def observe(self, driver, token, expected, label):
        """Compare complete old public/owner reads with new actual Torch values."""
        got = driver.n.tensor(token)
        keys = ('dtype', 'shape', 'strides', 'numel', 'element_size',
                'storage_bytes', 'version', 'requires_grad', 'is_leaf',
                'is_view', 'retains_grad')
        for key in keys:
            driver.check(label + '/' + key, got[key], expected[key])
        driver.check(label + '/device', got['device'], 'cpu')
        driver.check(label + '/storage-offset-elements', got['storage_offset_elements'],
                     expected['storage_offset'])
        driver.check(label + '/storage-offset-bytes', got['storage_offset_bytes'],
                     expected['storage_offset'] * expected['element_size'])
        driver.check(label + '/history-present', got['grad_fn'] is not None,
                     expected['history_present'])
        driver.check(label + '/grad-is-none', got['grad_is_none'], True)
        driver.check(label + '/base-absent', got['base_tensor_identity'], None)
        driver.check(label + '/all-logical-bits', got['logical'],
                     self.reference['blob'](expected['logical']))
        driver.check(label + '/all-storage-bits', got['storage'],
                     self.reference['blob'](expected['storage']))
        return got

    def exchange(self, driver, label, body, left, right, row):
        """Extend the old F64-only metadata branch for this exact new operation.

        All original snapshot and frame observers still execute. The former
        same-dtype-only F64 binary branch is replaced during this one admitted
        multiplication by explicit promoted-dtype and unchanged/fresh ownership
        checks. The original branch is restored even when a check fails.
        """
        owner = driver.owner
        original = owner.f64_command_metadata
        expected_command = [2, 3, left, right, 0, 0, 0, 0]
        result_dtype = {'torch.float32': 0, 'torch.float64': 4,
                        'torch.float16': 5, 'torch.bfloat16': 6}[row['result']['dtype']]
        left_dtype, right_dtype = self.dtypes[label]

        def metadata(before, after, command, record):
            require(command == expected_command and record['opcode'] == 2,
                    'only the exact current successor command enters the owner extension')
            sources = [owner.public_node(before, token) for token in (left, right)]
            observed = [owner.w(before['nodes'][str(index)], 576) for index in sources]
            require(observed == [left_dtype, right_dtype]
                    and any(dtype in (5, 6) for dtype in observed)
                    and all(dtype in (0, 1, 2, 4, 5, 6) for dtype in observed),
                    'exact two public noAD successor operand types')
            result = owner.public_node(after, record['value'])
            node = owner._live_node(after, result, 'half-mul-successor-result')
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
            return dict(operation=2, binary_operation=3, role=role,
                        input_dtypes=observed, result_dtype=result_dtype,
                        input_all84words_unchanged=True,
                        exact_successor_label=label,
                        actual_Torch_result_dtype_used=True,
                        numerical_expected_values_computed_here=False)

        owner.f64_command_metadata = metadata
        try:
            return driver.n.exchange(body, label)
        finally:
            owner.f64_command_metadata = original

    def wrap_entry(self, original_entry):
        """Replace only the exact selected rejection calls in the old code."""
        original_reject = original_entry.__globals__['reject']
        names = {0: 'F32', 1: 'Bool', 2: 'I64', 4: 'F64', 5: 'F16', 6: 'BF16'}
        accepted = []

        def reject(driver, label, body, expected, **kwargs):
            if label not in self.rows:
                return original_reject(driver, label, body, expected, **kwargs)
            require(not kwargs and expected == 7
                    and label == self.proof['labels'][len(accepted)],
                    'only the exact ordered original22 unsupported multiplication requests change')
            row = self.rows[label]
            left_dtype, right_dtype = self.dtypes[label]
            left = driver.live[names[left_dtype]]
            right = driver.live[names[right_dtype]]
            exact = b'HTORCH03' + struct.pack('<3Q', 1, 96, 0)
            exact += struct.pack('<8Q', 2, 3, left, right, 0, 0, 0, 0)
            require(body == exact, 'complete original opcode, operands, padding and envelope')
            before = [self.observe(driver, token, wanted, label + '/input-' + side)
                      for token, wanted, side in zip((left, right), row['inputs'], ('left', 'right'))]
            status, records = self.exchange(driver, label, body, left, right, row)
            driver.operations.append(dict(label=label, request_sha256=sha(body), actual_status=status,
                expected_status=0, record_count=len(records), predecessor_expected_status=7,
                successor_reference_index=row['index']))
            driver.check(label + '/actual-frame-status', status, 0)
            driver.check(label + '/one-multiplication-result', [r['opcode'] for r in records], [2])
            result = records[0]['value']
            role = 'half-mul-successor-result/' + label
            require(role not in driver.live, 'fresh temporary result role')
            driver.live[role] = result
            got = self.observe(driver, result, row['result'], label + '/result')
            for key in ('tensor_identity', 'storage_identity', 'native_version_identity'):
                driver.check(label + '/fresh-' + key, got[key] not in [x[key] for x in before], True)
            for token, saved, side in zip((left, right), before, ('left', 'right')):
                after = driver.n.tensor(token)
                # Owner endpoints identify different complete snapshots. Every
                # Tensor observation and its complete Storage must be identical.
                require(set(after) == set(saved), 'complete original input observation fields')
                for key in saved:
                    if key != 'owner':
                        driver.check(label + '/' + side + '-input-unchanged/' + key,
                                     after[key], saved[key])
            driver.rng_unchanged(label + '/successful-multiplication')
            driver.drop(role)
            accepted.append(label)
            return status

        namespace = dict(original_entry.__globals__)
        namespace['reject'] = reject
        cloned = types.FunctionType(original_entry.__code__, namespace,
            original_entry.__name__, original_entry.__defaults__, original_entry.__closure__)
        cloned.__kwdefaults__ = original_entry.__kwdefaults__
        require(cloned.__code__ is original_entry.__code__, 'complete original entry recipe code object')

        def execute(driver):
            require(not driver.operations and not accepted, 'fresh original entry-only driver')
            details = cloned(driver)
            require(details['actual_refusals'] == len(driver.operations) == 239
                    and accepted == self.proof['labels'], 'every original239 request executed in original order')
            successful = [row for row in driver.operations if row['expected_status'] == 0]
            retained = [row for row in driver.operations if row['expected_status'] != 0]
            require(len(successful) == 22 and len(retained) == 217
                    and [row['label'] for row in successful] == accepted
                    and all(row['actual_status'] == row['expected_status'] for row in driver.operations),
                    'exact22 successes and217 unchanged refusal outcomes')
            return dict(details, actual_refusals=217, actual_successful_multiplications=22,
                        original_entry_requests=239, multiplication_successor=self.proof)
        return execute


def adapter_type(original_type, repository, reader_directory, reference):
    """Bind the successor reference before executing the unchanged old main."""
    class Adapter(original_type):
        def bind_package(self, inputs):
            digest = super().bind_package(inputs)
            self.half_mul_successor = Successor(repository, reader_directory,
                reference['root'], reference['execution'], inputs)
            return digest

        def current_main(self):
            main = super().current_main()
            controls = self.old['controls']
            original = controls.entry_controls
            def entry(driver):
                return self.half_mul_successor.wrap_entry(original)(driver)
            def execute():
                require(controls.entry_controls is original, 'unmodified original controls before successor scope')
                controls.entry_controls = entry
                try:
                    return main()
                finally:
                    controls.entry_controls = original
            return execute
    return Adapter
