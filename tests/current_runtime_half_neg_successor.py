"""Add exactly two negations to the unchanged239-request Half47 recipe.

The complete existing22 multiplication adapter remains byte identical. Its
actual reject closure and owner hook are reused directly; only the final
partition now accounts for215 refusals,22 multiplications and2 negations.
Both live transport and independent physical replay use this same recipe.
"""
import ast
import hashlib
import importlib
from pathlib import Path
import struct
import sys
import types


MUL_HELPER = '47557e7730d00c374864d2e2948a511e86b47ae653034adaa0f776445fc556ad'
SUCCESSORS = (('unsupported-half5-neg', 5), ('unsupported-half6-neg', 6))
WORDS = {5: (0x3c00, 0x8000), 6: (0x3f80, 0x8000)}


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
                 reference_execution, inputs, multiplication):
        self.repository = Path(repository).resolve(strict=True)
        self.reader = Path(reader_directory).resolve(strict=True)
        self.reference_directory = Path(reference_directory).resolve(strict=True)
        self.reference_execution = Path(reference_execution).resolve(strict=True)
        self.multiplication = multiplication
        tests = self.repository / 'tests'
        inputs.bind(Path(__file__).resolve())
        helper = tests / 'current_runtime_half_mul_successor.py'
        inputs.bind(helper, MUL_HELPER)
        require(Path(multiplication.wrap_entry.__code__.co_filename).resolve() == helper
                and multiplication.repository == self.repository
                and multiplication.reader == self.reader,
                'the unchanged complete22 helper is loaded from this actual repository')
        for path in sorted((tests / 'half_neg').glob('*.py')):
            inputs.bind(path)
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(tests))
            run = importlib.import_module('half_neg.run')
            cases = importlib.import_module('half_neg.cases')
        finally:
            sys.path[:] = old_path
        for module in (run, cases):
            require(Path(module.__file__).resolve().parent == tests / 'half_neg',
                    'new reference modules belong to this actual repository')
        legacy = multiplication.legacy
        raw = inputs.bind(self.reader / legacy.ORIGINAL_MEMBER, cases.LEGACY_SOURCE_SHA256)
        functions = [node for node in ast.parse(raw).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'entry_controls']
        require(len(functions) == 1, 'one complete frozen239 original entry recipe')
        node = functions[0]
        entry = b''.join(raw.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
        require(cases.LEGACY_SOURCE_SHA256 == legacy.ORIGINAL_SOURCE_SHA256
                and cases.LEGACY_ENTRY_SHA256 == legacy.ORIGINAL_ENTRY_SHA256 == sha(entry)
                and len(entry) == legacy.ORIGINAL_ENTRY_BYTES == 7497
                and cases.SUCCESSORS == SUCCESSORS,
                'exact original bytes and two literal negation successor labels')
        self.reference = run.read_reference(self.reference_directory,
                                            self.reference_execution, inputs)
        rows = [row for row in self.reference['report']['rows']
                if 'legacy_refusal_label' in row['spec']]
        require(len(rows) == 2 and [row['spec']['legacy_refusal_label'] for row in rows]
                == [label for label, _ in SUCCESSORS],
                'all and only two original negation requests have fresh Torch results')
        self.rows, self.dtypes = {}, {}
        for row, (label, dtype) in zip(rows, SUCCESSORS):
            literal = struct.pack('<2H', *WORDS[dtype])
            require(row['accepted'] is True and row['spec']['name'] == 'successor-' + label
                    and row['spec']['operation'] == 'neg'
                    and row['spec']['input'] == cases.input_spec(dtype, [2], words=WORDS[dtype])
                    and cases.raw_input(row['spec']['input']) == literal
                    and self.reference['blob'](row['input']['logical']) == literal
                    and row['input']['shape'] == [2] and row['input']['strides'] == [1],
                    'actual Torch negates the complete unchanged original literal input')
            self.rows[label], self.dtypes[label] = row, dtype
        self.proof = dict(
            schema='heptatorch.half47.neg2-successor.v1',
            reference_directory=str(self.reference_directory),
            reference_execution=str(self.reference_execution),
            reference_sha256=self.reference['report_sha256'],
            reference_process_sha256=sha(inputs.bind(self.reference_execution / 'process.json')),
            case_sha256=cases.CASE_SHA256,
            original_controls_sha256=cases.LEGACY_SOURCE_SHA256,
            original_entry_sha256=cases.LEGACY_ENTRY_SHA256, original_entry_bytes=7497,
            original_request_count=239, predecessor_refusals=217, retained_refusals=215,
            existing_multiplication_successors=22, accepted_successors=2,
            labels=[label for label, _ in SUCCESSORS],
            multiplication_helper_sha256=MUL_HELPER,
            multiplication_reject_and_owner_hook_reused_unchanged=True,
            multiplication_terminal_partition_replaced_explicitly=True,
            original_entry_code_object_preserved=True,
            actual_Torch_reference_verified=True, host_expected_values_computed=False,
            original_frozen_recipe_modified=False, original_owner_methods_modified=False,
            every_original_snapshot_field_checked=True,
            input_all84words_unchanged_checked=True)

    def observe(self, driver, token, expected, label):
        # Reuse every existing observation predicate with this new reference.
        return type(self.multiplication).observe(self, driver, token, expected, label)

    def wrap_entry(self, original_entry):
        old_execute = self.multiplication.wrap_entry(original_entry)
        old_closure = closure(old_execute)
        require(set(old_closure) == {'accepted', 'cloned', 'self'}
                and old_closure['self'] is self.multiplication,
                'exact original multiplication terminal wrapper closure')
        mul_entry, accepted_mul = old_closure['cloned'], old_closure['accepted']
        require(mul_entry.__code__ is original_entry.__code__ and accepted_mul == [],
                'the unchanged multiplication adapter preserves the complete original entry code')
        mul_reject = mul_entry.__globals__['reject']
        reject_code = [code for code in self.multiplication.wrap_entry.__code__.co_consts
                       if isinstance(code, types.CodeType) and code.co_name == 'reject']
        require(len(reject_code) == 1 and mul_reject.__code__ is reject_code[0]
                and closure(mul_reject)['accepted'] is accepted_mul
                and closure(mul_reject)['original_reject'] is original_entry.__globals__['reject'],
                'the original22 rejection dispatcher and all215 other rejection predicates are reused')
        accepted_neg = []

        def reject(driver, label, body, expected, **kwargs):
            if label not in self.rows:
                return mul_reject(driver, label, body, expected, **kwargs)
            require(not kwargs and expected == 7 and len(accepted_neg) < 2
                    and label == self.proof['labels'][len(accepted_neg)],
                    'only the exact ordered original two negations change')
            row, dtype = self.rows[label], self.dtypes[label]
            source = driver.live['F16' if dtype == 5 else 'BF16']
            exact = b'HTORCH03' + struct.pack('<3Q', 1, 96, 0)
            exact += struct.pack('<8Q', 3, 6, source, 0, 0, 0, 0, 0)
            require(body == exact, 'complete original unary opcode, operand, padding and envelope')
            before = self.observe(driver, source, row['input'], label + '/input')
            owner_before, _, _ = driver.n.capture(label + '/native-negation-before')
            status, records = driver.n.exchange(body, label)
            owner_after, _, _ = driver.n.capture(label + '/native-negation-after')
            driver.operations.append(dict(label=label, request_sha256=sha(body), actual_status=status,
                expected_status=0, record_count=len(records), predecessor_expected_status=7,
                successor_reference_index=row['index']))
            driver.check(label + '/actual-frame-status', status, 0)
            driver.check(label + '/one-negation-result', [record['opcode'] for record in records], [3])
            result = records[0]['value']
            source_node = driver.owner.public_node(owner_before, source)
            result_node = driver.owner.public_node(owner_after, result)
            driver.owner._f64_source_unchanged(owner_before, owner_after, source_node)
            native_role = driver.owner.half_role(owner_after, result_node)
            require(native_role['dtype'] == dtype, 'negation retains the exact actual Half dtype')
            driver.owner._half_fresh_result(owner_before, owner_after, result_node, [source_node])
            role = 'half-neg-successor-result/' + label
            require(role not in driver.live, 'fresh temporary negation result role')
            driver.live[role] = result
            got = self.observe(driver, result, row['result'], label + '/result')
            for key in ('tensor_identity', 'storage_identity', 'native_version_identity'):
                driver.check(label + '/fresh-' + key, got[key] != before[key], True)
            after = driver.n.tensor(source)
            require(set(after) == set(before), 'complete original input observation fields')
            for key in before:
                if key != 'owner':
                    driver.check(label + '/input-unchanged/' + key, after[key], before[key])
            driver.rng_unchanged(label + '/successful-negation')
            driver.drop(role)
            accepted_neg.append(label)
            return status

        namespace = dict(mul_entry.__globals__)
        namespace['reject'] = reject
        cloned = types.FunctionType(original_entry.__code__, namespace,
            original_entry.__name__, original_entry.__defaults__, original_entry.__closure__)
        cloned.__kwdefaults__ = original_entry.__kwdefaults__
        require(cloned.__code__ is original_entry.__code__, 'complete original239 entry code object')

        def execute(driver):
            require(not driver.operations and not accepted_mul and not accepted_neg,
                    'fresh complete239 entry-only driver')
            details = cloned(driver)
            require(details['actual_refusals'] == len(driver.operations) == 239
                    and accepted_mul == self.multiplication.proof['labels']
                    and accepted_neg == self.proof['labels'],
                    'every original239 request and each exact successor runs in original order')
            successful = [row for row in driver.operations if row['expected_status'] == 0]
            retained = [row for row in driver.operations if row['expected_status'] != 0]
            require(len(successful) == 24 and len(retained) == 215
                    and [row['label'] for row in successful if row['label'] in self.multiplication.rows] == accepted_mul
                    and [row['label'] for row in successful if row['label'] in self.rows] == accepted_neg
                    and all(row['actual_status'] == row['expected_status'] for row in driver.operations),
                    'exact215 original refusals,22 original multiplication successors and2 negations')
            return dict(details, actual_refusals=215, actual_successful_multiplications=22,
                actual_successful_negations=2, original_entry_requests=239,
                multiplication_successor=self.multiplication.proof, negation_successor=self.proof)
        return execute


def adapter_type(original_type, multiplication_module, repository, reader_directory,
                 multiplication_reference, negation_reference):
    """Bind both actual references and scope only the unchanged original entry."""
    class Adapter(original_type):
        def bind_package(self, inputs):
            digest = super().bind_package(inputs)
            self.half_mul_successor = multiplication_module.Successor(repository, reader_directory,
                multiplication_reference['root'], multiplication_reference['execution'], inputs)
            self.half_neg_successor = Successor(repository, reader_directory,
                negation_reference['root'], negation_reference['execution'], inputs,
                self.half_mul_successor)
            return digest

        def current_main(self):
            main = super().current_main()
            controls = self.old['controls']
            original = controls.entry_controls
            def entry(driver):
                return self.half_neg_successor.wrap_entry(original)(driver)
            def execute():
                require(controls.entry_controls is original, 'unmodified original controls before successor scope')
                controls.entry_controls = entry
                try:
                    return main()
                finally:
                    controls.entry_controls = original
            return execute
    return Adapter
