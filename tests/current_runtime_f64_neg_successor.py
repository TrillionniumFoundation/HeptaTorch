"""One numerically checked negation in the unchanged87-request F6448 recipe.

The original entry code, original reject function and complete owner/frame
observers remain intact. Only unsupported-F64-neg receives an explicit numeric
successor, backed by the same independently executed reference as wide_neg405.
"""
import ast
import hashlib
import importlib
from pathlib import Path
import struct
import sys
import types

CONTROLS = '16daead20d5bf9934371a8283b18a94dc03a45d3390b346d7ef5d7019d8820d8'
ENTRY = 'ba632c235c28f95fa7464673d4ca5b4f556d727712154f5d8a349e4cf7c7078b'
LABEL = 'unsupported-F64-neg'
WORDS = (0x3ff0000000000000, 0x8000000000000000)
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Successor:
    def __init__(self, repository, reader_directory, reference, inputs):
        self.repository = Path(repository).resolve(strict=True)
        self.reader = Path(reader_directory).resolve(strict=True)
        self.reference_directory = Path(reference['root']).resolve(strict=True)
        self.reference_execution = Path(reference['execution']).resolve(strict=True)
        inputs.bind(Path(__file__).resolve())
        controls_path = self.reader / 'controls736.py'
        raw = inputs.bind(controls_path, CONTROLS)
        functions = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'entry_controls']
        require(len(functions) == 1, 'one complete original F64 entry recipe')
        node = functions[0]
        entry = b''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
        require(sha(entry) == ENTRY and len(entry) == 6998,
                'all original F64 request recipe bytes are unchanged')
        tests = self.repository / 'tests'
        for path in sorted((tests / 'wide_neg').glob('*.py')):
            inputs.bind(path)
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(tests))
            run = importlib.import_module('wide_neg.run')
            cases = importlib.import_module('wide_neg.cases')
        finally:
            sys.path[:] = old_path
        for module in (run, cases):
            require(Path(module.__file__).resolve().parent == tests / 'wide_neg',
                    'fresh reference modules belong to this repository')
        self.reference = run.read_reference(self.reference_directory, self.reference_execution, inputs)
        rows = [row for row in self.reference['report']['rows'] if 'legacy_refusal_label' in row['spec']]
        require(len(rows) == 1 and rows[0]['spec']['legacy_refusal_label'] == LABEL,
                'all and only the original F64 negation gets a new numeric result')
        row = rows[0]
        literal = struct.pack('<2Q', *WORDS)
        require(row['accepted'] is True and row['spec']['name'] == 'successor-' + LABEL
                and row['spec']['operation'] == 'neg'
                and row['spec']['input'] == cases.input_spec(4, [2], words=WORDS)
                and cases.raw_input(row['spec']['input']) == literal
                and self.reference['blob'](row['input']['logical']) == literal
                and row['input']['shape'] == [2] and row['input']['strides'] == [1],
                'real Torch negates the complete unchanged original F64 input')
        self.row = row
        self.proof = dict(schema='heptatorch.f6448.neg1-successor.v1',
            reference_directory=str(self.reference_directory),
            reference_execution=str(self.reference_execution),
            reference_sha256=self.reference['report_sha256'],
            reference_process_sha256=sha(inputs.bind(self.reference_execution/'process.json')),
            case_sha256=cases.CASE_SHA256, original_controls_sha256=CONTROLS,
            original_entry_sha256=ENTRY, original_entry_bytes=6998,
            original_request_count=87, retained_refusals=86, accepted_successors=1,
            labels=[LABEL], original_frozen_recipe_modified=False,
            original_entry_code_object_preserved=True, original_owner_methods_modified=False,
            every_original_snapshot_field_checked=True, input_all84words_unchanged_checked=True,
            actual_Torch_reference_verified=True, host_expected_values_computed=False)

    def observe(self, driver, token, expected, label):
        got = driver.n.tensor(token)
        for key in ('dtype','shape','strides','numel','element_size','storage_bytes','version',
                    'requires_grad','is_leaf','is_view','retains_grad'):
            driver.check(label+'/'+key, got[key], expected[key])
        driver.check(label+'/device', got['device'], 'cpu')
        driver.check(label+'/storage-offset-elements', got['storage_offset_elements'], expected['storage_offset'])
        driver.check(label+'/storage-offset-bytes', got['storage_offset_bytes'], expected['storage_offset']*8)
        driver.check(label+'/history-present', got['grad_fn'] is not None, expected['history_present'])
        driver.check(label+'/grad-is-none', got['grad_is_none'], True)
        driver.check(label+'/base-absent', got['base_tensor_identity'], None)
        driver.check(label+'/all-logical-bits', got['logical'], self.reference['blob'](expected['logical']))
        driver.check(label+'/all-storage-bits', got['storage'], self.reference['blob'](expected['storage']))
        return got

    def wrap_entry(self, original_entry):
        require(Path(original_entry.__code__.co_filename).resolve() == self.reader/'controls736.py'
                and original_entry.__name__ == 'entry_controls',
                'the original complete F64 entry code is used')
        original_reject = original_entry.__globals__['reject']
        accepted = []

        def reject(driver, label, body, expected, **kwargs):
            if label != LABEL:
                return original_reject(driver, label, body, expected, **kwargs)
            require(not kwargs and expected == 7 and not accepted,
                    'only the one original F64 negation refusal changes')
            source = driver.live['f64']
            exact = b'HTORCH03' + struct.pack('<3Q',1,96,0)
            exact += struct.pack('<8Q',3,6,source,0,0,0,0,0)
            require(body == exact, 'complete original negation request bytes')
            row = self.row
            before = self.observe(driver, source, row['input'], label+'/input')
            owner_before, _, _ = driver.n.capture(label+'/native-negation-before')
            status, records = driver.n.exchange(body,label)
            owner_after, _, _ = driver.n.capture(label+'/native-negation-after')
            driver.operations.append(dict(label=label, request_sha256=sha(body),actual_status=status,
                expected_status=0,record_count=len(records),predecessor_expected_status=7,
                successor_reference_index=row['index']))
            driver.check(label+'/actual-frame-status',status,0)
            driver.check(label+'/one-negation-result',[record['opcode'] for record in records],[3])
            result = records[0]['value']
            source_node = driver.owner.public_node(owner_before,source)
            result_node = driver.owner.public_node(owner_after,result)
            driver.owner._f64_source_unchanged(owner_before,owner_after,source_node)
            driver.owner.f64_role(owner_after,result_node)
            driver.owner._f64_fresh_result(owner_before,owner_after,result_node,[source_node])
            role = 'f64-neg-successor-result'
            require(role not in driver.live,'fresh temporary F64 negation output role')
            driver.live[role] = result
            got = self.observe(driver,result,row['result'],label+'/result')
            for key in ('tensor_identity','storage_identity','native_version_identity'):
                driver.check(label+'/fresh-'+key,got[key] != before[key],True)
            after = driver.n.tensor(source)
            require(set(after) == set(before),'complete source observation fields')
            for key in before:
                if key != 'owner':
                    driver.check(label+'/input-unchanged/'+key,after[key],before[key])
            driver.rng_unchanged(label+'/successful-negation')
            driver.drop(role)
            accepted.append(label)
            return status

        namespace = dict(original_entry.__globals__)
        namespace['reject'] = reject
        cloned = types.FunctionType(original_entry.__code__,namespace,original_entry.__name__,
                                    original_entry.__defaults__,original_entry.__closure__)
        cloned.__kwdefaults__ = original_entry.__kwdefaults__
        require(cloned.__code__ is original_entry.__code__,'all original87 recipe code retained')

        def execute(driver):
            require(not driver.operations and not accepted,'fresh complete87 entry-only driver')
            details = cloned(driver)
            require(details['actual_refusals'] == len(driver.operations) == 87 and accepted == [LABEL],
                    'every original request and exactly one numeric successor executed')
            successful = [row for row in driver.operations if row['expected_status'] == 0]
            retained = [row for row in driver.operations if row['expected_status'] != 0]
            require(len(retained) == 86 and [row['label'] for row in successful] == [LABEL]
                    and all(row['actual_status'] == row['expected_status'] for row in driver.operations),
                    '86 original refusals and one independently checked result')
            return dict(details,actual_refusals=86,actual_successful_negations=1,
                        original_entry_requests=87,negation_successor=self.proof)
        return execute


def adapter_type(original_type, repository, reader_directory, reference):
    class Adapter(original_type):
        def bind_package(self, inputs):
            digest = super().bind_package(inputs)
            self.wide_neg_successor = Successor(repository,reader_directory,reference,inputs)
            # Keep the frozen current_controls module and code available for
            # exact binding and independent replay. Only the active import
            # namespace gains the scoped numeric successor.
            active = types.ModuleType('current_f64_negation_successor_controls')
            active.__dict__.update(self.current_controls.__dict__)
            active.entry_controls = self.wide_neg_successor.wrap_entry(self.current_controls.entry_controls)
            sys.modules['controls'] = active
            return digest

        def candidate_admission(self, directory, inputs):
            result = super().candidate_admission(directory,inputs)
            result['wide_neg_successor'] = self.wide_neg_successor.proof
            result['other85_expected_statuses_unchanged'] = False
            result['unchanged_other_refusal_expectations'] = 84
            result['retained_refusals'] = 86
            result['actual_numeric_successors'] = 1
            return result
    return Adapter
