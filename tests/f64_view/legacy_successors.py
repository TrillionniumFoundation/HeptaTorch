"""Three explicit public F64 view successors for the original mandatory gates.

The frozen87 recipe code is retained. Only its two named view requests receive
new contracts; the other85 requests still reach their existing observers. The
public192 successor retains its original two scope inputs and the AD refusal.
Mutation/root-lifetime coverage is supplied by the unchanged view188 comparisons
for rows174/175/176 on the same separately admitted actual native build.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
import lzma
from pathlib import Path
import struct
import types

from . import native as view_native

CONTROLS = '16daead20d5bf9934371a8283b18a94dc03a45d3390b346d7ef5d7019d8820d8'
ENTRY = 'ba632c235c28f95fa7464673d4ca5b4f556d727712154f5d8a349e4cf7c7078b'
LABELS = ('unsupported-F64-view-111', 'same-packet-F64-public-alias-view')
PUBLIC = 'public-f64-as-strided'
INDEX = dict(zip((*LABELS, PUBLIC), (174, 175, 176)))
ORIGINAL_CASES = '57c00c0f5803d905975c14138c1868e0d9e8c5f62d63b71146db2338bc54bf90'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def definitions(raw):
    lines = raw.splitlines(keepends=True)
    return {node.name: b''.join(lines[node.lineno-1:node.end_lineno]) for node in ast.parse(raw).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class Family:
    """One exact two-element no-AD family amid all unrelated live owner objects.

Every old snapshot and base frame predicate runs. The old dense-only F64 role
predicate has an explicit, temporary extension for this root and this one view.
No owner snapshot, descriptor, or table is projected or normalized.
"""
    def __init__(self, owner, before, before_raw, source, row, reference, factory):
        self.owner, self.before, self.before_raw = owner, before, before_raw
        self.row, self.reference, self.factory = row, reference, factory
        self.old_role = owner.f64_role
        self.counter = before['header']['incarnation']
        self.next = before['header']['next']
        self.root_index = self.next if factory else owner.public_node(before, source)
        self.view_index = self.next + int(factory)
        self.original_source = source
        self.source = source
        self.view = None
        require(row['spec']['base']['shape'] == [2] and len(row['spec']['base']['words']) == 2,
                'only exact legacy two-element inputs enter this owner extension')
        operation = row['spec']['operations'][0]
        require(operation == dict(operation='view', name='view', source='root', shape=[2], strides=[1], offset=0),
                'only the exact original view geometry enters this owner extension')

    def node(self, snapshot, index):
        return snapshot['nodes'].get(str(index))

    def live(self, snapshot, index):
        node = self.node(snapshot, index)
        return node is not None and self.owner.w(node, 304) != 0

    def exact_family(self, snapshot):
        owner = self.owner
        require(self.live(snapshot, self.root_index) and self.live(snapshot, self.view_index),
                'the successful alias retains both exact native descriptors')
        root, view = [self.node(snapshot, index) for index in (self.root_index, self.view_index)]
        storage, version = root[27:29]
        storage_row = owner._live_row(snapshot, 2, storage, 'legacy-view-Storage')
        version_row = owner._live_row(snapshot, 3, version, 'legacy-view-Version')
        require(storage_row[3:5] == [16, 16] and version_row[2] == 0,
                'original two-element Storage extent and unchanged Version value')
        root_token, view_token = root[24], view[24]
        if self.factory:
            expected_root = [0] * 84
            expected_root[0:3] = [1, 2, (1 << 63) + storage_row[2]]
            expected_root[8] = self.root_index
            expected_root[12] = 2
            expected_root[16] = 1
            expected_root[24:31] = [root_token, 1, 1, storage, version, 0, 1]
            expected_root[38] = 1
            expected_root[40] = self.counter + 1
            expected_root[72] = 4
            expected_root[75] = 1
        else:
            expected_root = list(self.before['nodes'][str(self.root_index)])
            require(expected_root[24] == self.original_source and expected_root[26] == expected_root[75] == 0,
                    'original dense source has no hidden preexisting aliases')
            expected_root[26] += 1
            expected_root[75] += 1
        require(list(root) == expected_root, 'all84 root words: only the declared retained-view edges change')
        expected_view = [0] * 84
        expected_view[0:3] = [1, 2, (1 << 63) + storage_row[2]]
        expected_view[8] = self.root_index
        expected_view[12] = 2
        expected_view[16] = 1
        expected_view[24:31] = [view_token, 1, 0, storage, version, 0, 1]
        expected_view[38] = 1
        expected_view[40] = self.counter + 1 + int(self.factory)
        expected_view[51] = 1
        expected_view[72] = 4
        expected_view[80:84] = [self.root_index, root[40], version_row[2], 1]
        require(list(view) == expected_view, 'all84 independently derived alias words, no AD/Saved edges')
        require(root_token != view_token and root[40] != view[40], 'fresh Tensor capability and incarnation')
        require(snapshot['header']['next'] == self.next + 1 + int(self.factory)
                and snapshot['header']['incarnation'] == self.counter + 1 + int(self.factory),
                'exact original factory/view constructor census')
        require(storage_row[5:] == [2, 1, 0] and version_row[3:] == [2, 0, 0, 0, 0],
                'exact shared Storage/Version ownership counts')
        require(owner._live_row(snapshot, 1, root_token, 'legacy-root-Tensor') ==
                [root_token >> 32, 1, storage, version, 0, 16, self.root_index, 0],
                'complete actual root Tensor owner row')
        require(owner._live_row(snapshot, 1, view_token, 'legacy-view-Tensor') ==
                [view_token >> 32, 1, storage, version, 0, 16, self.view_index, 0],
                'complete actual view Tensor owner row')
        return root_token, view_token

    def role(self, snapshot, index):
        if index in (self.root_index, self.view_index) and self.live(snapshot, self.view_index):
            root, view = self.exact_family(snapshot)
            node = self.node(snapshot, index)
            return dict(node=index, dtype=4, width=8, storage=node[27], version=node[28],
                actual_version_value=0, storage_bytes=16, requires_grad=False, autograd_operation=0,
                ViewInfo_absent=index == self.root_index, Saved_absent=True,
                exact_original_view_successor=self.row['spec']['legacy_label'],
                all84_family_descriptor_words_checked=True, root_token=root, view_token=view)
        return self.old_role(snapshot, index)

    def unrelated(self, after):
        owner = self.owner
        for key, before_node in self.before['nodes'].items():
            if owner.w(before_node, 304) and int(key) != self.root_index:
                require(after['nodes'].get(key) == before_node, 'all84 words of each unrelated live descriptor unchanged')
        before_storage = None if self.factory else self.before['nodes'][str(self.root_index)][27]
        before_version = None if self.factory else self.before['nodes'][str(self.root_index)][28]
        for kind, rows in self.before['tables'].items():
            for slot, old in enumerate(rows):
                if not old[1]:
                    continue
                token = (old[0] << 32) | (int(kind) << 24) | (slot + 1)
                if (int(kind), token) in ((2, before_storage), (3, before_version)):
                    continue
                require(after['tables'][kind][slot] == old, 'every unrelated live owner row is unchanged')

    def inventory(self, raw, active):
        before = view_native.Memory(self.before_raw)
        after = view_native.Memory(raw)
        expected = {kind: view_native.live_rows(before, kind) for kind in range(1, 9)}
        if not self.factory:
            node = before.node(self.original_source)
            expected[1].pop(self.original_source)
            expected[2].pop(node[27])
            expected[3].pop(node[28])
        if active:
            root = view_native.indexed_node(after, self.root_index)
            storage, version = root[27:29]
            physical = after.row(2, storage)
            expected[2][storage] = (storage >> 32, 1, physical[2], 16, 16, len(active), 1, 0)
            expected[3][version] = (version >> 32, 1, 0, len(active), 0, 0, 0, 0)
            for index in active:
                node = view_native.indexed_node(after, index)
                token = node[24]
                expected[1][token] = (token >> 32, 1, storage, version, 0, 16, index, 0)
        for page_index in range((after.q(8) + 255) // 256):
            address = after.q(4096 + page_index * 8)
            buffer = after.q(6144 + page_index * 8)
            require(address >> 63 == 1, 'actual complete descriptor-page table')
            storage = after.row(6, buffer)[2]
            page_storage = (storage >> 32, 1, address & ((1 << 63) - 1), 172032, 172032, 1, 1, 0)
            page_buffer = (buffer >> 32, 1, storage, 1, 0, 0, 0, 0)
            if storage in expected[2]:
                require(expected[2][storage] == page_storage, 'unchanged preexisting descriptor-page Storage')
            if buffer in expected[6]:
                require(expected[6][buffer] == page_buffer, 'unchanged preexisting descriptor-page Buffer')
            expected[2][storage], expected[6][buffer] = page_storage, page_buffer
        for kind in range(1, 9):
            require(view_native.live_rows(after, kind) == expected[kind],
                    'entire actual live owner inventory has only the declared family/page additions: ' + str(kind))

    def committed(self, after, after_raw, records):
        owner = self.owner
        root, view = self.exact_family(after)
        require([r['opcode'] for r in records] == ([97, 111] if self.factory else [111])
                and all(r['payload'] == b'' for r in records)
                and [r['value'] for r in records] == ([root, view] if self.factory else [view]),
                'all original records publish the actual family capabilities')
        self.source, self.view = root, view
        self.unrelated(after)
        self.inventory(after_raw, [self.root_index, self.view_index])
        old_live = {int(key) for key, node in self.before['nodes'].items() if owner.w(node, 304)}
        new_live = {int(key) for key, node in after['nodes'].items() if owner.w(node, 304)}
        require(new_live == old_live | ({self.root_index, self.view_index} if self.factory else {self.view_index}),
                'exact live descriptor additions, without a hidden temporary tensor')
        for token, expected in ((root, self.row['stages'][1]['tensors']['root']),
                                (view, self.row['stages'][1]['tensors']['view'])):
            memory = view_native.Memory(after_raw)
            node = memory.node(token)
            storage = memory.row(2, node[27])
            physical = after_raw[memory.start + storage[2]:memory.start + storage[2] + storage[3]]
            require(physical == self.reference['blob'](expected['storage']) == self.reference['blob'](expected['logical']),
                    'complete committed native Storage and logical bits equal actual Torch observations')
        return dict(original_request_unchanged=True, all84_family_words_checked=True,
            all_unrelated_live_descriptors_and_owner_rows_unchanged=True,
            actual_fresh_Tensor_shared_Storage_shared_Version=True, retained_true_root=True,
            no_AD_or_Saved=True, exact_record_and_descriptor_census=True,
            all8_live_owner_table_inventories_checked=True,
            numerical_expected_values_computed=False)

    @contextmanager
    def installed(self, driver, request):
        owner = self.owner
        old_boundary = owner.verify_frame_boundary
        owner.f64_role = self.role
        def boundary(before, after, commands, status, records, body, **kwargs):
            if body != request:
                return old_boundary(before, after, commands, status, records, body, **kwargs)
            require(status == 0, 'this explicit successor requires a real successful original request')
            if self.factory:
                expected = [list(struct.unpack_from('<8Q', request, 32 + i*64)) for i in range(2)]
                require(commands == expected, 'the one unchanged97/111 FE request')
                # The prior wrapper explicitly has a single97 scope. Its complete
                # predecessor still checks both actual owners and every record;
                # this method supplies the full new joint factory/view contract.
                previous = owner._f64_original_verify_frame_boundary(before, after, commands, status,
                    records, body, **kwargs)
            else:
                previous = old_boundary(before, after, commands, status, records, body, **kwargs)
            require(kwargs['before_raw'] == self.before_raw and before == self.before,
                    'the original before owner is the exact independently captured source')
            return dict(previous, public_f64_view_successor=self.committed(after, kwargs['after_raw'], records))
        owner.verify_frame_boundary = boundary
        try:
            yield self
        finally:
            owner.verify_frame_boundary = old_boundary
            owner.f64_role = self.old_role


class Successor:
    def __init__(self, reference, actual_build, inputs, source_path=None):
        self.reference, self.actual_build, self.inputs = reference, actual_build, inputs
        if source_path is not None:
            inputs.bind(source_path)
        report = reference['report']
        require(report['cases'] == len(report['rows']) == 188 and report['original_case_sha256'] == ORIGINAL_CASES,
                'one complete actual original188 view reference')
        require(len(actual_build['source_sha256']) == len(actual_build['native_sha256']) == 64
                and actual_build['all_inputs_unchanged'] is True
                and actual_build['objects_reused'] is False,
                'caller must supply one explicitly admitted complete actual build')
        self.rows = {}
        for label, index in INDEX.items():
            row = report['rows'][index]
            require(row['index'] == index and row['spec']['legacy_label'] == label
                    and row['accepted'] and row['all_Torch_Tensors_released'] and row['rng_unchanged'],
                    'exact original legacy input and actual accepted view/lifetime observations')
            expected_words = ((0x3ff0000000000000, 0x8000000000000000) if index == 174 else
                              (0x3ff0000000000000, 0x4000000000000000) if index == 175 else
                              (0, 0x8000000000000000))
            require(row['spec']['base'] == dict(shape=[2], words=list(expected_words))
                    and row['spec']['legacy_same_frame'] is (index == 175), 'exact original literal inputs and FE selection')
            initial = row['stages'][0]['tensors']['root']
            raw = struct.pack('<2Q', *expected_words)
            require(reference['blob'](initial['logical']) == reference['blob'](initial['storage']) == raw,
                    'actual reference consumes original raw signed-zero/F64 bits')
            self.rows[label] = row
        self.proof = dict(schema='heptatorch.f64-view.original3-successors.v1',
            source_sha256=actual_build['source_sha256'], native_sha256=actual_build['native_sha256'],
            manifest_sha256=actual_build['manifest_sha256'], reference_sha256=reference['report_sha256'],
            original_case_sha256=ORIGINAL_CASES, original_reference_indices=INDEX,
            original87_requests=87, retained87_refusals=61, retained_numeric_successors=24,
            new87_view_successors=2, original192_cases=192, original192_scope_controls=2,
            retained192_scope_refusals=1, new192_view_successors=1,
            full_lifetime_companion_indices=[174,175,176], companion_execution_required=True,
            strict_reference_qualified=reference.get('strict_reference_qualified', False),
            original_frozen_recipe_code_modified=False, all_inputs_and_expected_bytes_original=True,
            native_or_Torch_executed_by_constructor=False, full_pytorch_rewrite_complete=False)

    def observe(self, driver, token, expected, label):
        got = driver.n.tensor(token)
        for key in ('dtype', 'shape', 'strides', 'numel', 'element_size', 'storage_bytes', 'version',
                    'requires_grad', 'is_leaf', 'is_view', 'retains_grad'):
            driver.check(label + '/' + key, got[key], expected[key])
        driver.check(label + '/device', got['device'], 'cpu')
        driver.check(label + '/storage-offset-elements', got['storage_offset_elements'], expected['storage_offset'])
        driver.check(label + '/storage-offset-bytes', got['storage_offset_bytes'], expected['storage_offset']*8)
        driver.check(label + '/no-history', got['grad_fn'], None)
        driver.check(label + '/no-gradient', got['grad_is_none'], True)
        driver.check(label + '/all-logical-bits', got['logical'], self.reference['blob'](expected['logical']))
        driver.check(label + '/all-storage-bits', got['storage'], self.reference['blob'](expected['storage']))
        return got

    def success(self, driver, label, body, expected):
        require(label in LABELS and expected == 7, 'only the two exact original refusal labels change')
        row = self.rows[label]
        factory = label == LABELS[1]
        source = None if factory else driver.live['f64']
        exact = view_native.factory_view_packet(self.wire, row['spec']) if factory else view_native.view_packet(self.wire, source, row['spec']['operations'][0])
        require(body == exact, 'complete original legacy request bytes, no replaced input or geometry')
        original = None if factory else self.observe(driver, source, row['stages'][0]['tensors']['root'], label + '/source-before')
        before, endpoint, raw = driver.n.capture(label + '/successor-before')
        family = Family(driver.owner, before, raw, source, row, self.reference, factory)
        with family.installed(driver, body):
            status, records = driver.n.exchange(body, label)
            driver.operations.append(dict(label=label, request_sha256=sha(body), actual_status=status,
                expected_status=0, record_count=len(records), predecessor_expected_status=7,
                successor_reference_index=row['index'], successor_kind='public-F64-view'))
            driver.check(label + '/actual-frame-status', status, 0)
            root, alias = family.source, family.view
            require(root is not None and alias is not None, 'actual committed capabilities captured by frame predicate')
            source_now = self.observe(driver, root, row['stages'][1]['tensors']['root'], label + '/source-after')
            view_now = self.observe(driver, alias, row['stages'][1]['tensors']['view'], label + '/view')
            driver.check(label + '/fresh-Tensor', view_now['tensor_identity'] != source_now['tensor_identity'], True)
            for key in ('storage_identity', 'native_version_identity'):
                driver.check(label + '/shared-' + key, view_now[key], source_now[key])
            driver.check(label + '/retained-original-root', view_now['base_tensor_identity'], source_now['tensor_identity'])
            driver.n.one(41, alias)
            restored = driver.n.tensor(root)
            baseline = source_now if factory else original
            for key in baseline:
                if key != 'owner':
                    driver.check(label + '/source-unchanged-after-view-release/' + key, restored[key], baseline[key])
            cleaned, _, cleaned_raw = driver.n.capture(label + '/view-released')
            require(not family.live(cleaned, family.view_index), 'view descriptor and all owner edges released')
            family.unrelated(cleaned)
            family.inventory(cleaned_raw, [family.root_index])
            if not factory:
                require(cleaned['nodes'][str(family.root_index)] == before['nodes'][str(family.root_index)],
                        'all84 original source words restored after releasing only the temporary alias')
                for kind, offset in (('2', 27), ('3', 28)):
                    token = before['nodes'][str(family.root_index)][offset]
                    require(driver.owner._live_row(cleaned, int(kind), token, 'released-alias-object') ==
                            driver.owner._live_row(before, int(kind), token, 'original-shared-object'),
                            'original Storage and Version owner rows restored exactly')
            else:
                driver.n.one(41, root)
                final, _, final_raw = driver.n.capture(label + '/entire-FE-family-released')
                require(not family.live(final, family.root_index) and not family.live(final, family.view_index),
                        'both original FE-created capabilities are explicitly released')
                family.unrelated(final)
                family.inventory(final_raw, [])
            driver.rng_unchanged(label + '/successful-view')
        return status

    def prepare_entry(self, original, wire):
        self.wire = wire
        raw = self.inputs.bind(Path(original.__code__.co_filename), CONTROLS)
        require(sha(definitions(raw)['entry_controls']) == ENTRY and original.__name__ == 'entry_controls',
                'complete unchanged original87 recipe code')
        prior = original.__globals__['reject']
        accepted = []
        def reject(driver, label, body, expected, **kwargs):
            if label not in LABELS:
                return prior(driver, label, body, expected, **kwargs)
            require(not kwargs and label == LABELS[len(accepted)], 'exact ordered two-view successor inventory')
            result = self.success(driver, label, body, expected)
            accepted.append(label)
            return result
        namespace = dict(original.__globals__)
        namespace['reject'] = reject
        changed = types.FunctionType(original.__code__, namespace, original.__name__,
            original.__defaults__, original.__closure__)
        changed.__kwdefaults__ = original.__kwdefaults__
        require(changed.__code__ is original.__code__, 'all original87 request recipe bytecode retained')
        return changed, accepted


class EntryOverlay:
    """Compose two view handlers with the existing exact23 arithmetic/one negation handlers."""
    def __init__(self, arithmetic, view):
        self.arithmetic, self.view = arithmetic, view

    def __getattr__(self, name):
        return getattr(self.arithmetic, name)

    def wrap_entry(self, original_entry):
        require(Path(original_entry.__code__.co_filename).resolve()==self.reader/'controls736.py'
            and original_entry.__name__=='entry_controls','complete unchanged original87 entry code')
        original_entry,accepted_views=self.view.prepare_entry(original_entry,original_entry.__globals__['WIRE'])
        original_reject=original_entry.__globals__['reject'];accepted=[]
        arithmetic_globals=self.arithmetic.wrap_entry.__func__.__globals__
        arithmetic_labels=arithmetic_globals['LABELS'];fe=arithmetic_globals['FE'];retained_bool=arithmetic_globals['RETAINED_BOOL']
        def reject(driver,label,body,expected,**kwargs):
            if label not in self.rows:return original_reject(driver,label,body,expected,**kwargs)
            require(not kwargs and expected==7 and label==arithmetic_labels[len(accepted)],
                'only the23 exact ordered original arithmetic requests receive successors')
            status=(self.fe_success(driver,label,body,expected,original_entry.__globals__['WIRE'])
                if label==fe else self.single_success(driver,label,body,expected))
            accepted.append(label);return status
        namespace=dict(original_entry.__globals__);namespace['reject']=reject
        clone=types.FunctionType(original_entry.__code__,namespace,original_entry.__name__,
            original_entry.__defaults__,original_entry.__closure__);clone.__kwdefaults__=original_entry.__kwdefaults__
        cloned,accepted_neg=self.negation.prepare_entry(clone)
        require(cloned.__code__ is original_entry.__code__,'all original87 recipe code preserved through all three hooks')
        def execute(driver):
            require(not driver.operations and not accepted and not accepted_neg and not accepted_views,'fresh complete87 entry-only driver')
            details=cloned(driver)
            require(details['actual_refusals']==len(driver.operations)==87 and accepted==list(arithmetic_labels)
                and accepted_neg==['unsupported-F64-neg'] and accepted_views==list(LABELS),
                'every original request and each exact successor executed')
            successful=[row for row in driver.operations if row['expected_status']==0]
            retained=[row for row in driver.operations if row['expected_status']!=0]
            expected_labels=list(arithmetic_labels[:-1])+['unsupported-F64-neg',fe]
            require(len(retained)==61 and [row['label'] for row in successful if row['label'] not in LABELS]==expected_labels
                and [row['label'] for row in successful if row['label'] in LABELS]==list(LABELS)
                and len(successful)==26 and all(row['actual_status']==row['expected_status'] for row in driver.operations),
                'exact61 original refusals,24 original numeric successors and2 explicit view successors')
            bool_rows=[row for row in retained if row['label'] in retained_bool]
            require([row['label'] for row in bool_rows]==list(retained_bool)
                and all(row['actual_status']==7 and row['record_count']==0 for row in bool_rows),
                'both actual Bool subtraction refusals retain the exact original contract')
            return dict(details,actual_refusals=61,actual_successful_negations=1,
                actual_successful_mixed_arithmetic=23,actual_successful_FE_arithmetic=1,
                actual_successful_views=2,original_entry_requests=87,negation_successor=self.negation.proof,
                public_f64_arithmetic_successor=self.proof,public_f64_view_successor=self.view.proof)
        return execute


class ScopeOwnerRecording:
    """Retain every actual owner read needed to replay the new scope predicate."""
    def __init__(self, session, folder):
        self.session, self.folder, self.owners = session, folder, []

    def __getattr__(self, name):
        return getattr(self.session, name)

    def owner(self):
        actual = self.session.owner()
        raw = actual['bytes']
        digest = sha(raw)
        path = self.folder / (digest + '.owner.xz')
        if not path.exists():
            packed = lzma.compress(raw, preset=1)
            require(lzma.decompress(packed) == raw, 'complete additional owner transport roundtrip')
            path.write_bytes(packed)
        self.owners.append(dict(frame=self.session.frames, sha256=digest, pointer=actual['pointer'],
                               length=actual['length'], live=actual['live']))
        return actual

    def finish(self):
        self.session.finish()
        transport = json.loads((self.folder / 'raw/transport.json').read_bytes())
        require(transport['exit_code'] == 0, 'actual scope successor child exit zero')
        finish = dict(exit_code=0, stdout_EOF=True, stderr_EOF=True,
            executable_sha256=transport['executable_sha256'], frames=transport['frames'],
            owner_reads=len(self.owners), mapping_reads=0, forced_cleanup=False)
        for name, value in (('owners.json', self.owners), ('maps.json', []), ('finish.json', finish)):
            (self.folder / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def scope_controls(args, wire, folder, old_native, successor):
    """Exact original192 two-scope inventory, with one observed view successor."""
    n = old_native.Native(args, wire, folder)
    if not getattr(n.s, 'scope_owner_replay', False):
        n.s = ScopeOwnerRecording(n.s, folder)
    rows = []
    row = successor.rows[PUBLIC]
    observer = view_native.Observer(n.s, wire, successor.reference)
    observer.check = n.check
    try:
        source = n.create(4, [2], struct.pack('<2Q', 0, 0x8000000000000000), 0)
        before, _ = n.tensor(source)
        gen = n.one(60, 1729)['value']
        rng = n.one(63, gen)['payload']
        n.check('boundary-complete-RNG-width', len(rng), 5056)
        cfg = struct.pack('<8Q', 1, 1, 2, 0, 0, 0, 160, 16)
        payload = wire.Payload(cfg + struct.pack('<2Q', 0, 0x8000000000000000))
        name = 'requires-grad-public-f64-create'
        status, records, _, _ = n.issue([(97, 4, payload)], name)
        n.check(name + '/status', status, 5)
        n.check(name + '/no-records', records, [])
        after, _ = n.tensor(source)
        n.check(name + '/source-unchanged', after, before)
        rows.append(dict(name=name, status='PASS_NATIVE_BOUNDARY_ONLY', error_status=status,
                         complete_owner_rollback=True, torch_compatibility_claimed=False))
        memory = view_native.Memory(n.s.owner()['bytes'])
        counter = memory.node(source)[40]
        model = view_native.OwnerModel(observer, source, row['stages'][0]['tensors']['root'], counter - 1)
        model.observe(observer, {'root': source}, row['stages'][0])
        view = wire.Payload(view_native.view_config(row['spec']['operations'][0]))
        request = wire.packet([(111, source, view, 80)])
        status, records, _, _ = n.issue([(111, source, view, 80)], PUBLIC)
        n.check(PUBLIC + '/status', status, 0)
        n.check(PUBLIC + '/one-view-result', [record['opcode'] for record in records], [111])
        alias = records[0]['value']
        model.register(observer, 'view', alias, row['stages'][1]['tensors']['view'], counter)
        model.observe(observer, {'root': source, 'view': alias}, row['stages'][1])
        n.one(41, alias)
        after, _ = n.tensor(source)
        n.check(PUBLIC + '/source-unchanged', after, before)
        released = view_native.Memory(n.s.owner()['bytes'])
        n.check(PUBLIC + '/view-descriptor-released', view_native.indexed_node(released, model.known['view']['index'])[38], 0)
        n.check(PUBLIC + '/root-has-no-view-edges', [released.node(source)[26], released.node(source)[75]], [0, 0])
        rows.append(dict(name=PUBLIC, status='PASS_EXPLICIT_VIEW_SUCCESSOR', actual_status=status,
            predecessor_error_status=7, request_sha256=sha(request), reference_index=176,
            complete_owner_rollback=False, all84_family_words_checked=True,
            all8_live_owner_inventories_checked=True, original_source_unchanged=True,
            temporary_alias_released=True, torch_compatibility_claimed=successor.proof['strict_reference_qualified']))
        n.check('boundary-complete-RNG-unchanged', n.one(63, gen)['payload'], rng)
        n.one(41, source)
        n.one(66, gen)
        model.finish(observer)
        return dict(status='PASS_SCOPE_WITH_EXPLICIT_VIEW_SUCCESSOR', controls=2, rows=rows,
                    checks=n.checks, frames=n.s.frames, retained_refusals=1, view_successors=1,
                    view_successor=successor.proof, torch_compatibility_claimed=successor.proof['strict_reference_qualified'])
    finally:
        n.finish()


def replay_scope(args, wire, folder, old_native, successor, inputs, expected):
    """Run the complete new two-scope predicate over original physical bytes.

The original Native construction/finish I/O is replaced explicitly. Its wire,
create, check, one, tensor and issue function bodies remain identical. Every
owner used by issue is first read from an existing verified Replay blob, so its
original conditional compression route cannot create a missing evidence file.
"""
    original_type = old_native.Native
    class Session(view_native.Replay):
        scope_owner_replay = True
        @property
        def frames(self):
            return self.at
        def run(self, commands):
            status, records = self.wire.parse(self.exchange(self.wire.packet(commands)))
            require(status == 0, 'original single public scope command succeeds')
            return records
    class Native(original_type):
        def __init__(self, selected, actual_wire, actual_folder):
            require(actual_folder == folder and selected.binary == args.binary,
                    'same admitted actual scope evidence and image paths')
            self.wire, self.folder = actual_wire, actual_folder
            self.s = Session(selected.binary, actual_wire, actual_folder, inputs.bind)
            self.checks, self.journal = 0, []
        def finish(self):
            self.s.finish()
            observed = json.loads(inputs.bind(self.folder / 'observations.json'))
            require(observed == dict(checks=self.checks, frames=self.s.frames, journal=self.journal,
                exit_code=0, stdout_EOF=True, stderr_EOF=True),
                'complete scope observation reconstructed from actual frames and all full owner reads')
    for name in ('check', 'one', 'create', 'view', 'setup', 'tensor', 'command', 'issue'):
        require(getattr(Native, name).__code__ is getattr(original_type, name).__code__,
                'original public Native method bytecode retained: ' + name)
    result = scope_controls(args, wire, folder, types.SimpleNamespace(Native=Native), successor)
    require(result == expected, 'all original/new scope assertions and counts replay exactly')
    return dict(status='PASS_PURE_SCOPE_SUCCESSOR_REPLAY', original_scope_controls=2,
        retained_scope_refusals=1, actual_view_successors=1, frames=result['frames'], checks=result['checks'],
        original_request_and_response_and_all_owner_bytes_consumed=True,
        native_executed=False, Torch_executed=False, reference_sha256=successor.proof['reference_sha256'])
