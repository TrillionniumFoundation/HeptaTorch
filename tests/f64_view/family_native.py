"""Final signed-empty F64 views, exact raw offsets and original owner predicates.

Only the signed-shape and canonical tagged-empty representation are decoded.
Physical metadata13 packets, all84 words and all8 owner inventories stay exact.
"""
import hashlib
import inspect
import struct

from . import native as original
from numeric_predicates import native as numeric
from .family_cases import (HIGH, CAP, WORD, EMPTY_ALLOCATION_SPEC, EMPTY_ALLOCATION_SHA256,
    HIGH_ALLOCATION_SPEC, HIGH_ALLOCATION_SHA256, ADDITIONAL_FAULTS, OFFSET_FAULTS,
    EMPTY_CASES, EMPTY_CASE_SHA256, WIDE_CASES, WIDE_CASE_SHA256,
    OFFSET_CASES, OFFSET_CASE_SHA256, CHAINS, CHAIN_SHA256)

sha = lambda raw: hashlib.sha256(raw).hexdigest()
require = original.require

RAW_SOURCE = inspect.getsource(numeric.Observer.tensor)
RAW_SUBSTITUTIONS = [
    ("        require(node[29]%width==0 and tensor_row[7]<=3,'aligned offset and retained flags')",
     """        encoded_offset = node[29] >= HIGH
        if encoded_offset:
            decoded_element_offset = node[29] & (HIGH - 1)
            require(dtype == 4 and width == 8 and node[1] == 0 and node[4] == 0
                    and node[51] == node[83] == 1 and CAP < decoded_element_offset <= HIGH - 1,
                    'canonical tag belongs only to an actual no-AD empty F64 view')
            self.check('tagged empty Tensor row has no span or AD retain flags', list(tensor_row[5:6]) + list(tensor_row[7:8]), [0, 0])
            self.check('tagged empty data pointer is actual Storage base', node[2], HIGH + storage[2])
            self.check('tagged empty has all original no-AD fields zero',
                [node[offset // 8] for offset in (24,32,40,48,56,288,384,584,592,608,616,624,632)], [0] * 13)
            self.check('tagged metadata word round trips exact independent element offset',
                node[29], HIGH | decoded_element_offset)
        else:
            require(node[29]%width==0 and tensor_row[7]<=3,'aligned offset and retained flags')"""),
    ("storage_bytes=storage[3],storage_offset=node[29]//width,version=version[2],",
     "storage_bytes=storage[3],storage_offset=decoded_element_offset if encoded_offset else node[29]//width,version=version[2],"),
]
DECODED_SOURCE = RAW_SOURCE
for old, new in RAW_SUBSTITUTIONS:
    assert DECODED_SOURCE.count(old) == 1
    DECODED_SOURCE = DECODED_SOURCE.replace(old, new)
inverse = DECODED_SOURCE
for old, new in reversed(RAW_SUBSTITUTIONS):
    assert inverse.count(new) == 1
    inverse = inverse.replace(new, old)
assert inverse == RAW_SOURCE
context = dict(numeric.__dict__, HIGH=HIGH, CAP=CAP)
exec(compile('class OffsetRawObserver(Observer):\n' + DECODED_SOURCE,
             '<original raw observer with explicit tagged-empty representation>', 'exec'), context)
OffsetRawObserver = context['OffsetRawObserver']


class SignedObserver(OffsetRawObserver):
    def tensor(self, token):
        actual, node = super().tensor(token)
        words = actual['shape']
        if any(value >= (1 << 63) for value in words):
            self.check('signed metadata remains an actual no-AD empty F64 view',
                       [actual['dtype'], actual['numel'], actual['requires_grad'], actual['is_view']],
                       ['torch.float64', 0, False, True])
            signed = [value if value < (1 << 63) else value - (1 << 64) for value in words]
            self.check('every signed dimension round-trips to the exact wire word',
                       struct.pack('<' + 'q' * len(signed), *signed), struct.pack('<' + 'Q' * len(words), *words))
            actual = dict(actual, shape=signed)
        return actual, node


OWNER_SOURCE = inspect.getsource(original.OwnerModel)
OWNER_OLD_EXPRESSION = 'expected[12:12 + len(shape)] = shape'
OWNER_NEW_EXPRESSION = 'expected[12:12 + len(shape)] = [dimension & MASK for dimension in shape]'
assert OWNER_SOURCE.count(OWNER_OLD_EXPRESSION) == 1
SIGNED_OWNER_SOURCE = OWNER_SOURCE.replace(OWNER_OLD_EXPRESSION, OWNER_NEW_EXPRESSION)
assert SIGNED_OWNER_SOURCE.replace(OWNER_NEW_EXPRESSION, OWNER_OLD_EXPRESSION) == OWNER_SOURCE

PRIOR_SIGNED_OWNER_SOURCE = SIGNED_OWNER_SOURCE
OWNER_SUBSTITUTIONS = [
    ("                offset = geometry['storage_offset'] * 8",
     """                element_offset = geometry['storage_offset']
                encoded_offset = geometry['numel'] == 0 and element_offset > CAP
                if encoded_offset:
                    require(name != 'root' and geometry['dtype'] == 'torch.float64'
                            and not geometry['requires_grad'] and geometry['is_view']
                            and CAP < element_offset <= HIGH - 1,
                            'independent Torch geometry permits only canonical high-offset empty F64 view encoding')
                offset = (HIGH | element_offset) if encoded_offset else element_offset * 8
                data_offset = 0 if encoded_offset else offset
                remaining_span = 0 if encoded_offset else max(self.base['storage_bytes'] - offset, 0)"""),
    ("(1 << 63) + self.storage_geometry[0] + offset]", "(1 << 63) + self.storage_geometry[0] + data_offset]"),
    ("max(self.base['storage_bytes'] - offset, 0), item['index'], 0)", "remaining_span, item['index'], 0)"),
]
OFFSET_OWNER_SOURCE = PRIOR_SIGNED_OWNER_SOURCE
for old, new in OWNER_SUBSTITUTIONS:
    assert OFFSET_OWNER_SOURCE.count(old) == 1
    OFFSET_OWNER_SOURCE = OFFSET_OWNER_SOURCE.replace(old, new)
inverse = OFFSET_OWNER_SOURCE
for old, new in reversed(OWNER_SUBSTITUTIONS):
    assert inverse.count(new) == 1
    inverse = inverse.replace(new, old)
assert inverse == PRIOR_SIGNED_OWNER_SOURCE
context = dict(original.__dict__, HIGH=HIGH, CAP=CAP)
exec(compile(OFFSET_OWNER_SOURCE, '<all84/all8 predicates with explicitly derived high-offset empty representation>', 'exec'), context)
SignedOwnerModel = context['OwnerModel']


def compare(n, row, original187=False):
    spec = row['spec']; stages = row['stages']
    generator, rng = n.begin()
    before_counter = original.Memory(n.s.owner()['bytes']).q(120)
    root = n.create(4, spec['base']['shape'], original.raw_input(spec['base']))
    model = SignedOwnerModel(n, root, stages[0]['tensors']['root'], before_counter)
    handles = {'root': root}
    model.observe(n, handles, stages[0])
    counter = original.Memory(n.s.owner()['bytes']).q(120)
    status, records, _, _ = n.issue(original.view_packet(n.wire, root, spec['operations'][0]))
    gap = False  # All accepted actual Torch rows require native success.
    if not row['accepted'] or gap:
        n.check('observed Torch refusal or explicitly retained native capacity refusal', status, spec['native_error'])
        n.check('rejected native view publishes no capability', records, [])
        model.observe(n, handles, stages[0])
        n.one(41, handles.pop('root')); n.end(generator, rng); model.finish(n)
        if gap:
            return dict(index=row['index'], compatible=False, counted_as_Torch_error=False,
                counted_as_compatible=False, known_native_capacity_gap=spec['comparison_scope'],
                actual_Torch_accepted=True, actual_native_status=status,
                actual_Torch_metadata=stages[1]['tensors']['view'], complete_original_owner_rollback=True)
        return dict(index=row['index'], compatible=True, accepted=False,
            actual_Torch_error=row['error'], actual_native_status=status, complete_original_owner_rollback=True)
    n.check('actual signed-empty F64 view accepted by native', status, 0)
    n.check('exact one view result', [record['opcode'] for record in records], [111])
    handles['view'] = records[0]['value']
    model.register(n, 'view', handles['view'], stages[1]['tensors']['view'], counter)
    model.observe(n, handles, stages[1])
    if original187:
        n.one(41, handles.pop('root')); model.observe(n, handles, stages[2])
        n.one(41, handles.pop('view')); model.observe(n, handles, stages[3])
        n.end(generator, rng); model.finish(n)
        return dict(index=187, compatible=True, accepted=True, original_recipe_unchanged=True,
                    historical_failure_preserved_as_before_evidence=True, actual_signed_metadata=True,
                    supplemental_public_root_and_view_release=True)
    n.one(100, handles['view'], WORD, 0); model.observe(n, handles, stages[2])
    n.one(41, handles.pop('root')); model.observe(n, handles, stages[3])
    counter = original.Memory(n.s.owner()['bytes']).q(120)
    operation = dict(shape=[2], strides=[1], offset=0)
    handles['child'] = n.one(111, handles['view'], n.wire.Payload(original.view_config(operation)), 80)['value']
    model.register(n, 'child', handles['child'], stages[4]['tensors']['child'], counter)
    model.observe(n, handles, stages[4])
    n.one(12, handles['child'], 0, WORD); model.observe(n, handles, stages[5])
    n.one(41, handles.pop('view')); model.observe(n, handles, stages[6])
    n.one(41, handles.pop('child')); model.observe(n, handles, stages[7])
    n.end(generator, rng); model.finish(n)
    return dict(index=row['index'], compatible=True, accepted=True, actual_signed_metadata=True,
        empty_fill_shared_version_bump=True, root_lifetime_matches_actual_Torch=True,
        recreated_nonempty_child_and_mutation_match_actual_Torch=True,
        all84_live_descriptor_words_and_all8_owner_tables_checked=True)


def specialized_original(function, **bindings):
    source = inspect.getsource(function)
    context = dict(original.__dict__, OwnerModel=SignedOwnerModel, **bindings)
    exec(compile(source, '<unchanged original control with tagged-empty owner model>', 'exec'), context)
    return context[function.__name__]


def compare_chain(n,row):
    spec=row['spec'];stages=row['stages'];at=0
    generator,rng=n.begin()
    counter=original.Memory(n.s.owner()['bytes']).q(120)
    root=n.create(4,spec['base']['shape'],original.raw_input(spec['base']))
    model=SignedOwnerModel(n,root,stages[at]['tensors']['root'],counter)
    handles={'root':root};model.observe(n,handles,stages[at]);source='root';at+=1
    for index,offset in enumerate(spec['offsets']):
        name='alias%d'%index
        operation=dict(shape=spec['shape'],strides=spec['strides'],offset=offset)
        counter=original.Memory(n.s.owner()['bytes']).q(120)
        handles[name]=n.one(111,handles[source],n.wire.Payload(original.view_config(operation)),80)['value']
        model.register(n,name,handles[name],stages[at]['tensors'][name],counter)
        model.observe(n,handles,stages[at]);at+=1
        n.one(100,handles[name],WORD,0);model.observe(n,handles,stages[at]);at+=1
        if index==0:
            n.one(41,handles.pop('root'));model.observe(n,handles,stages[at]);at+=1
        source=name
    counter=original.Memory(n.s.owner()['bytes']).q(120)
    operation=dict(shape=[2],strides=[1],offset=0)
    handles['child']=n.one(111,handles[source],n.wire.Payload(original.view_config(operation)),80)['value']
    model.register(n,'child',handles['child'],stages[at]['tensors']['child'],counter)
    model.observe(n,handles,stages[at]);at+=1
    n.one(12,handles['child'],0,WORD);model.observe(n,handles,stages[at]);at+=1
    for index in (2,0,4,1,3):
        n.one(41,handles.pop('alias%d'%index));model.observe(n,handles,stages[at]);at+=1
    n.one(41,handles.pop('child'));model.observe(n,handles,stages[at]);at+=1
    n.check('every independent Torch alias-chain stage checked',at,len(stages))
    n.end(generator,rng);model.finish(n)
    return dict(index=row['index'],compatible=True,accepted=True,
        aliases=len(spec['offsets']),actual_Torch_stages=len(stages),
        encoded_and_ordinary_empty_aliases_created_after_public_root_release=True,
        nonempty_child_recreated_from_high_offset_empty=True,
        empty_mutations_share_original_Version=True,
        all84_live_descriptor_words_and_all8_owner_tables_checked=True)


def replace_words(n,replacements):
    before=n.s.owner();raw=before['bytes'];after=bytearray(raw)
    old=[]
    assert len({offset for offset,value in replacements})==len(replacements)
    for offset,value in replacements:
        assert offset%8==0 and 0<=offset<=len(raw)-8 and 0<=value<(1<<64)
        previous=raw[offset:offset+8];encoded=struct.pack('<Q',value)
        assert previous!=encoded
        old.append((offset,struct.unpack('<Q',previous)[0]))
        after[offset:offset+8]=encoded
    if not n.s.replay:
        with open('/proc/%d/mem'%n.s.pid,'r+b',buffering=0) as stream:
            for offset,value in replacements:
                stream.seek(before['pointer']+offset)
                assert stream.read(8)==raw[offset:offset+8]
                stream.seek(before['pointer']+offset)
                assert stream.write(struct.pack('<Q',value))==8
    n.check('exactly the explicitly declared physical diagnostic words changed',
            n.s.owner(),dict(before,bytes=bytes(after)))
    return old,before


def fault_control(n,reference,fault):
    label,setup,changes=fault
    generator,rng=n.begin()
    model=None
    if setup in ('f32','nonempty'):
        expected=reference['f32_fault_setup' if setup=='f32' else 'nonempty_fault_setup']
        root=n.create(0,[2],struct.pack('<II',0x3f800000,0x80000000)) if setup=='f32' else n.create(4,[2],struct.pack('<QQ',0x3ff0000000000000,0x8000000000000000))
        operation=dict(shape=[0] if setup=='f32' else [2],strides=[1],offset=0)
        view=n.one(111,root,n.wire.Payload(original.view_config(operation)),80)['value']
        handles={'root':root,'view':view}
        n.check('uncorrupted special fault-setup audit',n.one(53)['value'],1)
        for name,token in handles.items():
            n.check('actual uncorrupted special reference metadata and bits: '+name,
                    n.tensor(token)[0],n.expected(expected[name]))
    else:
        row=reference['rows'][1 if setup=='low' else 2]
        spec=row['spec'];counter=original.Memory(n.s.owner()['bytes']).q(120)
        root=n.create(4,spec['base']['shape'],original.raw_input(spec['base']))
        model=SignedOwnerModel(n,root,row['stages'][0]['tensors']['root'],counter)
        handles={'root':root};stage=row['stages'][0]
        if setup!='root':
            token=n.one(111,root,n.wire.Payload(original.view_config(spec['operations'][0])),80)['value']
            handles['view']=token;stage=row['stages'][1]
            model.register(n,'view',token,stage['tensors']['view'],counter+1)
        model.observe(n,handles,stage)
    initial=original.Memory(n.s.owner()['bytes']);root_node=initial.node(root)
    root_index=initial.row(1,root)[6];root_storage,root_version=root_node[27:29]
    replacements=[];details=[]
    for kind,target,field,value in changes:
        token=handles[target]
        if kind=='descriptor':
            assert field%8==0 and 0<=field<672
            index=initial.row(1,token)[6];page=initial.q(4096+(index>>8)*8)
            assert page>>63==1
            offset=initial.start+(page&((1<<63)-1))+(index&255)*672+field
        else:
            assert kind=='row' and field%8==0 and 0<=field<64
            slot=token&0xffffff;assert 0<slot<=initial.slots
            offset=139264+128+(slot-1)*64+field
        bad=initial.q(offset)+value['add'] if isinstance(value,dict) else value
        replacements.append((offset,bad))
        details.append(dict(kind=kind,target=target,field=field,absolute_owner_offset=offset,
                            original=initial.q(offset),injected=bad))
    saved,before=replace_words(n,replacements)
    try:
        status,records,_,_=n.issue(n.wire.packet([(53,)]))
        n.check('actual native owner audit rejects tag forgery: '+label,status,9)
        n.check('tag forgery audit publishes no capabilities',records,[])
    finally:
        replaced,_=replace_words(n,saved)
        n.check('restore exactly every declared diagnostic word',replaced,replacements)
        n.check('complete original owner identity and bytes restored',n.s.owner(),before)
    if model is not None:
        model.observe(n,handles,stage)
    else:
        n.check('restored special fault-setup audit',n.one(53)['value'],1)
        for name,token in handles.items():
            n.check('restored special reference metadata and bits: '+name,
                    n.tensor(token)[0],n.expected(expected[name]))
    for name in reversed(list(handles)):n.one(41,handles.pop(name))
    n.end(generator,rng)
    if model is not None:model.finish(n)
    else:n.check('all special Tensor family references released',
        original.descriptor_only_contract(n.s.owner()['bytes'],root_index,root_storage,root_version),True)
    return dict(label=label,setup=setup,actual_status=status,diagnostic_words=details,
        candidate_binary_modified=False,all_unlisted_bytes_unchanged=True,
        exact_original_owner_restored=True,complete_refused_request_owner_rollback=True,
        clean_recovery_compared_to_actual_Torch=True)


empty_allocation_control = specialized_original(original.allocation_control,
    ALLOCATION_SPEC=EMPTY_ALLOCATION_SPEC, ALLOCATION_SHA256=EMPTY_ALLOCATION_SHA256)
high_allocation_control = specialized_original(original.allocation_control,
    ALLOCATION_SPEC=HIGH_ALLOCATION_SPEC, ALLOCATION_SHA256=HIGH_ALLOCATION_SHA256)
empty_audit_fault_control = specialized_original(original.audit_fault_control)
Observer = SignedObserver

_FUNCTION_RELATIONS = [{'source': 'empty_native.py', 'name': 'SignedObserver', 'original_sha256': '5d4cfe497963bfe1fc9c9b1727615526fb2963fe6b7a08b4c886254e9d27e3a9', 'derived_sha256': '292b0bfb349c2f265fbdf823c25c5912286875c42565873f6e58b22dd8091235', 'replacements': [{'before': '(original.Observer)', 'after': '(OffsetRawObserver)'}], 'whole_function_inverse_exact': True}, {'source': 'empty_native.py', 'name': 'compare', 'original_sha256': 'e975ad177a6b4007faf9fbba77761cd473a3ab6ff9bd8c7e7912320a9187d4ab', 'derived_sha256': 'fc2a8e4618d2660a4c87ce7749228084e2e49a53dbb96acbade32327d61b0ead', 'replacements': [{'before': "gap = row['accepted'] and spec.get('comparison_scope', '').startswith('known-native-')", 'after': 'gap = False  # All accepted actual Torch rows require native success.'}], 'whole_function_inverse_exact': True}, {'source': 'offset_native.py', 'name': 'specialized_original', 'original_sha256': 'c1b9f9b83424861cedeef08794fe4df65ebb48864c2f0df3b4e9411b2337861b', 'derived_sha256': 'c1b9f9b83424861cedeef08794fe4df65ebb48864c2f0df3b4e9411b2337861b', 'replacements': [], 'whole_function_inverse_exact': True}, {'source': 'offset_controls.py', 'name': 'compare_chain', 'original_sha256': 'f61937150e5373142a4908c0a65163604fc4f3b0a5e2003b7f208a349dd25fc4', 'derived_sha256': 'ae021b1973cf732f4951ba86ddb6d2315fd5a34ab82810b872120738d443399a', 'replacements': [{'before': 'extension.SignedOwnerModel', 'after': 'SignedOwnerModel'}], 'whole_function_inverse_exact': True}, {'source': 'offset_controls.py', 'name': 'replace_words', 'original_sha256': '4172cf775f59ca78e61ee910de472b991c10859cfd1c1e304ab4f7a5de1f7212', 'derived_sha256': '4172cf775f59ca78e61ee910de472b991c10859cfd1c1e304ab4f7a5de1f7212', 'replacements': [], 'whole_function_inverse_exact': True}, {'source': 'offset_controls.py', 'name': 'fault_control', 'original_sha256': 'a3f5a782c0dcb03d9fad83bcf6fecb95674be9fc7fb0bb2aa54173e0a8239866', 'derived_sha256': '33e7070c45b984954609c844db0afd32efc7df1cbe515c0f5f84ce924f66b132', 'replacements': [{'before': 'extension.SignedOwnerModel', 'after': 'SignedOwnerModel'}], 'whole_function_inverse_exact': True}]

def source_relations():
    result = []
    for row in _FUNCTION_RELATIONS:
        derived = inspect.getsource(globals()[row['name']])
        require(sha(derived.encode()) == row['derived_sha256'], 'complete consolidated function source')
        restored = derived
        for edit in reversed(row['replacements']):
            require(edit['after'] in restored, 'exact declared consolidated function inverse')
            restored = restored.replace(edit['after'], edit['before'])
        require(sha(restored.encode()) == row['original_sha256'], 'whole original family function inverse')
        result.append(dict(row, independently_reconstructed_original_sha256=sha(restored.encode())))
    return dict(functions=result,
        original_raw_observer_sha256=sha(RAW_SOURCE.encode()),
        tagged_raw_observer_sha256=sha(DECODED_SOURCE.encode()),
        raw_observer_inverse_exact=True, raw_metadata13_comparison_unchanged=True,
        raw_observer_substitutions=[dict(before=a, after=b) for a,b in RAW_SUBSTITUTIONS],
        original_owner_model_sha256=sha(OWNER_SOURCE.encode()),
        signed_owner_model_sha256=sha(SIGNED_OWNER_SOURCE.encode()),
        tagged_owner_model_sha256=sha(OFFSET_OWNER_SOURCE.encode()),
        whole_original_owner_model_inverse_exact=True,
        signed_shape_substitution=dict(before=OWNER_OLD_EXPRESSION, after=OWNER_NEW_EXPRESSION),
        tagged_owner_substitutions=[dict(before=a, after=b) for a,b in OWNER_SUBSTITUTIONS],
        all84_descriptor_and_all8_owner_table_comparisons_unchanged=True,
        original_allocation_function_sha256=sha(inspect.getsource(original.allocation_control).encode()),
        original_audit_fault_function_sha256=sha(inspect.getsource(original.audit_fault_control).encode()),
        original_allocation_and_corruption_function_bodies_unchanged=True,
        every_accepted_Torch_row_requires_actual_native_success=True,
        old_capacity_gap_evidence_unchanged=True)


def recipes(reference, families):
    """All1175 independent native/replay sessions, with complete observed blobs."""
    data = families['report']
    require(len(data['empty_rows']) == len(EMPTY_CASES) == 841
            and len(data['wide_rows']) == len(WIDE_CASES) == 20
            and len(data['offset_rows']) == len(OFFSET_CASES) == 75
            and len(data['chains']) == len(CHAINS) == 3,
            'all unchanged family denominators')
    plans = [(name, function, parameters, reference) for name, function, parameters in original.recipes(reference)]
    require(len(plans) == 198 and plans[187][0] == 'case-187', 'all original188 and original controls retained')
    plans[187] = ('case-187', compare, (data['original187'], True), families)
    plans += [('empty-family-%04d' % row['index'], compare, (row,), families) for row in data['empty_rows']]
    plans.append(('empty-allocation-control', empty_allocation_control, (data['empty_allocation'],), families))
    selected = [row for row in data['empty_rows'] if row['accepted'] and row['spec']['operations'][0] ==
                dict(operation='view', name='view', source='root', shape=[0, -1], strides=[1, 1], offset=0)]
    require(bool(selected), 'the original signed-empty corruption input is present')
    for prefix, faults in [('empty-audit-', original.FAULTS), ('additional-audit-', ADDITIONAL_FAULTS)]:
        plans += [(prefix + fault[0], empty_audit_fault_control, (selected[0], fault), families) for fault in faults]
    plans += [('wide-%02d' % row['index'], compare, (row,), families) for row in data['wide_rows']]
    plans += [('offset-%03d' % row['index'], compare, (row,), families) for row in data['offset_rows']]
    plans += [('chain-%02d' % row['index'], compare_chain, (row,), families) for row in data['chains']]
    plans.append(('high-offset-allocation-control', high_allocation_control, (data['high_allocation'],), families))
    offset = dict(rows=data['offset_rows'], f32_fault_setup=data['f32_fault_setup'],
                  nonempty_fault_setup=data['nonempty_fault_setup'])
    plans += [('offset-owner-' + fault[0], fault_control, (offset, fault), families) for fault in OFFSET_FAULTS]
    require(len(plans) == 1175 and len({row[0] for row in plans}) == 1175,
            'exact full session denominator and no duplicated session names')
    return plans
