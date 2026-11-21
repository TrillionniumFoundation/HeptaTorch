"""Twenty-three actual mixed-F64 results in the unchanged87-request F6448 recipe.

The existing negation predicate is reused through its exact preparation
interface. Twenty-two mixed single requests and one unchanged factory/FE
packet receive independently observed Torch results. Both Bool subtraction
requests and all remaining original refusals retain their original observers.
"""
import ast
import hashlib
import importlib
import importlib.util
from pathlib import Path
import struct
import sys
import types

CONTROLS='16daead20d5bf9934371a8283b18a94dc03a45d3390b346d7ef5d7019d8820d8'
ENTRY='ba632c235c28f95fa7464673d4ca5b4f556d727712154f5d8a349e4cf7c7078b'
FE='same-packet-F64-public-alias-mixed-pair'
WORDS={0:(0x3f800000,0x80000000),1:(1,0),2:(1,0),4:(0x3ff0000000000000,0x8000000000000000)}
PUBLIC=tuple((f'mixed-{name}-op{operation}-F64-{side}',operation,left,right)
    for name,other in (('F32',0),('Bool',1),('I64',2)) for operation in (1,2,3,4)
    for side,left,right in (('left',4,other),('right',other,4)))+((FE,1,4,0),)
LABELS=tuple(label for label,operation,left,right in PUBLIC if not(operation==2 and 1 in (left,right)))
RETAINED_BOOL=tuple(label for label,operation,left,right in PUBLIC if operation==2 and 1 in (left,right))
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def module(path,name,inputs):
    inputs.bind(path);spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

class Successor:
    def __init__(self,repository,reader_directory,reference,negation,inputs):
        self.repository=Path(repository).resolve(strict=True)
        self.reader=Path(reader_directory).resolve(strict=True)
        self.reference_directory=Path(reference['root']).resolve(strict=True)
        self.reference_execution=Path(reference['execution']).resolve(strict=True)
        self.negation=negation
        inputs.bind(Path(__file__).resolve())
        raw=inputs.bind(self.reader/'controls736.py',CONTROLS)
        definitions=[node for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef) and node.name=='entry_controls']
        require(len(definitions)==1,'one entire original F64 entry recipe')
        node=definitions[0];entry=b''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
        require(sha(entry)==ENTRY and len(entry)==6998,'all original87 request recipe bytes retained')
        helper=self.repository/'tests/current_runtime_f64_neg_successor.py'
        observer=module(self.repository/'tools/current_runtime/public_f64_arithmetic_observer_transition.py',
            'f6448_negation_preparation_exact_inverse',inputs)
        _,observer_proof=observer.restore_negation_helper(inputs.bind(helper),inputs.bind)
        require(Path(negation.prepare_entry.__code__.co_filename).resolve()==helper,
            'actual prior negation uses the exact interface-only refactor')
        relu=module(self.repository/'tools/current_runtime/typed_relu_source_transition.py',
            'f6448_typed_relu_whole_source_inverse',inputs)
        indexing=module(self.repository/'tools/current_runtime/index_signbit_source_transition.py',
            'f6448_index_signbit_whole_source_inverse',inputs)
        parent775,index_inverse=indexing.restore_source(
            inputs.bind(self.repository/'tools/current_runtime/source/runtime.hepta'),inputs.bind)
        require(index_inverse['functions']==778 and index_inverse['parent_functions']==775
            and index_inverse['whole_source_inverse_exact'] is True
            and index_inverse['whole_round_trip_exact'] is True,'complete778 to775 source relation')
        same773,relu_source_proof=relu.restore_source(parent775,inputs.bind)
        require(relu_source_proof['functions']==775 and relu_source_proof['parent_functions']==773
            and relu_source_proof['changed_functions']==['unary']
            and relu_source_proof['whole_source_inverse_exact'] is True
            and relu_source_proof['whole_round_trip_exact'] is True,
            'complete typed ReLU775 inverse before the frozen same-F64 source')
        same=module(self.repository/'tools/current_runtime/same_f64_arithmetic_source_transition.py',
            'f6448_same_f64_whole_source_inverse',inputs)
        previous773,same_source_proof=same.restore_source(same773,inputs.bind)
        require(same.SOURCE==relu.PARENT_SOURCE
            and same_source_proof['functions']==same_source_proof['parent_functions']==773
            and same_source_proof['changed_functions']==['half_mul_priority','public_f64_binary']
            and same_source_proof['whole_source_inverse_exact'] is True
            and same_source_proof['whole_round_trip_exact'] is True,
            'actual same-F64 successor admitted without transferring mixed773 execution results')
        source=module(self.repository/'tools/current_runtime/public_f64_arithmetic_source_transition.py',
            'f6448_arithmetic_whole_source_inverse',inputs)
        _,source_proof=source.restore_source(previous773,inputs.bind)
        require(source.SOURCE==same.PARENT_SOURCE,'exact current-to-mixed773-to771 source chain')
        self.fe_owner=module(self.repository/'tests/current_runtime_f64_arithmetic_fe_owner.py',
            'f6448_exact_FE_promotion_owner',inputs)
        tests=self.repository/'tests'
        for path in sorted((tests/'public_f64_arithmetic').glob('*.py')):inputs.bind(path)
        old_path=list(sys.path)
        try:
            sys.path.insert(0,str(tests))
            run=importlib.import_module('public_f64_arithmetic.run')
            cases=importlib.import_module('public_f64_arithmetic.cases')
        finally:sys.path[:]=old_path
        require(all(Path(value.__file__).resolve().parent==tests/'public_f64_arithmetic' for value in (run,cases)),
            'shared actual arithmetic reference modules belong to this repository')
        self.reference=run.read_reference(self.reference_directory,self.reference_execution,inputs)
        actual=[row for row in self.reference['report']['rows'] if 'legacy_refusal_label' in row['spec']]
        require([row['spec']['legacy_refusal_label'] for row in actual]==[item[0] for item in PUBLIC],
            'all25 original mixed requests in their complete original order')
        self.rows={};self.dtypes={}
        for row,(label,operation,left,right) in zip(actual,PUBLIC):
            spec=row['spec'];left_words=WORDS[left] if label!=FE else (0x3ff0000000000000,0x4000000000000000)
            require(spec['operation']==operation and spec['left']==cases.input_spec(left,[2],words=left_words)
                and spec['right']==cases.input_spec(right,[2],words=WORDS[right])
                and spec['legacy_same_frame']==(label==FE),
                'every original literal operand, opcode, layout and FE selection retained')
            for item,wanted,dtype in ((spec['left'],row['inputs'][0],left),(spec['right'],row['inputs'][1],right)):
                require(self.reference['blob'](wanted['logical'])==cases.raw_input(item)
                    and wanted['shape']==[2] and wanted['strides']==[1]
                    and wanted['element_size']==cases.WIDTHS[dtype],
                    'actual Torch consumes the complete original raw input bits')
            require(row['accepted']==(label in LABELS),'only the23 actually supported original requests change')
            if label in LABELS:
                require(row['result']['dtype']=='torch.float64' and row['fresh_tensor'] and row['fresh_storage'],
                    'actual original-shape promoted F64 result has independent ownership')
                self.rows[label]=row;self.dtypes[label]=(left,right)
            else:
                require(label in RETAINED_BOOL and row['error']['type']=='NotImplementedError',
                    'both original Bool subtraction refusals remain independently established')
        require(tuple(self.rows)==LABELS and len(LABELS)==23 and len(RETAINED_BOOL)==2,
            'complete23 arithmetic successors and two retained Bool refusals')
        self.fe_rows={FE:self.rows[FE]};self.fe_dtypes={FE:4}
        self.proof=dict(schema='heptatorch.f6448.public-f64-arithmetic23-successor.v1',
            reference_directory=str(self.reference_directory),reference_execution=str(self.reference_execution),
            reference_sha256=self.reference['report_sha256'],reference_process_sha256=sha(inputs.bind(self.reference_execution/'process.json')),
            case_sha256=cases.CASE_SHA256,original_controls_sha256=CONTROLS,original_entry_sha256=ENTRY,
            original_entry_bytes=6998,original_request_count=87,original_mixed_request_count=25,
            retained_refusals=63,accepted_successors=24,new_arithmetic_successors=23,
            retained_negation_successors=1,retained_bool_subtraction_refusals=2,
            labels=list(LABELS),retained_bool_labels=list(RETAINED_BOOL),single_command_successors=22,
            index_signbit_source_inverse=index_inverse,
            original_FE_successors=1,source_inverse=relu_source_proof,
            same_f64_predecessor_source_inverse=same_source_proof,
            mixed_predecessor_source_inverse=source_proof,negation_observer_inverse=observer_proof,
            prior_negation_reference_sha256=negation.proof['reference_sha256'],
            original_frozen_recipe_modified=False,original87_entry_code_object_preserved=True,
            original_negation_numeric_predicates_unchanged=True,actual_Torch_reference_verified=True,
            every_original_snapshot_field_checked=True,input_all84words_unchanged_checked=True,
            exact_FE_all84word_descriptors_and_all_owner_tables_checked=True,
            same_f64_numerical_predicates_unchanged=True,same_f64_NaN_payload_qualified=False,
            host_expected_values_computed=False)

    def observe(self,driver,token,expected,label):
        got=driver.n.tensor(token)
        for key in ('dtype','shape','strides','numel','element_size','storage_bytes','version',
                    'requires_grad','is_leaf','is_view','retains_grad'):
            driver.check(label+'/'+key,got[key],expected[key])
        driver.check(label+'/device',got['device'],'cpu')
        driver.check(label+'/storage-offset-elements',got['storage_offset_elements'],expected['storage_offset'])
        driver.check(label+'/storage-offset-bytes',got['storage_offset_bytes'],expected['storage_offset']*expected['element_size'])
        driver.check(label+'/history-present',got['grad_fn'] is not None,expected['history_present'])
        driver.check(label+'/grad-is-none',got['grad_is_none'],True)
        driver.check(label+'/base-absent',got['base_tensor_identity'],None)
        driver.check(label+'/all-logical-bits',got['logical'],self.reference['blob'](expected['logical']))
        driver.check(label+'/all-storage-bits',got['storage'],self.reference['blob'](expected['storage']))
        return got

    def exchange(self,driver,label,body,left,right,row):
        owner=driver.owner;original=owner.f64_command_metadata
        operation=row['spec']['operation'];expected_command=[2,operation,left,right,0,0,0,0]
        def metadata(before,after,command,record):
            require(command==expected_command and record['opcode']==2,'only this exact original mixed command')
            sources=[owner.public_node(before,token) for token in (left,right)]
            observed=[owner.w(before['nodes'][str(index)],576) for index in sources]
            require(observed==list(self.dtypes[label]) and 4 in observed and observed[0]!=observed[1]
                and all(dtype in (0,1,2,4) for dtype in observed),'exact mixed public F64 operand types')
            result=owner.public_node(after,record['value']);role=owner.f64_role(after,result)
            require(row['result']['dtype']=='torch.float64','actual fixed Torch promoted result dtype')
            for source in sources:
                require(owner.w(before['nodes'][str(source)],32)==0,'both original inputs have no gradients')
                owner._f64_source_unchanged(before,after,source)
            owner._f64_fresh_result(before,after,result,sources)
            return dict(operation=2,binary_operation=operation,role=role,input_dtypes=observed,result_dtype=4,
                input_all84words_unchanged=True,exact_successor_label=label,
                actual_Torch_result_dtype_used=True,numerical_expected_values_computed_here=False)
        owner.f64_command_metadata=metadata
        try:return driver.n.exchange(body,label)
        finally:owner.f64_command_metadata=original

    def single_success(self,driver,label,body,expected):
        names={0:'f32',1:'boolean',2:'integer',4:'f64'}
        row=self.rows[label];left_dtype,right_dtype=self.dtypes[label]
        left,right=driver.live[names[left_dtype]],driver.live[names[right_dtype]]
        exact=b'HTORCH03'+struct.pack('<3Q',1,96,0)+struct.pack('<8Q',2,row['spec']['operation'],left,right,0,0,0,0)
        require(expected==7 and body==exact,'complete original96-byte mixed binary request')
        before=[self.observe(driver,token,wanted,label+'/input-'+side)
            for token,wanted,side in zip((left,right),row['inputs'],('left','right'))]
        status,records=self.exchange(driver,label,body,left,right,row)
        driver.operations.append(dict(label=label,request_sha256=sha(body),actual_status=status,
            expected_status=0,record_count=len(records),predecessor_expected_status=7,successor_reference_index=row['index']))
        driver.check(label+'/actual-frame-status',status,0)
        driver.check(label+'/one-arithmetic-result',[item['opcode'] for item in records],[2])
        result=records[0]['value'];role='f64-arithmetic-successor-result/'+label
        require(role not in driver.live,'fresh temporary mixed F64 result role');driver.live[role]=result
        got=self.observe(driver,result,row['result'],label+'/result')
        for key in ('tensor_identity','storage_identity','native_version_identity'):
            driver.check(label+'/fresh-'+key,got[key] not in [item[key] for item in before],True)
        for token,saved,side in zip((left,right),before,('left','right')):
            after=driver.n.tensor(token)
            require(set(after)==set(saved),'every original input observation field retained')
            for key in saved:
                if key!='owner':driver.check(label+'/'+side+'-input-unchanged/'+key,after[key],saved[key])
        driver.rng_unchanged(label+'/successful-arithmetic');driver.drop(role)
        return status

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
            require(actual_body == body and len(body) == 240 and actual_commands == commands
                    and status == 0 and [item['opcode'] for item in records] == [97, 2],
                    'only the exact original F64 two-command FE-add success enters this joint contract')
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
            factory_role = owner.f64_role(after, left)
            require(factory_role['dtype'] == dtype == 4, 'the FE factory has its exact public F64 dtype')
            owner._f64_fresh_result(before, after, left, [existing])
            owner._f64_fresh_result(before, after, result, [existing])
            output = owner._live_node(after, result, 'FE-add-fresh-F64-result')
            require(owner.w(output, 576) == 4 and owner.w(output, 240) == 1
                    and not any(owner.w(output, field) for field in owner.F64_ZERO_FIELDS),
                    'the actual Torch F64 result has no AD, Saved, view or retained input')
            left_node = after['nodes'][str(left)]
            for offset in (192, 216, 224, 320):
                require(owner.w(left_node, offset) != owner.w(output, offset),
                        'the two new Tensor/Storage/Version/incarnation identities are distinct')
            # The pinned native path creates the public F64 factory, one
            # temporary F64 right promotion, then the public F64 output. The
            # already-F64 factory operand is reused. Commit releases promotion.
            first = before['header']['next']
            allocated = {first, first + 1, first + 2}
            require(after['header']['next'] == first + 3
                    and after['header']['incarnation'] == before['header']['incarnation'] + 3
                    and left == first and result == first + 2
                    and owner.w(left_node, 320) == before['header']['incarnation'] + 1
                    and owner.w(output, 320) == before['header']['incarnation'] + 3,
                    'exact factory, native right-promotion and result descriptor allocation order')
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
                          exact_original_packet_bytes=240, actual_record_opcodes=[97, 2],
                          factory_dtype=dtype, result_dtype=4, allocated_descriptors=3,
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
                        frame_admission='exact-original1-F64-FE-add-success',
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
        right = driver.live['f32']
        commands = [[97, dtype, 160, 0, 0, 0, 0, 0],
                    [2, 1, wire.reference(1), right, 0, 0, 0, 0]]
        exact = b'HTORCH03' + struct.pack('<3Q', 2, 240, 0)
        exact += b''.join(struct.pack('<8Q', *command) for command in commands)
        exact += struct.pack('<8Q', 1, 0, 2, 0, 0, 0, 224, 16) + struct.pack('<2Q', 0x3ff0000000000000, 0x4000000000000000)
        require(expected == 7 and body == exact, 'the entire original240-byte factory/FE-add request')
        old_right = self.observe(driver, right, row['inputs'][1], label + '/original-right')
        status, records, transition = self.exchange_fe(driver, label, body, commands, dtype, right)
        driver.operations.append(dict(label=label, request_sha256=sha(body), actual_status=status,
            expected_status=0, record_count=len(records), predecessor_expected_status=7,
            successor_reference_index=row['index'], original_same_frame_FE_packet=True))
        driver.check(label + '/actual-frame-status', status, 0)
        driver.check(label + '/factory-then-add-result', [item['opcode'] for item in records], [97, 2])
        factory, result = [item['value'] for item in records]
        roles = ['f64-FE-add-factory/' + label, 'f64-FE-add-result/' + label]
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


    def wrap_entry(self,original_entry):
        require(Path(original_entry.__code__.co_filename).resolve()==self.reader/'controls736.py'
            and original_entry.__name__=='entry_controls','complete unchanged original87 entry code')
        original_reject=original_entry.__globals__['reject'];accepted=[]
        def reject(driver,label,body,expected,**kwargs):
            if label not in self.rows:return original_reject(driver,label,body,expected,**kwargs)
            require(not kwargs and expected==7 and label==LABELS[len(accepted)],
                'only the23 exact ordered original arithmetic requests receive successors')
            status=(self.fe_success(driver,label,body,expected,original_entry.__globals__['WIRE'])
                if label==FE else self.single_success(driver,label,body,expected))
            accepted.append(label);return status
        namespace=dict(original_entry.__globals__);namespace['reject']=reject
        clone=types.FunctionType(original_entry.__code__,namespace,original_entry.__name__,
            original_entry.__defaults__,original_entry.__closure__);clone.__kwdefaults__=original_entry.__kwdefaults__
        cloned,accepted_neg=self.negation.prepare_entry(clone)
        require(cloned.__code__ is original_entry.__code__,'all original87 recipe code preserved through both hooks')
        def execute(driver):
            require(not driver.operations and not accepted and not accepted_neg,'fresh complete87 entry-only driver')
            details=cloned(driver)
            require(details['actual_refusals']==len(driver.operations)==87 and accepted==list(LABELS)
                and accepted_neg==['unsupported-F64-neg'],'every original request and each exact successor executed')
            successful=[row for row in driver.operations if row['expected_status']==0]
            retained=[row for row in driver.operations if row['expected_status']!=0]
            expected_labels=list(LABELS[:-1])+['unsupported-F64-neg',FE]
            require(len(retained)==63 and [row['label'] for row in successful]==expected_labels
                and all(row['actual_status']==row['expected_status'] for row in driver.operations),
                'exact63 original refusals,22 mixed arithmetic,one original negation andone original FE result')
            bool_rows=[row for row in retained if row['label'] in RETAINED_BOOL]
            require([row['label'] for row in bool_rows]==list(RETAINED_BOOL)
                and all(row['actual_status']==7 and row['record_count']==0 for row in bool_rows),
                'both actual Bool subtraction refusals retain the exact original contract')
            return dict(details,actual_refusals=63,actual_successful_negations=1,
                actual_successful_mixed_arithmetic=23,actual_successful_FE_arithmetic=1,
                original_entry_requests=87,negation_successor=self.negation.proof,
                public_f64_arithmetic_successor=self.proof)
        return execute

def adapter_type(original_type,repository,reader_directory,reference):
    class Adapter(original_type):
        def bind_package(self,inputs):
            digest=super().bind_package(inputs)
            self.public_f64_arithmetic_successor=Successor(repository,reader_directory,reference,
                self.wide_neg_successor,inputs)
            active=types.ModuleType('current_f64_arithmetic_successor_controls')
            active.__dict__.update(self.current_controls.__dict__)
            active.entry_controls=self.public_f64_arithmetic_successor.wrap_entry(self.current_controls.entry_controls)
            sys.modules['controls']=active
            return digest
        def candidate_admission(self,directory,inputs):
            result=super().candidate_admission(directory,inputs)
            result.update(public_f64_arithmetic_successor=self.public_f64_arithmetic_successor.proof,
                wide_neg_successor_is_preserved_predecessor_scoped_contract=True,
                retained_refusals=63,actual_numeric_successors=24,unchanged_other_refusal_expectations=61)
            return result
    return Adapter
