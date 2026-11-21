"""Admit the exact22 division requests in the unchanged239-request Half entry.

All original code bytes and every earlier arithmetic rejection closure stay
in use. The terminal partition is151 refusals,22 multiplications,2 negations,
42 additions/subtractions and22 independently observed true divisions.
"""
import ast
import hashlib
import importlib
from pathlib import Path
import struct
import sys
import types

ADDSUB_HELPER='c48d62f7922942891d208e7e99573da43b0456a7a2288bc74fb305440349a145'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def closure(function):
    return dict(zip(function.__code__.co_freevars,(cell.cell_contents for cell in function.__closure__ or ())))


class Successor:
    def __init__(self,repository,reader_directory,reference_directory,reference_execution,inputs,addsub):
        self.repository=Path(repository).resolve(strict=True)
        self.reader=Path(reader_directory).resolve(strict=True)
        self.reference_directory=Path(reference_directory).resolve(strict=True)
        self.reference_execution=Path(reference_execution).resolve(strict=True)
        self.addsub=addsub;self.negation=addsub.negation;self.multiplication=addsub.multiplication
        inputs.bind(Path(__file__).resolve())
        helper=self.repository/'tests/current_runtime_half_addsub_successor.py'
        inputs.bind(helper,ADDSUB_HELPER)
        require(Path(addsub.wrap_entry.__code__.co_filename).resolve()==helper
            and addsub.repository==self.repository and addsub.reader==self.reader,
            'complete bound predecessor arithmetic helper from the same actual repository')
        tests=self.repository/'tests'
        for package in ('half_div','numeric_predicates'):
            for path in sorted((tests/package).glob('*.py')):inputs.bind(path)
        saved_path=list(sys.path)
        try:
            sys.path.insert(0,str(tests))
            run=importlib.import_module('half_div.run');cases=importlib.import_module('half_div.cases')
        finally:sys.path[:]=saved_path
        for module in (run,cases):
            require(Path(module.__file__).resolve().parent==tests/'half_div','actual bound division modules')
        legacy=self.multiplication.legacy
        raw=inputs.bind(self.reader/legacy.ORIGINAL_MEMBER,legacy.ORIGINAL_SOURCE_SHA256)
        functions=[node for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef) and node.name=='entry_controls']
        require(len(functions)==1,'one complete immutable original239 entry recipe')
        node=functions[0];entry=b''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
        require(sha(entry)==legacy.ORIGINAL_ENTRY_SHA256 and len(entry)==legacy.ORIGINAL_ENTRY_BYTES==7497,
            'every byte of the original entry remains frozen')
        expected=tuple((label.rsplit('-',1)[0]+'-4',left,right) for label,left,right in legacy.SUCCESSORS)
        require(cases.LEGACY_CONTROLS==expected and len(expected)==22,
            'all and only22 original public division requests in original order')
        self.reference=run.read_reference(self.reference_directory,self.reference_execution,inputs)
        rows=[row for row in self.reference['report']['rows'] if 'legacy_refusal_label' in row['spec']]
        require(len(rows)==22 and [row['spec']['legacy_refusal_label'] for row in rows]==[item[0] for item in expected],
            'all22 original requests have independent real division observations')
        self.rows={};self.dtypes={}
        for row,(label,left,right) in zip(rows,expected):
            require(row['accepted'] is True and row['spec']['name']=='successor-'+label
                and row['spec']['operation']==4 and row['spec']['same_handle'] is (left==right),
                'exact original operation and same-handle relation')
            for key,dtype,observed in zip(('left','right'),(left,right),row['inputs']):
                literal=b''.join(word.to_bytes(cases.WIDTHS[dtype],'little') for word in legacy.WORDS[dtype])
                require(row['spec'][key]==cases.input_spec(dtype,[2],words=legacy.WORDS[dtype])
                    and cases.raw_input(row['spec'][key])==literal
                    and self.reference['blob'](observed['logical'])==literal
                    and observed['shape']==[2] and observed['strides']==[1],
                    'the entire original literal operand was observed by actual Torch')
            self.rows[label]=row;self.dtypes[label]=(left,right)
        self.proof=dict(schema='heptatorch.half47.div22-successor.v1',
            reference_directory=str(self.reference_directory),reference_execution=str(self.reference_execution),
            reference_sha256=self.reference['report_sha256'],
            reference_process_sha256=sha(inputs.bind(self.reference_execution/'process.json')),
            case_sha256=cases.CASE_SHA256,reference_cases=1196,
            original_controls_sha256=legacy.ORIGINAL_SOURCE_SHA256,
            original_entry_sha256=legacy.ORIGINAL_ENTRY_SHA256,original_entry_bytes=7497,
            original_request_count=239,predecessor_refusals=173,retained_refusals=151,
            accepted_successors=22,existing_multiplication_successors=22,existing_negation_successors=2,
            existing_addsub_successors=42,labels=[item[0] for item in expected],
            predecessor_addsub_helper_sha256=ADDSUB_HELPER,
            all_previous_reject_closures_reused_unchanged=True,original_entry_code_object_preserved=True,
            actual_Torch_reference_verified=True,host_expected_values_computed=False,
            original_frozen_recipe_modified=False,every_original_snapshot_field_checked=True,
            scoped_owner_extension='exact22-Half-true-division-public-promotion',
            input_all84words_unchanged_checked=True,original_same_handle_relations_checked=True,
            prior_unrecovered_native769_evidence_used=False)

    def observe(self,driver,token,expected,label):
        return type(self.multiplication).observe(self,driver,token,expected,label)

    def exchange(self,driver,label,body,left,right,row):
        """Retain every full original owner predicate around one exact div frame."""
        owner=driver.owner;original=owner.f64_command_metadata
        expected_command=[2,4,left,right,0,0,0,0]
        result_dtype={'torch.float32':0,'torch.float64':4,'torch.float16':5,'torch.bfloat16':6}[row['result']['dtype']]
        left_dtype,right_dtype=self.dtypes[label]
        def metadata(before,after,command,record):
            require(command==expected_command and record['opcode']==2,
                'only the exact original true-division request enters the owner extension')
            sources=[owner.public_node(before,token) for token in (left,right)]
            observed=[owner.w(before['nodes'][str(index)],576) for index in sources]
            require(observed==[left_dtype,right_dtype] and any(dtype in (5,6) for dtype in observed)
                and all(dtype in (0,1,2,4,5,6) for dtype in observed),'exact public Half division operand types')
            result=owner.public_node(after,record['value']);node=owner._live_node(after,result,'half-div-successor-result')
            require(owner.w(node,576)==result_dtype,'division result dtype equals independent Torch')
            for source in sources:
                require(owner.w(before['nodes'][str(source)],32)==0,'division operands have no gradients')
                owner._f64_source_unchanged(before,after,source)
            if result_dtype in (5,6):
                role=owner.half_role(after,result);owner._half_fresh_result(before,after,result,sources)
            else:
                if result_dtype==4:role=owner.f64_role(after,result)
                else:
                    require(not any(owner.w(node,field) for field in owner.F64_ZERO_FIELDS)
                        and owner.w(node,240)==1,'promoted F32 result owns one live Tensor and no retained AD or view state')
                    role=dict(node=result,dtype=0,width=4,storage=owner.w(node,216),version=owner.w(node,224))
                owner._f64_fresh_result(before,after,result,sources)
            return dict(operation=2,binary_operation=4,role=role,input_dtypes=observed,result_dtype=result_dtype,
                input_all84words_unchanged=True,exact_successor_label=label,actual_Torch_result_dtype_used=True,
                numerical_expected_values_computed_here=False)
        owner.f64_command_metadata=metadata
        try:return driver.n.exchange(body,label)
        finally:owner.f64_command_metadata=original

    def wrap_entry(self,original_entry):
        old_execute=self.addsub.wrap_entry(original_entry);old=closure(old_execute)
        require(set(old)=={'accepted_mul','accepted_neg','accepted_addsub','accepted_fe','cloned','self'}
            and old['self'] is self.addsub,'exact predecessor terminal wrapper closure')
        entry=old['cloned'];accepted_mul=old['accepted_mul'];accepted_neg=old['accepted_neg']
        accepted_addsub=old['accepted_addsub'];accepted_fe=old['accepted_fe']
        require(entry.__code__ is original_entry.__code__ and accepted_mul==accepted_neg==accepted_addsub==accepted_fe==[],
            'all previous adapters retain the complete original code object and fresh ordered logs')
        old_reject=entry.__globals__['reject']
        codes=[code for code in self.addsub.wrap_entry.__code__.co_consts
            if isinstance(code,types.CodeType) and code.co_name=='reject']
        old_cells=closure(old_reject)
        require(len(codes)==1 and old_reject.__code__ is codes[0] and old_cells['self'] is self.addsub
            and old_cells['accepted_addsub'] is accepted_addsub and old_cells['accepted_fe'] is accepted_fe,
            'the exact previous full arithmetic rejection dispatcher is reused unchanged')
        accepted_div=[];names={0:'F32',1:'Bool',2:'I64',4:'F64',5:'F16',6:'BF16'}
        def reject(driver,label,body,expected,**kwargs):
            if label not in self.rows:return old_reject(driver,label,body,expected,**kwargs)
            require(not kwargs and expected==7 and len(accepted_div)<22
                and label==self.proof['labels'][len(accepted_div)],'only the exact ordered original22 division requests change')
            row=self.rows[label];left_dtype,right_dtype=self.dtypes[label]
            left,right=driver.live[names[left_dtype]],driver.live[names[right_dtype]]
            exact=b'HTORCH03'+struct.pack('<3Q',1,96,0)+struct.pack('<8Q',2,4,left,right,0,0,0,0)
            require(body==exact,'complete original division opcode operands padding and envelope')
            driver.check(label+'/same-handle-inputs',left==right,row['spec']['same_handle'])
            before=[self.observe(driver,token,wanted,label+'/input-'+side)
                for token,wanted,side in zip((left,right),row['inputs'],('left','right'))]
            status,records=self.exchange(driver,label,body,left,right,row)
            driver.operations.append(dict(label=label,request_sha256=sha(body),actual_status=status,
                expected_status=0,record_count=len(records),predecessor_expected_status=7,
                successor_reference_index=row['index']))
            driver.check(label+'/actual-frame-status',status,0)
            driver.check(label+'/one-division-result',[item['opcode'] for item in records],[2])
            result=records[0]['value'];role='half-div-successor-result/'+label
            require(role not in driver.live,'fresh temporary true-division result role');driver.live[role]=result
            got=self.observe(driver,result,row['result'],label+'/result')
            for key in ('tensor_identity','storage_identity','native_version_identity'):
                driver.check(label+'/fresh-'+key,got[key] not in [item[key] for item in before],True)
            for token,saved,side in zip((left,right),before,('left','right')):
                after=driver.n.tensor(token);require(set(after)==set(saved),'every complete original input observation field')
                for key in saved:
                    if key!='owner':driver.check(label+'/'+side+'-input-unchanged/'+key,after[key],saved[key])
            driver.rng_unchanged(label+'/successful-division');driver.drop(role);accepted_div.append(label)
            return status
        namespace=dict(entry.__globals__);namespace['reject']=reject
        cloned=types.FunctionType(original_entry.__code__,namespace,original_entry.__name__,original_entry.__defaults__,original_entry.__closure__)
        cloned.__kwdefaults__=original_entry.__kwdefaults__
        require(cloned.__code__ is original_entry.__code__,'complete original239 request code object')
        def execute(driver):
            require(not driver.operations and not any((accepted_mul,accepted_neg,accepted_addsub,accepted_fe,accepted_div)),
                'fresh complete239 entry driver and all five successor logs')
            details=cloned(driver)
            require(details['actual_refusals']==len(driver.operations)==239
                and accepted_mul==self.multiplication.proof['labels'] and accepted_neg==self.negation.proof['labels']
                and accepted_addsub==self.addsub.proof['single_command_labels']
                and accepted_fe==self.addsub.proof['same_frame_FE_labels'] and accepted_div==self.proof['labels'],
                'all239 original requests and all88 actual arithmetic successors execute in exact original order')
            successful=[row for row in driver.operations if row['expected_status']==0]
            retained=[row for row in driver.operations if row['expected_status']!=0]
            domains=((self.multiplication.rows,accepted_mul),(self.negation.rows,accepted_neg),
                (self.addsub.rows,accepted_addsub),(self.addsub.fe_rows,accepted_fe),(self.rows,accepted_div))
            require(len(successful)==88 and len(retained)==151
                and all([row['label'] for row in successful if row['label'] in domain]==labels for domain,labels in domains)
                and [row['label'] for row in successful if row['label'] in self.addsub.rows or row['label'] in self.addsub.fe_rows]==self.addsub.proof['labels']
                and all(row['actual_status']==row['expected_status'] for row in driver.operations),
                'exact151 refusals plus22/2/40/2/22 genuine successes with no unmatched request')
            for labels in (self.addsub.proof['retained_actual_Bool_error_labels'],self.addsub.proof['retained_private3_protocol_refusal_labels']):
                actual=[row for row in retained if row['label'] in labels]
                require([row['label'] for row in actual]==labels and all(row['expected_status']==row['actual_status']==7 for row in actual),
                    'all four actual Bool subtraction errors and eight private3 add/sub refusals stay intact')
            private_div=[f'half-binary-{side}-{half}-3-4' for half in (5,6) for side in ('left','right')]
            actual=[row for row in retained if row['label'] in private_div]
            require([row['label'] for row in actual]==private_div and all(row['actual_status']==row['expected_status']==7 for row in actual),
                'all four original private3 division requests remain explicit refusals')
            require(sum(self.addsub.operations[label]==1 for label in accepted_addsub)==22
                and sum(self.addsub.operations[label]==2 for label in accepted_addsub)==18,'unchanged actual add/sub partition')
            return dict(details,actual_refusals=151,actual_successful_multiplications=22,
                actual_successful_negations=2,actual_successful_additions=24,actual_successful_subtractions=18,
                actual_successful_addsub=42,actual_successful_single_command_addsub=40,actual_successful_FE_addsub=2,
                actual_successful_divisions=22,original_entry_requests=239,
                multiplication_successor=self.multiplication.proof,negation_successor=self.negation.proof,
                addsub_successor=self.addsub.proof,division_successor=self.proof)
        return execute


def adapter_type(original_type,multiplication_module,negation_module,addsub_module,repository,reader_directory,
                 multiplication_reference,negation_reference,addsub_reference,division_reference):
    class Adapter(original_type):
        def bind_package(self,inputs):
            digest=super().bind_package(inputs)
            self.half_mul_successor=multiplication_module.Successor(repository,reader_directory,
                multiplication_reference['root'],multiplication_reference['execution'],inputs)
            self.half_neg_successor=negation_module.Successor(repository,reader_directory,
                negation_reference['root'],negation_reference['execution'],inputs,self.half_mul_successor)
            self.half_addsub_successor=addsub_module.Successor(repository,reader_directory,
                addsub_reference['root'],addsub_reference['execution'],inputs,self.half_neg_successor)
            self.half_div_successor=Successor(repository,reader_directory,
                division_reference['root'],division_reference['execution'],inputs,self.half_addsub_successor)
            return digest
        def current_main(self):
            main=super().current_main();controls=self.old['controls'];original=controls.entry_controls
            def entry(driver):return self.half_div_successor.wrap_entry(original)(driver)
            def execute():
                require(controls.entry_controls is original,'unmodified original controls before successor scope')
                controls.entry_controls=entry
                try:return main()
                finally:controls.entry_controls=original
            return execute
    return Adapter
