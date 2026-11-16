from pathlib import Path
import argparse,hashlib,json,os,re,struct,subprocess,time
parser=argparse.ArgumentParser(description='Native borrowed-module compilation, link admission and execution tests.')
parser.add_argument('--compiler',type=Path,required=True)
parser.add_argument('--linker',type=Path,required=True)
parser.add_argument('--legacy-compiler',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
options=parser.parse_args()
OUT=options.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
COMP=options.compiler.resolve();LINK=options.linker.resolve();LEGACY=options.legacy_compiler.resolve()
START=time.monotonic();ROWS=[]
sha=lambda b:hashlib.sha256(b).hexdigest()
CPIN=sha(COMP.read_bytes());LPIN=sha(LINK.read_bytes())
assert CPIN=='e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
assert LPIN=='1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8'
assert sha(LEGACY.read_bytes())=='c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'

def function(name,params,result,body,effects='read,write,trap',tail=''):
 sig=','.join(f'{n}:{t}' for n,t in params)
 return f'fn {name}({sig}) -> ({result}) effects({effects}) entry(b0){tail}\nblock b0({sig})\n'+''.join('  '+line+'\n' for line in body)+'end\n'
def declaration(name,params,result,effects='read,write,trap'):
 return f'fn {name}('+','.join(f'{n}:{t}' for n,t in params)+f') -> ({result}) effects({effects}) external()\n'
def module(name,exports,*parts):
 return ('hia-asm.v1 hepta.hia.program.v8 '+name+'\nexports('+','.join(exports)+')\n'+''.join(parts)).encode()
def const(name,typ,value):return f'{name}:{typ} = const() '+json.dumps({'value':value},separators=(',',':'))
def call(output,typ,args,target):return (f'{output}:{typ}' if typ else '_')+' = call('+','.join(args)+') '+json.dumps({'function':target},separators=(',',':'))
ENTRY=[('s','readbuf'),('o','mutbuf'),('w','mutbuf')]

def compile_source(label,raw,expected=0):
 (OUT/(label+'.hepta')).write_bytes(raw)
 p=subprocess.run([str(COMP)],input=raw,capture_output=True,timeout=20,env={'PATH':''})
 (OUT/(label+'.compile.stdout')).write_bytes(p.stdout);(OUT/(label+'.compile.stderr')).write_bytes(p.stderr)
 assert p.returncode==expected and not p.stderr,(label,p.returncode,p.stderr)
 if expected:assert not p.stdout,label
 else:(OUT/(label+'.o')).write_bytes(p.stdout)
 if expected:ROWS.append({'case':label,'kind':'native-compiler-refusal','exit':p.returncode,'stdout_bytes':len(p.stdout),'source_sha256':sha(raw),'status':'PASS'})
 return p.stdout

def frame(objects,entry=b'entry'):
 return b'HIABLK01'+struct.pack('<QQQ',len(objects),len(entry),0)+entry+b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)

def link_objects(label,objects,entry=b'entry',expected=0,raw=None):
 request=frame(objects,entry) if raw is None else raw
 (OUT/(label+'.request')).write_bytes(request)
 p=subprocess.run([str(LINK)],input=request,capture_output=True,timeout=20,env={'PATH':''})
 (OUT/(label+'.link.stdout')).write_bytes(p.stdout);(OUT/(label+'.link.stderr')).write_bytes(p.stderr)
 assert p.returncode==expected and not p.stderr,(label,p.returncode,p.stderr)
 if expected:assert not p.stdout,label
 else:
  path=OUT/(label+'.native');path.write_bytes(p.stdout);path.chmod(0o700)
 ROWS.append({'case':label,'kind':'native-link-refusal' if expected else 'native-link-success','exit':p.returncode,'stdout_bytes':len(p.stdout),'request_sha256':sha(request),'object_sha256':[sha(obj) for obj in objects],'status':'PASS'})
 return OUT/(label+'.native') if not expected else None

def execute(label,exe,data,expected):
 p=subprocess.run([str(exe)],input=data,capture_output=True,timeout=20,env={'PATH':''})
 assert p.returncode==0 and not p.stderr and p.stdout==expected,(label,p.returncode,p.stderr,p.stdout.hex(),expected.hex())
 ROWS.append({'case':label,'kind':'linked-execution','input_hex':data.hex(),'expected_hex':expected.hex(),'observed_hex':p.stdout.hex(),'executable_sha256':sha(exe.read_bytes()),'status':'PASS'})

# Actual source/output/workspace resources cross the unit boundary.
mut_provider=function('kernel',ENTRY,'u32',[
 const('z','usize',0),'a:u8 = buffer.load(s,z)',const('one','u8',1),'b:u8 = int.add.wrap(a,one)','_ = buffer.store(o,z,b)',const('two','u8',2),'c:u8 = int.add.wrap(a,two)','_ = buffer.store(w,z,c)',const('n','u32',1),'ret(n)'])
mut_entry=function('entry',ENTRY,'u32',[call('n','u32',['s','o','w'],'kernel'),const('z','usize',0),const('i','usize',1),'a:u8 = buffer.load(w,z)','_ = buffer.store(o,i,a)',const('length','u32',2),'ret(length)'])
MUT_A=module('mut_caller',['entry'],declaration('kernel',ENTRY,'u32'),mut_entry)
MUT_B=module('mut_provider',['kernel'],mut_provider)
ma=compile_source('mut-caller',MUT_A);mb=compile_source('mut-provider',MUT_B)
exe=link_objects('cross-resource-mutation',[ma,mb]);execute('cross-resource-mutation-41',exe,b'\x29',b'\x2a\x2b');execute('cross-resource-mutation-wrap',exe,b'\xff',b'\x00\x01')
reverse=link_objects('reordered-objects',[mb,ma]);execute('reordered-resources',reverse,b'\x7f',b'\x80\x81')

shared_params=[('a','readbuf'),('b','readbuf'),('o','mutbuf')]
shared=function('shared',shared_params,'u32',[const('z','usize',0),'x:u8 = buffer.load(a,z)','y:u8 = buffer.load(b,z)','v:u8 = int.add.wrap(x,y)','_ = buffer.store(o,z,v)',const('n','u32',1),'ret(n)'])
shared_entry=function('entry',ENTRY,'u32',[call('n','u32',['s','s','o'],'shared'),'ret(n)'])
sa=compile_source('shared-read-caller',module('shared_a',['entry'],declaration('shared',shared_params,'u32'),shared_entry));sb=compile_source('shared-read-provider',module('shared_b',['shared'],shared))
execute('shared-read-alias-permitted',link_objects('shared-read-alias',[sa,sb]),b'\x83',b'\x06')

loan_params=[('b','mutbuf'),('x','u8')]
loan_provider=function('borrow_kernel',loan_params,'u64',['child:mutbuf = borrow.mut(b)',const('z','usize',0),'_ = buffer.store(child,z,x)','_ = loan.end(child)','wide:u64 = int.cast(x) {"mode":"wrap"}','ret(wide)'])
loan_entry=function('entry',ENTRY,'u32',['child:mutbuf = borrow.mut(w)',const('x','u8',91),call('r','u64',['child','x'],'borrow_kernel'),'_ = loan.end(child)',const('z','usize',0),'value:u8 = buffer.load(w,z)','_ = buffer.store(o,z,value)',const('n','u32',1),'ret(n)'])
la=compile_source('loan-caller',module('loan_a',['entry'],declaration('borrow_kernel',loan_params,'u64'),loan_entry));lb=compile_source('loan-provider',module('loan_b',['borrow_kernel'],loan_provider))
execute('nested-borrow-ends-and-parent-resumes',link_objects('nested-borrow',[la,lb]),b'',b'\x5b')

error_params=[('b','mutbuf')]
error_provider=function('error_result',error_params,'u64',[const('r','u64',17),'ret(r)'])
error_entry=function('entry',ENTRY,'u32',[call('r','u64',['w'],'error_result'),const('z','usize',0),const('i','usize',1),const('nine','u8',9),'_ = buffer.store(w,z,nine)','v:u8 = int.cast(r) {"mode":"wrap"}','_ = buffer.store(o,z,v)','v2:u8 = buffer.load(w,z)','_ = buffer.store(o,i,v2)',const('n','u32',2),'ret(n)'])
ea=compile_source('error-caller',module('error_a',['entry'],declaration('error_result',error_params,'u64'),error_entry));eb=compile_source('error-provider',module('error_b',['error_result'],error_provider))
execute('scalar-error-does-not-consume-loan',link_objects('scalar-error',[ea,eb]),b'',b'\x11\x09')

# Full 16-slot private ABI, including narrow/signed/bool scalar normalization.
scalar_specs=[('u8',255),('i8',-1),('u16',65535),('i16',-2),('u32',4294967295),('i32',-3),('u64',(1<<64)-1),('i64',-4),('usize',13),('bool',True),('u16',17),('i8',-5),('u32',0),('u64',1)]
params=[('s','readbuf'),('o','mutbuf')]+[(f'a{i}',t) for i,(t,_) in enumerate(scalar_specs)]
body=[const('z','usize',0),'input:u8 = buffer.load(s,z)','sum0:u64 = int.cast(input) {"mode":"wrap"}']
for i,(typ,value) in enumerate(scalar_specs):
 if typ=='bool':body += [const('yes','u64',1),const('no','u64',0),f'wide{i}:u64 = select(a{i},yes,no)']
 else:body += [f'wide{i}:u64 = int.cast(a{i}) {{"mode":"wrap"}}']
 body += [f'sum{i+1}:u64 = int.add.wrap(sum{i},wide{i})']
for i in range(8):body += [const(f'shift{i}','u64',8*i),f'part{i}:u64 = int.shr.checked(sum14,shift{i})',f'byte{i}:u8 = int.cast(part{i}) {{"mode":"wrap"}}',const(f'at{i}','usize',i),f'_ = buffer.store(o,at{i},byte{i})']
body += [const('length','u32',8),'ret(length)']
provider=function('slots16',params,'u32',body)
callerbody=[const(f'v{i}',t,v) for i,(t,v) in enumerate(scalar_specs)]+[call('length','u32',['s','o']+[f'v{i}' for i in range(14)],'slots16'),'ret(length)']
caller=function('entry',ENTRY,'u32',callerbody)
six_a=compile_source('slots16-caller',module('six_a',['entry'],declaration('slots16',params,'u32'),caller));six_b=compile_source('slots16-provider',module('six_b',['slots16'],provider))
expected=(7+sum(int(v) for t,v in scalar_specs))&((1<<64)-1)
execute('sixteen-slots-normalization',link_objects('slots16',[six_a,six_b]),b'\x07',struct.pack('<Q',expected))

# Identically named private functions remain local to their original unit.
def private_provider(public,value):
 helper=function('private_impl',[],'u64',[const('v','u64',value),'ret(v)'])
 public_body=function(public,[],'u64',[call('v','u64',[],'private_impl'),'ret(v)'])
 return module('module_'+public,[public],public_body,helper)
private_entry=function('entry',ENTRY,'u32',[call('a','u64',[],'left'),call('b','u64',[],'right'),'sum:u64 = int.add.wrap(a,b)','v:u8 = int.cast(sum) {"mode":"wrap"}',const('z','usize',0),'_ = buffer.store(o,z,v)',const('n','u32',1),'ret(n)'])
pa=compile_source('private-caller',module('private_a',['entry'],declaration('left',[],'u64'),declaration('right',[],'u64'),private_entry));pb=compile_source('private-left',private_provider('left',7));pc=compile_source('private-right',private_provider('right',11))
execute('private-names-do-not-collide',link_objects('private-scope',[pa,pb,pc]),b'',b'\x12')

# Last explicit interface record and symbol257 carry borrowed resources.
capacity_fns=[]
for i in range(256):
 capacity_fns.append(function('pub'+str(i),[('b','mutbuf'),('x','u8')],'u64',[const('z','usize',0),'_ = buffer.store(b,z,x)',const('r','u64',i),'ret(r)']))
cap_kernel=compile_source('interfaces-256',module('capacity',[f'pub{i}' for i in range(256)],*capacity_fns))
cap_entry=function('entry',ENTRY,'u32',[const('x','u8',91),call('r','u64',['w','x'],'pub255'),'rv:u8 = int.cast(r) {"mode":"wrap"}',const('z','usize',0),const('i','usize',1),'_ = buffer.store(o,z,rv)','wv:u8 = buffer.load(w,z)','_ = buffer.store(o,i,wv)',const('n','u32',2),'ret(n)'])
cap_caller=compile_source('interfaces-256-caller',module('capacity_user',['entry'],declaration('pub255',[('b','mutbuf'),('x','u8')],'u64'),cap_entry))
execute('last-borrowed-interface',link_objects('interfaces-256-link',[cap_caller,cap_kernel]),b'',b'\xff\x5b')
compile_source('interfaces-257-refused',module('too_many',[f'pub{i}' for i in range(257)],*(capacity_fns+[capacity_fns[-1].replace('pub255','pub256')])),2)

# Caller and callee resource checks are the actual native HIA checks.
alias_decl=declaration('alias_target',[('a','mutbuf'),('b','mutbuf')],'u64')
alias_entry=function('entry',ENTRY,'u32',[call('r','u64',['w','w'],'alias_target'),const('n','u32',0),'ret(n)'])
compile_source('mutable-alias-refused',module('alias',['entry'],alias_decl,alias_entry),2)
compile_source('ended-loan-call-refused',module('ended',['entry'],declaration('borrow_kernel',loan_params,'u64'),loan_entry.replace('  _ = loan.end(child)\n','').replace('  x:u8 =','  _ = loan.end(child)\n  x:u8 =')),2)
compile_source('suspended-parent-access-refused',module('suspended',['entry'],declaration('borrow_kernel',loan_params,'u64'),loan_entry.replace('  _ = loan.end(child)\n','')),2)
compile_source('double-end-refused',module('double_end',['entry'],declaration('borrow_kernel',loan_params,'u64'),loan_entry.replace('  _ = loan.end(child)\n','  _ = loan.end(child)\n  _ = loan.end(child)\n')),2)
root_end=function('entry',ENTRY,'u32',['_ = loan.end(w)',const('n','u32',0),'ret(n)'])
compile_source('external-root-end-refused',module('root_end',['entry'],root_end),2)
compile_source('caller-effect-understatement-refused',module('missing_effect',['entry'],declaration('kernel',ENTRY,'u32'),mut_entry.replace('effects(read,write,trap)','effects(read,trap)')),2)
compile_source('provider-effect-understatement-refused',module('missing_effect_provider',['kernel'],mut_provider.replace('effects(read,write,trap)','effects(read,trap)')),2)
compile_source('call-buffer-type-mismatch-refused',module('bad_type',['entry'],declaration('error_result',error_params,'u64'),error_entry.replace('call(w)','call(s)')),2)
compile_source('external-declaration-body-refused',module('body',['entry'],declaration('error_result',error_params,'u64')+'block b0(b:mutbuf)\n  ret(b)\nend\n',error_entry),2)
compile_source('external-declaration-export-refused',module('bad_export',['entry','error_result'],declaration('error_result',error_params,'u64'),error_entry),2)
compile_source('external-no-module-refused',MUT_A.replace(b'exports(entry)\n',b''),2)
compile_source('missing-declaration-refused',module('missing',['entry'],mut_entry),2)
compile_source('duplicate-declaration-refused',module('duplicate',['entry'],declaration('kernel',ENTRY,'u32'),declaration('kernel',ENTRY,'u32'),mut_entry),2)
compile_source('import-borrowed-result-refused',module('borrow_return',['entry'],declaration('resource_result',[('s','readbuf')],'readbuf'),function('entry',ENTRY,'u32',[const('n','u32',0),'ret(n)'])),2)
borrowed_export=function('borrow_result',[('s','readbuf')],'readbuf',['ret(s)'],tail=' return_loans(["s"])')
compile_source('export-borrowed-result-refused',module('borrow_export',['borrow_result'],borrowed_export),2)
compile_source('owned-buffer-param-refused',module('owner',['owner'],function('owner',[('b','buffer')],'u64',[const('r','u64',0),'ret(r)'])),2)
compile_source('owned-chain-param-refused',module('chain_owner',['owner'],function('owner',[('b','chain')],'u64',[const('r','u64',0),'ret(r)'])),2)
compile_source('allocation-effect-refused',module('alloc_effect',['entry'],mut_entry.replace('effects(read,write,trap)','effects(alloc,read,write,trap)')),2)

# Individually admitted interfaces that disagree only across the unit boundary.
def pair(label,itypes,iret,ifx,dtypes,dret,dfx):
 ip=[('p'+str(i),t) for i,t in enumerate(itypes)];dp=[('p'+str(i),t) for i,t in enumerate(dtypes)]
 cb=[];args=[];mut=0
 for i,typ in enumerate(itypes):
  if typ=='readbuf':args.append('s')
  elif typ=='mutbuf':args.append(['o','w'][mut]);mut+=1
  else:cb.append(const('v'+str(i),typ,False if typ=='bool' else 0));args.append('v'+str(i))
 cb+=[call('result',iret,args,'target'),const('n','u32',0),'ret(n)']
 c=module('pair_caller',['entry'],declaration('target',ip,iret,ifx),function('entry',ENTRY,'u32',cb))
 db=['ret()'] if not dret else [const('r',dret,None if dret=='unit' else (False if dret=='bool' else 0)),'ret(r)']
 d=module('pair_provider',['target'],function('target',dp,dret,db,dfx))
 ca=compile_source(label+'-caller',c);de=compile_source(label+'-provider',d)
 link_objects(label,[ca,de],expected=2)
 return ca,de
pair('link-scalar-width-mismatch',['u32'],'u64','read,write,trap',['u64'],'u64','read,write,trap')
pair('link-buffer-mode-mismatch',['mutbuf'],'u64','read,write,trap',['readbuf'],'u64','read,write,trap')
pair('link-result-mismatch',['mutbuf'],'u64','read,write,trap',['mutbuf'],'u32','read,write,trap')
pair('link-arity-mismatch',['mutbuf'],'u64','read,write,trap',['mutbuf','u64'],'u64','read,write,trap')
pair('link-effects-mismatch',['mutbuf'],'u64','read,write,trap',['mutbuf'],'u64','read,trap')
pair('link-void-unit-mismatch',[],'','read,write,trap',[],'unit','read,write,trap')
link_objects('unresolved-import',[ma],expected=2)
link_objects('duplicate-strong-provider',[ma,mb,mb],expected=2)
link_objects('private-entry-refused',[ma,mb],entry=b'kernel_private',expected=2)
link_objects('wrong-entry-signature',[ea,eb],entry=b'error_result',expected=2)
private_missing=compile_source('private-import-provider',module('private_provider',['visible'],function('visible',[],'u64',[const('v','u64',0),'ret(v)']),mut_provider))
link_objects('private-import-refused',[ma,private_missing],expected=2)
cycle_sig=[('w','mutbuf')]
cycle_a=module('cycle_a',['entry','cycle_left'],declaration('cycle_right',cycle_sig,'u64'),function('entry',ENTRY,'u32',[call('v','u64',['w'],'cycle_left'),const('n','u32',0),'ret(n)']),function('cycle_left',cycle_sig,'u64',[call('v','u64',['w'],'cycle_right'),'ret(v)']))
cycle_b=module('cycle_b',['cycle_right'],declaration('cycle_left',cycle_sig,'u64'),function('cycle_right',cycle_sig,'u64',[call('v','u64',['w'],'cycle_left'),'ret(v)']))
cy_a=compile_source('cyclic-a',cycle_a);cy_b=compile_source('cyclic-b',cycle_b);link_objects('cross-module-cycle-refused',[cy_a,cy_b],expected=2)

# Corruption probes target one concrete field at a time, retaining the rest.
def section(raw,index):
 table=struct.unpack_from('<Q',raw,40)[0]
 return struct.unpack_from('<IIQQQQIIQQ',raw,table+64*index)
def patch(raw,at,fmt,value):
 b=bytearray(raw);struct.pack_into(fmt,b,at,value);return bytes(b)
meta=section(ma,7)[4]
for label,at,fmt,value in [('abi-version',meta+8,'<Q',2),('metadata-magic',meta,'<Q',0),('record-ordinal',meta+64,'<Q',3),('unknown-effect',meta+64+32,'<Q',30),('reserved-field',meta+64+176,'<Q',1),('unused-parameter-slot',meta+64+48+3*8,'<Q',1),('owned-interface',meta+64+48,'<Q',8),('borrowed-result',meta+64+40,'<Q',7),('multiple-results',meta+64+24,'<Q',2)]:
 link_objects('corrupt-'+label,[patch(ma,at,fmt,value),mb],expected=2)
rel=section(ma,6)[4];textsec=section(ma,1)
offset=struct.unpack_from('<Q',ma,rel)[0]
for label,at,fmt,value in [('relocation-kind',rel+8,'<I',2),('relocation-addend',rel+16,'<q',0),('relocation-outside-text',rel,'<Q',textsec[5]),('relocation-to-definition',rel+12,'<I',2),('relocation-not-call',textsec[4]+offset-1,'<B',0x90),('relocation-nonzero-placeholder',textsec[4]+offset,'<I',1)]:
 link_objects('corrupt-'+label,[patch(ma,at,fmt,value),mb],expected=2)
# A true overlapping relocation: two separately emitted call fixups exist.
twice_entry=mut_entry.replace('  n:u32 = call(s,o,w) {"function":"kernel"}\n','  n:u32 = call(s,o,w) {"function":"kernel"}\n  again:u32 = call(s,o,w) {"function":"kernel"}\n')
twice=compile_source('two-fixups',module('twice',['entry'],declaration('kernel',ENTRY,'u32'),twice_entry))
r=section(twice,6)[4];assert section(twice,6)[5]==48
first=struct.unpack_from('<Q',twice,r)[0]
link_objects('overlapping-relocations-refused',[patch(twice,r+24,'<Q',first),mb],expected=2)
link_objects('trailing-object-bytes-refused',[ma+b'\0',mb],expected=2)
link_objects('wrong-frame-magic',[],raw=b'HIAXLK01'+frame([ma,mb])[8:],expected=2)
link_objects('reserved-frame-field',[],raw=patch(frame([ma,mb]),24,'<Q',1),expected=2)
link_objects('trailing-frame-byte',[],raw=frame([ma,mb])+b'\0',expected=2)
# A valid upstream object with only legacy f0 has no module interface authority.
stock_source=MUT_A.replace(b'exports(entry)\n',b'').replace(declaration('kernel',ENTRY,'u32').encode(),b'').replace(b'v8 mut_caller\n',b'v8 selfhost_elf\n')+mut_provider.encode()
legacy=LEGACY
p=subprocess.run([str(legacy)],input=stock_source,capture_output=True,timeout=20,env={'PATH':''});assert p.returncode==0 and not p.stderr
link_objects('legacy-object-without-interface-refused',[p.stdout],expected=2)
assert sha(COMP.read_bytes())==CPIN and sha(LINK.read_bytes())==LPIN
report={'schema':'heptatorch.native-borrowed-module-boundaries.v1','status':'PASS','compiler_sha256':CPIN,'linker_sha256':LPIN,'rows':ROWS,'cases':len(ROWS),'seconds':time.monotonic()-START,'runtime_path':'empty','native_compilation_and_admission_and_linking':True,'host_role':'source/packet construction, expected-byte comparison and subprocess transport only','machine_code_authenticity_not_proved':True,'foreign_pointer_provenance_not_proved':True}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
