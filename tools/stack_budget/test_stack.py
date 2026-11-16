#!/usr/bin/env python3
"""Independent executable qualification of ABI3 native stack admission."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import runpy
import signal
import struct
import subprocess
import sys

sys.dont_write_bytecode=True
SPEC=Path(__file__).resolve().parent
sys.path.insert(0,str(SPEC))
from inspect_stack import inspect,longest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources',type=Path,required=True)
parser.add_argument('--native',type=Path,required=True)
parser.add_argument('--service-sources',type=Path,nargs=5,required=True)
parser.add_argument('--owner-client',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
S=args.sources.resolve();N=args.native.resolve();O=args.output.resolve();O.mkdir(parents=True,exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
bootstrap=json.loads((N/'bootstrap.json').read_text());adapt=json.loads((S/'adaptation.json').read_text())
C=N/'stack-module-compiler';assert sha(C.read_bytes())==bootstrap['compiler_sha256']
tools={label:N/('stack-linker-'+label) for label in adapt['linkers']}
for label,path in tools.items():assert sha(path.read_bytes())==bootstrap['linker_sha256'][label]
client_raw=args.owner_client.read_bytes();assert sha(client_raw)=='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
client=runpy.run_path(str(args.owner_client),run_name='qualification_client')
assert args.owner_client.read_bytes()==client_raw
rows=[]


def compile_source(name,raw):
    (O/(name+'.hepta')).write_bytes(raw)
    process=subprocess.run([str(C)],input=raw,capture_output=True,env={'PATH':''},timeout=30)
    (O/(name+'.compiler.stdout')).write_bytes(process.stdout)
    (O/(name+'.compiler.stderr')).write_bytes(process.stderr)
    assert process.returncode==0 and not process.stderr and process.stdout.startswith(b'\x7fELF'),(name,process.returncode)
    (O/(name+'.o')).write_bytes(process.stdout)
    return {'source':raw,'object':process.stdout,'inspection':inspect(raw,process.stdout)}


def native_link(name,units,label='qualification',accepted=True,raw_objects=None):
    objects=raw_objects if raw_objects is not None else [unit['object'] for unit in units]
    magic=adapt['linkers'][label]['request_magic'].encode()
    request=magic+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
    (O/(name+'.request')).write_bytes(request)
    process=subprocess.run([str(tools[label])],input=request,capture_output=True,env={'PATH':''},timeout=30)
    (O/(name+'.link.stdout')).write_bytes(process.stdout);(O/(name+'.link.stderr')).write_bytes(process.stderr)
    if not accepted:
        assert process.returncode==2 and not process.stdout and not process.stderr,(name,process.returncode)
        rows.append({'case':name,'kind':'native-refusal','status':'PASS','request_sha256':sha(request),'exit':2})
        return None
    assert process.returncode==0 and not process.stderr and process.stdout.startswith(b'\x7fELF'),(name,process.returncode)
    path=O/name;path.write_bytes(process.stdout);path.chmod(0o700)
    expected=longest([unit['inspection'] for unit in units])
    startup=json.loads((S/('linker-'+label)/'startup.json').read_text())
    emitted=struct.unpack_from('<Q',process.stdout,startup['stack']['budget_hole'])[0]
    assert emitted==expected['required_stack_bytes'],(name,emitted,expected)
    assert emitted<=adapt['linkers'][label]['stack_bytes']
    (O/(name+'.independent-stack.json')).write_text(json.dumps(expected,indent=2)+'\n')
    return path,expected


def run_small(name,path,stack_limit=32768):
    request=b'HTOWNR01'+struct.pack('<Q',1)+b'\x5a'
    request+=b'HTOWNR01'+struct.pack('<Q',1)+b'\xa3'
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_STACK,(stack_limit,stack_limit))
    process=subprocess.run([str(path)],input=request,capture_output=True,env={'PATH':''},preexec_fn=limits,timeout=30)
    expected=b''.join(b'HTOWNS01'+struct.pack('<Q',32)+bytes([v])+bytes(31) for v in (0x5a,0xa3))
    (O/(name+'.input')).write_bytes(request);(O/(name+'.stdout')).write_bytes(process.stdout);(O/(name+'.stderr')).write_bytes(process.stderr)
    assert process.returncode==0 and not process.stderr and process.stdout==expected,(name,process.returncode,process.stdout[:64])
    return {'RLIMIT_STACK_bytes':stack_limit,'frames':2,'exact_response_bytes':len(expected),'executable_sha256':sha(path.read_bytes())}


def entry(target,external=True,second=None,tail=False):
    raw='hia-asm.v1 hepta.hia.program.v8 stack_entry\nexports(step,close)\n'
    if external:
        raw+=f'fn {target}(x:u8'+(',n:u64' if tail else '')+') -> (u8) effects() external()\n'
    raw+='''fn step(s:readbuf,o:mutbuf,state:buffer) -> (buffer,u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,state:buffer)
  z:usize = const() {"value":0}
  value:u8 = buffer.load(s,z)
'''
    if tail:
        raw+='  n:u64 = const() {"value":10000}\n'
    raw+='  first:u8 = call(value'+(',n' if tail else '')+') {"function":"'+target+'"}\n'
    value='first'
    if second:
        raw+='  second:u8 = call(first) {"function":"'+second+'"}\n';value='second'
    raw+='  zero:u8 = const() {"value":0}\n  _ = buffer.fill(o,zero)\n'
    raw+='  _ = buffer.store(o,z,'+value+')\n  size:u32 = const() {"value":32}\n  ret(state,size)\nend\n'
    raw+='fn close(state:buffer) -> () effects(alloc,trap) entry(b0)\nblock b0(state:buffer)\n  _ = owner.drop(state)\n  ret()\nend\n'
    return raw


def helper(name,padding,target=None,module=True):
    raw=('hia-asm.v1 hepta.hia.program.v8 '+name+'_unit\nexports('+name+')\n') if module else ''
    if target:raw+=f'fn {target}(x:u8) -> (u8) effects() external()\n'
    raw+=f'fn {name}(x:u8) -> (u8) effects() entry(b0)\nblock b0(x:u8)\n'
    raw+=''.join(f'  p{i}:u64 = const() {{"value":{i}}}\n' for i in range(padding))
    if target:raw+='  result:u8 = call(x) {"function":"'+target+'"}\n  ret(result)\n'
    else:raw+='  ret(x)\n'
    return (raw+'end\n').encode()


root=compile_source('chain-entry',entry('deep0').encode())
chain=[compile_source('chain-'+str(i),helper('deep'+str(i),1200,'deep'+str(i+1) if i<2 else None)) for i in range(3)]
base=longest([root['inspection'],*[u['inspection'] for u in chain]])
adjust=(61440-base['maximum_path_bytes'])//16
assert base['maximum_path_bytes']+adjust*16==61440 and 0<=1200+adjust<=2000
for delta,label in ((-1,'below-limit'),(0,'at-limit'),(1,'over-limit')):
    leaf=compile_source(label+'-leaf',helper('deep2',1200+adjust+delta))
    units=[root,*chain[:2],leaf]
    expected=longest([u['inspection'] for u in units])
    assert expected['maximum_path_bytes']==61440+delta*16
    linked=native_link(label,units,accepted=delta<=0)
    if linked:
        path,observed=linked
        rows.append({'case':label,'kind':'native-execution','status':'PASS',**observed,**run_small(label,path)})
        reverse=native_link(label+'-reversed',list(reversed(units)))
        rows.append({'case':label+'-reversed','kind':'native-execution','status':'PASS',**reverse[1],**run_small(label+'-reversed',reverse[0])})
    else:
        path,observed=native_link(label+'-production',units,label='session')
        rows.append({'case':label+'-production','kind':'native-execution','status':'PASS',**observed,**run_small(label+'-production',path)})

fan=entry('left',external=False,second='right').encode()+helper('left',2000,module=False)+helper('right',2000,module=False)
fan_unit=compile_source('fanout',fan)
path,expected=native_link('fanout-path-not-sum',[fan_unit])
assert expected['sum_all_frames_bytes']>65536 and expected['required_stack_bytes']<65536
rows.append({'case':'fanout-path-not-sum','kind':'native-execution','status':'PASS',**expected,**run_small('fanout-path-not-sum',path)})

tail=entry('tail',external=False,tail=True)
tail+='fn tail(x:u8,n:u64) -> (u8) effects(trap) entry(b0)\nblock b0(x:u8,n:u64)\n'
tail+=''.join(f'  p{i}:u64 = const() {{"value":{i}}}\n' for i in range(2000))
tail+='''  zero:u64 = const() {"value":0}
  done:bool = compare.eq(n,zero)
  branch done finish(x) again(x,n)
block finish(v:u8)
  ret(v)
block again(v2:u8,n2:u64)
  one:u64 = const() {"value":1}
  next:u64 = int.sub.wrap(n2,one)
  result:u8 = call(v2,next) {"function":"tail"}
  ret(result)
end
'''
tail_unit=compile_source('tail-self',tail.encode())
path,expected=native_link('tail-self-10000',[tail_unit])
assert expected['required_stack_bytes']<65536
rows.append({'case':'tail-self-10000','kind':'native-execution','status':'PASS',**expected,**run_small('tail-self-10000',path)})

maximum=entry('maximum',external=False).encode()+helper('maximum',2047,module=False)
maximum_unit=compile_source('maximum-ssa-slot',maximum)
maximum_record=next(r for r in maximum_unit['inspection']['records'] if r['name']=='maximum')
assert maximum_record['decoded_frame']==33360
path,expected=native_link('maximum-ssa-slot-frame',[maximum_unit])
rows.append({'case':'maximum-ssa-slot-frame','kind':'native-execution','status':'PASS',**expected,**run_small('maximum-ssa-slot-frame',path)})
too_many=entry('maximum',external=False).encode()+helper('maximum',2048,module=False)
process=subprocess.run([str(C)],input=too_many,capture_output=True,env={'PATH':''},timeout=30)
(O/'too-many-ssa.hepta').write_bytes(too_many)
assert process.returncode==2 and not process.stdout and not process.stderr
rows.append({'case':'2049-values-still-refused','kind':'native-compiler-refusal','status':'PASS'})
bad=bytearray(maximum_unit['object']);at=maximum_record['stack_record_offset']
struct.pack_into('<Q',bad,at+24,33376);struct.pack_into('<Q',bad,at+48,2048)
code=maximum_unit['inspection']['text_at']+maximum_record['offset']
last_probe=code+4+8*11
assert bad[last_probe:last_probe+3]==bytes.fromhex('48 81 ec')
struct.pack_into('<I',bad,last_probe+3,608)
native_link('beyond-maximum-native-frame',[maximum_unit],accepted=False,raw_objects=[bytes(bad)])

# Independently forged metadata is never counted as a production program.
# These probes exercise structural/budget gate refusals, not foreign-code authentication.
unit=fan_unit;raw=unit['object'];view=unit['inspection'];stack=view['stack_at']
offsets=[(stack,0),(stack+8,2),(stack+16,257),(stack+24,0),(stack+32,63),(stack+40,0),(stack+48,39),(stack+56,1)]
for field in (0,8,16,24,32,40,48,56):
    offsets.append((stack+64+field,(1<<64)-1))
for index,(offset,value) in enumerate(offsets):
    bad=bytearray(raw);struct.pack_into('<Q',bad,offset,value)
    native_link('malformed-stack-'+str(index),[unit],accepted=False,raw_objects=[bytes(bad)])
for delta in (-1,1):
    bad=bytearray(raw);row=view['records'][0];at=row['stack_record_offset']
    oldslots=struct.unpack_from('<Q',bad,at+48)[0]
    struct.pack_into('<Q',bad,at+48,oldslots+delta)
    struct.pack_into('<Q',bad,at+24,row['decoded_frame']+delta*16)
    native_link('frame-prologue-mismatch-'+str(delta),[unit],accepted=False,raw_objects=[bytes(bad)])
n=view['function_count'];width=view['row_bytes'];matrix=stack+64+n*64
bad=bytearray(raw);bad[matrix]|=1
native_link('self-call-metadata-cycle',[unit],accepted=False,raw_objects=[bytes(bad)])
left=next(r['ordinal'] for r in view['records'] if r['name']=='left')
right=next(r['ordinal'] for r in view['records'] if r['name']=='right')
bad=bytearray(raw);bad[matrix+left*width+right//8]|=1<<(right%8);bad[matrix+right*width+left//8]|=1<<(left%8)
native_link('private-two-function-cycle',[unit],accepted=False,raw_objects=[bytes(bad)])
bad=bytearray(raw);bad[matrix+width-1]|=0x80
native_link('unused-high-matrix-bit',[unit],accepted=False,raw_objects=[bytes(bad)])
if (n*width)%8:
    bad=bytearray(raw);bad[matrix+n*width]=1
    native_link('nonzero-matrix-padding',[unit],accepted=False,raw_objects=[bytes(bad)])
for cut in (len(raw)-1,stack+63):
    native_link('truncated-object-'+str(cut),[unit],accepted=False,raw_objects=[raw[:cut]])

bad_owner=entry('left',external=False).replace('  ret(state,size)','  _ = owner.drop(state)\n  ret(state,size)').encode()+helper('left',0,module=False)
process=subprocess.run([str(C)],input=bad_owner,capture_output=True,env={'PATH':''},timeout=30)
(O/'use-after-drop.hepta').write_bytes(bad_owner);(O/'use-after-drop.stdout').write_bytes(process.stdout)
assert process.returncode==2 and not process.stdout and not process.stderr
rows.append({'case':'owned-use-after-drop-still-refused','kind':'native-compiler-refusal','status':'PASS'})

service=[]
for index,path in enumerate(args.service_sources):service.append(compile_source('service-'+str(index),path.read_bytes()))
service_path,service_budget=native_link('owned-storage-service',service,label='session')
probe_path,_=native_link('owned-storage-cleanup',service,label='qualification-cleanup')
guard_paths=[]
for side in ('low','high'):
    path,_=native_link('guard-'+side,service,label='qualification-guard-'+side)
    def no_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    process=subprocess.run([str(path)],input=b'',capture_output=True,env={'PATH':''},preexec_fn=no_core,timeout=10)
    assert process.returncode==-signal.SIGSEGV and not process.stdout and not process.stderr,(side,process.returncode)
    rows.append({'case':'real-'+side+'-guard-fault','kind':'qualification-native-fault','status':'PASS',
                 'observed_signal':'SIGSEGV','fault_before_owner_becomes_live':True,'executable_sha256':sha(path.read_bytes())})

for name,status,probe,mode in [('live-normal',0,False,'eof'),('cleanup-eof',0,True,'eof'),
                               ('cleanup-invalid',2,True,'invalid'),('cleanup-oversize',3,True,'oversize'),
                               ('cleanup-epipe',4,True,'epipe')]:
    process=client['OwnerSession'](probe_path if probe else service_path,probe=probe,directory=O/name)
    make=client['request']([(1,64,0x5a),(20,)],slots=8,pool=128)
    answer=client['response'](process.exchange(make));assert answer[0]==0
    tensor=answer[1][0]['values'][0]
    answer=client['response'](process.exchange(client['request']([(12,tensor,7),(20,)])))
    assert answer[0]==0 and answer[1][0]['values'][0]==0x5a
    base_address=process.control_base();owner=process.owner(False)
    stack_base,stack_mapping,stack_top,required,old_rsp=struct.unpack('<5Q',process.memory(base_address+200,40))
    assert stack_mapping==8388608+8192 and stack_top==stack_base+4096+8388608
    assert required==service_budget['required_stack_bytes'] and not stack_base<=old_rsp<stack_base+stack_mapping
    mapping=Path(f'/proc/{process.pid}/maps').read_text()
    (O/name/'maps-live.txt').write_text(mapping)
    def protection(address):
        for line in mapping.splitlines():
            a,b=(int(v,16) for v in line.split()[0].split('-'))
            if a<=address<b:return line.split()[1]
        return None
    assert protection(stack_base)=='---p' and protection(stack_base+4096)=='rw-p' and protection(stack_top)=='---p'
    syscall=process.wait_syscall(0,0)
    assert stack_base+4096<=syscall[-2]<stack_top,syscall
    if mode=='invalid':process.send(b'INVALID!'+bytes(8))
    elif mode=='oversize':process.send(b'HTOWNR01'+struct.pack('<Q',(1<<64)-1))
    elif mode=='epipe':
        process.close_output();body=client['request']();process.send(b'HTOWNR01'+struct.pack('<Q',len(body))+body)
    else:process.close_input()
    if probe:
        marker=process.read_exact(process.marker,16)
        assert marker==b'HTSTKCL1'+struct.pack('<Q',status),marker
        process.wait_syscall(0,4)
        assert process.poll() is None
        assert not process.mapped(stack_base) and not process.mapped(stack_top) and not process.mapped(base_address)
        assert not process.mapped(owner['pointer'])
    process.finish(expected=status)
    rows.append({'case':name,'kind':'real-owned-service','status':'PASS','frames_before_EOF':2,
                 'native_required_stack_bytes':required,'guard_maps_observed':True,
                 'execution_rsp_inside_private_stack':True,'pre_exit_unmap_observed':probe,
                 'normal_exit':status,'owner_stack_and_io_unmapped_before_exit':probe})

for label,path in tools.items():assert sha(path.read_bytes())==bootstrap['linker_sha256'][label]
assert sha(C.read_bytes())==bootstrap['compiler_sha256'] and args.owner_client.read_bytes()==client_raw
report={'schema':'heptatorch.stack-budget-independent-qualification.v1','status':'PASS','cases':len(rows),
        'compiler_sha256':bootstrap['compiler_sha256'],'linkers_sha256':bootstrap['linker_sha256'],
        'reviewer_sha256':sha(Path(__file__).read_bytes()),'source_call_graph_and_machine_frame_oracle_sha256':sha((SPEC/'inspect_stack.py').read_bytes()),
        'service_budget':service_budget,'rows':rows,'producer_bytes_verified_after':True,
        'scope':'Pinned ABI3 compiler products; exact native path budget plus private stack guards and normal release. No foreign machine-code authentication or asynchronous signal recovery claim.'}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
