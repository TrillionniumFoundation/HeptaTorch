#!/usr/bin/env python3
"""Exercise actual production 8 MiB boundaries and all ABI3 table dimensions."""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import struct
import subprocess
import sys

sys.dont_write_bytecode=True
SPEC=Path(__file__).resolve().parent;sys.path.insert(0,str(SPEC))
from inspect_stack import inspect,longest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources',type=Path,required=True)
parser.add_argument('--native',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();S=args.sources.resolve();N=args.native.resolve();O=args.output.resolve();O.mkdir(parents=True,exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pin=json.loads((N/'bootstrap.json').read_text());C=N/'stack-module-compiler';L=N/'stack-linker-session'
assert sha(C.read_bytes())==pin['compiler_sha256'] and sha(L.read_bytes())==pin['linker_sha256']['session']
startup=json.loads((S/'linker-session/startup.json').read_text());rows=[]


def compile(name,source):
    (O/(name+'.hepta')).write_bytes(source)
    p=subprocess.run([str(C)],input=source,capture_output=True,env={'PATH':''},timeout=30)
    (O/(name+'.compiler.stdout')).write_bytes(p.stdout);(O/(name+'.compiler.stderr')).write_bytes(p.stderr)
    assert p.returncode==0 and not p.stderr,(name,p.returncode)
    (O/(name+'.o')).write_bytes(p.stdout)
    return {'source':source,'object':p.stdout,'inspection':inspect(source,p.stdout)}


def entry(target):
    return ('''hia-asm.v1 hepta.hia.program.v8 deep_entry
exports(step,close)
fn TARGET(x:u8) -> (u8) effects() external()
fn step(s:readbuf,o:mutbuf,state:buffer) -> (buffer,u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,state:buffer)
  zero:usize = const() {"value":0}
  value:u8 = buffer.load(s,zero)
  result:u8 = call(value) {"function":"TARGET"}
  fill:u8 = const() {"value":0}
  _ = buffer.fill(o,fill)
  _ = buffer.store(o,zero,result)
  size:u32 = const() {"value":32}
  ret(state,size)
end
fn close(state:buffer) -> () effects(alloc,trap) entry(b0)
block b0(state:buffer)
  _ = owner.drop(state)
  ret()
end
''').replace('TARGET',target).encode()


def chain_unit(number,definitions,successor=None,padding=0):
    head='u'+str(number)
    source=f'hia-asm.v1 hepta.hia.program.v8 unit{number}\nexports({head})\n'
    if successor is not None:source+=f'fn u{successor}(x:u8) -> (u8) effects() external()\n'
    for i in range(definitions):
        name=head if i==0 else 'n'+str(i)
        source+=f'fn {name}(x:u8) -> (u8) effects() entry(b0)\nblock b0(x:u8)\n'
        if i+1==definitions:
            source+=''.join(f'  p{j}:u64 = const() {{"value":{j}}}\n' for j in range(padding))
        target='n'+str(i+1) if i+1<definitions else (None if successor is None else 'u'+str(successor))
        if target:source+=f'  result:u8 = call(x) {{"function":"{target}"}}\n  ret(result)\n'
        else:source+='  ret(x)\n'
        source+='end\n'
    return source.encode()


def link(name,units,accepted=True):
    objects=[u['object'] for u in units]
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
    assert len(request)<=8388608
    (O/(name+'.request')).write_bytes(request)
    expected=longest([u['inspection'] for u in units])
    (O/(name+'.independent-stack.json')).write_text(json.dumps(expected,indent=2)+'\n')
    p=subprocess.run([str(L)],input=request,capture_output=True,env={'PATH':''},timeout=60)
    (O/(name+'.link.stdout')).write_bytes(p.stdout);(O/(name+'.link.stderr')).write_bytes(p.stderr)
    if not accepted:
        assert p.returncode==2 and not p.stdout and not p.stderr,(name,p.returncode)
        rows.append({'case':name,'status':'PASS','native_refusal':True,'required_stack_bytes':expected['required_stack_bytes'],
                     'maximum_path_bytes':expected['maximum_path_bytes'],'objects':len(units),'request_bytes':len(request)})
        return expected
    assert p.returncode==0 and not p.stderr,(name,p.returncode)
    observed=struct.unpack_from('<Q',p.stdout,startup['stack']['budget_hole'])[0]
    assert observed==expected['required_stack_bytes']<=8388608
    path=O/name;path.write_bytes(p.stdout);path.chmod(0o700)
    body=b'HTOWNR01'+struct.pack('<Q',1)+b'\x7b'
    def low_stack():
        resource.setrlimit(resource.RLIMIT_STACK,(32768,32768));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    run=subprocess.run([str(path)],input=body,capture_output=True,env={'PATH':''},preexec_fn=low_stack,timeout=30)
    (O/(name+'.stdout')).write_bytes(run.stdout);(O/(name+'.stderr')).write_bytes(run.stderr)
    assert run.returncode==0 and not run.stderr and run.stdout==b'HTOWNS01'+struct.pack('<Q',32)+b'\x7b'+bytes(31),(name,run.returncode)
    rows.append({'case':name,'status':'PASS','native_execution':True,'required_stack_bytes':observed,
                 'maximum_path_bytes':expected['maximum_path_bytes'],'objects':len(units),'request_bytes':len(request),
                 'functions_including_declarations':sum(u['inspection']['function_count'] for u in units),
                 'RLIMIT_STACK_bytes':32768,'executable_sha256':sha(p.stdout)})
    return expected


root=compile('entry',entry('u0'))
last256=compile('unit-256-functions',chain_unit(0,256))
assert last256['inspection']['function_count']==256
link('all-256-function-slots',[root,last256])
units64=[root]+[compile('object-'+str(i),chain_unit(i,1,i+1 if i<62 else None)) for i in range(63)]
link('all-64-object-slots',units64)
link('all-64-object-slots-reversed',list(reversed(units64)))

# 52 independently compiled units contain 13,099 ordinary native frames.
# This reaches the real production budget without exceeding any source-unit,
# object, input, output or workspace limit.
definition_counts=[255]*51+[94]
deep=[compile('deep-'+str(i),chain_unit(i,count,i+1 if i<51 else None)) for i,count in enumerate(definition_counts)]
baseline=longest([root['inspection'],*[u['inspection'] for u in deep]])
extra=(8384512-baseline['maximum_path_bytes'])//16
assert 0<=extra<=2040 and baseline['maximum_path_bytes']+extra*16==8384512,(extra,baseline['maximum_path_bytes'])
for delta,label in ((0,'production-at-limit'),(1,'production-over-limit')):
    last=compile(label+'-last',chain_unit(51,94,padding=extra+delta))
    expected=link(label,[root,*deep[:-1],last],accepted=delta==0)
    assert expected['maximum_path_bytes']==8384512+16*delta

assert sha(C.read_bytes())==pin['compiler_sha256'] and sha(L.read_bytes())==pin['linker_sha256']['session']
report={'schema':'heptatorch.stack-budget-production-boundaries.v1','status':'PASS','cases':len(rows),
        'compiler_sha256':pin['compiler_sha256'],'linker_sha256':pin['linker_sha256']['session'],
        'reviewer_sha256':sha(Path(__file__).read_bytes()),'oracle_sha256':sha((SPEC/'inspect_stack.py').read_bytes()),
        'rows':rows,'native_limit_is_8MiB':True,'source_and_native_table_limits_unchanged':True}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
