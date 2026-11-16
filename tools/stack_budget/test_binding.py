#!/usr/bin/env python3
"""Native machine-call binding attacks, with independent GNU disassembly."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import resource
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
SPEC = Path(__file__).resolve().parent
sys.path.insert(0, str(SPEC))
from inspect_stack import inspect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources', type=Path, required=True)
parser.add_argument('--native', type=Path, required=True)
parser.add_argument('--review', type=Path, required=True)
parser.add_argument('--owned-flow', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
S, N, R, O = (p.resolve() for p in (args.sources, args.native, args.review, args.output))
O.mkdir(parents=True, exist_ok=False)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
pins = json.loads((N/'bootstrap.json').read_text())
adaptation = json.loads((S/'adaptation.json').read_text())
C, L = N/'stack-module-compiler', N/'stack-linker-runtime'
assert sha(C.read_bytes()) == pins['compiler_sha256']
assert sha(L.read_bytes()) == pins['linker_sha256']['runtime']
tool_before = {p: p.read_bytes() for p in (C, L, N/'stack-linker-qualification')}
objdump = shutil.which('objdump')
assert objdump
version = subprocess.run([objdump, '--version'], capture_output=True, check=True).stdout.decode().splitlines()[0]
rows = []


def compile_source(name, raw):
    (O/(name+'.hepta')).write_bytes(raw)
    result = subprocess.run([str(C)], input=raw, capture_output=True, env={'PATH':''}, timeout=30)
    (O/(name+'.compile.stdout')).write_bytes(result.stdout)
    (O/(name+'.compile.stderr')).write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr, (name,result.returncode)
    path = O/(name+'.o'); path.write_bytes(result.stdout)
    unit = {'raw':result.stdout, 'view':inspect(raw,result.stdout), 'name':name}
    disassembly = subprocess.run([objdump,'--disassemble','--wide','--disassemble-zeroes','--insn-width=16',str(path)], capture_output=True, check=True).stdout
    (O/(name+'.disassembly.txt')).write_bytes(disassembly)
    instructions = []
    for line in disassembly.decode().splitlines():
        match = re.match(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s)+)\s*(.*)$',line)
        if match:
            instructions.append((int(match[1],16),bytes.fromhex(match[2]),match[3]))
    text_at = unit['view']['text_at']
    for offset,raw_instruction,_ in instructions:
        assert result.stdout[text_at+offset:text_at+offset+len(raw_instruction)] == raw_instruction
    unit['instructions'] = instructions
    return unit


def request(objects, entry=b'entry', magic=b'HIASTKR1'):
    return magic+struct.pack('<3Q',len(objects),len(entry),0)+entry+b''.join(struct.pack('<Q',len(o))+o for o in objects)


def link(name, objects, accepted=False, entry=b'entry', magic=b'HIASTKR1', tool=L):
    raw = request(objects,entry,magic)
    (O/(name+'.request')).write_bytes(raw)
    result = subprocess.run([str(tool)],input=raw,capture_output=True,env={'PATH':''},timeout=30)
    (O/(name+'.link.stdout')).write_bytes(result.stdout)
    (O/(name+'.link.stderr')).write_bytes(result.stderr)
    assert not result.stderr
    if accepted:
        assert result.returncode == 0 and result.stdout.startswith(b'\x7fELF'),(name,result.returncode)
        path = O/name; path.write_bytes(result.stdout); path.chmod(0o700)
        return path
    assert result.returncode == 2 and not result.stdout, (name,result.returncode)
    rows.append({'case':name,'kind':'native-refusal','status':'PASS','request_sha256':sha(raw),'exit':2})


def run(name, path, raw, expected):
    def limits():
        resource.setrlimit(resource.RLIMIT_STACK,(32768,32768))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    result = subprocess.run([str(path)],input=raw,capture_output=True,env={'PATH':''},preexec_fn=limits,timeout=30)
    (O/(name+'.input')).write_bytes(raw); (O/(name+'.stdout')).write_bytes(result.stdout)
    assert result.returncode == 0 and result.stdout == expected and not result.stderr,(name,result.returncode,result.stdout)
    rows.append({'case':name,'kind':'native-execution','status':'PASS','RLIMIT_STACK_bytes':32768,
                 'executable_sha256':sha(path.read_bytes()),'exact_output_hex':expected.hex()})


root_source = '''hia-asm.v1 hepta.hia.program.v8 binding_root
exports(entry)
fn remote(x:u8) -> (u8) effects() external()
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  zero:usize = const() {"value":0}
  fake_call_and_prologue:u64 = const() {"value":16539860742916653288}
  fake_prologue:u64 = const() {"value":3850979413}
  value:u8 = buffer.load(s,zero)
  local_result:u8 = call(value) {"function":"local"}
  first:u8 = call(local_result) {"function":"remote"}
  second:u8 = call(first) {"function":"remote"}
  z:u8 = const() {"value":0}
  flag:bool = compare.eq(value,z)
  branch flag left(o,second) right(o,second)
block left(a:mutbuf,x:u8)
  lzero:usize = const() {"value":0}
  _ = buffer.store(a,lzero,x)
  lone:u32 = const() {"value":1}
  ret(lone)
block right(b:mutbuf,y:u8)
  rzero:usize = const() {"value":0}
  _ = buffer.store(b,rzero,y)
  rone:u32 = const() {"value":1}
  ret(rone)
end
fn local(x:u8) -> (u8) effects() entry(b0)
block b0(x:u8)
  one:u8 = const() {"value":1}
  answer:u8 = int.add.wrap(x,one)
  ret(answer)
end
'''.encode()
# Separate literals contain E8/zero rel32 and a complete prologue byte sequence.
marker = bytes.fromhex('e8 00 00 00 00 90 90 90')
root_source = root_source.replace(b'16539860742916653288',str(int.from_bytes(marker,'little')).encode())
provider_source = b'''hia-asm.v1 hepta.hia.program.v8 binding_provider
exports(remote)
fn remote(x:u8) -> (u8) effects() entry(b0)
block b0(x:u8)
  mask:u8 = const() {"value":85}
  answer:u8 = bit.xor(x,mask)
  ret(answer)
end
'''
root = compile_source('binding-root',root_source)
provider = compile_source('binding-provider',provider_source)
objects = [root['raw'],provider['raw']]
exe = link('valid-private-and-external',objects,accepted=True)
run('both-call-kinds-zero-branch',exe,b'\0',b'\1')
run('both-call-kinds-other-branch',exe,b'\xff',b'\0')
view = root['view']; records = {r['name']:r for r in view['records']}
matrix = view['stack_at']+64+view['function_count']*64
for target,name in (('remote','clear-external-call-bit'),('local','clear-private-call-bit')):
    bad = bytearray(root['raw']); caller=records['entry']['ordinal']; callee=records[target]['ordinal']
    at = matrix+caller*view['row_bytes']+callee//8
    before = bad[at];bad[at] &= ~(1<<(callee%8))
    assert before ^ bad[at] == 1<<(callee%8)
    link(name,[bytes(bad),provider['raw']])
    rows[-1]['changed_object_bytes'] = 1
    rows[-1]['machine_code_unchanged'] = True
bad=bytearray(root['raw']);bad[matrix+records['local']['ordinal']*view['row_bytes']] |= 1
link('invent-absent-call-edge',[bytes(bad),provider['raw']])
calls=[(off,raw) for off,raw,_ in root['instructions'] if raw[0]==0xe8]
assert len(calls)==3
private_call=calls[0][0]
assert private_call+5+struct.unpack('<i',calls[0][1][1:])[0]==records['local']['offset']
for name,target in [('private-call-to-interior',records['local']['offset']+1),
                    ('private-call-to-self',records['entry']['offset']),
                    ('private-call-outside-text',1<<20)]:
    bad=bytearray(root['raw']);struct.pack_into('<i',bad,view['text_at']+private_call+1,target-private_call-5)
    link(name,[bytes(bad),provider['raw']])
for name,replacement in [('indirect-call',bytes.fromhex('ff d0 90 90 90')),
                         ('unadmitted-syscall',bytes.fromhex('0f 05 90 90 90')),
                         ('unknown-opcode',bytes.fromhex('06 90 90 90 90')),
                         ('hidden-second-prologue',bytes.fromhex('55 48 89 e5 90'))]:
    bad=bytearray(root['raw']);bad[view['text_at']+private_call:view['text_at']+private_call+5]=replacement
    link(name,[bytes(bad),provider['raw']])
table=struct.unpack_from('<Q',root['raw'],40)[0]
relocs,reloc_size=struct.unpack_from('<2Q',root['raw'],table+384+24)
assert reloc_size==48
bad=bytearray(root['raw']);bad[relocs+24:relocs+48]=bad[relocs:relocs+24]
link('duplicate-relocation',[bytes(bad),provider['raw']])
bad=bytearray(root['raw']);a=bad[relocs:relocs+24];bad[relocs:relocs+24]=bad[relocs+24:relocs+48];bad[relocs+24:relocs+48]=a
link('reversed-relocations',[bytes(bad),provider['raw']])
marker_at=root['raw'].index(marker,view['text_at'])-view['text_at']
bad=bytearray(root['raw']);struct.pack_into('<Q',bad,relocs,marker_at+1)
link('relocation-inside-immediate',[bytes(bad),provider['raw']])
branches=[(off,raw) for off,raw,_ in root['instructions'] if raw[0]==0xe9 or (raw[:1]==b'\x0f' and 0x80<=raw[1]<=0x8f)]
assert branches
branch,encoded=branches[0]
operand=branch+len(encoded)-4
for name,target in [('branch-to-prologue',records['entry']['offset']),
                    ('branch-to-other-function',records['local']['offset']),
                    ('branch-inside-immediate',marker_at)]:
    bad=bytearray(root['raw']);struct.pack_into('<i',bad,view['text_at']+operand,target-operand-4)
    link(name,[bytes(bad),provider['raw']])
bad=bytearray(root['raw']);struct.pack_into('<Q',bad,view['stack_at']+56,2)
link('invent-absent-allocator-tail',[bytes(bad),provider['raw']])

# The actual over-budget witness is regenerated by the independent stack suite.
over = (R/'over-limit.request').read_bytes()
cursor=32+struct.unpack_from('<Q',over,16)[0];over_objects=[]
for _ in range(struct.unpack_from('<Q',over,8)[0]):
    size=struct.unpack_from('<Q',over,cursor)[0];cursor+=8
    over_objects.append(over[cursor:cursor+size]);cursor+=size
assert cursor==len(over)
bad=bytearray(over_objects[0]);v=inspect((R/'chain-entry.hepta').read_bytes(),bad)
row=next(r for r in v['records'] if r['name']=='step');callee=next(r for r in v['records'] if r['name']=='deep0')
at=v['stack_at']+64+v['function_count']*64+row['ordinal']*v['row_bytes']+callee['ordinal']//8
bad[at]&=~(1<<(callee['ordinal']%8))
link('over-budget-external-edge-cleared',[bytes(bad),*over_objects[1:]],entry=b'step',magic=b'HIASTKQ1',tool=N/'stack-linker-qualification')
rows[-1]['machine_code_unchanged']=True
rows[-1]['changed_object_bytes']=1

# Independently authored prior owned-flow fixtures are loaded as literal data only.
flow_raw=args.owned_flow.read_bytes()
assert sha(flow_raw)=='a4d02ac5caa4c13733333b233c16342803a35f82ce55c7764affe4551af336c0'
flow={}
for node in ast.parse(flow_raw).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('CALLER','FACTORY','TRANSFORM'):
        flow[node.targets[0].id]=ast.literal_eval(node.value).encode()
assert len(flow)==3
owned=[compile_source('owned-'+name.lower(),flow[name]) for name in ('CALLER','FACTORY','TRANSFORM')]
exe=link('three-owner-cfg',[u['raw'] for u in owned],accepted=True)
run('three-owner-false-cfg',exe,b'\0',bytes.fromhex('0c44083355550101'))
run('three-owner-true-cfg',exe,b'\1',bytes.fromhex('0c22081155550101'))
owner=owned[0];v=owner['view'];flags=struct.unpack_from('<Q',owner['raw'],v['stack_at']+56)[0]
assert flags==3
last=max(r['offset']+r['size'] for r in v['records'] if not r['imported'])
drop_len=len(bytes.fromhex(adaptation['machine_call_binding']['closed_chain_drop_hex']))
for name,offset in [('alter-chain-destructor',last),('alter-calloc',last+drop_len),('alter-free',last+drop_len+149)]:
    bad=bytearray(owner['raw']);bad[v['text_at']+offset]^=1
    link(name,[bytes(bad),owned[1]['raw'],owned[2]['raw']])
for flags in (0,1,2,4):
    bad=bytearray(owner['raw']);struct.pack_into('<Q',bad,v['stack_at']+56,flags)
    link('false-helper-flags-'+str(flags),[bytes(bad),owned[1]['raw'],owned[2]['raw']])
helper_calls=[(off,raw) for off,raw,_ in owner['instructions'] if raw[0]==0xe8 and off<last and off+5+struct.unpack('<i',raw[1:])[0]>=last]
assert helper_calls
off,raw=helper_calls[0];bad=bytearray(owner['raw']);value=struct.unpack('<i',raw[1:])[0]
struct.pack_into('<i',bad,v['text_at']+off+1,value+1)
link('call-into-helper-interior',[bytes(bad),owned[1]['raw'],owned[2]['raw']])
instruction_source = b'''hia-asm.v1 hepta.hia.program.v8 decoder_operations
exports(entry)
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  zero:usize = const() {"value":0}
  one:usize = const() {"value":1}
  two:usize = const() {"value":2}
  three:usize = const() {"value":3}
  four:usize = const() {"value":4}
  seven:usize = const() {"value":7}
  eight:usize = const() {"value":8}
  nine:usize = const() {"value":9}
  ten:usize = const() {"value":10}
  eleven:usize = const() {"value":11}
  twelve:usize = const() {"value":12}
  _ = buffer.copy(o,s,eight)
  _ = buffer.move_within(o,one,zero,four)
  _ = buffer.move_within(o,zero,two,three)
  _ = buffer.swap(o,zero,seven)
  a:i8 = const() {"value":-7}
  ad:i8 = const() {"value":2}
  aq:i8 = int.div.checked(a,ad)
  av:u8 = int.cast(aq) {"mode":"wrap"}
  b:i16 = const() {"value":-300}
  bd:i16 = const() {"value":2}
  bq:i16 = int.div.checked(b,bd)
  bv:u8 = int.cast(bq) {"mode":"wrap"}
  c:i32 = const() {"value":-70000}
  cd:i32 = const() {"value":2}
  cq:i32 = int.div.checked(c,cd)
  cv:u8 = int.cast(cq) {"mode":"wrap"}
  d:i64 = const() {"value":-9}
  dq:i64 = int.neg.checked(d)
  dv:u8 = int.cast(dq) {"mode":"wrap"}
  e:u16 = const() {"value":65535}
  ev:u8 = int.cast(e) {"mode":"wrap"}
  _ = buffer.store(o,eight,av)
  _ = buffer.store(o,nine,bv)
  _ = buffer.store(o,ten,cv)
  _ = buffer.store(o,eleven,dv)
  _ = buffer.store(o,twelve,ev)
  size:u32 = const() {"value":13}
  ret(size)
end
'''
instructions=compile_source('decoder-operations',instruction_source)
exe=link('all-emitted-decoder-families',[instructions['raw']],accepted=True)
for i,raw in enumerate((bytes(range(1,9)),bytes([255,0,42,33,22,11,244,177]))):
    expected=bytearray(raw);expected[1:5]=expected[0:4];expected[0:3]=expected[2:5]
    expected[0],expected[7]=expected[7],expected[0]
    expected+=bytes([253,106,72,9,255])
    run('signed-narrow-and-overlap-'+str(i),exe,raw,bytes(expected))
for path,before in tool_before.items():assert path.read_bytes()==before
assert args.owned_flow.read_bytes()==flow_raw
report={'schema':'heptatorch.stack-budget-machine-call-binding.v1','status':'PASS','cases':len(rows),
        'compiler_sha256':pins['compiler_sha256'],'linker_sha256':pins['linker_sha256']['runtime'],
        'qualification_linker_sha256':pins['linker_sha256']['qualification'],
        'reviewer_sha256':sha(Path(__file__).read_bytes()),'independent_disassembler':version,
        'owned_flow_fixture_authoring_sha256':sha(flow_raw),'rows':rows,
        'current_producer_metadata_single_bit_deletions_refused':True,
        'scope':'Native instruction-boundary call/matrix binding and closed-helper admission; not a validator of arbitrary foreign effects or arbitrary register/stack manipulation.'}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
