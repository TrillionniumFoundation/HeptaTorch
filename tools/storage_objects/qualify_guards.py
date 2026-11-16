#!/usr/bin/env python3
"""Native HIA edge-state witnesses, with no fabricated owner capability.

The fixture driver obtains a real buffer from step, changes only scalar state
through an actual mutable loan, and consumes/returns that owner through step.
It snapshots bytes for observation; snapshots are never cast back to owners.
These injected internal states are explicit qualification fixtures, not public
wire features and not PyTorch API compatibility tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import body,handle,parse_body
from qualify import check_dump

COMPILER_SHA='0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
LINKER_SHA='23eed229e329fb5b7e2116012f74f210d5de4acd2b4f8475e364890748da86ee'
sha=lambda b:hashlib.sha256(b).hexdigest()


class Source:
    def __init__(self):
        self.n=0
        self.lines=['hia-asm.v1 hepta.hia.program.v8 storage_guard', 'exports(entry)',
          'fn step(s:readbuf,o:mutbuf,state:buffer) -> (buffer,u32) effects(alloc,read,write,trap) external()',
          'fn close(state:buffer) -> () effects(alloc,trap) external()',
          'fn uput(w:mutbuf,at:u64,value:u64) -> (u64) effects(read,write,trap) external()',
          'fn audit(w:mutbuf) -> (u64) effects(read,write,trap) external()',
          'fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)',
          'block b0(s:readbuf,o:mutbuf,w:mutbuf)']

    def emit(self,op,args=(),types=(),**attrs):
        names=[]
        for typ in types:
            name='v'+str(self.n);self.n+=1;names.append(name)
        left=','.join(n+':'+t for n,t in zip(names,types)) or '_'
        self.lines.append('  '+left+' = '+op+'('+','.join(args)+')'+(' '+json.dumps(attrs,separators=(',',':')) if attrs else ''))
        return names[0] if len(names)==1 else names

    def c(self,value,typ='u64'):
        return self.emit('const',types=[typ],value=value)

    def put(self,mut,at,value):
        if isinstance(at,int):at=self.c(at)
        if isinstance(value,int):value=self.c(value)
        return self.emit('call',[mut,at,value],['u64'],function='uput')

    def cast(self,value,typ):
        return self.emit('int.cast',[value],[typ],mode='wrap')

    def add(self,a,b):
        if isinstance(b,int):b=self.c(b)
        return self.emit('int.add.wrap',[a,b],['u64'])


def driver(second,mutations,fills):
    g=Source()
    zero=g.c(0,'usize')
    empty=g.emit('buffer.new',[zero],['buffer'])
    state,first_len=g.emit('call',['s','o',empty],['buffer','u32'],function='step')
    m=g.emit('borrow.mut',[state],['mutbuf'])
    for at,value in mutations:g.put(m,at,value)
    for at,size,value in fills:
        g.emit('buffer.fill_range',[m,g.c(at,'usize'),g.c(size,'usize'),g.c(value,'u8')])
    before_audit=g.emit('call',[m],['u64'],function='audit')
    g.emit('loan.end',[m])
    original_size=g.emit('buffer.len',[state],['usize'])
    snapshot=g.emit('buffer.new',[original_size],['buffer'])
    source=g.emit('borrow.read',[state],['readbuf'])
    target=g.emit('borrow.mut',[snapshot],['mutbuf'])
    g.emit('buffer.copy_range',[target,source,zero,zero,original_size])
    g.emit('loan.end',[target]);g.emit('loan.end',[source])
    request=g.emit('buffer.new',[g.c(len(second),'usize')],['buffer'])
    target=g.emit('borrow.mut',[request],['mutbuf'])
    assert len(second)%8==0
    for at in range(0,len(second),8):g.put(target,at,struct.unpack_from('<Q',second,at)[0])
    g.emit('loan.end',[target])
    source=g.emit('borrow.read',[request],['readbuf'])
    state,result_len=g.emit('call',[source,'o',state],['buffer','u32'],function='step')
    g.emit('loan.end',[source]);g.emit('owner.drop',[request])
    m=g.emit('borrow.mut',[state],['mutbuf'])
    after_audit=g.emit('call',[m],['u64'],function='audit')
    g.emit('loan.end',[m])
    final_size=g.emit('buffer.len',[state],['usize'])
    at=g.cast(result_len,'u64')
    source=g.emit('borrow.read',[snapshot],['readbuf'])
    g.emit('buffer.copy_range',['o',source,g.cast(at,'usize'),zero,original_size])
    g.emit('loan.end',[source]);g.emit('owner.drop',[snapshot])
    at=g.add(at,g.cast(original_size,'u64'))
    source=g.emit('borrow.read',[state],['readbuf'])
    g.emit('buffer.copy_range',['o',source,g.cast(at,'usize'),zero,final_size])
    g.emit('loan.end',[source])
    g.emit('call',[state],function='close')
    at=g.add(at,g.cast(final_size,'u64'))
    for value in [g.cast(result_len,'u64'),g.cast(original_size,'u64'),g.cast(final_size,'u64'),before_audit,after_audit]:
        g.put('o',at,value);at=g.add(at,8)
    result=g.cast(at,'u32')
    g.lines+=['  ret('+result+')','end']
    return '\n'.join(g.lines)+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--linker',type=Path,required=True)
    p.add_argument('--module-dir',type=Path,required=True)
    p.add_argument('--legacy-module-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    compiler,linker=a.compiler.resolve(),a.linker.resolve()
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    T=lambda s,g=1:handle(1,s,g)
    S=lambda s,g=1:handle(2,s,g)
    V=lambda s,g=1:handle(3,s,g)
    slot=lambda k,s:128+((k-1)*4+s-1)*64
    initial=body([(1,16,7)],4,64)
    cases=[
      {'name':'exhausted-tensor-generation-slot-is-skipped','initial':initial,
       'second':body([(1,16,8),(20,)]),'mutations':[(slot(1,2),0xffffffff)],'fills':[],
       'status':0,'create':[T(3),S(2),V(2),16,16]},
      {'name':'all-old-free-generations-exhausted-table-grows','initial':initial,
       'second':body([(1,16,8),(20,)]),'mutations':[(slot(1,s),0xffffffff) for s in (2,3,4)],'fills':[],
       'status':0,'create':[T(5),S(2),V(2),16,16]},
      {'name':'version-counter-does-not-wrap','initial':initial,
       'second':body([(11,T(1),0,9)]),'mutations':[(slot(3,1)+16,0xffffffffffffffff)],'fills':[],
       'status':8,'unchanged':True},
      {'name':'resize-epoch-does-not-wrap','initial':initial,
       'second':body([(17,8,128)]),'mutations':[(40,0xffffffffffffffff)],'fills':[],
       'status':8,'unchanged':True}]
    base=424*4+144
    free=128+384*4
    mutations=[(32,5)]
    for s in range(1,5):mutations.append((slot(2,s)+16,16+(s-1)*32))
    for i in range(5):mutations.extend([(free+i*16,i*32),(free+i*16+8,16)])
    fills=[(base,144,0)]+[(base+16+i*32,16,65+i) for i in range(4)]
    full={'name':'full-free-table-release-coalesces-without-extra-slot',
          'initial':body([(1,16,65+i) for i in range(4)],4,144),
          'second':body([(3,T(2)),(20,),(12,T(1),15),(12,T(3),15),(12,T(4),15),(16,)]),
          'mutations':mutations,'fills':fills,'status':0,'full_free':True}
    cases.append(full)
    if a.legacy_module_dir:
        cases.append({**full,'name':'preserved-legacy-full-free-table-refusal',
                      'status':9,'unchanged':True,'legacy':True,'full_free':False})
    rows=[]
    for case in cases:
        out=a.output/case['name'];out.mkdir()
        source=driver(case['second'],case['mutations'],case['fills']).encode()
        (out/'driver.hepta').write_bytes(source)
        compiled=subprocess.run([str(compiler)],input=source,capture_output=True,env={'PATH':''},timeout=30)
        (out/'compile.stdout').write_bytes(compiled.stdout);(out/'compile.stderr').write_bytes(compiled.stderr)
        assert compiled.returncode==0 and not compiled.stderr and compiled.stdout.startswith(b'\x7fELF'),case['name']
        module_dir=a.legacy_module_dir if case.get('legacy') else a.module_dir
        names=['bytes','records','audit','ops','owners']
        objects=[(module_dir/(n+'.o')).read_bytes() for n in names]+[compiled.stdout]
        request=b'HIAOWN02'+struct.pack('<3Q',len(objects),5,0)+b'entry'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
        (out/'link-request.bin').write_bytes(request)
        linked=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=30)
        (out/'link.stderr').write_bytes(linked.stderr)
        assert linked.returncode==0 and not linked.stderr and linked.stdout.startswith(b'\x7fELF'),case['name']
        executable=out/'probe';executable.write_bytes(linked.stdout);executable.chmod(0o700)
        (out/'input.bin').write_bytes(case['initial'])
        result=subprocess.run([str(executable.resolve())],input=case['initial'],capture_output=True,env={'PATH':''},timeout=10)
        (out/'actual.bin').write_bytes(result.stdout);(out/'stderr.bin').write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr and len(result.stdout)>=40,case['name']
        n,before_len,after_len,before_audit,after_audit=struct.unpack_from('<5Q',result.stdout,len(result.stdout)-40)
        assert n+before_len+after_len+40==len(result.stdout) and before_audit==after_audit==0
        response=parse_body(result.stdout[:n]);assert response['status']==case['status'],(case['name'],response['status'])
        before=result.stdout[n:n+before_len];after=result.stdout[n+before_len:n+before_len+after_len]
        before_info,after_info=check_dump(before),check_dump(after)
        (out/'before.bin').write_bytes(before);(out/'after.bin').write_bytes(after)
        if case.get('unchanged'):assert before==after
        if 'create' in case:assert response['records'][0]['values']==tuple(case['create'])
        if case.get('full_free'):
            assert before_info['free_extents']==[(i*32,16) for i in range(5)]
            assert after_info['free_extents']==[(0,16),(32,48),(96,16),(128,16)]
            assert [r['values'][0] for r in response['records'][2:5]]==[65,67,68]
        row={'case':case['name'],'pass':True,'driver_source_sha256':sha(source),
             'native_executable_sha256':sha(linked.stdout),'business_module_objects':{n:sha(o) for n,o in zip(names,objects)},
             'status':response['status'],'native_audit_before':before_audit,'native_audit_after':after_audit,
             'before_state_sha256':sha(before),'after_state_sha256':sha(after),
             'entire_state_unchanged':before==after,
             'preserved_legacy_refusal':bool(case.get('legacy'))}
        (out/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(case['name']+' PASS',flush=True)
    report={'schema':'heptatorch.native-storage-object-extreme-state-guards.v1','status':'PASS',
            'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,
            'script_sha256':sha(Path(__file__).read_bytes()),'cases':len(rows),'rows':rows,
            'fixtures_are_internal_state_witnesses':True,'owner_descriptor_fabrication':False,
            'public_scalar_to_owner_operation_added':False,'pytorch_compatibility_claimed':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
