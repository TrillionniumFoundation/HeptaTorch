"""Execute native module chains beyond unit instruction and old pack limits.

These are synthetic ABI/capacity probes, not additional PyTorch operators.
All arithmetic executes in the compiled HIA functions; Python derives a small
closed-form expected integer and transports independent framed bytes.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,select,struct,subprocess,time

sha=lambda raw:hashlib.sha256(raw).hexdigest()
ADDS=513
FUNCTIONS=8


def function(name,params,result,body):
    signature=','.join(f'{n}:{t}' for n,t in params)
    return f'fn {name}({signature}) -> ({result}) effects(read,write,trap) entry(b0)\nblock b0({signature})\n'+''.join('  '+line+'\n' for line in body)+'end\n'


def const(name,typ,value):return f'{name}:{typ} = const() '+json.dumps({'value':value},separators=(',',':'))
def call(name,typ,arg,target):return f'{name}:{typ} = call({arg}) '+json.dumps({'function':target},separators=(',',':'))


def program(units):
    sources=[];all_functions=[]
    for unit in range(units):
        bodies=[]
        if unit==0:
            body=[const('z','usize',0),'byte:u8 = buffer.load(s,z)',
                  'input:u64 = int.cast(byte) {"mode":"wrap"}',call('result','u64','input','step_0_0')]
            header=b'HTORSP01'+struct.pack('<QQQ',1,0,0)
            for i,value in enumerate(header):
                body += [const('at'+str(i),'usize',i),const('v'+str(i),'u8',value),
                         '_ = buffer.store(o,at'+str(i)+',v'+str(i)+')']
            for i in range(8):
                body += [const('shift'+str(i),'u64',8*i),
                         f'part{i}:u64 = int.shr.checked(result,shift{i})',
                         f'out{i}:u8 = int.cast(part{i}) {{"mode":"wrap"}}',
                         const('dest'+str(i),'usize',32+i),f'_ = buffer.store(o,dest{i},out{i})']
            body += [const('length','u32',40),'ret(length)']
            bodies.append(function('entry',[('s','readbuf'),('o','mutbuf'),('w','mutbuf')],'u32',body))
            public='entry'
        else:
            public='advance_'+str(unit)
            bodies.append(function(public,[('x','u64')],'u64',
                                   [call('v','u64','x',f'step_{unit}_0'),'ret(v)']))
        for index in range(FUNCTIONS):
            body=[const('one','u64',1)];last='x'
            for n in range(ADDS):
                name='v'+str(n);body.append(f'{name}:u64 = int.add.wrap({last},one)');last=name
            if index+1<FUNCTIONS:
                body += [call('next','u64',last,f'step_{unit}_{index+1}'),'ret(next)']
            elif unit+1<units:
                body += [call('next','u64',last,'advance_'+str(unit+1)),'ret(next)']
            else:body.append('ret('+last+')')
            bodies.append(function(f'step_{unit}_{index}',[('x','u64')],'u64',body))
        external='' if unit+1==units else (
            f'fn advance_{unit+1}(x:u64) -> (u64) effects(read,write,trap) external()\n')
        sources.append((f'hia-asm.v1 hepta.hia.program.v8 capacity_{unit}\nexports({public})\n'+
                        external+''.join(bodies)).encode())
        all_functions.extend(bodies)
    return sources,('hia-asm.v1 hepta.hia.program.v8 selfhost_exec\n'+''.join(all_functions)).encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compiler','legacy-compiler','flat-linker','packer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiler=args.compiler.resolve();legacy=args.legacy_compiler.resolve()
    linker=args.flat_linker.resolve();packer=args.packer.resolve()
    assert sha(compiler.read_bytes())=='e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
    assert sha(legacy.read_bytes())=='c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
    assert sha(linker.read_bytes())=='4c36e3409b397788664f1501896aff2719f667306795cb5fa95061d710fd6139'
    assert sha(packer.read_bytes())=='22600cee8853b6a49dfbf66154ddcdc99bdbbc6fcfb2face2605ad7117b20a44'
    rows=[]
    def run(exe,raw):return subprocess.run([str(exe)],input=raw,capture_output=True,env={'PATH':''},timeout=45)
    for count in (2,12):
        folder=out/str(count);folder.mkdir()
        sources,combined=program(count);objects=[];unit_counts=[]
        if count==2:
            (folder/'combined.hepta').write_bytes(combined)
            instructions=len(re.findall(rb'^  .* = ',combined,re.M))
            assert len(combined)<1<<20 and instructions>8192
            result=run(legacy,combined)
            assert result.returncode==2 and not result.stdout and not result.stderr
            rows.append({'case':'single-unit-instruction-bound','pass':True,
                         'source_bytes':len(combined),'instructions':instructions,
                         'function_count':combined.count(b'\nfn '),'expected_refusal_exit':2})
        for i,source in enumerate(sources):
            (folder/f'unit-{i}.hepta').write_bytes(source)
            result=run(compiler,source)
            assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
            (folder/f'unit-{i}.o').write_bytes(result.stdout);objects.append(result.stdout)
            n=len(re.findall(rb'^  .* = ',source,re.M));assert n<=8192
            unit_counts.append(n)
        request=b'HIAFLK01'+struct.pack('<QQQ',count,5,0)+b'entry'+b''.join(
            struct.pack('<Q',len(obj))+obj for obj in objects[::-1])
        (folder/'link-request.bin').write_bytes(request)
        result=run(linker,request)
        assert result.returncode==0 and not result.stderr and not result.stdout.startswith(b'\x7fELF')
        image=result.stdout;(folder/'linked.image').write_bytes(image)
        entry=69+struct.unpack_from('<i',image,65)[0];assert entry>1024
        if count==12:assert len(image)>1<<20
        packed=run(packer,b'HIAFLT01'+struct.pack('<Q',len(image))+image)
        assert packed.returncode==0 and not packed.stderr
        exe=folder/'session-probe';exe.write_bytes(packed.stdout);exe.chmod(0o500)
        child=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,env={'PATH':''},bufsize=0)
        def receive(n):
            data=bytearray();deadline=time.monotonic()+10
            while len(data)<n:
                ready,_,_=select.select([child.stdout],[],[],max(0,deadline-time.monotonic()))
                assert ready and child.poll() is None
                part=os.read(child.stdout.fileno(),n-len(data));assert part;data.extend(part)
            return bytes(data)
        outputs=[]
        for value in (7,255):
            raw=b'HTOSES01'+struct.pack('<Q',1)+bytes([value])
            assert os.write(child.stdin.fileno(),raw)==len(raw)
            header=receive(16);assert header==b'HTOSRP01'+struct.pack('<Q',40)
            actual=receive(40)
            wanted=b'HTORSP01'+struct.pack('<QQQQ',1,0,0,value+count*FUNCTIONS*ADDS)
            assert actual==wanted and not child.stdin.closed and child.poll() is None
            outputs.append({'input':value,'expected_u64':value+count*FUNCTIONS*ADDS,
                            'actual_body_hex':actual.hex(),'stdin_open_at_response':True})
        child.stdin.close();assert child.wait(timeout=5)==0 and not child.stderr.read() and not child.stdout.read()
        rows.append({'case':f'{count}-unit-executed-native-chain','pass':True,
                     'objects':count,'unit_instructions':unit_counts,'aggregate_instructions':sum(unit_counts),
                     'actually_executed_add_operations_per_frame':count*FUNCTIONS*ADDS,
                     'flat_image_bytes':len(image),'selected_entry_offset':entry,
                     'flat_image_sha256':sha(image),'packed_executable_sha256':sha(packed.stdout),
                     'packed_executable_bytes':len(packed.stdout),'link_request_bytes':len(request),
                     'before_eof_responses_same_process':True,'outputs':outputs})
    report={'schema':'heptatorch.session-native-module-growth-review.v1','pass':True,
            'cases':len(rows),'rows':rows,'module_compiler_sha256':sha(compiler.read_bytes()),
            'flat_linker_sha256':sha(linker.read_bytes()),'wide_packer_sha256':sha(packer.read_bytes()),
            'synthetic_capacity_probe_not_pytorch_operator_coverage':True,
            'runtime_path':'empty','no_host_numerical_runtime':True}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
