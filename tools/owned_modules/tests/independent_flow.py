#!/usr/bin/env python3
"""Native ABI2 triple-owner/CFG witnesses and independent lifetime controls.

This script authors existing HIA syntax, invokes native compiler/linker tools,
and compares fixed response bytes. It supplies no allocation or HIA interpreter.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess

COMPILER_SHA = '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
LINKER_SHA = '23eed229e329fb5b7e2116012f74f210d5de4acd2b4f8475e364890748da86ee'

FACTORY = '''hia-asm.v1 hepta.hia.program.v8 factory
exports(make3,take_head)
fn make3() -> (buffer,chain,buffer) effects(alloc,read,write,trap) entry(b0)
block b0()
  n0:usize = const() {"value":8}
  n1:usize = const() {"value":12}
  n2:usize = const() {"value":17}
  a:buffer = buffer.new(n0)
  b:buffer = buffer.new(n1)
  p:buffer = buffer.new(n2)
  m:mutbuf = borrow.mut(p)
  value:u8 = const() {"value":85}
  _ = buffer.fill(m,value)
  _ = loan.end(m)
  empty:chain = chain.empty()
  c:chain,rest:buffer,ok:bool = chain.try_push(empty,p)
  _ = owner.drop(rest)
  ret(a,c,b)
end
fn take_head(c:chain) -> (chain,buffer,bool) effects(alloc,read,write,trap) entry(b0)
block b0(c:chain)
  tail:chain,payload:buffer,present:bool = chain.pop(c)
  ret(tail,payload,present)
end
'''

TRANSFORM = '''hia-asm.v1 hepta.hia.program.v8 transform
exports(change3)
fn change3(a:buffer,c:chain,b:buffer,flag:bool) -> (buffer,chain,buffer) effects(read,write,trap) entry(b0)
block b0(a:buffer,c:chain,b:buffer,flag:bool)
  branch flag yes(a,c,b) no(a,c,b)
block yes(ay:buffer,cy:chain,by:buffer)
  my0:mutbuf = borrow.mut(ay)
  vy0:u8 = const() {"value":17}
  _ = buffer.fill(my0,vy0)
  _ = loan.end(my0)
  my1:mutbuf = borrow.mut(by)
  vy1:u8 = const() {"value":34}
  _ = buffer.fill(my1,vy1)
  _ = loan.end(my1)
  jump joined(ay,cy,by)
block no(an:buffer,cn:chain,bn:buffer)
  mn0:mutbuf = borrow.mut(an)
  vn0:u8 = const() {"value":51}
  _ = buffer.fill(mn0,vn0)
  _ = loan.end(mn0)
  mn1:mutbuf = borrow.mut(bn)
  vn1:u8 = const() {"value":68}
  _ = buffer.fill(mn1,vn1)
  _ = loan.end(mn1)
  jump joined(an,cn,bn)
block joined(x:buffer,y:chain,z:buffer)
  ret(z,y,x)
end
'''

CALLER = '''hia-asm.v1 hepta.hia.program.v8 caller
exports(entry)
fn make3() -> (buffer,chain,buffer) effects(alloc,read,write,trap) external()
fn change3(a:buffer,c:chain,b:buffer,flag:bool) -> (buffer,chain,buffer) effects(read,write,trap) external()
fn take_head(c:chain) -> (chain,buffer,bool) effects(alloc,read,write,trap) external()
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  zero:usize = const() {"value":0}
  zbyte:u8 = const() {"value":0}
  choice:u8 = buffer.load(s,zero)
  flag:bool = compare.ne(choice,zbyte)
  a:buffer,c:chain,b:buffer = call() {"function":"make3"}
  aa:buffer,cc:chain,bb:buffer = call(a,c,b,flag) {"function":"change3"}
  la:usize = buffer.len(aa)
  lb:usize = buffer.len(bb)
  la8:u8 = int.cast(la) {"mode":"wrap"}
  lb8:u8 = int.cast(lb) {"mode":"wrap"}
  ra:readbuf = borrow.read(aa)
  av:u8 = buffer.load(ra,zero)
  _ = loan.end(ra)
  rb:readbuf = borrow.read(bb)
  bv:u8 = buffer.load(rb,zero)
  _ = loan.end(rb)
  tail:chain,p:buffer,present:bool = call(cc) {"function":"take_head"}
  isempty:bool = chain.is_empty(tail)
  rp:readbuf = borrow.read(p)
  pv0:u8 = buffer.load(rp,zero)
  last:usize = const() {"value":16}
  pv1:u8 = buffer.load(rp,last)
  _ = loan.end(rp)
  _ = owner.drop(aa)
  _ = owner.drop(bb)
  _ = owner.drop(tail)
  _ = owner.drop(p)
  one:usize = const() {"value":1}
  two:usize = const() {"value":2}
  three:usize = const() {"value":3}
  four:usize = const() {"value":4}
  five:usize = const() {"value":5}
  six:usize = const() {"value":6}
  seven:usize = const() {"value":7}
  yes:u8 = const() {"value":1}
  exists:u8 = select(present,yes,zbyte)
  empty:u8 = select(isempty,yes,zbyte)
  _ = buffer.store(o,zero,la8)
  _ = buffer.store(o,one,av)
  _ = buffer.store(o,two,lb8)
  _ = buffer.store(o,three,bv)
  _ = buffer.store(o,four,pv0)
  _ = buffer.store(o,five,pv1)
  _ = buffer.store(o,six,exists)
  _ = buffer.store(o,seven,empty)
  count:u32 = const() {"value":8}
  ret(count)
end
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def function_module(body, params='', results='', effects='alloc,read,write,trap', imports=''):
    return ('hia-asm.v1 hepta.hia.program.v8 refusal\nexports(check)\n'+imports
            +f'fn check({params}) -> ({results}) effects({effects}) entry(b0)\n'
            +f'block b0({params})\n'+body+'end\n')


def refusal_sources():
    consume = 'fn consume(b:buffer) -> () effects(alloc,trap) external()\n'
    controls = {
        'external-consume-with-live-child': function_module(
            '  child:readbuf = borrow.read(b)\n'
            '  _ = call(b) {"function":"consume"}\n'
            '  _ = loan.end(child)\n  ret()\n', 'b:buffer', imports=consume),
        'use-after-external-consume': function_module(
            '  _ = call(b) {"function":"consume"}\n'
            '  size:usize = buffer.len(b)\n  ret(size)\n', 'b:buffer', 'usize', imports=consume),
        'duplicate-owner-external-arguments': function_module(
            '  _ = call(b,b) {"function":"take2"}\n  ret()\n', 'b:buffer',
            imports='fn take2(a:buffer,b:buffer) -> () effects(alloc,trap) external()\n'),
        'duplicate-three-result-owner': function_module(
            '  c:chain = chain.empty()\n  ret(b,c,b)\n', 'b:buffer', 'buffer,chain,buffer'),
        'owner-return-with-live-child': function_module(
            '  child:readbuf = borrow.read(b)\n  ret(b)\n', 'b:buffer', 'buffer'),
        'return-leaks-sibling-chain': function_module(
            '  c:chain = chain.empty()\n  ret(b)\n', 'b:buffer', 'buffer'),
        'cfg-omits-live-owner': function_module(
            '  c:chain = chain.empty()\n  jump next(b)\n'
            'block next(moved:buffer)\n  ret(moved)\n', 'b:buffer', 'buffer'),
        'drop-missing-new-trap-effect': function_module(
            '  _ = owner.drop(b)\n  ret()\n', 'b:buffer', effects='alloc'),
        'chain-pop-missing-new-trap-effect': function_module(
            '  c2:chain,b:buffer,ok:bool = chain.pop(c)\n  ret(c2,b,ok)\n',
            'c:chain', 'chain,buffer,bool', effects='alloc'),
        'chain-pop-with-live-head-loan': function_module(
            '  head:readbuf = chain.head_read(c)\n'
            '  c2:chain,b:buffer,ok:bool = chain.pop(c)\n'
            '  _ = loan.end(head)\n  ret(c2,b,ok)\n', 'c:chain', 'chain,buffer,bool'),
    }
    return controls


def link_request(objects):
    entry = b'entry'
    return (b'HIAOWN02'+struct.pack('<3Q', len(objects), len(entry), 0)+entry
            +b''.join(struct.pack('<Q', len(obj))+obj for obj in objects))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--linker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compiler, linker = args.compiler.resolve(), args.linker.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert sha(compiler.read_bytes()) == COMPILER_SHA
    assert sha(linker.read_bytes()) == LINKER_SHA
    rows = []

    def compile_source(name, source, expect=True):
        raw = source.encode()
        (out/(name+'.hepta')).write_bytes(raw)
        result = subprocess.run([str(compiler)], input=raw, capture_output=True,
                                env={'PATH': ''}, timeout=30)
        (out/(name+'.compile.stdout')).write_bytes(result.stdout)
        (out/(name+'.compile.stderr')).write_bytes(result.stderr)
        if expect:
            assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), name
            (out/(name+'.o')).write_bytes(result.stdout)
        else:
            assert result.returncode != 0 and not result.stdout, name
            rows.append({'case': name, 'kind': 'native_compilation_refusal',
                         'source_sha256': sha(raw), 'exit_code': result.returncode,
                         'output_bytes': len(result.stdout), 'pass': True})
        return result.stdout

    objects = [compile_source('caller', CALLER), compile_source('factory', FACTORY),
               compile_source('transform', TRANSFORM)]
    request = link_request(objects)
    (out/'valid.link-request.bin').write_bytes(request)
    linked = subprocess.run([str(linker)], input=request, capture_output=True,
                            env={'PATH': ''}, timeout=30)
    (out/'valid.link.stderr').write_bytes(linked.stderr)
    assert linked.returncode == 0 and not linked.stderr and linked.stdout.startswith(b'\x7fELF')
    application = out/'application'
    application.write_bytes(linked.stdout)
    application.chmod(0o700)
    for name, input_bytes, expected in (
            ('false-cfg-path', b'\x00', bytes([12, 68, 8, 51, 85, 85, 1, 1])),
            ('true-cfg-path', b'\x01', bytes([12, 34, 8, 17, 85, 85, 1, 1]))):
        result = subprocess.run([str(application)], input=input_bytes, capture_output=True,
                                env={'PATH': ''}, timeout=10)
        (out/(name+'.input.bin')).write_bytes(input_bytes)
        (out/(name+'.actual.bin')).write_bytes(result.stdout)
        (out/(name+'.expected.bin')).write_bytes(expected)
        (out/(name+'.execute.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr and result.stdout == expected, (name,result)
        rows.append({'case': name, 'kind': 'native_execution', 'input_hex': input_bytes.hex(),
                     'expected_hex': expected.hex(), 'actual_hex': result.stdout.hex(),
                     'exit_code': result.returncode, 'pass': True})
    for name, source in refusal_sources().items():
        compile_source(name, source, expect=False)

    mismatch_template = '''hia-asm.v1 hepta.hia.program.v8 mismatch
exports(entry)
fn make3() -> (RESULTS) effects(EFFECTS) external()
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  count:u32 = const() {"value":0}
  ret(count)
end
'''
    for name, results, effects in (
            ('ordered-owner-result-interface-mismatch', 'buffer,buffer,chain', 'alloc,read,write,trap'),
            ('callee-trap-effect-interface-mismatch', 'buffer,chain,buffer', 'alloc,read,write')):
        obj = compile_source(name, mismatch_template.replace('RESULTS', results).replace('EFFECTS', effects))
        raw = link_request([obj, objects[1]])
        (out/(name+'.link-request.bin')).write_bytes(raw)
        result = subprocess.run([str(linker)], input=raw, capture_output=True, env={'PATH': ''}, timeout=30)
        (out/(name+'.link.stdout')).write_bytes(result.stdout)
        (out/(name+'.link.stderr')).write_bytes(result.stderr)
        assert result.returncode != 0 and not result.stdout, (name,result.returncode,len(result.stdout))
        rows.append({'case': name, 'kind': 'native_link_refusal', 'request_sha256': sha(raw),
                     'exit_code': result.returncode, 'output_bytes': len(result.stdout), 'pass': True})
    report = {'schema': 'heptatorch.independent-owned-module-flow-review.v1',
              'status': 'PASS', 'cases': len(rows), 'execution_cases': 2,
              'compilation_refusals': 10, 'link_refusals': 2,
              'compiler_sha256': COMPILER_SHA, 'linker_sha256': LINKER_SHA,
              'observer_script_sha256': sha(Path(__file__).read_bytes()),
              'valid_application_sha256': sha(linked.stdout),
              'valid_source_sha256': dict((name,sha(source.encode())) for name,source in
                                          [('caller',CALLER),('factory',FACTORY),('transform',TRANSFORM)]),
              'valid_object_sha256': [sha(obj) for obj in objects],
              'native_allocation_provider': 'Candidate ABI2 Linux mmap/munmap private allocator',
              'runtime_path': '', 'host_allocation_or_numerical_runtime': False,
              'reference': 'Fixed byte expectations derived from literal fills, swapped lengths and chain payload; no native output used to establish expected bytes',
              'scope': 'Three true modules, three mixed owner slots, two input-dependent CFG paths, selected lifetime/effect/signature refusals; not general storage or complete compiler proof',
              'rows': rows}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
