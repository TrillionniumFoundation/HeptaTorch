"""Independent source-level alias/effect audit of native external declarations.

The host only constructs HIA and packets; all admission and linking is native.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compiler', type=Path, required=True)
parser.add_argument('--linker', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
arguments = parser.parse_args()
ROOT = arguments.output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
COMPILER = arguments.compiler.resolve()
LINKER = arguments.linker.resolve()
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha(COMPILER.read_bytes()) == 'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
assert sha(LINKER.read_bytes()) == '1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8'
PINS = {'compiler_sha256': sha(COMPILER.read_bytes()), 'linker_sha256': sha(LINKER.read_bytes())}
ROWS = []
ENTRY = 's:readbuf,o:mutbuf,w:mutbuf'


def module(name, body, exports='entry'):
    return ('hia-asm.v1 hepta.hia.program.v8 ' + name + '\nexports(' + exports + ')\n' + body).encode()


def fn(name, params, result, lines, effects='read,write,trap', tail=''):
    return ('fn ' + name + '(' + params + ') -> (' + result + ') effects(' + effects + ') entry(b0)' + tail + '\n'
            'block b0(' + params + ')\n' + ''.join('  ' + line + '\n' for line in lines) + 'end\n')


def external(name, params, result, effects='read,write,trap'):
    return 'fn ' + name + '(' + params + ') -> (' + result + ') effects(' + effects + ') external()\n'


def call(out, typ, args, name):
    return out + ':' + typ + ' = call(' + args + ') {"function":"' + name + '"}'


def const(name, typ, value):
    return name + ':' + typ + ' = const() ' + json.dumps({'value': value}, separators=(',', ':'))


def compile_source(name, text, accepted):
    (ROOT / (name + '.hepta')).write_bytes(text)
    result = subprocess.run([str(COMPILER)], input=text, capture_output=True, timeout=20, env={'PATH': ''})
    (ROOT / (name + '.stdout')).write_bytes(result.stdout)
    (ROOT / (name + '.stderr')).write_bytes(result.stderr)
    assert result.returncode == (0 if accepted else 2) and not result.stderr, (name, result.returncode, result.stderr)
    assert result.stdout.startswith(b'\x7fELF') if accepted else not result.stdout, name
    ROWS.append({'case': name, 'kind': 'compiler-accept' if accepted else 'compiler-refuse', 'status': 'PASS',
                 'source_sha256': sha(text), 'exit_code': result.returncode, 'output_bytes': len(result.stdout)})
    return result.stdout


def link_and_execute(name, objects, expected):
    entry = b'entry'
    packet = b'HIABLK01' + struct.pack('<QQQ', len(objects), len(entry), 0) + entry
    packet += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    result = subprocess.run([str(LINKER)], input=packet, capture_output=True, timeout=20, env={'PATH': ''})
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), (name, result.returncode)
    binary = ROOT / (name + '.native')
    binary.write_bytes(result.stdout); binary.chmod(0o700)
    execution = subprocess.run([str(binary)], input=b'input', capture_output=True, timeout=10, env={'PATH': ''})
    assert execution.returncode == 0 and not execution.stderr and execution.stdout == expected, (name, execution.returncode, execution.stdout)
    (ROOT / (name + '.request')).write_bytes(packet)
    ROWS.append({'case': name, 'kind': 'linked-execution', 'status': 'PASS', 'expected_hex': expected.hex(),
                 'observed_hex': execution.stdout.hex(), 'executable_sha256': sha(result.stdout),
                 'link_request_sha256': sha(packet)})


read_decl = external('read_byte', 'r:readbuf', 'u8', 'read,trap')
read_provider = fn('read_byte', 'r:readbuf', 'u8', [const('zero', 'usize', 0), 'value:u8 = buffer.load(r,zero)', 'ret(value)'], 'read,trap')
rp = compile_source('read-provider', module('read_provider', read_provider, 'read_byte'), True)
init = [const('zero', 'usize', 0), const('first', 'u8', 91), '_ = buffer.store(w,zero,first)']
finish = [const('next', 'u8', 92), '_ = buffer.store(w,zero,next)', '_ = buffer.store(o,zero,value)',
          const('one', 'usize', 1), 'again:u8 = buffer.load(w,zero)', '_ = buffer.store(o,one,again)',
          const('n', 'u32', 2), 'ret(n)']

# A private returned loan remains tied to its caller's parent across the import.
local_loan = fn('local_loan', 'parent:mutbuf', 'readbuf', ['child:readbuf = borrow.read(parent)', 'ret(child)'],
                tail=' return_loans(["parent"])')
body = init + [call('child', 'readbuf', 'w', 'local_loan'), call('value', 'u8', 'child', 'read_byte'), '_ = loan.end(child)'] + finish
consumer = module('private_loan_user', fn('entry', ENTRY, 'u32', body) + local_loan + read_decl)
obj = compile_source('private-returned-loan-to-external', consumer, True)
link_and_execute('private-loan-parent-resumes', [obj, rp], b'\x5b\x5c')

# Two distinct read loans with one mutable parent may cross together.
read_pair = external('read_pair', 'a:readbuf,b:readbuf', 'u8', 'read,trap')
pair_provider = fn('read_pair', 'a:readbuf,b:readbuf', 'u8', [const('zero', 'usize', 0),
                   'av:u8 = buffer.load(a,zero)', 'bv:u8 = buffer.load(b,zero)', 'sum:u8 = int.add.wrap(av,bv)', 'ret(sum)'], 'read,trap')
pp = compile_source('read-pair-provider', module('read_pair_provider', pair_provider, 'read_pair'), True)
body = init + ['left:readbuf = borrow.read(w)', 'right:readbuf = borrow.read(w)',
               call('value', 'u8', 'left,right', 'read_pair'), '_ = loan.end(left)', '_ = loan.end(right)'] + finish
obj = compile_source('sibling-read-loans-cross-external', module('sibling_reads', fn('entry', ENTRY, 'u32', body) + read_pair), True)
link_and_execute('sibling-reads-parent-resumes', [pp, obj], b'\xb6\x5c')

# CFG incoming-resource state, followed by an external call and explicit end.
cfg = ('fn entry(' + ENTRY + ') -> (u32) effects(read,write,trap) entry(b0)\n'
       'block b0(' + ENTRY + ')\n'
       '  z0:usize = const() {"value":0}\n'
       '  first:u8 = const() {"value":91}\n'
       '  _ = buffer.store(w,z0,first)\n'
       '  condition:bool = const() {"value":true}\n'
       '  branch condition b1(s,o,w) b2(s,o,w)\n'
       'block b1(s1:readbuf,o1:mutbuf,w1:mutbuf)\n'
       '  r1:readbuf = borrow.read(w1)\n'
       '  jump b3(w1,r1,o1,s1)\n'
       'block b2(s2:readbuf,o2:mutbuf,w2:mutbuf)\n'
       '  r2:readbuf = borrow.read(w2)\n'
       '  jump b3(w2,r2,o2,s2)\n'
       'block b3(q:mutbuf,r:readbuf,out:mutbuf,src:readbuf)\n'
       '  got:u8 = call(r) {"function":"read_byte"}\n'
       '  _ = loan.end(r)\n'
       '  z3:usize = const() {"value":0}\n'
       '  _ = buffer.store(out,z3,got)\n'
       '  next:u8 = const() {"value":92}\n'
       '  _ = buffer.store(q,z3,next)\n'
       '  n:u32 = const() {"value":1}\n'
       '  ret(n)\nend\n')
obj = compile_source('cfg-joined-loan-cross-external', module('cfg_user', cfg + read_decl), True)
link_and_execute('cfg-parent-resumes-after-call', [obj, rp], b'\x5b')

# Invalid aliases are rejected even if the external signature claims no effects.
no_effect_pair = external('pair', 'a:mutbuf,b:mutbuf', 'u32', '')
compile_source('same-mutable-root-even-pure-declaration', module('same_root', fn('entry', ENTRY, 'u32', [call('n', 'u32', 'w,w', 'pair'), 'ret(n)']) + no_effect_pair), False)
mixed = external('mixed', 'r:readbuf,m:mutbuf', 'u8')
bad = init + ['child:readbuf = borrow.read(w)', call('value', 'u8', 'child,w', 'mixed'), '_ = loan.end(child)'] + finish
compile_source('read-child-and-suspended-mutable-parent', module('mixed_alias', fn('entry', ENTRY, 'u32', bad) + mixed), False)
bad = init + [call('child', 'readbuf', 'w', 'local_loan'), call('value', 'u8', 'child,w', 'mixed'), '_ = loan.end(child)'] + finish
compile_source('private-returned-alias-and-parent', module('returned_alias', fn('entry', ENTRY, 'u32', bad) + local_loan + mixed), False)
compile_source('cfg-joined-alias-and-parent', module('cfg_alias', cfg.replace('call(r) {"function":"read_byte"}', 'call(r,q) {"function":"mixed"}') + mixed), False)

# Forward declarations cannot hide required effects or resource results.
late = external('writes', 'w:mutbuf', 'u32', 'write')
compile_source('late-declaration-effect-understatement', module('late_effect', fn('entry', ENTRY, 'u32', [call('n', 'u32', 'w', 'writes'), 'ret(n)'], effects='') + late), False)
compile_source('unused-external-allocation-effect', module('unused_alloc', fn('entry', ENTRY, 'u32', [const('n', 'u32', 0), 'ret(n)']) + external('unused', 'w:mutbuf', 'u64', 'alloc')), False)
compile_source('unused-external-multiple-results', module('multi_result', fn('entry', ENTRY, 'u32', [const('n', 'u32', 0), 'ret(n)']) + external('unused', 'w:mutbuf', 'u64,u32')), False)
compile_source('unused-private-owned-parameter', module('private_owned', fn('entry', ENTRY, 'u32', [const('n', 'u32', 0), 'ret(n)']) + fn('owner', 'x:buffer', 'u64', [const('r', 'u64', 0), 'ret(r)'])), False)
illegal = fn('illegal', 'r:readbuf', 'u64', ['m:mutbuf = borrow.mut(r)', '_ = loan.end(m)', const('n', 'u64', 0), 'ret(n)'])
compile_source('provider-cannot-upgrade-read-resource', module('read_upgrade', illegal, 'illegal'), False)
write_hidden = fn('write_hidden', 'w:mutbuf', 'u32', ['m:mutbuf = borrow.mut(w)', const('z', 'usize', 0), const('v', 'u8', 1), '_ = buffer.store(m,z,v)', '_ = loan.end(m)', const('n', 'u32', 0), 'ret(n)'], effects='read,trap')
compile_source('write-through-child-still-requires-write', module('hidden_effect', write_hidden, 'write_hidden'), False)
compile_source('external-tail-loan-annotation-refused', module('external_tail', fn('entry', ENTRY, 'u32', [const('n', 'u32', 0), 'ret(n)']) + read_decl.rstrip() + ' return_loans(["r"])\n'), False)

assert sha(COMPILER.read_bytes()) == PINS['compiler_sha256'] and sha(LINKER.read_bytes()) == PINS['linker_sha256']
report = {'status': 'PASS', **PINS, 'rows': ROWS, 'cases': len(ROWS),
          'scope': 'Source compiler external alias/effect rules, private loan provenance and CFG joins; no malicious ELF-object admission claim',
          'native_runtime_path': '', 'canonical_files_modified': False}
(ROOT / 'external-capability-review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'rows'}))
