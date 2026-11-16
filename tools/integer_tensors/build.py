"""Emit and compile a genuine native HIA int64 successor of frozen523.

Python checks byte custody and authors modules. Tensor operations, conversion,
ownership and allocation execute exclusively in the generated HIA program.
"""
import argparse
import ast
import json
import re
import struct
import subprocess
from pathlib import Path

from authoring_support import CALL, COMPILER, EXTERNAL, LINKER, PARENT, Emitter, functions, once, sha
import storage

HERE = Path(__file__).resolve().parent
PARENT_AUTHOR = '2127b5726f70b5a605942258857b4c8e38cc0a2600194d519ffdf2813711d032'
PARENT_IMAGE = '830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'repository', 'upstream', 'compactor', 'compiler', 'linker', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    compiler, linker = a.compiler.resolve(), a.linker.resolve()
    assert sha(compiler.read_bytes()) == COMPILER
    assert sha(linker.read_bytes()) == LINKER
    parent = a.parent.resolve()
    assert sha((parent/'tensor-dtype-dropout-session').read_bytes()) == PARENT_IMAGE
    author = (parent/'overlay-authoring.py').read_bytes()
    assert sha(author) == PARENT_AUTHOR
    e = Emitter(parent, a.repository.resolve(), a.upstream.resolve(), a.compactor.resolve())
    original = {n.name: ast.unparse(n) for n in ast.parse(author).body if isinstance(n, ast.FunctionDef)}
    own = {name: (HERE/name).read_bytes() for name in ('build.py', 'storage.py', 'arithmetic.py', 'dropout.py', 'scalar.py', 'view_roots.py', 'authoring_support.py')}
    for name, raw in own.items():
        (out/name).write_bytes(raw)
    changes = storage.patches(original, once)
    changes['entry'] = once(original['entry'],
        '        if code >= 96 and code <= 102:',
        '        if code == 106:\n            next = integer_scalar_command(s,o,w,32+i*64,at)\n        elif code == 105:\n            next = integer_dropout_command(s,o,w,scratch,32+i*64,at)\n        elif code == 103 or code == 104:\n            next = integer_command(s,o,w,32+i*64,at)\n        elif code >= 96 and code <= 102:')
    changes['entry'] = once(changes['entry'],
        '        at = next\n',
        '        if code == 105 and getw(o,at+32) != 0:\n            halted = 1\n        at = next\n')
    for name in changes:
        regenerated = functions(e.emit(original[name]))
        assert regenerated == {name: e.parent[name]}, ('baseline authoring differs', name)
    aliases = {}
    alias_inverse = {}
    for name in ('pointwise_axis_order', 'pointwise_layout'):
        body = e.parent[name]
        revised = once(body, ('fn '+name+'(').encode(), ('fn integer_'+name+'(').encode())
        callee = 'pointwise_compare_axes' if name == 'pointwise_axis_order' else 'pointwise_axis_order'
        revised = once(revised, ('{"function":"'+callee+'"}').encode(), ('{"function":"integer_'+callee+'"}').encode())
        inverse = once(revised, ('{"function":"integer_'+callee+'"}').encode(), ('{"function":"'+callee+'"}').encode())
        inverse = once(inverse, ('fn integer_'+name+'(').encode(), ('fn '+name+'(').encode())
        assert inverse == body
        aliases['integer_'+name] = revised
        e.add_context('integer_'+name,revised)
        alias_inverse[name] = {'original_sha256':sha(body),'alias_sha256':sha(revised),'inverse_exact':True}
    authored = '\n\n'.join(changes.values()) + '\n\n' + own['arithmetic.py'].decode() + '\n\n' + own['dropout.py'].decode() + '\n\n' + own['scalar.py'].decode() + '\n\n' + own['view_roots.py'].decode()
    (out/'overlay-authoring.py').write_text(authored)
    try:
        fragment = e.emit(authored)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'authoring', 'error':repr(error),
            'native_attempted':False}, indent=2)+'\n')
        raise
    (out/'overlay.hepta').write_bytes(fragment)
    combined = dict(e.parent)
    combined.update(aliases)
    combined.update(functions(fragment))
    tag_edits = []
    for name in ('init_state', 'state_header', 'so_state_init', 'so_state_header'):
        outer = name in ('init_state', 'state_header')
        before = b'HTDTYP03' if outer else b'HTODTY03'
        after = b'HTINT002' if outer else b'HTOINT02'
        old = ('{"value":' + str(int.from_bytes(before, 'little')) + '}').encode()
        new = ('{"value":' + str(int.from_bytes(after, 'little')) + '}').encode()
        raw = combined[name]
        revised = once(raw, old, new)
        assert once(revised, new, old) == raw
        combined[name] = revised
        tag_edits.append({'function':name, 'before':sha(raw), 'after':sha(revised), 'inverse_exact':True})
    raw = b'hia-asm.v1 hepta.hia.program.v8 integer_tensor_runtime\n' + b''.join(combined.values())
    (out/'runtime.hepta').write_bytes(raw)
    (out/'authoring-custody.json').write_text(json.dumps({
        'parent_source_sha256':PARENT, 'parent_authoring_sha256':PARENT_AUTHOR,
        'baseline_reemission_all_exact':True, 'native_tags':tag_edits, 'layout_alias_inverse':alias_inverse,
        'changes':{n:{'before':original[n], 'after':body} for n,body in changes.items()},
        'compactor_records':e.checks}, indent=2)+'\n')
    (out/'source-map.json').write_text(json.dumps([
        {'name':n, 'sha256':sha(body), 'parent_sha256':sha(e.parent[n]) if n in e.parent else None,
         'unchanged_parent':e.parent.get(n)==body} for n,body in combined.items()], indent=2)+'\n')

    ordered, done, active = [], set(), set()
    def visit(name):
        if name in EXTERNAL:
            return
        assert name in combined and name not in active, ('missing/recursive function',name)
        if name in done:
            return
        active.add(name)
        for target in sorted({n.decode() for n in CALL.findall(combined[name])}):
            visit(target)
        active.remove(name)
        done.add(name)
        ordered.append(name)
    for name in combined:
        visit(name)
    groups, group, instructions, blocks = [], [], 0, 0
    for name in ordered:
        body = combined[name]
        ni, nb = body.count(b' = '), len(re.findall(rb'^block ',body,re.M))
        if group and (instructions+ni>3600 or blocks+nb>1000 or len(group)>=65):
            groups.append(group)
            group,instructions,blocks = [],0,0
        group.append(name)
        instructions += ni
        blocks += nb
    if group:
        groups.append(group)
    units = [('rng03',e.rng_raw)]
    for i,names in enumerate(groups):
        calls = {n.decode() for name in names for n in CALL.findall(combined[name])}
        imports = sorted(calls-set(names))
        assert len(names)+len(imports)<=256
        declarations = []
        for name in imports:
            body = e.rng[name] if name in EXTERNAL else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n')[0])+b'\n')
        unit = f'integer_tensor_{i:02d}'
        source = f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,source))
    units.append(('owners',e.owner_raw))
    report = {'schema':'heptatorch.integer-tensor-native-build.v1', 'status':'BUILDING',
        'parent_source_sha256':PARENT, 'parent_native_sha256':PARENT_IMAGE,
        'source_sha256':sha(raw), 'descriptor_bytes':640, 'state_format':'HTINT002',
        'backing_format':'HTOINT02', 'functions':len(combined),
        'blocks':len(re.findall(rb'^block ',raw,re.M)), 'instructions':raw.count(b' = '),
        'compiler_sha256':COMPILER, 'linker_sha256':LINKER,
        'changed_parent_functions':[n for n in e.parent if combined[n]!=e.parent[n]],
        'added_functions':[n for n in combined if n not in e.parent],
        'inputs_sha256':{str(p):sha(data) for p,data in e.inputs.items()},
        'authoring_inputs_sha256':{n:sha(data) for n,data in own.items()}, 'units':[],
        'runtime_qualified':False, 'full_pytorch_rewrite_complete':False, 'pytorch_api_denominator':2590}
    def save():
        (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    objects = []
    for name,source in units:
        (out/(name+'.hepta')).write_bytes(source)
        result = subprocess.run([str(compiler)],input=source,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout)
        (out/(name+'.stderr')).write_bytes(result.stderr)
        row = {'name':name, 'source_sha256':sha(source), 'object_sha256':sha(result.stdout),
            'functions':len(functions(source)), 'blocks':len(re.findall(rb'^block ',source,re.M)),
            'instructions':source.count(b' = '), 'compiler_exit':result.returncode}
        report['units'].append(row)
        save()
        if result.returncode!=0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
            report['status']='NATIVE_COMPILER_REFUSAL';save();raise AssertionError(row)
        objects.append(result.stdout)
    request = b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
    (out/'link.request.bin').write_bytes(request)
    result = subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stderr').write_bytes(result.stderr)
    report['link_exit']=result.returncode
    if result.returncode!=0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
        (out/'link.refusal.stdout').write_bytes(result.stdout)
        report['status']='NATIVE_LINKER_REFUSAL';save();raise AssertionError(result.returncode)
    app=out/'tensor-integer-session';app.write_bytes(result.stdout);app.chmod(0o500)
    e.unchanged()
    assert (parent/'overlay-authoring.py').read_bytes()==author
    for name,data in own.items():
        assert (HERE/name).read_bytes()==data
    assert sha(compiler.read_bytes())==COMPILER and sha(linker.read_bytes())==LINKER
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout))
    save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256','authoring_inputs_sha256','units')},indent=2))


if __name__=='__main__':
    main()
