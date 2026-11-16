"""Author real HIA mixed-autograd code, then compile every module natively."""
import argparse
import ast
import json
import re
import struct
import subprocess
from pathlib import Path

from authoring_support import CALL, COMPILER, EXTERNAL, LINKER, PARENT, Emitter, functions, once, sha

HERE = Path(__file__).resolve().parent
PARENT_AUTHOR = '27a1dd831a43dbb1167f078052adf9a836c1be5ce81f077ee17fd85de55cd0bd'
PARENT_IMAGE = 'b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08'


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
    assert sha((parent/'tensor-integer-session').read_bytes()) == PARENT_IMAGE
    author = (parent/'overlay-authoring.py').read_bytes()
    assert sha(author) == PARENT_AUTHOR
    e = Emitter(parent, a.repository.resolve(), a.upstream.resolve(), a.compactor.resolve())
    original = {n.name: ast.unparse(n) for n in ast.parse(author).body if isinstance(n, ast.FunctionDef)}
    own = {name: (HERE/name).read_bytes() for name in ('build.py', 'saved_inputs.py', 'gradient_detach_cleanup.py', 'view_row_release.py', 'authoring_support.py', 'PROTOCOL.md')}
    for name, raw in own.items():
        (out/name).write_bytes(raw)
    changes = {}
    changes['entry'] = once(original['entry'],
        '        if code == 106:',
        '        if code == 107:\n            next = binary_saved_command(s,o,w,32+i*64,at)\n        elif code == 106:')
    changes['integer_result_dtype'] = once(original['integer_result_dtype'],
        '    if getw(w, dptr(w, a) + 32) != 0 or getw(w, dptr(w, b) + 32) != 0:\n        return terror(w, 5)\n', '')
    binary = once(original['integer_binary'],
        '    dtype = integer_result_dtype(w, op, a, b)',
        '    requires = getw(w,dptr(w,a)+32) | getw(w,dptr(w,b)+32)\n    if inplace != 0 and requires != 0:\n        return terror(w,5)\n    dtype = integer_result_dtype(w, op, a, b)')
    binary = once(binary, 'target = typed_descriptor(w, rank, 128, 0, 1, dtype)',
                  'target = typed_descriptor(w, rank, 128, requires, 1, dtype)')
    binary = once(binary, '    return target',
        '    if inplace == 0 and requires != 0:\n        return record(w,target,choose(op == 7,4,op),a,b)\n    return target')
    changes['integer_binary'] = binary
    audit = original['dtype_audit']
    begin = audit.index('            if mask != 0:\n')
    end = audit.index('            elif getw(w, p + 592)', begin)
    previous = audit[begin:end]
    replacement = '            if mask > 65535:\n                status = binary_saved_audit(w,p)\n' + previous.replace('            if mask != 0:', '            elif mask != 0:', 1)
    changes['dtype_audit'] = once(audit,previous,replacement)
    changes['dtype_audit'] = once(changes['dtype_audit'],
        '                    if integer_view_root_valid(w, p, q) == 0:',
        '                    if getw(w,p+240) == 0 or integer_view_root_valid(w, p, q) == 0:')
    changes['dtype_audit'] = once(changes['dtype_audit'],
        '            elif getw(w, p + 592) != 0 or (code == 2 and getw(w, p + 408) != 0):',
        '            elif getw(w, p + 592) != 0 or (code == 2 and getw(w, p + 408) != 0 and getw(w,p+240) != 0):')
    cleanup = {n.name: ast.unparse(n) for n in ast.parse(own['gradient_detach_cleanup.py']).body
               if isinstance(n, ast.FunctionDef)}['gradient_detach_inplace']
    original['gradient_detach_inplace'] = once(cleanup,
        '        if dtype_mask_release(w, g) == 0:\n            return 0\n', '')
    changes['gradient_detach_inplace'] = cleanup
    for name in changes:
        assert functions(e.emit(original[name])) == {name:e.parent[name]}, ('baseline authoring differs',name)
    aliases = {}
    alias_inverse = {}
    for name,new_name in (('record','mixed_original_record'), ('dtype_mask_release','mixed_original_mask_release'), ('tensor_row_release','mixed_original_tensor_row_release')):
        body = e.parent[name]
        revised = once(body, ('fn '+name+'(').encode(), ('fn '+new_name+'(').encode())
        assert once(revised, ('fn '+new_name+'(').encode(), ('fn '+name+'(').encode()) == body
        aliases[new_name] = revised
        e.add_context(new_name,revised)
        alias_inverse[name] = {'original_sha256':sha(body),'alias_sha256':sha(revised),'inverse_exact':True}
    authored = '\n\n'.join(changes.values())+'\n\n'+own['saved_inputs.py'].decode()+'\n\n'+own['view_row_release.py'].decode()
    (out/'overlay-authoring.py').write_text(authored)
    try:
        fragment = e.emit(authored)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'authoring','error':repr(error),'native_attempted':False},indent=2)+'\n')
        raise
    (out/'overlay.hepta').write_bytes(fragment)
    combined = dict(e.parent)
    combined.update(aliases)
    combined.update(functions(fragment))
    old_read = b'{"function":"tget"}'
    new_read = b'{"function":"binary_backward_read"}'
    before = combined['back_binary']
    assert before.count(old_read) == 5
    combined['back_binary'] = before.replace(old_read,new_read)
    assert combined['back_binary'].replace(new_read,old_read) == before
    tag_edits = []
    for name in ('init_state','state_header','so_state_init','so_state_header'):
        outer = name in ('init_state','state_header')
        before_tag = b'HTINT002' if outer else b'HTOINT02'
        after_tag = b'HTMXAD02' if outer else b'HTOMXD02'
        old = ('{"value":'+str(int.from_bytes(before_tag,'little'))+'}').encode()
        new = ('{"value":'+str(int.from_bytes(after_tag,'little'))+'}').encode()
        raw = combined[name]
        combined[name] = once(raw,old,new)
        assert once(combined[name],new,old) == raw
        tag_edits.append({'function':name,'before':sha(raw),'after':sha(combined[name]),'inverse_exact':True})
    raw = b'hia-asm.v1 hepta.hia.program.v8 mixed_integer_autograd_runtime\n'+b''.join(combined.values())
    (out/'runtime.hepta').write_bytes(raw)
    (out/'authoring-custody.json').write_text(json.dumps({
        'parent_source_sha256':PARENT,'parent_authoring_sha256':PARENT_AUTHOR,
        'baseline_reemission_all_exact':True,'native_tags':tag_edits,'alias_inverse':alias_inverse,
        'back_binary_five_reads_inverse_exact':True,
        'changes':{n:{'before':original[n],'after':body} for n,body in changes.items()},
        'compactor_records':e.checks},indent=2)+'\n')
    (out/'source-map.json').write_text(json.dumps([
        {'name':n,'sha256':sha(body),'parent_sha256':sha(e.parent[n]) if n in e.parent else None,
         'unchanged_parent':e.parent.get(n)==body} for n,body in combined.items()],indent=2)+'\n')

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
        unit = f'mixed_integer_{i:02d}'
        source = f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,source))
    units.append(('owners',e.owner_raw))
    report = {'schema':'heptatorch.mixed-integer-autograd-native-build.v1', 'status':'BUILDING',
        'parent_source_sha256':PARENT, 'parent_native_sha256':PARENT_IMAGE,
        'source_sha256':sha(raw), 'descriptor_bytes':640, 'state_format':'HTMXAD02',
        'backing_format':'HTOMXD02', 'functions':len(combined),
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
    app=out/'tensor-mixed-autograd-session';app.write_bytes(result.stdout);app.chmod(0o500)
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
