"""Isolated checked-source authoring, followed by actual native admission/linking."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, re, struct, subprocess, sys

HERE = Path(__file__).resolve().parent
REPO = UPSTREAM = COMPILER = LINKER = BASELINE = None
CORE_SHA = '264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233'
V2_SHA = '91e5cb7d80299e735dcac2dc3520bafbf3e272465ccdced6aa2f732800127b07'
COMPILER_SHA = '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
LINKER_SHA = '55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
FUNCTION = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
functions = lambda raw: {name.decode(): body for body,name in FUNCTION.findall(raw)}


def py_functions(source):
    tree = ast.parse(source)
    return {n.name: n for n in tree.body if isinstance(n,ast.FunctionDef)}


def imported_source(path):
    tree = ast.parse(path.read_text())
    parts=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'SOURCE' for t in n.targets):
            parts.append(ast.literal_eval(n.value))
        if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='SOURCE':
            parts.append(ast.literal_eval(n.value))
    if parts: return '\n'.join(parts)
    return path.read_text()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--linker',type=Path,required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--stack-profile', action='store_true',
                        help='Use the separately pinned native ABI3 whole-program stack admission profile')
    args = parser.parse_args()
    global REPO,UPSTREAM,COMPILER,LINKER,BASELINE
    REPO=args.repository.resolve();UPSTREAM=args.source_root.resolve()
    COMPILER=args.compiler.resolve();LINKER=args.linker.resolve();BASELINE=args.baseline.resolve()
    profile_path=HERE/'profile.json'
    own_profile_raw=profile_path.read_bytes();own_profile=json.loads(own_profile_raw)
    dependency_snapshots={}
    for relative,expected_hash in own_profile['dependency_sources'].items():
        path=REPO/relative;data=path.read_bytes()
        assert sha(data)==expected_hash,relative
        dependency_snapshots[path]=data
    selected_profile=own_profile['stack_profile'] if args.stack_profile else own_profile
    expected_compiler=selected_profile['compiler_sha256']
    expected_linker=selected_profile['linker_sha256']
    if not args.stack_profile:
        assert expected_compiler==COMPILER_SHA and expected_linker==LINKER_SHA
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    core_bytes = (REPO/'src/heptatorch.hepta').read_bytes()
    v2_bytes = (BASELINE/'runtime.hepta').read_bytes()
    assert sha(core_bytes) == CORE_SHA and sha(v2_bytes) == V2_SHA
    original, current = functions(core_bytes), functions(v2_bytes)
    assert sha(COMPILER.read_bytes()) == expected_compiler and sha(LINKER.read_bytes()) == expected_linker
    profile = json.loads((REPO/'tools/owned_tensors/profile.json').read_text())
    custody = {}
    for rel,digest in profile['upstream_authoring_sources'].items():
        raw = (UPSTREAM/rel).read_bytes(); assert sha(raw) == digest, rel
        custody[str(UPSTREAM/rel)] = raw
    inputs = {p: p.read_bytes() for p in HERE.glob('*.py')}
    inputs.update(dependency_snapshots)
    inputs[profile_path]=own_profile_raw
    for path in (REPO/'docs/functions.json',REPO/'tools/owned_tensors/profile.json',BASELINE/'runtime.hepta'):
        inputs[path]=path.read_bytes()
    inputs[HERE/'owners.hepta'] = (HERE/'owners.hepta').read_bytes()
    sys.path.insert(0,str(UPSTREAM))
    emitter_path = REPO/'tools/owned_tensors/authoring.py'
    inputs[emitter_path] = emitter_path.read_bytes()
    spec = importlib.util.spec_from_file_location('object_emitter',emitter_path)
    emitter = importlib.util.module_from_spec(spec); spec.loader.exec_module(emitter); emitter.ROOT=REPO
    ParentBuilder = emitter.NativeBuilder
    class ObjectBuilder(ParentBuilder):
        def expr(self,node):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id == 'copy_range':
                dest,source,da,sa,size = [self.expr(arg) for arg in node.args]
                self.op('buffer.copy_range',[dest,source,self.cast(da,'usize'),self.cast(sa,'usize'),self.cast(size,'usize')])
                return self.c(0)
            return super().expr(node)
    emitter.NativeBuilder = ObjectBuilder
    signature = {}
    for name,body in current.items():
        line = body.split(b'\n',1)[0].decode()
        sig = re.search(r'^fn \w+\((.*?)\) -> \((.*?)\)',line)
        signature[name]=([p.split(':')[1] for p in sig[1].split(',') if p],sig[2])
    source_nodes = {}
    for filename in ('memory.py','tensors.py','graph.py','objects.py','commands.py','matmul_saved.py'):
        source_nodes.update(py_functions((HERE/filename).read_text()))

    # The object implementation is the existing admitted storage service with
    # a fixed enclosing-buffer base. No host storage operation is substituted.
    storage_sources = {}
    for module in ('bytes','records','audit'):
        path=REPO/'tools/storage_objects'/f'{module}.authoring'
        raw=path.read_bytes(); inputs[path]=raw
        assert sha(raw)==own_profile['storage_sources'][module+'.authoring']
        storage_sources.update(py_functions(raw.decode()))
    selected = {'rget','uget','uput','align16','data_base','state_size'} | set(py_functions((REPO/'tools/storage_objects/records.authoring').read_text())) | set(py_functions((REPO/'tools/storage_objects/audit.authoring').read_text()))
    selected -= {'plan_slot','plan_pool'}
    class Rebase(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            node.name='so_'+node.name
            return self.generic_visit(node)
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name):
                if node.func.id in selected: node.func.id='so_'+node.func.id
                elif node.func.id == 'fill_range': node.func.id='so_fill_range'
                elif node.func.id == 'copy_range': node.func.id='so_copy_range'
            return node
        def visit_Name(self,node):
            if node.id == 'STATE_MAGIC': return ast.Constant(int.from_bytes(b'HTOSST01','little'))
            return node
    rebased=[]
    for name,node in storage_sources.items():
        if name not in selected: continue
        node=Rebase().visit(node); ast.fix_missing_locations(node)
        text=ast.unparse(node)
        if name in ('rget','uget','uput'):
            text=text.replace('at + i','139264 + at + i')
        if name == 'state_header':
            text=text.replace('size != length(w)','size + 139264 != length(w)')
        if name == 'audit_records':
            text=text.replace('if so_uget(w, p + 56) != 0 or (kind == 1 and so_uget(w, p + 48) != 0):',
                'if kind == 1 and (so_uget(w,p+56) != 0 or so_uget(w,p+48) == 0 or so_uget(w,p+48) >= getw(w,8)):\n                        return 9\n                    if kind == 5 and so_uget(w,p+56) != 0 and private_hold(w,so_uget(w,p+56),128) == 0:')
            old='j = 24\n                    while j < 64:'
            assert text.count(old)==1
            text=text.replace(old,
                'j = 24\n                    if kind == 6:\n                        if so_uget(w,p+24) > 1:\n                            return 9\n                        j = 32\n                    while j < 64:')
        rebased.append(text)
    rebased.extend(['''def so_fill_range(w:mutbuf, at:u64, size:u64, value:u64) -> u64:
    fill_range(w,139264+at,size,value)
    return 0
''','''def so_copy_range(w:mutbuf, s:readbuf, at:u64, source:u64, size:u64) -> u64:
    copy_range(w,s,139264+at,139264+source,size)
    return 0
'''])
    storage_authoring='\n\n'.join(rebased)+'\n'
    (out/'storage-adapter.py').write_text(storage_authoring)
    source_nodes.update(py_functions(storage_authoring))

    old_python={}
    sources=[HERE/'kernel_baselines.py']
    for path in sources:
        inputs[path]=path.read_bytes(); old_python.update(py_functions(imported_source(path)))
    v2_authoring=BASELINE/'overlay-authoring.py'
    inputs[v2_authoring]=v2_authoring.read_bytes()
    assert sha(inputs[v2_authoring])==own_profile['baseline_authoring_sha256']
    v2_python=py_functions(v2_authoring.read_text())
    # Precise read/control changes; scalar binary32 primitives are untouched.
    back_unary=ast.unparse(old_python['back_unary'])
    assert back_unary.count('x = tget(w, a, i)') == 1
    back_unary=back_unary.replace('x = tget(w, a, i)','x = tget(w, t, i)',1)
    back_unary=back_unary.replace('        if op == 6:\n            g = negf(g)','        g = choose(op == 6,negf(g),g)')
    source_nodes.update(py_functions(back_unary))
    back_matmul=ast.unparse(old_python['back_matmul'])
    # Both independent gradient branches have to guard their data reads.
    (out/'back-matmul-baseline.py').write_text(back_matmul+'\n')
    baseline_checks={}
    for name in ('conv2d','pool2d','index_select','tensor_join','concat_extent','concat_shape','back_unary','back_matmul','indexed_add','index_select_command','gather_command','indexed_add_command'):
        node=old_python[name]
        emitted,_=emitter.emit(ast.unparse(node),signature)
        body=functions(emitted.encode())[name]
        baseline_checks[name]={'authoring_sha256':sha(ast.unparse(node).encode()),'emitted_sha256':sha(body),'canonical_sha256':sha(original[name]),'canonical_identical':body==original[name]}
        if name not in ('pool2d','back_unary'):
            assert body==original[name], ('authoring baseline differs from canonical',name)
    # These two historical authoring snippets predate exact canonical edits.
    # Verify those edits before applying the separate v3 ownership overlay.
    canonical_authoring = ast.unparse(old_python['back_unary'])
    canonical_authoring = canonical_authoring.replace('        if op == 6:\n            g = negf(g)',
        '        g = choose(op == 6,negf(g),g)')
    emitted,_ = emitter.emit(canonical_authoring,signature)
    assert functions(emitted.encode())['back_unary']==original['back_unary']
    baseline_checks['back_unary']['canonical_normalization']='Existing choose(op==6,negf(g),g) expression'
    baseline_checks['back_unary']['canonical_after_normalization']=True
    canonical_authoring = ast.unparse(old_python['pool2d'])
    needle='    n = getw(w, p + 8)\n'
    assert canonical_authoring.count(needle)==1
    canonical_authoring=canonical_authoring.replace(needle,needle+'    if n > idiv(16777216,getw(w,config+104)):\n        return terror(w,3)\n')
    emitted,_ = emitter.emit(canonical_authoring,signature)
    normalized=functions(emitted.encode())['pool2d']
    baseline_checks['pool2d']['canonical_normalization']='Existing 16777216-work budget guard'
    baseline_checks['pool2d']['normalized_sha256']=sha(normalized)
    def alpha_rename(body):
        names={};counts={b'v':0,b'b':0}
        def rename(match):
            token=match[0]
            if token not in names:
                prefix=token[:1];names[token]=prefix+str(counts[prefix]).encode();counts[prefix]+=1
            return names[token]
        return re.sub(rb'\b(?:v[0-9]+|b[0-9]+)\b',rename,body)
    baseline_checks['pool2d']['canonical_after_ssa_alpha_rename']=alpha_rename(normalized)==alpha_rename(original['pool2d'])
    assert alpha_rename(normalized)==alpha_rename(original['pool2d']), 'pool2d canonical guard differs beyond SSA names'
    (out/'baseline-checks.json').write_text(json.dumps(baseline_checks,indent=2)+'\n')
    # Auxiliary allocations acquire actual internal StrongHold records. Capture
    # the handle before tnew can allocate gradient/work storage of its own.
    attachments={'conv2d':[('config',264)],'pool2d':[('config',264),('indices',272)],
                 'index_select':[('indices',264)],'gather':[('indices',264)],
                 'tensor_join':[('saved',264)],'indexed_add':[('aux',264)]}
    for name,attaches in attachments.items():
        node=v2_python[name] if name=='gather' else old_python[name]
        source=ast.unparse(node)
        if name=='pool2d':
            needle='    n = getw(w, p + 8)\n'
            assert source.count(needle)==1
            source=source.replace(needle,needle+'    if n > idiv(16777216,getw(w,config+104)):\n        return terror(w,3)\n')
        if name=='tensor_join':
            source=source.replace('>= 256','>= 65536')
            begin=source.index('    cursor = getw(w, 16)\n')
            end=source.index('    saved = alloc(',begin)
            source=source[:begin]+source[end:]
        lines=source.splitlines(); result=[]
        for line in lines:
            result.append(line)
            if line.startswith('def '):
                result.extend('    '+var+'_hold = 0' for var,_ in attaches)
            for var,field in attaches:
                if re.match(r'\s*'+var+r' = alloc\(',line):
                    indent=line[:len(line)-len(line.lstrip())]
                    result.append(indent+var+'_hold = getw(w,80)')
        source='\n'.join(result)+'\n'
        for var,field in attaches:
            # All auxiliaries are attached after their allocation and before
            # returning success. A failed frame discards the whole trial owner.
            marker='    return t\n'
            if name=='conv2d': marker='    return binary(w, 1, t, bias_view)\n'
            # Attach immediately before the operation's record call, which is
            # after every auxiliary allocation and before later composition.
            record_lines=[line for line in source.splitlines() if line.startswith('    record(w, t,')]
            if record_lines:
                needle=record_lines[0]+'\n'
                attach=f'    putw(w,dptr(w,t)+{field},{var}_hold)\n'
                source=source.replace(needle,attach+needle,1)
            else:
                raise AssertionError((name,var,'record attachment point'))
        source_nodes.update(py_functions(source))
    # Config data carries external IDs only at these explicit list/command sites.
    for name in ('concat_extent','concat_shape'):
        source=ast.unparse(old_python[name])
        source=source.replace('getr(s, offset + j * 8)','wire_node(s,w,offset+j*8)')
        if name=='concat_shape':
            # The second occurrence is the aggregate numel bound, not perdim.
            source=source.replace('if n > 262144:', 'if n > 16777216:')
        source_nodes.update(py_functions(source))
    join=ast.unparse(source_nodes['tensor_join'])
    join=join.replace('getr(s, config + 8 + j * 8)','wire_node(s,w,config+8+j*8)')
    join=join.replace('getr(s, config + 8)', 'wire_node(s,w,config+8)')
    source_nodes.update(py_functions(join))
    for name,fields in (('index_select_command',(8,)),('gather_command',(8,)),('indexed_add_command',(8,16))):
        source=ast.unparse(old_python[name])
        for field in fields: source=source.replace(f'getr(s, cp + {field})',f'wire_node(s,w,cp+{field})')
        source_nodes.update(py_functions(source))
    # ReLU reads the preserved result; mm/bmm gradient reads are conditional.
    bm=ast.unparse(old_python['back_matmul_point'])
    bm=bm.replace('        gadd(w, a, ai, mulf(grad, tget(w, b, bi)))',
        '        if getw(w,dptr(w,a)+32) != 0:\n            gadd(w,a,ai,mulf(grad,matmul_saved_get(w,t,1,bi)))')
    bm=bm.replace('        gadd(w, b, bi, mulf(grad, tget(w, a, ai)))',
        '        if getw(w,dptr(w,b)+32) != 0:\n            gadd(w,b,bi,mulf(grad,matmul_saved_get(w,t,0,ai)))')
    (out/'back-matmul-current.py').write_text(bm+'\n')
    source_nodes.update(py_functions(bm))
    backward=(HERE/'backward_base.py').read_text()
    inputs[HERE/'backward_base.py']=backward.encode()
    backward=backward.replace('t & 255','t').replace('fill_range(w,','memory_fill(w,')
    source_nodes.update(py_functions(backward))
    for name in ('frame_valid','reply_error','tensor_initialize','full_tensor'):
        source=ast.unparse(v2_python[name])
        source=source.replace(str(int.from_bytes(b'HTORCH02','little')),str(int.from_bytes(b'HTORCH03','little')))
        source=source.replace(str(int.from_bytes(b'HTORSP02','little')),str(int.from_bytes(b'HTORSP03','little')))
        if name=='reply_error': source=source.replace('putw(o, 8, 2)','putw(o, 8, 3)')
        source_nodes.update(py_functions(source))
    command_node=v2_python['command']
    removed={'code == 10 or code == 11','code == 13','code == 39 or code == 40'}
    before=len(command_node.body)
    command_node.body=[node for node in command_node.body if not (isinstance(node,ast.If) and ast.unparse(node.test) in removed)]
    assert before-len(command_node.body)==3
    command=ast.unparse(command_node)
    command=command.replace('    a = getr(s, cp + 8)', '    if code >= 41:\n        return object_command(s,o,w,cp,at)\n    putw(w,88,getw(w,8))\n    a = getr(s, cp + 8)',1)
    command=command.replace('    putw(w,88,getw(w,8))', '    if code == 10 or code == 11 or code == 13 or code == 39 or code == 40:\n        return inspect_command(s,o,w,cp,at)\n    putw(w,88,getw(w,8))',1)
    needle='    result = 0\n'
    command=command.replace(needle,'    a = command_argument(w,code,1,a)\n    b = command_argument(w,code,2,b)\n    c = command_argument(w,code,3,c)\n'+needle,1)
    command=command.replace('batchnorm_eval(w, a, b, c, getr(s, cp + 32), getr(s, cp + 40), getr(s, cp + 48))',
        'batchnorm_eval(w,a,b,c,public_optional(w,getr(s,cp+32)),public_optional(w,getr(s,cp+40)),getr(s,cp+48))')
    command=command.replace('length(o) - at - 32 < 104','length(o) - at - 32 < 128')
    command=command.replace('putw(o, at + 120, getw(w, dp + 64))\n                putw(o, at + 128, tversion(w, a))',
        'putw(o,at+120,getw(w,dp+216))\n                putw(o,at+128,getw(w,dp+224))\n                putw(o,at+136,tversion(w,a))\n                putw(o,at+144,getw(w,dp+232))\n                sp = so_storage_live(w,getw(w,dp+216))\n                putw(o,at+152,so_uget(w,sp+24))')
    command=command.replace('payload = 104','payload = 128')
    tensor_codes={1,2,3,4,5,6,13,15,16,17,18,19,20,21,22,23,26,27,28,30,31,32,34,35,36,37,38}
    mask=sum(1<<code for code in tensor_codes)
    needle='    status = getw(w, 0)\n'
    command=command.replace(needle,f'    if getw(w,0) == 0:\n        if ({mask} & (1 << code)) != 0:\n            result = publish_tensor(w,result)\n        if getw(w,8) != getw(w,88):\n            finish_command(w,getw(w,88))\n        putw(w,88,0)\n'+needle,1)
    source_nodes.update(py_functions(command))

    authoring='\n\n'.join(ast.unparse(n) for n in source_nodes.values())+'\n'
    (out/'overlay-authoring.py').write_text(authoring)
    fragment,mapping=emitter.emit(authoring,signature)
    replacement=functions(fragment.encode())
    combined=dict(current)
    combined.update(replacement)
    # Import the separately qualified absolute-Adam-counter correction without
    # maintaining another numerical implementation or reinterpreting its tests.
    version_root=REPO/'tools/owned_tensor_versions'
    version_profile=json.loads((version_root/'profile.json').read_text())
    producer=(version_root/'build.py').read_bytes()
    helper=(version_root/'adam_versions.hepta').read_bytes()
    assert sha(producer)==version_profile['builder_sha256']
    assert sha(helper)==version_profile['helper_sha256']=='ea5e5a4c36139303f2379113adc2df878b02345a862b7ecc7a6246194090c4f0'
    inputs[version_root/'build.py']=producer
    inputs[version_root/'adam_versions.hepta']=helper
    inputs[version_root/'profile.json']=(version_root/'profile.json').read_bytes()
    changes=next(ast.literal_eval(node.value) for node in ast.walk(ast.parse(producer))
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='changes' for t in node.targets))
    adam=combined['adam']
    for old,new in changes:
        assert adam.count(old)==1
        adam=adam.replace(old,new,1)
    combined['adam']=adam
    combined.update(functions(helper))
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_object_runtime\n'+b''.join(combined.values())
    assert sha(raw)==own_profile['candidate_source_sha256']
    (out/'runtime.hepta').write_bytes(raw)
    (out/'overlay.hepta').write_text(fragment)
    (out/'functions.json').write_text(json.dumps(mapping,indent=2)+'\n')
    # Small native units remove source-profile aggregate limits. Every import
    # is emitted from the exact defining signature and checked by native tools.
    # The frozen native linker admits acyclic module dependencies as well as
    # the compiler's acyclic function graph. Keep every callee in the same or
    # an earlier unit; arbitrary slices of canonical source order form cycles.
    ordered=[]; active=set(); done=set()
    def visit(name):
        assert name not in active, ('recursive source',name)
        if name in done: return
        active.add(name)
        calls=sorted({n.decode() for n in re.findall(rb'\{"function":"(\w+)"\}',combined[name])})
        for callee in calls:
            assert callee in combined, (name,callee)
            visit(callee)
        active.remove(name); done.add(name); ordered.append(name)
    for name in combined: visit(name)
    groups=[]; group=[]; ins=0; blocks=0
    for name in ordered:
        body=combined[name]
        count=body.count(b' = '); nblocks=len(re.findall(rb'^block ',body,re.M))
        if group and (ins+count>3600 or blocks+nblocks>1000 or len(group)>=65):
            groups.append(group); group=[]; ins=0; blocks=0
        group.append(name); ins+=count; blocks+=nblocks
    if group: groups.append(group)
    units=[]
    for index,names in enumerate(groups):
        local=set(names)
        calls={c.decode() for name in names for c in re.findall(rb'\{"function":"(\w+)"\}',combined[name])}
        imports=sorted(calls-local)
        assert all(name in combined for name in imports), set(imports)-set(combined)
        declarations=[]
        for name in imports:
            line=combined[name].split(b'\n',1)[0]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',line)+b'\n')
        name=f'objects_{index:02d}'
        source=f'hia-asm.v1 hepta.hia.program.v8 {name}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        assert len(names)+len(imports)<=256
        units.append((name,source))
    units.append(('objects_owners',(HERE/'owners.hepta').read_bytes()))
    rows=[]; objects=[]
    report={'schema':'heptatorch.tensor-object-native-build.v1','status':'SOURCE_ONLY',
        'native_profile':'owned-stack-abi3' if args.stack_profile else 'owned-session-abi2',
        'compiler_sha256':expected_compiler,'linker_sha256':expected_linker,
        'canonical_source_sha256':CORE_SHA,'v2_source_sha256':V2_SHA,'candidate_source_sha256':sha(raw),
        'function_count':len(combined),'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = '),
        'canonical_functions_byte_identical':sum(combined[n]==body for n,body in original.items()),
        'function_map':[{'name':n,'canonical_sha256':sha(body),'candidate_sha256':sha(combined[n]),'byte_identical':body==combined[n]} for n,body in original.items()],
        'input_sources':{str(p):sha(b) for p,b in inputs.items()},'upstream_sources':{p:sha(b) for p,b in custody.items()},'units':rows}
    def save(): (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    expected_units={row['name']:row for row in selected_profile['units']}
    assert [name for name,_ in units]==list(expected_units)
    for name,source in units:
        assert sha(source)==expected_units[name]['source_sha256'],name
        (out/(name+'.hepta')).write_bytes(source)
        row={'name':name,'source_sha256':sha(source),'source_bytes':len(source),'functions':len(functions(source)),
             'blocks':len(re.findall(rb'^block ',source,re.M)),'instructions':source.count(b' = ')}
        rows.append(row); save()
        if args.source_only: continue
        result=subprocess.run([str(COMPILER)],input=source,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.compile.stdout')).write_bytes(result.stdout);(out/(name+'.compile.stderr')).write_bytes(result.stderr)
        row['compiler_exit']=result.returncode; save()
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),(name,result.returncode,result.stderr)
        assert sha(result.stdout)==expected_units[name]['object_sha256'],name
        (out/(name+'.o')).write_bytes(result.stdout);objects.append(result.stdout)
        row['object_sha256']=sha(result.stdout);save()
    if not args.source_only:
        magic=b'HIASTKS1' if args.stack_profile else b'HIAOWNS1'
        request=magic+struct.pack('<QQQ',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
        (out/'link.request.bin').write_bytes(request)
        result=subprocess.run([str(LINKER)],input=request,capture_output=True,env={'PATH':''},timeout=120)
        (out/'link.stdout').write_bytes(result.stdout);(out/'link.stderr').write_bytes(result.stderr)
        report['link_exit']=result.returncode;save()
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),(result.returncode,result.stderr)
        assert sha(result.stdout)==selected_profile['executable_sha256']
        app=out/'tensor-object-session';app.write_bytes(result.stdout);app.chmod(0o500)
        report.update(status='PASS',executable_sha256=sha(result.stdout),executable_bytes=len(result.stdout))
    assert (out/'runtime.hepta').read_bytes()==raw
    for name,source in units:
        assert (out/(name+'.hepta')).read_bytes()==source,name
    if not args.source_only:
        for (name,source),object_bytes in zip(units,objects):
            assert (out/(name+'.o')).read_bytes()==object_bytes,name
        assert (out/'link.request.bin').read_bytes()==request
        assert (out/'tensor-object-session').read_bytes()==result.stdout
    for p,raw_input in inputs.items(): assert p.read_bytes()==raw_input,p
    for p,raw_input in custody.items(): assert Path(p).read_bytes()==raw_input,p
    assert (REPO/'src/heptatorch.hepta').read_bytes()==core_bytes
    assert sha(COMPILER.read_bytes())==expected_compiler and sha(LINKER.read_bytes())==expected_linker
    save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('function_map','input_sources','upstream_sources')},indent=2))


if __name__=='__main__': main()
