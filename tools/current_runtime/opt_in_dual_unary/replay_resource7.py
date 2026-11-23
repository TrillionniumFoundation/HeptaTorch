#!/usr/bin/env python3
"""Retain all seven resource cases on the exact joint image and read original physical evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import types

BASE_SOURCE='aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
BASE_ELF='083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
SOURCE='f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a'
ELF='5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
PROFILE='tools/current_runtime/qualification.py'
ADMIT='tools/current_runtime/opt_in_admission/admit.py'
HELPER='tools/current_runtime/fusion_resource_transition.py'
LEGACY='tools/current_runtime/source/runtime.hepta'
DIR='tests/current_runtime_resources/'
READER='tests/qualify_current_runtime.py'
BINDER='tests/current_runtime_original35/binding.py'
PIN_SHA='04a72be721ef3a8179bca741f5d465d6f3cfa945b60bc36c7e29a87884186ee0'
IDENTITIES=((BASE_SOURCE,SOURCE),(BASE_ELF,ELF))

def need(ok,message):
    if not ok:
        raise ValueError(message)

def sha(b):return hashlib.sha256(b).hexdigest()

def exact(raw, old, new):
    need(raw.count(old)==1 and new not in raw,'fixed reversible source anchor changed')
    out=raw.replace(old,new,1)
    need(out.replace(new,old,1)==raw,'source edit did not invert')
    return out

def load(path, raw, name, **values):
    m=types.ModuleType(name);m.__file__=str(path);m.__dict__.update(values)
    exec(compile(raw,str(path),'exec',dont_inherit=True,optimize=0), m.__dict__)
    return m

def bodies(raw):
    rows=list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z_0-9_]*)\(.*?^end\n',raw,re.M|re.S))
    funcs={m[1].decode():m[0] for m in rows}
    head=raw.split(b'\n',1)[0]+b'\n'
    need(len(rows)==len(funcs)==800 and head+b''.join(funcs.values())==raw,'entire 800-function source envelope')
    return head,funcs

def identity(raw):
    for old,new in IDENTITIES:
        raw=exact(raw,old.encode(),new.encode())
    return raw

class Joint:
    def __init__(self, repo, build, source, receipt):
        self.repo,self.build,self.source,self.receipt=repo,build,source,receipt
        catalog=Path(__file__).with_name('resource7-pins.json').read_bytes()
        need(sha(catalog)==PIN_SHA,'wrong fixed resource source pin catalog')
        self.pins=json.loads(catalog)
        self.check_pins()
        self.offline=load(repo/ADMIT,self.raw(ADMIT),'resource_fixed_offline_admission')
        self.id=self.admission()
        need(self.id['native_image_sha256']==ELF and self.id['source_sha256']==SOURCE and
             sha((source/'manifest.json').read_bytes())==self.id['manifest_sha256'],
             'wrong exact candidate build or source profile')
        self.legacy=load(repo/PROFILE,self.raw(PROFILE),'resource_fixed_legacy_admission')
        rebound=exact(self.raw(PROFILE),
          b'PROFILE_PATH = ROOT / "tools/current_runtime/source/manifest.json"',b'PROFILE_PATH = resource_manifest')
        self.profile=load(repo/PROFILE,rebound,'resource_exact_full_admission',resource_manifest=source/'manifest.json')
        original=self.profile.admit_build
        def admit_selected(path,bind,expected):
            need(path==build and expected==self.id['build_report_sha256'],'physical build identity differs')
            try:
                self.legacy.admit_build(path,bind,expected)
            except ValueError as e:
                need(str(e)=='build used another source manifest','unexpected canonical refusal: '+str(e))
            else:
                raise ValueError('canonical gate must still refuse joint image')
            out=original(path,bind,expected)
            need((out['source_sha256'],out['native_sha256'])==(SOURCE,ELF),'full candidate admission differs')
            return out
        self.profile.admit_build=admit_selected

    def check_pins(self):
        for n,h in self.pins.items():
            p=self.repo/n
            need(p==p.resolve() and sha(p.read_bytes())==h,'fixed original source differs: '+n)
            need(not (p.parent/'__pycache__').exists(),'need fresh checkout without bytecode caches')

    def raw(self,p):return (self.repo/p).read_bytes()

    def admission(self):
        return self.offline.admit('dual', self.repo/'tools/current_runtime/source/manifest.json',
                                  self.build, {'dual_scoped':self.receipt})

    def prove(self, actual, bind):
        canonical=bind(self.repo/LEGACY,BASE_SOURCE)
        need(actual==bind(self.build/'runtime.hepta',SOURCE),'physical candidate source not bound')
        h,base=bodies(canonical);hh,new=bodies(actual)
        keys={name for name in base if base[name]!=new[name]}
        need(h==hh and tuple(base)==tuple(new) and keys=={'entry','predicate_command','predicate_bits','wide_neg_tensor'}
             and h+b''.join(base[k] if k in keys else new[k] for k in new)==canonical
             and h+b''.join(new[k] if k in keys else base[k] for k in base)==actual,
             'whole source candidate inverse differs')
        # Prove that both changed dispatchers are the original body with a
        # single added equality guard and the exact same successor operands.
        for name, oldvar, stem, rid in (
            ('entry','predicate_entry_admitted','typed_logical_not_entry','v139'),
            ('predicate_command','predicate_admitted','typed_logical_not_predicate','v6')):
            old_line,=[line for line in base[name].splitlines(keepends=True)
                       if line.startswith(f'  branch {oldvar} '.encode())]
            new_line=(f'  {stem}_opcode:u64 = const() {{"value":127}}\n'
                      f'  {stem}_selected:bool = compare.eq({rid},{stem}_opcode)\n'
                      f'  {stem}_admitted:bool = bool.or({oldvar},{stem}_selected)\n').encode()
            new_line+=old_line.replace(oldvar.encode(),(stem+'_admitted').encode(),1)
            need(exact(new[name],new_line,old_line)==base[name],
                 'actual added opcode-127 guard or its unchanged argument routes differ')
        module=load(self.repo/HELPER,bind(self.repo/HELPER,self.pins[HELPER]),'resource_fixed_historical_proof')
        proof=module.prove(canonical,bind)
        need(proof['source_sha256']==BASE_SOURCE and proof['native_sha256']==BASE_ELF
             and proof['historical_bodies_substituted_for_actual_bodies'] is False
             and proof['unary_original_domain_route']['original_resource_requests_must_exclude_unary_opcode']==3,
             'original handwritten allocation/source proof changed')
        proof['legacy_source_sha256'],proof['legacy_native_sha256']=proof['source_sha256'],proof['native_sha256']
        proof['source_sha256'],proof['native_sha256']=SOURCE,ELF
        route=proof['obligations']['to_command']['predicate_entry_route']
        route['canonical_entry_sha256']=route['actual_entry_sha256']
        route['canonical_predicate_command_sha256']=route['actual_predicate_command_sha256']
        route['actual_entry_sha256']=sha(new['entry'])
        route['actual_predicate_command_sha256']=sha(new['predicate_command'])
        proof['joint_original_resource_route']=dict(source_inverse_exact=True,changed_bodies=sorted(keys),
             original_argument_forwarding_unchanged=True,excluded_dispatch_opcode=127,excluded_unary_opcode=3,
             canonical_ancestor_is_not_actual_executable=True,
             wide_abs_body_reachable_with_opcode3_only=True)
        return proof

def worker(repo,build,source,receipt):
    need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize,'worker isolation')
    ctx=Joint(*(Path(x) for x in (repo,build,source,receipt)))
    p=DIR+'adapter.py'
    raw=identity(ctx.raw(p))
    raw=exact(raw,b'self.pins[\'target\'] == dict(source_sha256=SOURCE, native_sha256=IMAGE,',
                  b'self.pins[\'target\'] == dict(source_sha256=canonical_source, native_sha256=canonical_image,')
    raw=exact(raw,b'source_helper.prove(',b'joint_runtime_prove(')
    raw=exact(raw,b"view_route['source_sha256'] == SOURCE",b"view_route['source_sha256'] == canonical_source")
    m=load(ctx.repo/p,raw,'scope_resource_worker',canonical_source=BASE_SOURCE,canonical_image=BASE_ELF,
           joint_runtime_prove=ctx.prove)
    plain=m.Adapter.plain
    def wrap_plain(self,path,name):
        orig=plain(self,path,name)
        return ctx.profile if path==ctx.repo/PROFILE else orig
    m.Adapter.plain=wrap_plain
    code=m.main()
    if code==0:
        ctx.check_pins()
        need(ctx.admission()==ctx.id,'candidate worker inputs changed')
    return code

def run(a):
    ctx=Joint(a.repository,a.dual_build,a.dual_source,a.dual_scoped)
    repo,build,out=ctx.repo,ctx.build,a.output
    build_sha=ctx.id['build_report_sha256']
    h=Path(__file__).resolve()
    cfg=tuple(map(str,(repo,build,a.dual_source,a.dual_scoped)))
    launch=out/'scoped-adapter.py'
    launch.write_text('import hashlib,runpy\nfrom pathlib import Path\n'
          + f'p=Path({str(h)!r})\nif hashlib.sha256(p.read_bytes()).hexdigest()!={sha(h.read_bytes())!r}: raise ValueError("wrapper changed")\n'
          + f'raise SystemExit(runpy.run_path(str(p))["worker"](*{cfg!r}))\n')
    launch_sha=sha(launch.read_bytes())
    p=DIR+'qualify.py'
    raw=identity(ctx.raw(p))
    raw=exact(raw,b'source_helper.prove(',b'joint_runtime_prove(')
    raw=exact(raw,b"str(HERE/'adapter.py')",b"str(scoped_adapter)")
    raw=exact(raw,b"require(3 not in opcodes,",b"require(3 not in opcodes and 127 not in opcodes,")
    m=load(repo/p,raw,'resource_scoped_outer',joint_runtime_prove=ctx.prove,scoped_adapter=launch)
    orig_module=m.module
    def wrap_module(path,name):
        orig=orig_module(path,name)
        return ctx.profile if path==repo/PROFILE else orig
    m.module=wrap_module
    sys.argv=[str(repo/p),'--repository',str(repo),'--build-directory',str(build),
              '--expected-build-sha256',build_sha,'--output',str(out/'observer')]
    code=m.main()
    native=(out/'observer/report.json').read_bytes()
    report=json.loads(native)
    result=dict(status='FAIL_RESOURCE7',observer_exit=code,observer_status=report['status'],
         original_report_sha256=sha(native),native_image_sha256=ELF,source_sha256=SOURCE,
         actual_build_sha256=build_sha,full_candidate_suite_qualification=False,production_activated=False)
    if code:
        result['error']=report.get('error') or report['failures']
        return result
    need(sha(launch.read_bytes())==launch_sha,'actual launched worker source changed')
    for s in report['suites']:
        argv=s['process']['argv']
        need(argv[1:5]==['-I','-S','-B',str(launch)] and
             argv[0]==sys.executable,'actual isolated opt-in worker argv changed')
    # Execute only the original independent resource readback function,
    # avoiding unrelated top-level suites. Its historical proof call is
    # explicitly redirected to the scoped proof on the actual candidate.
    compact=ast.parse(ctx.raw(READER))
    fn,=[node for node in compact.body if isinstance(node,ast.FunctionDef) and node.name=='qualify_resource7']
    only=ast.get_source_segment(ctx.raw(READER).decode(),fn).encode()+b'\n'
    only=exact(only,b"helper.prove(inputs.bind(ROOT / 'tools/current_runtime/source/runtime.hepta', SOURCE), inputs.bind)",
                    b"joint_prove(inputs.bind(joint_build / 'runtime.hepta', SOURCE), inputs.bind)")
    readback=load(repo/READER,only,'resource_exact_retained_readback',ROOT=repo,Path=Path,
         SOURCE=SOURCE,IMAGE=ELF,FUNCTIONS=800,UNITS=15,FUSION_RESOURCE_SHA=ctx.pins[HELPER],
         require=need,json=json,importlib=__import__('importlib.util'),joint_prove=ctx.prove,joint_build=build)
    inputs=load(repo/BINDER,ctx.raw(BINDER),'resource_physical_reader_inputs').Inputs()
    physical=readback.qualify_resource7(report,build_sha,out/'observer',inputs)
    need(not inputs.recheck() and ctx.admission()==ctx.id and physical==dict(
         cases=7, frames=378, assertions=1060, native_sessions=7, worker_sessions=3, allocation_controls=3),
         'complete unchanged physical readback denominator required')
    ctx.check_pins()
    result.update(status='PASS_RETAINED_RESOURCE7',readback=physical,physical_worker_sha256=launch_sha,
                  physical_reader_in_memory_sha256=sha(only), source_catalog_sha256=PIN_SHA)
    return result

def main():
    p=argparse.ArgumentParser()
    for key in ('repository','dual-build','dual-source','dual-scoped','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'use python3 -I -B')
    for k,v in vars(a).items():
        need(not v.is_symlink(),'unexpected symlink')
        setattr(a,k,v.resolve())
    a.output.mkdir()
    try:
        r=run(a)
    except Exception as e:
        r=dict(status='BLOCKED_OR_FAILED',error=f'{type(e).__name__}: {e}')
    (a.output/'receipt.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    print(r['status'])
    return 0 if r['status']=='PASS_RETAINED_RESOURCE7' else 1

if __name__=='__main__':
    raise SystemExit(main())
