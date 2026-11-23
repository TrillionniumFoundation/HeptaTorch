#!/usr/bin/env python3
"""Scope the unchanged PublicLogical192 readers to one exact candidate, with frozen original custody."""
import argparse, contextlib, hashlib, importlib, json, runpy, struct, sys
from pathlib import Path
RESOURCE_SHA='831f5f5ddfc78a87cfbd981d31273c25037e63a914ca5f907cfdfab32eaccf13'

def sha(b):return hashlib.sha256(b).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def setup(paths):
    repo,viewrepo,build,source,receipt,resource,_=map(Path,paths)
    need(sha(resource.read_bytes())==RESOURCE_SHA,'resource exact shared proof changed')
    R=runpy.run_path(str(resource))
    ctx=R['Joint'](repo,build,source,receipt)
    pin=R['PROFILE']; profile=repo/pin
    raw=R['exact'](profile.read_bytes(),b'PROFILE_PATH = ROOT / "tools/current_runtime/source/manifest.json"',b'PROFILE_PATH = scoped_manifest')
    real=R['load'](profile,raw,'exact_public192_current_build',scoped_manifest=source/'manifest.json')
    op=repo/'tests/current_runtime_public_f64_logical192.py'
    gates=R['load'](op,op.read_bytes(),'original_public192_coordinator')
    bp=repo/'tests/current_runtime_original35/binding.py'
    binder=R['load'](bp,bp.read_bytes(),'public192_isolated_exact_ancestor_binder').Inputs()
    ctx.prove(binder.bind(build/'runtime.hepta',R['SOURCE']),binder.bind)
    ctx.profile.admit_build(build,binder.bind,ctx.id['build_report_sha256'])
    need(not binder.recheck(),'complete exact scoped pair inputs changed')
    # The shared view reference is bound to its actual producing checkout, not
    # claimed as a new native execution. Same strict reference and predicates.
    sys.path.insert(0,str(viewrepo/'tests'))
    for name in ('f64_view.evidence','f64_view.legacy_successors','f64_view.legacy_binding'):
        importlib.import_module(name)
    binding=sys.modules['f64_view.legacy_binding']
    f=binding.__file__; b=Path(f).read_bytes()
    need(sha(b)=='6297b1ce89cad1ac37601cd2c15a3c12b0251b5b1e1e5ed946711a2b898c64db',
         'unchanged original view binding differs')
    before=(b"    spec = importlib.util.spec_from_file_location('legacy_view_complete_current_build', profile_path)\n"
            b"    profile = importlib.util.module_from_spec(spec)\n"
            b"    exec(compile(profile_raw, str(profile_path), 'exec', dont_inherit=True, optimize=0), profile.__dict__)")
    b=R['exact'](b,before,b'    profile = exact_scoped_candidate_profile')
    scope=dict(binding.__dict__,exact_scoped_candidate_profile=real)
    exec(compile(b,f,'exec',dont_inherit=True,optimize=0),scope)
    def from_arguments(args,inputs):
        need(args.repository==repo and args.build_directory==build,'actual candidate and original reader path changed')
        return scope['strict_successor'](viewrepo,build,args.expected_build_sha256,
              args.f64_view_reference,args.f64_view_reference_execution,inputs)
    binding.from_arguments=from_arguments
    return R,ctx,real,gates

@contextlib.contextmanager
def scoped_readers(paths,outer=None,inner=None):
    repo,*_=map(Path,paths)
    R,ctx,real,gates=setup(paths)
    with gates.frozen_reader(repo,R['load'](repo/'tests/current_runtime_original35/binding.py',
          (repo/'tests/current_runtime_original35/binding.py').read_bytes(),'frozen192_loader_binder').Inputs()) as q:
        m=q.run
        # Keep the historical wrapper's pinned support check by proving the
        # exact physical loader-only inverse used in the earlier reviewed lane.
        p=Path(m.__file__)
        raw=R['exact'](p.read_bytes(),
              b"    inputs.bind(ROOT / 'tools/current_runtime/qualification.py', ADMISSION_SHA)",
              b"    exact_loader_support_inverse(inputs)")
        def exact_loader_support_inverse(inputs):
            current=inputs.bind(repo/R['PROFILE'],ctx.pins[R['PROFILE']])
            original=R['exact'](current,
                b"    exec(compile(builder_raw, str(builder_path), 'exec', dont_inherit=True, optimize=0), builder.__dict__)\n",
                b"    spec.loader.exec_module(builder)\n")
            need(sha(original)==m.ADMISSION_SHA,'historical loader-only complete inverse differs')
        scope=dict(m.__dict__,exact_loader_support_inverse=exact_loader_support_inverse)
        exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),scope)
        # Restrict the only rewritten function to original immutable admission.
        scope['SOURCE_SHA'],scope['IMAGE_SHA']=R['SOURCE'],R['ELF']
        m.admit=scope['admit']; m.SOURCE_SHA,m.IMAGE_SHA=R['SOURCE'],R['ELF']
        def cold(dir,expected,inputs):
            need(dir==ctx.build and expected==ctx.id['build_report_sha256'],'wrong actual cold profile')
            inputs.bind(repo/R['PROFILE'],ctx.pins[R['PROFILE']])
            built=real.admit_build(dir,inputs.bind,expected)
            return built,R['bodies'](inputs.bind(dir/'runtime.hepta',R['SOURCE']))[1]
        m.common.bind_build=cold
        if inner is not None:
            run_native=q.run_native
            def exact_actual_launcher(argv,out):
                need(argv[4]==str(repo/'tests/public_f64_logical/run736.py'),'original inner launcher differs')
                return run_native(argv[:4]+[str(inner)]+argv[5:],out)
            q.run_native=exact_actual_launcher
        yield R,ctx,gates,q

def launch(paths,which,inner):
    with scoped_readers(paths,inner=Path(inner)) as (_,_,_,q):
        return q.main() if which=='outer' else q.run.main()

def census(out):
    seen=set();frames=0
    for path in (out/'native').glob('*/raw/requests.frames.bin'):
        raw=path.read_bytes();i=0
        while i<len(raw):
            m,n=struct.unpack_from('<8sQ',raw,i);i+=16
            need(m==b'HTOWNR01' and 32<=n<=len(raw)-i and raw[i:i+8]==b'HTORCH03','original complete raw request')
            c,size,r=struct.unpack_from('<3Q',raw,i+8)
            need(32+64*c<=size==n and r==0,'complete original packet census')
            seen.update(struct.unpack_from('<Q',raw,i+32+64*j)[0] for j in range(c));i+=n;frames+=1
    need(not seen&{3,127},'historical request leaves proven unchanged opcode domain')
    return frames,sorted(seen)

def main():
    p=argparse.ArgumentParser()
    items=('repository','view-repository','dual-build','dual-source','dual-scoped','resource-proof','output')
    for n in items:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    paths=tuple(str(getattr(a,k.replace('-','_')).resolve()) for k in items)
    repo,viewrepo,build,_,_,_,out=map(Path,paths)
    out.mkdir()
    controller=Path(__file__).resolve()
    outer,inner=out/'scope-qualifier.py',out/'scope-worker.py'
    for path,which in ((outer,'outer'),(inner,'inner')):
        path.write_text('import hashlib,runpy\nfrom pathlib import Path\n'
            f'p=Path({str(controller)!r})\nif hashlib.sha256(p.read_bytes()).hexdigest()!={sha(controller.read_bytes())!r}: raise ValueError("scope source changed")\n'
            f'raise SystemExit(runpy.run_path(str(p))["launch"]({paths!r},{which!r},{str(inner)!r}))\n')
    with scoped_readers(paths) as (R,ctx,gates,q):
        source=gates.restore_reference(viewrepo,out/'reference',q.common.Inputs())
        view=dict(root=viewrepo/'strict-view-full/reference',execution=viewrepo/'strict-view-full/reference-execution')
        cli=gates.reader_cli(repo,build,ctx.id['build_report_sha256'],source,view)
        actual=out/'actual'; execution=out/'execution';execution.mkdir()
        p=repo/'tests/current_runtime_process.py'
        process=R['load'](p,p.read_bytes(),'physical_public192_supervisor')
        command=[sys.executable,'-I','-S','-B',str(outer),*cli,'--output',str(actual)]
        result=process.execute(command+(['--prepare-only'] if a.prepare_only else []),execution,1200,cwd=repo,mirror_stderr=True)
        if result['exit_code']!=0 or a.prepare_only:
            print((actual/'report.json').read_text()[-3000:] if (actual/'report.json').exists() else result)
            return result['exit_code']
        report=json.loads((actual/'report.json').read_bytes())
        @contextlib.contextmanager
        def reuse_original(*args):
            yield q
        gates.frozen_reader=reuse_original
        orig_proc=gates.read_process
        def bind_actual_launcher(directory,inputs,tail,timeout):
            mapping={str(repo/'tests/public_f64_logical/qualify736.py'):outer,
                     str(repo/'tests/public_f64_logical/run736.py'):inner}
            need(tail[3] in mapping,'exact original runner coordinate differs')
            launcher=mapping[tail[3]]
            inputs.bind(launcher)
            return orig_proc(directory,inputs,tail[:3]+[str(launcher)]+tail[4:],timeout)
        # Package checks use the actual historical package repository, while
        # every original script and process predicate still binds the native reader checkout.
        orig_ref=gates.bind_reference
        gates.bind_reference=lambda repo_ignored,folder,inputs:orig_ref(viewrepo,folder,inputs)
        gates.read_process=bind_actual_launcher
        counts=gates.qualify_public192(report,repo,build,ctx.id['build_report_sha256'],actual,
             source['root'],q.common.Inputs(),execution)
        n,ops=census(actual)
        need(n==counts['frames'] and ctx.admission()==ctx.id,'candidate raw command denominator differs')
        summary=dict(status='PASS_RETAINED_PUBLIC_F64_LOGICAL192_ONE_PHYSICAL_COLD_BUILD',counts=counts,
            actual_build_sha256=ctx.id['build_report_sha256'],candidate_source=R['SOURCE'],candidate_image=R['ELF'],
            actual_report_sha256=sha((actual/'report.json').read_bytes()),retained_opcodes=ops,
            original_comparisons_changed=False,full_candidate_suite_qualification=False,native_pass_transferred=False)
        (out/'joint-result.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
        print(json.dumps(summary,sort_keys=True))
    return 0
if __name__=='__main__':raise SystemExit(main())
