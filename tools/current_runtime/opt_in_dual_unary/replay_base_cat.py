#!/usr/bin/env python3
"""Replay exact retained base270 or Cat4 on the joint candidate, with original observer code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

PIN_FILE = Path(__file__).with_name("base-cat-pins.json")
PIN_SHA = "21ca0a54b3ba51b823677b8064d0ada3d4aff9528a9d9a1b5940090d1654d3b1"
ELF = "5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72"
PROFILE = b'PROFILE_PATH = ROOT / "tools/current_runtime/source/manifest.json"'
REBIND = b'PROFILE_PATH = scoped_manifest_path'
CALL = b'built = module.admit_build(directory, inputs.bind, expected)'
NEW_CALL = b'built = scoped_admit(module, directory, inputs.bind, expected)'
BINDING = 'tests/current_runtime_original35/binding.py'
QUALIFICATION = 'tools/current_runtime/qualification.py'
DIRECT = 'tests/current_runtime270.py'
CAT = 'tests/current_runtime_cat4'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def check(ok, message):
    if not ok:
        raise ValueError(message)

def exact(raw, old, new):
    check(raw.count(old) == 1 and new not in raw, "fixed admission line changed")
    out = raw.replace(old, new, 1)
    check(out.replace(new, old, 1) == raw, "admission substitution not reversible")
    return out

def load(path, raw, name, **kwargs):
    m = types.ModuleType(name)
    m.__file__ = str(path)
    m.__dict__.update(kwargs)
    exec(compile(raw, str(path), "exec", dont_inherit=True, optimize=0), m.__dict__)
    return m

def digest_file(path):
    check(path.is_file() and path.resolve() == path, f"ordinary exact input required: {path}")
    return sha(path.read_bytes())

def run(a):
    repo, build, output = a.repository, a.dual_build, a.output
    pin_raw = PIN_FILE.read_bytes()
    check(sha(pin_raw) == PIN_SHA, "wrong fixed source pin catalog")
    pins = json.loads(pin_raw)
    active = pins['shared'] | (pins[a.suite])
    for name, h in active.items():
        check(digest_file(repo / name) == h, f"fixed input changed: {name}")
    for p in { (repo / name).parent for name in active }:
        check(not (p / '__pycache__').exists(), f"need a fresh checkout: {p}")
    def raw(name):
        return (repo / name).read_bytes()

    offline_path = 'tools/current_runtime/opt_in_admission/admit.py'
    offline = load(repo / offline_path, raw(offline_path), 'dual_artifact_identity')
    canonical = repo / 'tools/current_runtime/source/manifest.json'
    admission = offline.admit('dual', canonical, build, {'dual_scoped': a.dual_scoped})
    check(admission['native_image_sha256'] == ELF, 'not the exact joint candidate')
    check(digest_file(a.dual_source / 'manifest.json') == admission['manifest_sha256'], "candidate checkout differs")
    build_sha = admission['build_report_sha256']
    ctx_module = load(repo / QUALIFICATION, exact(raw(QUALIFICATION), PROFILE, REBIND),
                      'dual_full_cold_profile', scoped_manifest_path=a.dual_source / 'manifest.json')
    calls = []
    def admit(legacy, directory, bind, expected):
        check(directory == build and expected == build_sha, 'different exact candidate build')
        try:
            legacy.admit_build(directory, bind, expected)
        except ValueError as e:
            check(str(e) == 'build used another source manifest', f"unexpected canonical outcome: {e}")
            calls.append(str(e))
        else:
            raise ValueError("the canonical path erroneously admitted the opt-in candidate")
        result = ctx_module.admit_build(directory, bind, expected)
        check(result["native_sha256"] == ELF, "incorrect candidate image")
        return result
    bindm = load(repo / BINDING, exact(raw(BINDING), CALL, NEW_CALL), 'joint270_binding', scoped_admit=admit)
    def other_admission(adapter):
        adapter.bind_build = bindm.bind_build
        return adapter
    orig = load(repo / DIRECT, raw(DIRECT), 'joint_retained_270')
    other_admission(orig)
    common = ['--build-directory', str(build), '--expected-build-sha256', build_sha, '--output', str(output/'observer')]
    readback = None
    argv = sys.argv[:]
    try:
        if a.suite == 'base270':
            code = orig.main(common)
            if code == 0:
                file = 'tests/current_runtime_admission.py'
                repl = exact(raw(file), b'_profile_spec.loader.exec_module(current_profile)',
                             b'current_profile = scoped_reader_profile')
                reader = load(repo / file, repl, 'joint_retained_base270_readback',
                              scoped_reader_profile=ctx_module)
                report = json.loads((output / 'observer/report.json').read_bytes())
                readback = reader.qualify_current270(report, build_sha, output/'observer', bindm.Inputs())
        else:
            cat_path = CAT + '/run.py'
            runner = load(repo / cat_path, raw(cat_path), 'joint_retained_cat4')
            origload = runner.load
            def load_cat(path, name):
                m = origload(path, name)
                return other_admission(m) if path == repo / DIRECT else m
            runner.load = load_cat
            sys.argv[:] = [str(repo / cat_path), '--repository', str(repo), *common]
            code = runner.main()
            if code == 0:
                qual = CAT + '/qualify.py'
                reviewer = load(repo / qual, raw(qual), 'joint_retained_cat4_readback')
                bound = reviewer.load_bound
                def load_for_readback(path, name, inputs, expected=None):
                    m = bound(path, name, inputs, expected)
                    return other_admission(m) if path == repo / DIRECT else m
                reviewer.load_bound = load_for_readback
                report = json.loads((output / 'observer/report.json').read_bytes())
                readback = reviewer.qualify_cat4(report, repo, build, build_sha, output/'observer', bindm.Inputs())
    finally:
        sys.argv[:] = argv
    rfile = output/'observer/report.json'
    report_raw = rfile.read_bytes()
    report = json.loads(report_raw)
    if code == 0:
        cases, frames = {'base270': (270, 284), 'cat4': (4, 4)}[a.suite]
        check(readback['cases'] == cases and readback['frames'] == frames and len(calls) == 2
              and report['native_sha256'] == ELF and report['native_executed'] is True,
              "complete unchanged native observer and independent physical readback required")
    check(offline.admit('dual', canonical, build, {'dual_scoped': a.dual_scoped}) == admission,
          'the actual build changed during replay')
    check(all(digest_file(repo / name) == h for name,h in active.items()), 'fixed source evidence changed during replay')
    return dict(status='PASS_RETAINED_'+a.suite.upper() if code == 0 else 'FAIL_RETAINED_'+a.suite.upper(),
                observer_exit=code, observer_status=report.get('status'), observer_report_sha256=sha(report_raw),
                observer_counts={k:report[k] for k in ('cases','frames','checks','native_executed','Torch_executed') if k in report and not isinstance(report[k], list)},
                readback=readback, canonical_admission_refusals=len(calls),
                native_sha256=ELF, actual_build_sha256=build_sha,
                source_sha256=admission['source_sha256'], manifest_sha256=admission['manifest_sha256'],
                source_catalog_sha256=PIN_SHA, retained_suite=a.suite,
                full_candidate_suite_qualification=False, production_activated=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repository','dual-build','dual-source','dual-scoped','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--suite',choices=('base270','cat4'),required=True)
    a=p.parse_args()
    check(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'run with python -I -B')
    for k,v in vars(a).items():
        if isinstance(v,Path):
            check(not v.is_symlink(), f"unexpected symlink: {v}")
            setattr(a,k,v.resolve())
    a.output.mkdir()
    try:
        receipt=run(a)
    except Exception as e:
        receipt=dict(status='BLOCKED_OR_FAILED',error=str(e),error_type=type(e).__name__)
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(receipt['status'])
    return 0 if receipt['status'].startswith('PASS_') else 1

if __name__=='__main__':
    raise SystemExit(main())
