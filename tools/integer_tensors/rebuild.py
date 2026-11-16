"""Cold-build every native unit and require the frozen int64 profile bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('parent','repository','upstream','compiler','linker','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--compactor',type=Path)
    a=p.parse_args();out=a.output.resolve();assert not out.exists()
    manifest_path=HERE/'package-manifest.json';manifest_raw=manifest_path.read_bytes()
    manifest=json.loads(manifest_raw);profile_raw=(HERE/'profile.json').read_bytes();profile=json.loads(profile_raw)

    def custody():
        assert manifest_path.read_bytes()==manifest_raw and (HERE/'profile.json').read_bytes()==profile_raw
        for name,pin in manifest['files'].items():
            path=PACKAGE/name;raw=path.read_bytes()
            assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256'] and path.stat().st_mode&511==pin['mode'],name

    custody()
    assert sha(a.compiler.read_bytes())==profile['compiler_sha256']
    assert sha(a.linker.read_bytes())==profile['linker_sha256']
    compactor=(a.compactor or a.repository/'tools/native_dropout/compact_functions.py').resolve()
    command=[sys.executable,'-B',str(HERE/'build.py'),'--parent',str(a.parent.resolve()),
             '--repository',str(a.repository.resolve()),'--upstream',str(a.upstream.resolve()),
             '--compactor',str(compactor),'--compiler',str(a.compiler.resolve()),'--linker',str(a.linker.resolve()),'--output',str(out)]
    report={'schema':'heptatorch.int64-tensor-cold-profile-rebuild.v1','status':'STARTED','command':command,
            'profile_sha256':sha(profile_raw),'package_manifest_sha256':sha(manifest_raw),
            'native_build_process_started':False,'native_build_process_completed':False,
            'existing_integer_image_reused':False,'full_pytorch_rewrite_complete':False}
    try:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        report['native_build_process_started']=True
        stdout,stderr=process.communicate(timeout=900)
        report.update(native_build_process_completed=True,exit_code=process.returncode)
        out.mkdir(parents=True,exist_ok=True)
        (out/'builder.stdout').write_bytes(stdout);(out/'builder.stderr').write_bytes(stderr)
        report.update(stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
        assert process.returncode==0,('native cold builder',process.returncode)
        receipt=(out/'build.json').read_bytes();actual=json.loads(receipt)
        assert actual['status']=='PASS'
        for key in ('parent_source_sha256','parent_native_sha256','source_sha256','native_sha256',
                    'descriptor_bytes','state_format','backing_format','functions','blocks','instructions',
                    'compiler_sha256','linker_sha256','units'):
            assert actual[key]==profile[key],key
        for unit in profile['units']:
            for suffix,key in (('.hepta','source_sha256'),('.o','object_sha256')):
                assert sha((out/(unit['name']+suffix)).read_bytes())==unit[key]
        assert sha((out/'runtime.hepta').read_bytes())==profile['source_sha256']
        assert sha((out/'tensor-integer-session').read_bytes())==profile['native_sha256']
        custody()
        report.update(status='PASS',units_reproduced_exactly=len(profile['units']),
                      source_sha256=profile['source_sha256'],native_sha256=profile['native_sha256'],
                      actual_build_report_sha256=sha(receipt),all_package_members_unchanged=True)
        (out/'profile-rebuild.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report));return 0
    except BaseException as error:
        if 'process' in locals() and process.poll() is None:
            process.kill();stdout,stderr=process.communicate()
            out.mkdir(parents=True,exist_ok=True)
            (out/'builder.stdout').write_bytes(stdout);(out/'builder.stderr').write_bytes(stderr)
        out.mkdir(parents=True,exist_ok=True);report.update(status='FAIL',error=repr(error))
        (out/'profile-rebuild.json').write_text(json.dumps(report,indent=2)+'\n')
        raise


if __name__=='__main__':raise SystemExit(main())
