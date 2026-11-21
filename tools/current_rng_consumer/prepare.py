#!/usr/bin/env python3
"""Generate only reviewed build/ABI/identity adaptations of all old RNG checks."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import IMAGE, ENTRY, admit
from process_evidence import record_actual_process

PINS = {'rng_cpu_oracle.py':'5eef7d3dd12cf1c3816a36c1429ef584cc70cbfffa423c1a15e354b19a3ec239',
        'flat_probe.py':'9711bfad298285fba3d5176975a003e0fd944a37e811b1d0172acbe9a0ba862e',
        'review_rng.py':'1bacf93e172b6df18741823521aff01a4de1b5961d16fabc0283f8970cd73e7f'}
sha=lambda b:hashlib.sha256(b).hexdigest()


def main():
    record_actual_process(__file__)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--current-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); repo=a.repository.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    admission=admit(repo,a.build,a.current_build)
    for name in ('build.py','support.py','process_evidence.py','reference_evidence.py'):
        raw=(Path(__file__).resolve().parent/name).read_bytes()
        snapshot=out.parent/name
        assert not snapshot.exists()
        snapshot.write_bytes(raw)
    wheel_helper=(repo/'tests/pinned_reference_custody.py').read_bytes()
    assert sha(wheel_helper)=='fa91a644b8743493ed6ed8f29da966494983cf5eed7300faa715ee4e08c7212e'
    (out.parent/'pinned_reference_custody.py').write_bytes(wheel_helper)
    reports=[]
    for name, relative in [('rng_cpu_oracle.py','tests/rng_cpu_oracle.py'),
                           ('flat_probe.py','tests/rng_independent/flat_probe.py'),
                           ('review_rng.py','tests/rng_independent/review_rng.py')]:
        original=(repo/relative).read_text(); assert sha(original.encode())==PINS[name]
        raw=original; changes=[]
        def replace(before,after,reason):
            nonlocal raw
            assert raw.count(before)==1,(name,before[:100],raw.count(before))
            raw=raw.replace(before,after)
            changes.append({'reason':reason,'before':before,'after':after})
        if name=='rng_cpu_oracle.py':
            replace('import traceback\n','import traceback\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nfrom support import admit, capture, observed_torch\nfrom process_evidence import record_actual_process\nfrom reference_evidence import begin_torch_capture, finish_torch_capture, record_reference, begin_reference_wheel, finish_reference_wheel\n', 'local reviewed admission and actual process/reference evidence imports')
            replace("IMAGE_SHA = 'cffc7c2f6574bf08a184e7d4f84e41a38ce0fb9be2f645bf9c332c93c2621eac'",f"IMAGE_SHA = '{IMAGE}'",'actual current-object adapter ELF identity')
            replace("COMPILER_SHA = '87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54'", "COMPILER_SHA = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'", 'actual current native module compiler')
            replace('def main():\n', 'def main():\n    global ROOT\n    record_actual_process(__file__)\n', 'explicit repository path and actual child process receipt')
            replace("    p.add_argument('--compiler', type=Path, required=True)", "    p.add_argument('--native-build', type=Path, required=True)\n    p.add_argument('--current-build', type=Path, required=True)\n    p.add_argument('--repository', type=Path, required=True)\n    p.add_argument('--allow-unverified-local', action='store_true')", 'current build instead of old compiler; strict default unchanged')
            start=raw.index("    source = (ROOT/'tools/rng-cpu/rng.hepta').read_bytes()")
            stop=raw.index("    compiler = shutil.which('g++')",start)
            replace(raw[start:stop], "    ROOT = a.repository.resolve()\n    assert not a.allow_unverified_local or a.reference == 'pytorch'\n    migration = admit(ROOT, a.native_build, a.current_build)\n    source = (ROOT/'tools/rng-cpu/rng.hepta').read_bytes()\n    assert sha(source) == SOURCE_SHA\n    profile_bytes = (ROOT/'tools/rng-cpu/profile.json').read_bytes(); profile = json.loads(profile_bytes)\n    native = a.native_build.resolve()/'rng-native'\n    assert sha(native.read_bytes()) == IMAGE_SHA\n", 'replace only old monolithic compiler/build admission with exact same-current-object build reader')
            replace("    if a.reference == 'pytorch':\n", "    if a.reference == 'pytorch' and a.allow_unverified_local:\n        torch, provenance = observed_torch(ROOT, out)\n    elif a.reference == 'pytorch':\n", 'explicit exploratory alternative; entire original strict eleven-check branch remains unchanged')
            replace('    torch, provenance = None, None\n',
                    "    import_capture = begin_torch_capture(out) if a.reference == 'pytorch' and not a.allow_unverified_local else None\n    torch, provenance = None, None\n",
                    'capture complete actual strict oracle import streams without altering its original branch')
            replace("        assert all(x is True for x in provenance['checks'].values())\n",
                    "        assert all(x is True for x in provenance['checks'].values())\n    if import_capture is not None:\n        finish_torch_capture(import_capture)\n    if torch is not None:\n        record_reference(ROOT, out, torch, provenance, a.allow_unverified_local)\n        wheel_state = begin_reference_wheel(ROOT, out)\n",
                    'bind original strict provenance to actual interpreter/package/metadata/PEP610/pip input entities')
            replace('    rows, frames, value_count, state_bytes_compared = [], 0, 0, 0\n', '    rows, frames, value_count, state_bytes_compared = [], 0, 0, 0\n    captured = []\n', 'retain actual raw packet receipts')
            replace("        assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr)\n", "        captured.append(capture(out, len(captured), raw, result.stdout, result.stderr, result.returncode))\n        assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr)\n", 'record actual native input/stdout/stderr; preserve original exit assertions')
            replace("    raw = json.dumps(report, indent=2)+'\\n';", "    if torch is not None:\n        finish_reference_wheel(wheel_state)\n    report.update(migration=migration, captured_native_packets=captured,\n                  allow_unverified_local=a.allow_unverified_local,\n                  full_installation_provenance_verified=a.reference == 'pytorch' and not a.allow_unverified_local,\n                  production_oracle_qualified=a.reference == 'pytorch' and not a.allow_unverified_local)\n    if a.allow_unverified_local and not failed:\n        report['status'] = 'PASS_EXPLORATORY_NATIVE_ONLY'\n    assert len(rows) == (76 if a.reference == 'pytorch' else 74)\n    assert len(captured) == frames\n    assert admit(ROOT, a.native_build, a.current_build) == migration\n    raw = json.dumps(report, indent=2)+'\\n';", 'append exact scope/provenance and final custody; do not claim original legacy ELF')
            protected=[('all independent reference classes', original[original.index('def packet('):original.index('\ndef main():')]),
                       ('all 74/76 numerical, state and rejection comparisons',original[original.index('    def scenario('):original.index("    assert sha((ROOT/'tools/rng-cpu/rng.hepta').read_bytes())")])]
        elif name=='flat_probe.py':
            replace('import struct\n','import struct\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nfrom support import admit, capture_flat\nfrom process_evidence import record_actual_process\n','explicit admission and actual child identity imports')
            replace('def main():\n', 'def main():\n    record_actual_process(__file__)\n', 'actual independent flat observer process receipt')
            replace("'return-zero':'cffc7c2f6574bf08a184e7d4f84e41a38ce0fb9be2f645bf9c332c93c2621eac'", f"'return-zero':'{IMAGE}'",'actual current-object ELF identity')
            old="""    # This location is derived from the already-observed ELF startup call.
    assert image[:4]==b'\\x7fELF' and image[0x151]==0xe8
    bridge=0x156+struct.unpack_from('<i',image,0x152)[0]
    assert bridge==0x1ea and image[bridge:bridge+8]==bytes.fromhex('554889e54883ec40')
"""
            new=f"""    # Fixed ABI3 startup call and complete actual ELF identity, reviewed in build.json.
    assert SHA(image)=='{IMAGE}' and image[:4]==b'\\x7fELF' and image[477]==0xe8
    bridge=482+struct.unpack_from('<i',image,478)[0]
    assert bridge=={ENTRY} and image[bridge:bridge+15]==bytes.fromhex('554889e54881ec00100000488b0424')
"""
            replace(old,new,'review actual linked startup/ABI3 entry instead of old monolithic bridge')
            replace('    return result,out,work\n', '    capture_flat(image, request, incoming[:len(request)], raw, out_cap, work_cap, out, work, result)\n    return result,out,work\n', 'retain complete actual flat output and workspace bytes after every original observation')
            replace("    signature=ctypes.CFUNCTYPE(ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint64,\n                               ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_uint64)\n", "    signature=ctypes.CFUNCTYPE(None,ctypes.c_void_p,ctypes.c_void_p)\n", 'ABI3 signature takes argument and result arrays')
            replace("        value=function(address(incoming),len(request),address(output),out_cap,address(workspace),work_cap)\n", "        arguments=(ctypes.c_uint64*6)(address(incoming),len(request),address(output),out_cap,address(workspace),work_cap)\n        result=(ctypes.c_uint64*1)(0xffffffffffffffff)\n        function(ctypes.addressof(arguments),ctypes.addressof(result))\n        value=result[0]\n        assert 0<=value<1<<32\n", 'marshal exactly the original six span words; no native memory transformations')
            replace("    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)", "    p.add_argument('--repository',type=Path,required=True)\n    p.add_argument('--native-build',type=Path,required=True)\n    p.add_argument('--current-build',type=Path,required=True)\n    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)\n    migration=admit(a.repository,a.native_build,a.current_build)\n    assert a.native.resolve()==a.native_build.resolve()/'rng-native'\n    assert a.expect_short=='return-zero'", 'admit same-current-object build; execute complete current stable 10-case contract')
            replace("'bridge_offset':490,",f"'bridge_offset':{ENTRY},'interface_abi':3,'migration':migration,",'accurate bridge receipt')
            protected=[('all 10 original capacity/output/input/workspace comparisons', original[original.index('    rows=[]\n'):original.index("    report={")])]
        else:
            replace('import subprocess\n','import subprocess\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nfrom support import admit\nfrom process_evidence import record_actual_process\n', 'explicit local flat observer and admission import')
            replace('def main():\n', 'def main():\n    record_actual_process(__file__)\n', 'actual independent numerical reviewer process receipt')
            replace("      'cffc7c2f6574bf08a184e7d4f84e41a38ce0fb9be2f645bf9c332c93c2621eac':", f"      '{IMAGE}':",'actual current-object ELF to original logical source contract pin')
            replace("    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)", "    p.add_argument('--repository',type=Path,required=True)\n    p.add_argument('--native-build',type=Path,required=True)\n    p.add_argument('--current-build',type=Path,required=True)\n    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)\n    migration=admit(a.repository,a.native_build,a.current_build)\n    assert a.native.resolve()==a.native_build.resolve()/'rng-native'",'admit exact current object before original 25 comparisons')
            replace("    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\\n');", "    report['migration']=migration\n    assert admit(a.repository,a.native_build,a.current_build)==migration\n    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\\n');",'bind complete original review to exact current object')
            protected=[('all independent transform/reference/state logic',original[original.index('def temper('):original.index('\ndef main():')]),
                       ('all 25 original numerical, error, budget and failure atomicity comparisons',original[original.index('    def raw_reference('):original.index('    report={')])]
        for label,section in protected:
            assert raw.count(section)==1,(name,label)
        inverse=raw
        for change in reversed(changes):
            assert inverse.count(change['after'])==1
            inverse=inverse.replace(change['after'],change['before'])
        assert inverse==original
        ast.parse(raw)
        (out/name).write_text(raw)
        reports.append({'source':relative,'executed':name,'original_sha256':sha(original.encode()),'executed_sha256':sha(raw.encode()),
                        'whole_file_inverse_exact':True,'changes':changes,
                        'protected_original_sections':[{'description':label,'sha256':sha(section.encode()),'bytes':len(section.encode()),'exactly_unchanged':True} for label,section in protected]})
    cpp=(repo/'tests/rng_independent/state_reference.cpp').read_bytes()
    (out/'state_reference.cpp').write_bytes(cpp)
    report={'schema':'heptatorch.rng-consumer-validation-admission-delta.v1','status':'PASS_SOURCE_ADMISSION_ONLY',
            'migration':admission,'original_eleven_provenance_assertions_retained_in_strict_branch':True,
            'original_case_counts':{'cpython':74,'pytorch':76,'flat_capacity':10,'independent_review':25},
            'scripts':reports,'state_reference_cpp_sha256':sha(cpp),'old_files_modified':False,'native_tests_executed':False}
    (out/'source-inverse.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'scripts':len(reports),'full_file_inverses':True}))


if __name__=='__main__':main()
