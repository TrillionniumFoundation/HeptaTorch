"""Execute the frozen 523 behavior suites on a separately built int64 image.

Only private-tag/hash/executable-name observer adaptations are admitted. This
driver does not call the 523 builder or claim a new cold build of the int64
image. Its existing complete 11-unit native build is checked independently.
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import stat
import subprocess
import sys
import tarfile

SOURCE = '5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
IMAGE = '69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'
FLAT = 'c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native-build', 'repository', 'adapters', 'flat-compiler', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    repo, native, adapters = a.repository.resolve(), a.native_build.resolve(), a.adapters.resolve()
    tests, refs = repo/'tests/native_dropout_review', repo/'validation/native-dropout/reference'
    paths = [Path(__file__), native/'build.json', native/'runtime.hepta', native/'tensor-mixed-autograd-session', a.flat_compiler]
    paths += list(adapters.glob('*.py')) + [adapters/'custody.json']
    paths += list(tests.rglob('*.py')) + list(refs.glob('*.json'))
    inputs = {path.resolve(): path.read_bytes() for path in paths}
    assert sha(inputs[(native/'runtime.hepta').resolve()]) == SOURCE
    assert sha(inputs[(native/'tensor-mixed-autograd-session').resolve()]) == IMAGE
    assert sha(inputs[a.flat_compiler.resolve()]) == FLAT
    build = json.loads(inputs[(native/'build.json').resolve()])
    assert build['status'] == 'PASS' and build['source_sha256'] == SOURCE and build['native_sha256'] == IMAGE
    assert build['functions'] == 556 and build['blocks'] == 7420 and build['instructions'] == 26889
    assert build['descriptor_bytes'] == 640 and build['state_format'] == 'HTMXAD02' and build['backing_format'] == 'HTOMXD02'
    assert len(build['units']) == 11
    for unit in build['units']:
        for suffix, field in (('.hepta','source_sha256'),('.o','object_sha256')):
            path = native/(unit['name']+suffix)
            raw = path.read_bytes(); assert sha(raw) == unit[field]; inputs[path.resolve()] = raw
        assert unit['compiler_exit'] == 0
    custody = json.loads(inputs[(adapters/'custody.json').resolve()])
    for row in custody['rows']:
        original = repo/row['original_path']; raw = original.read_bytes(); inputs[original.resolve()] = raw
        revised = (adapters/row['file']).read_bytes()
        assert sha(raw) == row['original_sha256'] and sha(revised) == row['adapted_sha256']
        inverse = revised
        for edit in reversed(row['edits']):
            before, after = edit['before'].encode(), edit['after'].encode()
            assert inverse.count(after) == edit['count']; inverse = inverse.replace(after, before)
        assert inverse == raw
    report = {'schema':'heptatorch.int64-inherited523-execution.v1', 'status':'RUNNING',
              'source_sha256': SOURCE, 'native_sha256': IMAGE, 'steps': [],
              'native_build_receipt_sha256': sha(inputs[(native/'build.json').resolve()]),
              'existing_native_units_verified':11, 'cold_build_executed_by_this_driver':False,
              'source_adapter_custody_sha256':sha(inputs[(adapters/'custody.json').resolve()]),
              'counting_boundary':'The twelve named behavior gates overlap. No totals imply unique upstream APIs or cases.',
              'all_expected_statuses_and_counts_matched':False,
              'Torch_executed_here':False, 'full_pytorch_rewrite_complete':False, 'pytorch_api_denominator':2590}

    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

    def archive_step(name, directory):
        archive_dir = out/'evidence'; archive_dir.mkdir(exist_ok=True)
        archive = archive_dir/(name+'.tar.xz'); files = {}
        with lzma.open(archive, 'wb', preset=3) as xz:
            with tarfile.open(fileobj=xz, mode='w|') as tar:
                for path in sorted(directory.rglob('*')):
                    if not path.is_file(): continue
                    relative = path.relative_to(directory).as_posix(); raw = path.read_bytes()
                    info = tar.gettarinfo(str(path), arcname=relative); assert info.isfile()
                    files[relative] = {'bytes':len(raw),'sha256':sha(raw),'mode':stat.S_IMODE(path.stat().st_mode)}
                    with path.open('rb') as reader: tar.addfile(info, reader)
        seen = set()
        with tarfile.open(archive,'r:xz') as tar:
            for member in tar:
                pin = files[member.name]; raw = tar.extractfile(member).read()
                assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256'] and member.mode==pin['mode']
                seen.add(member.name)
        assert seen == set(files)
        with lzma.open(archive,'rb') as reader:
            while reader.read(1048576): pass
        removed = []
        for name2, pin in files.items():
            path = directory/name2
            if path.suffix in ('.json','.py','.md','.txt','.stdout','.stderr'): continue
            assert sha(path.read_bytes()) == pin['sha256'] and stat.S_IMODE(path.stat().st_mode)==pin['mode']
            path.unlink(); removed.append(name2)
        receipt = {'archive':archive.name,'archive_sha256':sha(archive.read_bytes()),'archive_bytes':archive.stat().st_size,
                   'members':files,'member_count':len(files),'original_bytes':sum(p['bytes'] for p in files.values()),
                   'all_member_bytes_sizes_modes_verified':True,'xz_complete_EOF_verified':True,
                   'verified_raw_expanded_copies_removed':removed,'restoration_base':str(directory)}
        (archive_dir/(name+'.manifest.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        (directory/'RESTORE.json').write_text(json.dumps({'archive':str(archive),'manifest':str(archive_dir/(name+'.manifest.json'))},indent=2)+'\n')
        return {key:value for key,value in receipt.items() if key not in ('members','verified_raw_expanded_copies_removed')}

    def run(name, script, args, expected_status, counts):
        command = [sys.executable, '-B', str(script), *map(str,args), '--output', str(out/name)]
        row = {'name':name,'command':command,'status':'STARTED'}; report['steps'].append(row); save()
        with (out/(name+'.stdout')).open('wb') as so, (out/(name+'.stderr')).open('wb') as se:
            completed = subprocess.run(command, stdout=so, stderr=se, timeout=900)
        row.update(exit_code=completed.returncode,stdout_sha256=sha((out/(name+'.stdout')).read_bytes()),stderr_sha256=sha((out/(name+'.stderr')).read_bytes()))
        receipt = out/name/'report.json'
        if receipt.exists():
            raw = receipt.read_bytes(); value = json.loads(raw)
            row.update(report_sha256=sha(raw),report=str(receipt.relative_to(out)),status=value['status'])
        else:
            value = None; row['status'] = 'NO_REPORT'
        save()
        assert completed.returncode == 0, (name, completed.returncode)
        assert value['status'] == expected_status, (name,value['status'],expected_status)
        for key,want in counts.items(): assert value[key] == want, (name,key,value[key],want)
        row['archive'] = archive_step(name,out/name); save()
        return value

    try:
        common = ['--binary',native/'tensor-mixed-autograd-session','--source',native/'runtime.hepta',
                  '--protocol',tests/'owned-tensor-object-extension/protocol.py']
        for name, script, extra, counts in (
            ('typed-slice',tests/'test_slice.py',[],{'frames':48}),
            ('typed-lifetimes',tests/'test_typed_lifetimes.py',[],{'frames':343}),
            ('bool-logical-not',tests/'test_logical_not.py',['--flat-compiler',a.flat_compiler.resolve()],{'frames':66}),
            ('dtype-admission101',tests/'test_dtype_admission_v3.py',[],{'frames':126}),
            ('owned-allocation',tests/'test_dtype_allocation_v2.py',[],{'frames':120}),
            ('history-mul-controls8',tests/'test_history_mul_controls.py',['--reference',refs/'torch-empty-dropout-identity.json'],{'cases':8,'passed':8,'frames':216}),
            ('actual-native-dropout28',tests/'replay_contract_identity.py',['--reference',refs/'torch-native-dropout.json'],{'cases':28,'frames':1816}),
            ('original-native-dropout8',tests/'replay_original_eight.py',['--reference',repo/'validation/gradient-adam-distributions/reference/torch-bernoulli-dropout.json'],{'cases':8,'passed':8,'frames':140}),
            ('actual-existing-optimizer6',adapters/'replay_existing_optimizer.py',['--observer',adapters/'identity_observer.py','--reference',refs/'torch-empty-dropout-existing-optimizer.json'],{'cases':6,'passed':6,'frames':2114}),
            ('actual-sgd-effective-layout6',adapters/'replay_sgd_effective_layout.py',['--observer',adapters/'identity_observer.py','--reference',refs/'torch-sgd-effective-layout.json'],{'cases':6,'passed':6,'frames':891})):
            run(name,script,common+extra,'PASS',counts)
        value = run('actual-retained12',adapters/'replay_retained_nonleaf.py',[
            '--observations',repo/'validation/retained-nonleaf-review/reference/torch-retained-nonleaf.json',
            '--probe',repo/'validation/retained-nonleaf-review/reference/pytorch_retained_nonleaf_probe.py',
            '--protocol',repo/'tests/retained_nonleaf_review/protocol.py','--binary',native/'tensor-mixed-autograd-session',
            '--source',native/'runtime.hepta','--expected-sha256',IMAGE,'--expected-source-sha256',SOURCE,
            '--state-tag','HTMXAD02','--backing-tag','HTOMXD02'],
            'QUALIFIED_EXPOSED_SCOPE',{'variants_executed':12,'variants_qualified':12,'unexpected_failures':0})
        assert value['inputs']['observations']=='ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3'
        assert len(value['reference_provenance_checks'])==11 and all(v is True for v in value['reference_provenance_checks'].values())
        assert value['all_inputs_unchanged'] is True
        run('inherited16',adapters/'run_inherited.py',[
            '--native-build',native,'--parent-package',repo,'--retained-package',repo,'--adapters',adapters],
            'PASS',{'step_count':16,'all_sources_unchanged':True})
        assert all(path.read_bytes()==raw for path,raw in inputs.items())
        report.update(status='PASS',step_count=len(report['steps']),all_inputs_unchanged=True,
                      all_expected_statuses_and_counts_matched=True,
                      inputs_sha256={str(path):sha(raw) for path,raw in inputs.items()})
        save(); print(json.dumps({key:value for key,value in report.items() if key not in ('steps','inputs_sha256')}))
        return 0
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));save();raise


if __name__=='__main__':
    raise SystemExit(main())
