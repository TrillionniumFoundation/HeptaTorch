#!/usr/bin/env python3
"""Integrate immutable retained-446 native qualification packages.

Default mode really executes the frozen 17-step producer, including compilation.
--producer-review explicitly verifies and copies an existing complete producer
directory instead; the aggregate never labels that branch a new cold build.
Every later runner still executes. This is integration orchestration, not a new
independent semantic review, a Torch run, or a production numerical fallback.
"""
import argparse
from collections import Counter
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import traceback


SOURCE = '666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b'
IMAGE = '7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5'
PARENT_SOURCE = '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
PARENT_IMAGE = '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
RETAIN_REFERENCE = 'ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3'
OPTIMIZER_REFERENCE = '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'
PROFILE = '8da746dea3493b3d21714eae3d8048a38b073796d519395b96f9dd783ac80758'
PARENT_MANIFEST = '3e8c80053365f7e5de1d24725c511f49f256e8437f5a6d3556f09ff07682dc69'
OWN_MANIFEST = 'd91e5fc032d6a38c262a09b73813668b5b087905d8d4ec27643c04546072dcad'
PROVENANCE = {
    'python_matches', 'platform_matches', 'version_backend_matches',
    'git_matches_oracle_build_pin', 'distribution_version_matches',
    'direct_url_matches', 'direct_url_archive_sha256_matches',
    'pip_report_url_matches', 'pip_report_archive_sha256_matches',
    'pip_report_version_matches', 'pip_report_requested',
}
MANIFESTS = (
    ('validation/retained-gradients/frozen-handoff-manifest.json',
     'ebb1bef63aee4f715dcf45399f454b8a82c94097f2d7147d05de281fcdf6540f', 79),
    ('validation/retained-nonleaf-review/frozen-handoff-manifest.json',
     '546b5bd8d401048f6d5eff81bfa797f81f5af9fd30cfc77f9f0edf6fa89f55ac', 22),
    ('validation/retained-resource-review/frozen-handoff-manifest.json',
     'a85c2785ca098202e233b4cc32a95e1a461c27fdfdb9c1299fad66d3611f9752', 32),
    ('validation/retained-extension-review/frozen-handoff-manifest.json',
     '9e4fcde490673ed4f0178e2d903022006ba6afb92290d0c346289489c6805e48', 12),
    ('tools/retained_gradients/package-manifest.json', OWN_MANIFEST, 13),
)
RELOCATIONS = {
    ('validation/retained-nonleaf-review/frozen-handoff-manifest.json', 'README.md'):
        'validation/retained-nonleaf-review/README.md',
    ('validation/retained-resource-review/frozen-handoff-manifest.json', 'README.md'):
        'validation/retained-resource-review/README.md',
    ('validation/retained-resource-review/frozen-handoff-manifest.json', 'handoff.json'):
        'validation/retained-resource-review/handoff.json',
    ('validation/retained-extension-review/frozen-handoff-manifest.json', 'README.md'):
        'validation/retained-extension-review/README.md',
    ('validation/retained-extension-review/frozen-handoff-manifest.json', 'handoff.json'):
        'validation/retained-extension-review/handoff.json',
}
STEPS = (
    ('cold-native-build', 'build.py', None),
    ('original-public-ten', 'replay_original_ten.py', 'torch-gradient-objects.json'),
    ('incoming-25', 'materialization_replay.py', 'torch-gradient-materialization.json'),
    ('alias-edge-12', 'replay_alias_edges.py', 'torch-gradient-alias-edge.json'),
    ('bernoulli-gradient-layout-8', 'replay_bernoulli_gradient_layout.py', 'torch-bernoulli-gradient-layout.json'),
    ('native-preserve-storage-4', 'test_preserve_storage_lifetimes.py', None),
    ('native-private-lifetimes-5', 'test_native_lifetimes.py', None),
    ('typed-gradient-references', 'test_gradient_references.py', None),
    ('signed-zero-13', 'replay_signed_zero.py', 'torch-gradient-signed-zero.json'),
    ('partial-16', 'replay_partial.py', 'torch-backward-partial.json'),
    ('presence-8', 'replay_presence.py', 'torch-gradient-presence.json'),
    ('gradient-clear-10', 'replay_grad_clear.py', 'torch-grad-clear.json'),
    ('distribution-atomic-183', 'replay_distribution_atomic.py', 'torch-bernoulli-dropout.json'),
    ('distribution-explicit-183', 'replay_distribution_explicit.py', 'torch-bernoulli-dropout.json'),
    ('original-optimizer-24-mapping', 'replay_optimizer_objects_v2.py', 'torch-optimizer-objects.json'),
    ('owned-adam-cases21-24', 'replay_adam_objects.py', 'torch-optimizer-objects.json'),
    ('optimizer-gradient-case20', 'replay_optimizer_gradient_alias.py', 'torch-optimizer-objects.json'),
)
ADAPTER_STEPS = {'native-private-lifetimes-5', 'typed-gradient-references', 'gradient-clear-10'}
OPEN_DIST = [4,9,14,19,24,29,34,39,109,110,167,168,169,170,171,172,173,174]
ATOMIC_DIFF = [101,103,105,179]


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def unique(pairs):
    result = {}
    for k,v in pairs:
        require(k not in result, ('duplicate JSON key', k))
        result[k] = v
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def relative(root, name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and str(p)==name,
            ('unsafe relative path', name))
    return root/name


class Custody:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None, size=None, mode=None):
        path = Path(path).resolve()
        require(path.is_file(), ('missing input', str(path)))
        value = {'bytes': path.stat().st_size, 'sha256': file_hash(path),
                 'mode': stat.S_IMODE(path.stat().st_mode)}
        require(expected is None or value['sha256']==expected, ('input SHA', str(path)))
        require(size is None or value['bytes']==size, ('input length', str(path)))
        require(mode is None or value['mode']==mode, ('input mode', str(path)))
        require(path not in self.files or self.files[path]==value, ('input changed', str(path)))
        self.files[path] = value
        return value

    def json(self, path, expected=None):
        self.bind(path, expected)
        return decode(Path(path).read_bytes())

    def record(self, path, expected=None):
        value = self.bind(path, expected)
        return {'path': str(Path(path).resolve()), **value, 'report': self.json(path)}

    def finish(self):
        for path,value in list(self.files.items()):
            self.bind(path, value['sha256'], value['bytes'], value['mode'])


def manifest(repo, name, pin, count, custody):
    data = custody.json(relative(repo,name),pin)
    entries = data['files']
    rows = [dict(v,path=k) for k,v in entries.items()] if isinstance(entries,dict) else entries
    require(len(rows)==count and len({r['path'] for r in rows})==count, ('manifest cardinality',name))
    relocation = {}
    for row in rows:
        installed = RELOCATIONS.get((name,row['path']),row['path'])
        custody.bind(relative(repo,installed),row['sha256'],row['bytes'],row.get('mode'))
        if installed!=row['path']:
            relocation[row['path']] = installed
    return {'path':str(repo/name),'sha256':pin,'file_count':count,
            'explicit_document_path_relocations':relocation,
            'all_member_bytes_hashes_declared_modes_verified':True}


def runtime_identity(data, source=SOURCE, image=IMAGE):
    sources = [data[k] for k in ('source_sha256','native_source_sha256','runtime_source_sha256') if k in data]
    images = [data[k] for k in ('native_sha256','native_executable_sha256','executable_sha256','binary_sha256') if k in data]
    require(sources and all(v==source for v in sources), 'report exact source identity')
    require(images and all(v==image for v in images), 'report exact native identity')


def check_build(directory, profile, custody, parent=False):
    data = custody.json(directory/'build.json')
    require(data['status']=='PASS' and data['link_exit']==0, 'native build terminal status')
    schema = 'heptatorch.gradient-adam-distributions-native-build.v1' if parent else 'heptatorch.retained-gradient-native-build.v1'
    require(data['schema']==schema, 'native build schema')
    for k in ('source_sha256','native_sha256','functions','blocks','instructions','descriptor_bytes','state_format','backing_format','units'):
        require(data[k]==profile[k], ('native profile field',k))
    require(data['compiler_sha256']==COMPILER and data['linker_sha256']==LINKER, 'native build compiler/linker identity')
    require(data['module_count']==len(data['units'])==9, 'nine native units')
    name = 'tensor-gradient-adam-distributions-session' if parent else 'tensor-retained-session'
    custody.bind(directory/'runtime.hepta',profile['source_sha256'])
    custody.bind(directory/name,profile['native_sha256'],profile['native_bytes'])
    require(len({r['name'] for r in data['units']})==9, 'unique native units')
    for row in data['units']:
        require(Path(row['name']).name==row['name'] and row['compiler_exit']==0, 'native unit admission')
        custody.bind(directory/(row['name']+'.hepta'),row['source_sha256'])
        custody.bind(directory/(row['name']+'.o'),row['object_sha256'])
    return {'path':str(directory),'receipt':custody.record(directory/'build.json'),
            'source_sha256':profile['source_sha256'],'native_sha256':profile['native_sha256'],
            'all_nine_module_sources_and_objects_verified':True}


def provenance(reference, strict=False):
    value = reference['oracle_provenance']
    require(value['status']=='PASS' and value['lock_sha256']==LOCK, 'fixed reference provenance')
    require(set(value['checks'])==PROVENANCE and all(x is True for x in value['checks'].values()),
            'all eleven original provenance checks')
    require(reference['status']=='OBSERVED', 'actual reference status')
    require(reference.get('probe_error_count',0)==0 and reference.get('probe_errors',0) in (0,[]), 'zero reference setup errors')
    if strict or 'native_executed' in reference:
        require(reference['native_executed'] is False, 'reference did not execute native target')
    if strict or 'source_files_unchanged' in reference:
        unchanged = reference['source_files_unchanged']
        require(unchanged is True or isinstance(unchanged,dict) and unchanged and all(v is True for v in unchanged.values()),
                'reference original source unchanged')
    require(value['observed']['version']=='2.15.0+cpu' and
            value['observed']['git_commit']=='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e', 'fixed actual Torch build')


def identities(value):
    keys = {'name','case','index','family','variant','status','reference_status','native_executed'}
    result = {k:value[k] for k in keys if k in value}
    for k in ('rows','cases','variants','records'):
        if isinstance(value.get(k),list):
            result[k] = [identities(x) for x in value[k] if isinstance(x,dict)]
    return result


def same_summary(actual, frozen):
    require(set(actual)==set(frozen), ('frozen report schema fields',set(actual)^set(frozen)))
    for k,v in frozen.items():
        if not isinstance(v,(dict,list)):
            require(type(actual[k]) is type(v) and actual[k]==v, ('frozen scalar summary',k,actual[k],v))
    require(identities(actual)==identities(frozen), 'frozen case classifications and order')


def flags(command):
    require(command[1:4]==['-I','-S','-B'] and len(command[5:])%2==0, 'producer Python flags')
    result = {}
    for name,value in zip(command[5::2],command[6::2]):
        require(name.startswith('--') and name not in result, 'unique producer CLI options')
        result[name] = value
    return result


def inventory(root, custody):
    result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), ('evidence symlink',str(path)))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = custody.bind(path)
        else:
            require(path.is_dir(), ('evidence special file',str(path)))
    return result


def copy_producer(source, destination, custody):
    require(source!=destination and not destination.exists() and source not in destination.parents,
            'producer reuse requires a new separate directory')
    before = inventory(source,custody)
    destination.mkdir()
    for name,value in before.items():
        target = relative(destination,name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(relative(source,name),target)
        target.chmod(value['mode'])
        custody.bind(target,value['sha256'],value['bytes'],value['mode'])
    require(inventory(source,custody)==before and inventory(destination,custody)==before,
            'producer copied all original bytes/modes and source remained unchanged')
    return {'source_directory':str(source),'copied_directory':str(destination),'file_count':len(before),
            'expanded_bytes':sum(v['bytes'] for v in before.values()),'members':before,
            'source_before_after_equal':True,'copy_bytes_hashes_modes_equal':True,
            'original_absolute_paths_or_argv_rewritten':False,'new_native_execution':False}


def producer_receipt(root, repo, parent_package, profile, custody):
    record = custody.record(root/'report.json')
    data = record['report']
    require(data['schema']=='heptatorch.retained-gradient-cold-qualification.v1' and data['status']=='PASS', 'producer17 status')
    runtime_identity(data)
    require(data['step_count']==len(data['steps'])==len(data['suite_reports'])==17, 'producer17 cardinality')
    require(data['native_executed'] is True and data['all_sources_unchanged'] is True and
            data['native_build_reproduced_exactly'] is True, 'producer actual execution receipt')
    require(data['parent_manifest_sha256']==PARENT_MANIFEST and data['package_manifest_sha256']==OWN_MANIFEST,
            'producer manifest identities')
    require([r['name'] for r in data['steps']]==[r[0] for r in STEPS], 'all17 fixed producer steps in order')
    recorded_repo = Path(data['steps'][0]['command'][4]).parents[2]
    recorded_parent = Path(data['steps'][1]['command'][4]).parents[2]
    recorded_native = Path(flags(data['steps'][0]['command'])['--output'])
    recorded_root = recorded_native.parent
    require(recorded_native.name=='native', 'producer native child location')
    own = custody.json(repo/'tools/retained_gradients/package-manifest.json',OWN_MANIFEST)
    require(data['own_inputs_sha256']=={str(recorded_repo/name):v['sha256'] for name,v in own['files'].items()},
            'producer all13 executed source inputs bound')
    children = {}
    for step,(name,script,reference_name) in zip(data['steps'],STEPS):
        options = flags(step['command'])
        if name=='cold-native-build':
            relative_script = 'tools/retained_gradients/build.py'
            script_root,recorded_script_root = repo,recorded_repo
            report_relative = 'native/build.json'
        elif name in ADAPTER_STEPS:
            relative_script = 'tests/retained_gradient_regression/'+script
            script_root,recorded_script_root = repo,recorded_repo
            report_relative = name+'/report.json'
        else:
            relative_script = 'tests/gradient_adam_distributions_review/'+script
            script_root,recorded_script_root = parent_package,recorded_parent
            report_relative = name+'/report.json'
        script_info = custody.bind(script_root/relative_script)
        require(Path(step['command'][4])==recorded_script_root/relative_script, ('producer exact executed script path',name))
        require(step['exit_code']==0 and step['report_path']==report_relative and
                data['suite_reports'][name]=={'path':report_relative,'sha256':step['report_sha256']}, 'producer child linkage')
        for suffix,field in (('.stdout','stdout_sha256'),('.stderr','stderr_sha256')):
            custody.bind(root/(name+suffix),step[field])
        path = relative(root,report_relative)
        child = custody.record(path,step['report_sha256'])
        expected_path = repo/'validation/retained-gradients/reports/retained-portable-qualification-01'/(name+'.json')
        expected = custody.record(expected_path)
        same_summary(child['report'],expected['report'])
        runtime_identity(child['report'])
        require(step['report_status']==child['report']['status'], 'producer child reported status')
        require(Path(options['--output'])==recorded_root/Path(report_relative).parent, 'producer child output argument')
        if name!='cold-native-build':
            if '--binary' in options:
                require(Path(options['--binary'])==recorded_native/'tensor-retained-session' and
                        Path(options['--source'])==recorded_native/'runtime.hepta', 'producer native argv identity')
                require(options.get('--expected-sha256',options.get('--expected-native-sha256'))==IMAGE and
                        options['--expected-source-sha256']==SOURCE, 'producer CLI native pins')
            else:
                require(Path(options['--executable'])==recorded_native/'tensor-retained-session', 'distribution executable argv')
        if reference_name is not None:
            ref_path = parent_package/'validation/gradient-adam-distributions/reference'/reference_name
            ref = custody.json(ref_path)
            provenance(ref)
            pin = file_hash(ref_path)
            fields = [child['report'][k] for k in ('reference_sha256','observations_sha256','actual_reference_sha256') if k in child['report']]
            require(fields and all(v==pin for v in fields), ('producer child actual reference pin',name))
            flag = '--reference' if '--reference' in options else '--observations'
            require(Path(options[flag])==recorded_parent/'validation/gradient-adam-distributions/reference'/reference_name,
                    ('producer exact fixed reference path',name))
        child.update(frozen_original_report_sha256=expected['sha256'],
                     actual_script_sha256=script_info['sha256'],
                     original_step=step,
                     original_streams={suffix:custody.bind(root/(name+'.'+suffix)) for suffix in ('stdout','stderr')})
        children[name] = child
    bridge = custody.json(root/'distribution-build-field-bridge.json')
    require(file_hash(root/'distribution-build-field-bridge.json')==data['distribution_receipt_bridge']['sha256'], 'distribution field bridge bytes')
    cold = children['cold-native-build']['report']
    require(bridge==dict(cold,runtime_source_sha256=SOURCE,executable_sha256=IMAGE), 'only two distribution receipt aliases')
    require(data['distribution_receipt_bridge']['original_cold_build_sha256']==children['cold-native-build']['sha256'], 'bridge original cold report identity')
    return {'original_report':record,'children':children,
            'recorded_producer_directory':str(recorded_root), 'current_receipt_directory':str(root),
            'original_absolute_paths_and_argv_unchanged':True,
            'native':check_build(root/'native',profile,custody)}


def check_distribution_ledger(report, reference, explicit):
    require(report['reference_cases']==len(report['rows'])==len(reference['rows'])==183, 'fixed183 corpus')
    for i,(row,original) in enumerate(zip(report['rows'],reference['rows'])):
        require((row['index'],row['name'],row['family'])==(i,original['name'],original['family']), '183 index/name/family')
        expected = 'OPEN_UNREPRESENTED' if i in OPEN_DIST else ('OPEN_TRANSACTION_DIFFERENCE' if not explicit and i in ATOMIC_DIFF else 'PASS')
        require(row['status']==expected, ('183 exact classification',i))
    expected = {'PASS':165,'OPEN_UNREPRESENTED':18} if explicit else {'PASS':161,'OPEN_TRANSACTION_DIFFERENCE':4,'OPEN_UNREPRESENTED':18}
    require(dict(Counter(x['status'] for x in report['rows']))==report['counts']==expected, '183 Counter and report')
    require(report['all_rng_complete'] is False and report['full_pytorch_rewrite_complete'] is False, 'distribution remains partial')


def frames(raw, magic):
    offset = 0
    result = []
    while offset<len(raw):
        require(len(raw)-offset>=16 and raw[offset:offset+8]==magic, 'complete transport frame header')
        size = struct.unpack_from('<Q',raw,offset+8)[0]
        require(size<=1048576 and size<=len(raw)-offset-16, 'complete transport frame body')
        result.append(raw[offset+16:offset+16+size])
        offset += 16+size
    require(offset==len(raw), 'no transport tail')
    return result


def typed_raw(root, report, custody):
    req = root/'transport/requests.frames.bin'
    res = root/'transport/responses.frames.bin'
    custody.bind(req);custody.bind(res)
    a,b = frames(req.read_bytes(),b'HTOWNR01'),frames(res.read_bytes(),b'HTOWNS01')
    require(len(a)==len(b)==len(report['frames'])==report['frame_count']==10, 'all ten fresh typed frames')
    for raw_request,raw_response,expected in zip(a,b,report['frames']):
        require(sha(raw_request)==expected['request_sha256'] and sha(raw_response)==expected['response_sha256'], 'typed original frame bytes')
    files = inventory(root,custody)
    return {'complete_fresh_raw_transport':True,'request_frames':10,'response_frames':10,
            'files':files,'file_count':len(files),'expanded_bytes':sum(x['bytes'] for x in files.values()),
            'original_interrupted_execution_reconstructed':False}


def archive_outputs(root, custody):
    """Preserve every child byte, excluding the three named self references."""
    excluded = {'report.json','evidence-manifest.json','raw-evidence.tar.xz'}
    require(not (root/'raw-evidence.tar.xz').exists() and not (root/'evidence-manifest.json').exists(),
            'new evidence archive and manifest required')
    members = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), ('output symlink',str(path)))
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in excluded:
                members[name] = custody.bind(path)
        else:
            require(path.is_dir(), ('output special file',str(path)))
    archive = root/'raw-evidence.tar.xz'
    with lzma.open(archive,'wb',format=lzma.FORMAT_XZ,preset=6) as packed:
        with tarfile.open(fileobj=packed,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for name,value in members.items():
                info = tarfile.TarInfo(name)
                info.size,info.mode,info.mtime = value['bytes'],value['mode'],0
                info.uid=info.gid=0;info.uname=info.gname=''
                with (root/name).open('rb') as stream:
                    tar.addfile(info,stream)
    seen = {}
    with tarfile.open(archive,'r|xz') as tar:
        for member in tar:
            require(member.isfile() and member.name in members and member.name not in seen,'safe unique evidence member')
            h,size = hashlib.sha256(),0
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    h.update(block);size+=len(block)
            value = {'bytes':size,'sha256':h.hexdigest(),'mode':member.mode}
            require(value==members[member.name] and member.size==size, ('archive member bytes',member.name))
            seen[member.name]=value
    require(seen==members,'complete archive membership')
    # Reading past the tar EOF independently validates the complete XZ trailer.
    with lzma.open(archive,'rb') as stream:
        tar_bytes=0
        for block in iter(lambda:stream.read(1024*1024),b''):tar_bytes+=len(block)
    for name,value in members.items():
        custody.bind(root/name,value['sha256'],value['bytes'],value['mode'])
    current = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
               and p.relative_to(root).as_posix() not in excluded}
    require(current==set(members),'no output member appeared or disappeared during archive')
    record = {'schema':'heptatorch.retained-integration-evidence.v1','status':'COMPLETE_BYTES_VERIFIED',
              'archive':{'path':'raw-evidence.tar.xz',**custody.bind(archive)},
              'file_count':len(members),'expanded_bytes':sum(v['bytes'] for v in members.values()),
              'tar_stream_bytes':tar_bytes,'files':members,
              'all_member_lengths_SHA256_modes_verified':True,'complete_XZ_trailer_verified':True,
              'expanded_originals_unchanged':True,'expanded_originals_deleted':False,
              'excluded_top_level_only':sorted(excluded),
              'self_reference_policy':'Final aggregate and evidence manifest are outside the archive; every child report and original stream is inside unchanged.'}
    dump(root/'evidence-manifest.json',record)
    return {'archive':{'path':str(archive),**custody.bind(archive)},
            'manifest':{'path':str(root/'evidence-manifest.json'),**custody.bind(root/'evidence-manifest.json')},
            'file_count':record['file_count'],'expanded_bytes':record['expanded_bytes'],
            'all_member_bytes_and_modes_verified':True,'complete_XZ_trailer_verified':True,
            'excluded_top_level_only':record['excluded_top_level_only']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--parent-package',type=Path,help='Repository root containing tools/gradient_adam_distributions and tests; defaults to --repo.')
    for name in ('parent-build','compiler','linker','stack-build','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--producer-review',type=Path,help='Verify/copy a complete prior producer directory; do not compile it again.')
    args = parser.parse_args()
    repo,out = args.repo.resolve(),args.output.resolve()
    parent_package = (args.parent_package or repo).resolve()
    parent_build = args.parent_build.resolve()
    out.mkdir(parents=True,exist_ok=False)
    script = Path(__file__).resolve()
    (out/'executed-integration.py').write_bytes(script.read_bytes())
    custody = Custody()
    report = {'schema':'heptatorch.retained-gradients-integration-qualification.v1',
              'status':'RUNNING','source_sha256':SOURCE,'native_sha256':IMAGE,
              'integration_script_sha256':file_hash(script),'commands':[],
              'role':'Integration of unchanged producer and independent tests; no new independent semantic review.',
              'producer_executed_this_invocation':args.producer_review is None,
              'producer_receipt_reused':args.producer_review is not None,
              'Torch_executed_here':False,'production_host_numerics':False,
              'live32_and_frozen35':'SEPARATE_ROOT_AND_HOSTED_GATES_NOT_EXECUTED_HERE',
              'overall_decision':'HOLD','pytorch_api_denominator':2590,
              'newly_closed_pytorch_api_rows':0,'full_schema_parity':0,'full_rewrite_complete':False}

    def save():
        dump(out/'report.json',report)

    def execute(name, runner, arguments, output, report_name='report.json'):
        custody.bind(runner)
        command = [sys.executable,'-I','-S','-B',str(runner),*map(str,arguments),'--output',str(output)]
        logs = out/'logs';logs.mkdir(exist_ok=True)
        with (logs/(name+'.stdout')).open('wb') as stdout,(logs/(name+'.stderr')).open('wb') as stderr:
            result = subprocess.run(command,cwd=repo,stdout=stdout,stderr=stderr,check=False)
        row = {'name':name,'command':command,'exit_code':result.returncode,
               'runner_sha256':file_hash(runner),
               'stdout':{'path':str(logs/(name+'.stdout')),**custody.bind(logs/(name+'.stdout'))},
               'stderr':{'path':str(logs/(name+'.stderr')),**custody.bind(logs/(name+'.stderr'))}}
        if (output/report_name).is_file():
            row['child'] = custody.record(output/report_name)
        report['commands'].append(row);save()
        print(json.dumps({'step':name,'exit_code':result.returncode,'status':row.get('child',{}).get('report',{}).get('status')}),flush=True)
        require(result.returncode==0 and 'child' in row, ('runner failed',name,result.returncode))
        return row['child']

    try:
        custody.bind(script)
        report['frozen_manifests'] = [manifest(repo,*row,custody) for row in MANIFESTS]
        report['frozen_manifests'].append(manifest(parent_package,'tools/gradient_adam_distributions/package-manifest.json',PARENT_MANIFEST,45,custody))
        custody.bind(repo/'compatibility/pytorch-oracle-lock.json',LOCK)
        profile = custody.json(repo/'tools/retained_gradients/profile.json',PROFILE)
        parent_profile = custody.json(parent_package/'tools/gradient_adam_distributions/profile.json')
        require(parent_profile['source_sha256']==PARENT_SOURCE and parent_profile['native_sha256']==PARENT_IMAGE,'parent428 profile')
        report['parent_native'] = check_build(parent_build,parent_profile,custody,True)
        for path,pin in ((args.compiler.resolve(),COMPILER),(args.linker.resolve(),LINKER),
                         (args.stack_build.resolve()/'native/stack-module-compiler',COMPILER),
                         (args.stack_build.resolve()/'native/stack-linker-session',LINKER)):
            custody.bind(path,pin)
        reference_root = repo/'validation/retained-nonleaf-review/reference'
        actual12 = custody.record(reference_root/'torch-retained-nonleaf.json',RETAIN_REFERENCE)
        optimizer24 = custody.record(reference_root/'original-optimizer-observations.json',OPTIMIZER_REFERENCE)
        for x in (actual12,optimizer24):provenance(x['report'],strict=True)
        require(actual12['report']['cases']==8 and actual12['report']['expected_variants']==12,'actual retain8 families/12 variants')
        require(optimizer24['report']['cases']==len(optimizer24['report']['rows'])==24,'original optimizer24')
        report['fixed_actual_references'] = {'retained12':actual12,'optimizer24':optimizer24,
                                            'eleven_provenance_checks_each':True,'Torch_reexecuted':False}
        producer = out/'producer'
        if args.producer_review is None:
            execute('producer17',repo/'tools/retained_gradients/qualify.py',[
                    '--parent-build',parent_build,'--parent-package',parent_package,
                    '--compiler',args.compiler.resolve(),'--linker',args.linker.resolve()],producer)
        else:
            previous = args.producer_review.resolve()
            if previous.is_file():
                require(previous.name=='report.json','--producer-review file must be report.json')
                previous = previous.parent
            verified = producer_receipt(previous,repo,parent_package,profile,custody)
            report['producer_reuse_original_report'] = verified['original_report']
            report['producer_reuse_copy'] = copy_producer(previous,producer,custody)
        report['producer'] = producer_receipt(producer,repo,parent_package,profile,custody)
        source,binary = producer/'native/runtime.hepta',producer/'native/tensor-retained-session'
        native = producer/'native'
        inherited = report['producer']['children']
        dist_ref = custody.json(parent_package/'validation/gradient-adam-distributions/reference/torch-bernoulli-dropout.json')
        for name,explicit in (('distribution-atomic-183',False),('distribution-explicit-183',True)):
            check_distribution_ledger(inherited[name]['report'],dist_ref,explicit)
        report['distribution_ledger'] = {'reference_cases':183,'atomic_counts':inherited['distribution-atomic-183']['report']['counts'],
                                         'explicit_counts':inherited['distribution-explicit-183']['report']['counts'],
                                         'unrepresented_zero_based_indices':OPEN_DIST,'atomic_difference_zero_based_indices':ATOMIC_DIFF,
                                         'all_rng_complete':False}
        report['historical_optimizer_mapping'] = {
            'original_24_report':inherited['original-optimizer-24-mapping'],
            'unchanged_original_classification':[18,1,5],
            'inherited_union_before_additive_retain':{'qualified_exposed_scope':23,'partial_cases':1,'partial_one_based_indices':[14]},
            'old_report_reclassified':False}
        oldfail = custody.record(repo/'validation/retained-gradients/reports/old-93-94-contract-failure.json')
        require(oldfail['report']['status']=='FAIL','original unknown93/94 negative must remain FAIL')
        report['original_93_94_negative'] = oldfail
        rawgap = custody.record(repo/'validation/retained-resource-review/qualification.json')
        require(rawgap['report']['recovery']['original_archive_complete'] is False and
                rawgap['report']['recovery']['original_typed_raw_frames_missing'] is True and
                rawgap['report']['recovery']['missing_original_bytes_reconstructed'] is False,'explicit old raw gap')
        report['historical_raw_gap'] = rawgap
        save()

        nonleaf = repo/'tests/retained_nonleaf_review'
        resource = repo/'tests/retained_resource_review'
        standard = ['--binary',binary,'--source',source,'--expected-sha256',IMAGE,'--expected-source-sha256',SOURCE]
        strict = execute('retained-actual12',nonleaf/'replay_retained_nonleaf.py',[
                *standard,'--observations',reference_root/'torch-retained-nonleaf.json',
                '--probe',reference_root/'pytorch_retained_nonleaf_probe.py','--protocol',nonleaf/'protocol.py',
                '--state-tag','HTRET001','--backing-tag','HTORET01'],out/'retained-actual12')
        d = strict['report'];runtime_identity(d)
        require(d['status']=='QUALIFIED_EXPOSED_SCOPE' and d['variants_executed']==d['variants_qualified']==len(d['rows'])==12
                and d['unexpected_failures']==0 and d['all_inputs_unchanged'] is True,'strict12 complete pass')
        require(d['reference_provenance_checks']==actual12['report']['oracle_provenance']['checks'],'strict12 original provenance')
        expected12 = custody.json(repo/'validation/retained-nonleaf-review/actual12-native-01.json')
        require([(r['case'],r['variant'],r['reference_json_pointer']) for r in d['rows']]==
                [(r['case'],r['variant'],r['reference_json_pointer']) for r in expected12['rows']], 'strict12 original case order')
        require(d['adam_parameter_moment_tolerance']==expected12['adam_parameter_moment_tolerance'],'original Adam tolerance unchanged')
        require(all(r['responses_before_stdin_eof'] is True and r['exit_code']==0 and r['status']=='QUALIFIED_EXPOSED_SCOPE'
                    for r in d['rows']), 'strict12 real EOF and all results')
        require(d['inputs']['observations']==RETAIN_REFERENCE and d['inputs']['binary']==IMAGE and d['inputs']['source']==SOURCE,
                'strict12 current and actual input pins')
        additive = execute('optimizer-case14-additive',nonleaf/'replay_optimizer_retained_admission.py',[
                *standard,'--observations',reference_root/'original-optimizer-observations.json',
                '--probe',reference_root/'pytorch_optimizer_object_probe.py','--old-replay',reference_root/'replay_optimizer_objects_v2.py',
                '--protocol',nonleaf/'protocol.py'],out/'optimizer-case14-additive')
        d = additive['report']
        require(d['status']=='ORIGINAL_RETAINED_NONLEAF_SUBOBSERVATION_QUALIFIED' and d['reference_outcomes_matched']==2
                and d['exit_code']==0 and d['all_responses_before_stdin_eof'] is True and d['all_inputs_unchanged'] is True,'additive original14')
        require(d['original_frozen_replay_modified'] is False and d['original_frozen_report_reclassified'] is False
                and d['other_original_24_observations_reexecuted'] is False,'old24 not rewritten or recounted')
        require(d['inputs']['observations']==OPTIMIZER_REFERENCE and d['inputs']['binary']==IMAGE and d['inputs']['source']==SOURCE,
                'additive14 current and actual input pins')
        resources = execute('resources8',resource/'test_resources.py',[
                '--binary',binary,'--source',source,'--expected-native-sha256',IMAGE,'--expected-source-sha256',SOURCE,
                '--controls',resource/'native_release_controls.py','--observer',resource/'owner_client.py',
                '--reference',repo/'validation/retained-resource-review/actual/torch-retained-nonleaf.json'],out/'resources8')
        d = resources['report'];runtime_identity(d)
        require(d['status']=='PASS' and d['case_count']==d['passed']==len(d['cases'])==8 and d['failed']==0
                and d['input_custody_after'] is True,'eight resource controls')
        typed = execute('typed-handoff2',resource/'test_typed_handoff.py',[
                '--build',native,'--previous-build',parent_build,'--stack-build',args.stack_build.resolve(),
                '--protocol',resource/'protocol.py','--controller',resource/'state_handoff.hepta'],out/'typed-handoff2')
        d = typed['report'];runtime_identity(d)
        require(d['status']=='PASS' and d['case_count']==d['passed']==len(d['cases'])==2 and d['failed']==0
                and d['linked_modules']==19 and len(d['units'])==18 and d['all_source_custody_unchanged'] is True,'two real typed directions')
        require(d['previous_source_sha256']==PARENT_SOURCE and d['previous_native_sha256']==PARENT_IMAGE
                and d['compiler_sha256']==COMPILER and d['linker_sha256']==LINKER,'typed parent/tools identity')
        require(d['raw_pointer_import'] is False and d['owner_descriptor_memcpy'] is False
                and all(r['inverse_byte_identical'] is True for r in d['units']), 'real linear owned handoff')
        report['fresh_typed_handoff_raw'] = typed_raw(out/'typed-handoff2',d,custody)
        source_review = execute('source-review',resource/'source_review.py',[
                '--build',native,'--previous-build',parent_build,'--peer-runner',nonleaf/'replay_retained_nonleaf.py',
                '--review-inputs',repo/'validation/retained-resource-review/source-inputs'],out/'source-review')
        d = source_review['report'];runtime_identity(d)
        require(d['status']=='PASS' and d['source_functions']==446 and d['parent_functions']==428
                and d['unchanged_parent_bodies']==410 and d['all446_core_bodies_in_exactly_one_actual_unit'] is True
                and d['all_inputs_unchanged'] is True and d['native_executed_by_this_review'] is False,'source/body/unit review scope')
        extensions = execute('frozen-extension-five-groups',repo/'tests/retained_extension_review/run_regressions.py',[
                '--repo',repo,'--build',native],out/'extension-regressions')
        d = extensions['report'];runtime_identity(d)
        require(d['status']=='PASS' and d['commands_executed']==d['successful_commands']==len(d['suites'])==6
                and d['regression_groups']==5 and d['input_custody_after'] is True,'all inherited extension groups executed')
        expected_extension = custody.json(repo/'validation/retained-extension-review/report.json')
        require([r['name'] for r in d['suites']]==[r['name'] for r in expected_extension['suites']],'all original extension children')
        for row in d['suites']:
            child_path = out/'extension-regressions/children'/row['name']/'report.json'
            child = custody.record(child_path,row['report_sha256'])
            require(child['report']==row['report'] and child['bytes']==row['report_bytes'] and row['exit_code']==0,'extension raw child bytes and status')
            for kind in ('stdout','stderr'):
                custody.bind(out/'extension-regressions/logs'/(row['name']+'.'+kind),row[kind]['sha256'],row[kind]['bytes'],row[kind]['mode'])
        require(d['cross6_adapter']['inverse_byte_identical'] is True and
                d['cross6_adapter']==expected_extension['cross6_adapter'],'only original two tag substitutions')
        report['new_review_children'] = {'retained_actual12':strict,'optimizer_case14_additive':additive,
                                         'resources8':resources,'typed_handoff2':typed,'source_review':source_review,
                                         'inherited_extensions':extensions}
        report['optimizer_case14_additive_boundary'] = {
            'same_original_reference_24_sha256':OPTIMIZER_REFERENCE,
            'original_report_keeps_18_qualified_1_partial_5_open':True,
            'inherited_union_keeps_23_qualified_plus_partial14':True,
            'additive_two_original_subobservations_passed':True,
            'separate_actual_retained_12_variants_passed':True,
            'full_optimizer_API_closure_claimed':False}
        report['unclosed_domains'] = {
            'distribution_unrepresented_cases':18,
            'ordinary_distribution_atomic_side_effect_differences':4,
            'incoming_materialization_unrepresented_variants':3,
            'legacy_presence_state_lifecycle_observations':[8,65],
            'legacy_gradient_clear_state_and_step_observations':[11,9,30],
            'legacy_gradient_clear_Adam_word_differences':6,
            'bool_native_dropout_other_dtypes_devices_higher_order_full_API':'OPEN',
            'historical_raw_archive_gap':'PRESERVED; fresh typed execution is separate complete evidence',
            'live32_frozen35':'SEPARATE; no admission inferred by this aggregate'}
        report['complete_current_execution_evidence'] = archive_outputs(out,custody)
        custody.finish()
        report.update(status='PASS',all_inputs_unchanged=True,
                      checked_inputs={str(p):v for p,v in custody.files.items()},
                      counts={'producer_steps':17,'retained_actual_variants':12,'additive_original_case14_subobservations':2,
                              'resource_cases':8,'typed_directions':2,'read_only_source_review':1,
                              'extension_groups':5,'extension_commands':6},
                      counting_boundary='All groups remain separately counted; overlapping regressions and reference reuse are not API rows.')
        save()
        print(json.dumps({'status':'PASS','report_sha256':file_hash(out/'report.json'),
                          'producer_executed_this_invocation':report['producer_executed_this_invocation'],
                          'checked_inputs':len(custody.files)}),flush=True)
        return 0
    except BaseException as error:
        primary_error,primary_traceback = repr(error),traceback.format_exc()
        report.update(status='FAIL',error=primary_error,traceback=primary_traceback,
                      requested_producer_mode='reuse' if args.producer_review else 'fresh',
                      producer_executed_this_invocation=any(x['name']=='producer17' for x in report['commands']),
                      producer_receipt_reused='producer_reuse_copy' in report,
                      completed_outer_commands=len(report['commands']),
                      successful_suite_completion_claimed=False)
        try:
            save()
        except BaseException as report_error:
            report['initial_failure_report_error'] = repr(report_error)
        try:
            partial = archive_outputs(out,custody)
            partial.update(scope='PARTIAL_FAILED_EXECUTION_BYTES_ONLY',
                           successful_suite_completion_claimed=False)
            report['failed_execution_evidence'] = partial
        except BaseException as archive_error:
            report['evidence_archive_error'] = repr(archive_error)
            report['evidence_archive_traceback'] = traceback.format_exc()
        report['checked_inputs'] = {str(p):v for p,v in custody.files.items()}
        try:
            save()
        except BaseException as report_error:
            report['final_failure_report_error'] = repr(report_error)
        print(json.dumps({'status':'FAIL','error':primary_error,'report':str(out/'report.json'),
                          'evidence_archive_error':report.get('evidence_archive_error'),
                          'final_failure_report_error':report.get('final_failure_report_error')}),flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
