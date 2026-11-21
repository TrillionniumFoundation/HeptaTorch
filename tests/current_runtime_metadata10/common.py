"""Fixed cat metadata recipes, reference custody and exact source adaptation."""
import ast
import hashlib
import importlib.util
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import textwrap
import types

HERE=Path(__file__).resolve().parent
SOURCE_MANIFEST = '359b0469fb44d4d5e5ad52fb008deb52696168d02c700f00c93da66ff90bf105'
CAT4_QUALIFIER='2d72ca5c8b8e92684e27b30505f32c9e8294ae27554311a68ae1eee6a31f4576'


def require(value,message):
    if not value:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def dump(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def load(path,name,inputs,expected=None):
    raw=inputs.bind(path,expected)
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def single_xz(raw):
    decoder=lzma.LZMADecompressor();result=decoder.decompress(raw)
    require(decoder.eof and not decoder.unused_data,'one complete XZ stream')
    return result


def source_manifest(inputs):
    value=json.loads(inputs.bind(HERE/'source.json',SOURCE_MANIFEST))
    require(value['schema']=='heptatorch.cat-metadata10.source-bindings.v1','fixed metadata10 source manifest')
    return value


def admit(repository,build,expected,inputs):
    pins=source_manifest(inputs)
    for name,pin in pins['dependencies'].items():
        path=repository/name;raw=inputs.bind(path,pin['sha256'])
        require(len(raw)==pin['bytes'] and stat.S_IMODE(path.stat().st_mode)==pin['mode'],'original dependency bytes/mode')
    profile=load(repository/'tools/current_runtime/qualification.py','metadata10_current_build',inputs)
    built=profile.admit_build(build,inputs.bind,expected)
    require(built['source_sha256']==pins['source_sha256'] and built['native_sha256']==pins['native_sha256'],
            'metadata10 must use the exact independently admitted current source/image')
    return pins,built


def reference_package(repository,destination,inputs,restore=False):
    pins=source_manifest(inputs)
    folder=repository/'validation/cat-metadata10-reference'
    manifest_raw=inputs.bind(folder/'manifest.json',pins['reference_manifest_sha256'])
    manifest=json.loads(manifest_raw)
    compressed=inputs.bind(folder/'complete.tar.xz',pins['reference_archive_sha256'])
    require(manifest['schema']=='heptatorch.cat-metadata10.reference-package.v1'
            and manifest['archive']==dict(path='complete.tar.xz',bytes=len(compressed),sha256=sha(compressed))
            and manifest['file_count']==len(manifest['files'])==15
            and manifest['payload_bytes']==sum(row['bytes'] for row in manifest['files'].values())==10351608,
            'entire original metadata10 reference package')
    if restore:destination.mkdir(parents=True,exist_ok=False)
    require(destination.is_dir() and not destination.is_symlink(),'physical reference package directory')
    seen=set()
    with tarfile.open(fileobj=io.BytesIO(single_xz(compressed)),mode='r:') as archive:
        for member in archive:
            name=PurePosixPath(member.name)
            require(name.parts and str(name)==member.name and not name.is_absolute()
                    and '..' not in name.parts and '\\' not in member.name
                    and member.isfile() and member.name not in seen and member.name in manifest['files'],
                    'only fifteen declared unique regular reference members')
            seen.add(member.name);row=manifest['files'][member.name];raw=archive.extractfile(member).read()
            require(len(raw)==member.size==row['bytes'] and sha(raw)==row['sha256']
                    and member.mode==row['mode'],'original reference member identity')
            path=destination.joinpath(*name.parts)
            if restore:
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:stream.write(raw)
                path.chmod(member.mode)
            require(inputs.bind(path,row['sha256'])==raw and stat.S_IMODE(path.stat().st_mode)==row['mode'],
                    'physical original reference member bytes/mode')
    require(seen==set(manifest['files'])=={str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()},
            'complete physical reference census')
    roles={name:destination/value for name,value in manifest['roles'].items()}
    require(len(roles)==15 and len(set(roles.values()))==15,'distinct exact original reference roles')
    reference=json.loads(inputs.bind(roles['report'],pins['reference_sha256']))
    producer=inputs.bind(roles['producer'],pins['producer_sha256'])
    module=types.ModuleType('metadata10_original_literal_recipes');module.__file__=str(roles['producer'])
    exec(compile(producer,module.__file__,'exec'),module.__dict__)
    recipes=list(module.CASES)
    require(reference['schema']=='heptatorch.cat-empty-reference10.v1'
            and reference['status']=='OBSERVED' and reference['cases']==reference['expected_cases']==len(recipes)==10
            and reference['recipes']==recipes and reference['case_manifest_sha256']==sha(canonical(recipes))
            and reference['source_sha256']==pins['producer_sha256'] and reference['inputs_unchanged'] is True
            and reference['rng_unchanged'] is True and reference['torch_executed'] is True
            and reference['native_executed'] is False and reference['versions_recorded_without_offset'] is True
            and reference['output_grad_retention_hooks_added'] is False
            and reference['wire_capacity_contract_inferred'] is False,'original actual Torch observations and recipes')
    provenance=reference['oracle_provenance']
    require(provenance['status']=='PASS' and len(provenance['checks'])==11
            and all(value is True for value in provenance['checks'].values()),'all eleven fixed original oracle checks')
    mapping=manifest['input_path_map']
    require(set(mapping)==set(reference['inputs']) and len(mapping)==len(set(mapping.values()))==10,
            'each original absolute input must have one physical custody mapping')
    for original,relative in mapping.items():
        row=reference['inputs'][original];raw=inputs.bind(destination/relative,row['sha256'])
        require(len(raw)==row['bytes'],'all ten original reference input entities physically present')
    require(manifest['original_absolute_paths_unchanged'] is True
            and manifest['original_path_strict_API_reexecuted'] is False
            and all(manifest[key] is False for key in ('packaging_native_executed','packaging_torch_executed','packaging_python_entity_executed')),
            'mapped immutable custody is not a replay of the original path-strict Torch API')
    execution=json.loads(inputs.bind(roles['execution'],manifest['original_execution_sha256']))
    # Its complete original byte identity and all referenced artifact bytes are
    # fixed. Preserve its original names; map only physical reads to this pack.
    artifact_mapping=manifest['execution_relative_path_map']
    require(set(artifact_mapping)=={'cat-empty-reference10-report.json','cat-empty-reference10-staging.json',
            'cat-empty-reference10.py','cat-empty-reference10.stderr','cat-empty-reference10.stdout'},
            'all five original public execution artifact paths')
    require(execution['schema']=='heptatorch.torch-reference-execution-public.v2'
            and execution['status']=='OBSERVED' and execution['process']['state']=='succeeded'
            and execution['process']['exit_code']==0 and execution['process']['terminal_publication_confirmed'] is True
            and execution['process']['cancelled'] is False and execution['process']['timed_out'] is False
            and execution['cases']==execution['expected_cases']==10 and execution['input_count']==10
            and execution['inputs_unchanged'] is True and execution['rng_unchanged'] is True
            and execution['native_executed'] is False and execution['oracle_provenance']==provenance,
            'original completed fixed Torch execution and provenance')
    for key in ('report','staging_report','source'):
        item=execution[key];path=destination/artifact_mapping[item['path']]
        raw=inputs.bind(path,item['sha256']);require(len(raw)==item['bytes'],'original public execution artifact')
    for name in ('stdout','stderr'):
        item=execution['streams'][name];raw=inputs.bind(destination/artifact_mapping[item['path']],item['sha256'])
        require(item['eof'] is True and item['offset']==0 and item['next_offset']==item['bytes']==len(raw),
                'entire original Torch execution stream and EOF')
    python=execution['python'];raw=inputs.bind(roles['python'],python['sha256'])
    require(len(raw)==python['bytes'] and stat.S_IMODE(roles['python'].stat().st_mode)==python['mode']
            and mapping[python['resolved']]==manifest['roles']['python'],
            'complete original reference interpreter entity, never executed by this custody check')
    return reference,recipes,roles,dict(manifest_sha256=sha(manifest_raw),archive_sha256=sha(compressed),
        files=15,original_reference_inputs=10,original_reference_paths_modified=False,
        original_path_strict_API_reexecuted=False,python_entity_executed=False,
        original_execution_sha256=sha(inputs.bind(roles['execution'])),reference_sha256=pins['reference_sha256'],
        producer_sha256=pins['producer_sha256'],provenance_checks=provenance['checks'])


def adapt_driver(built,inputs):
    pins=source_manifest(inputs);item=pins['original_native_source']
    original=inputs.bind(HERE/item['path'],item['sha256'])
    require(len(original)==item['bytes'],'complete original v2 native reader')
    derived=original;edits=[]
    for old,new in [(pins['original_source_pin'],built['source_sha256']),
                    (pins['original_native_pin'],built['native_sha256'])]:
        before,after=old.encode(),new.encode()
        require(derived.count(before)==1 and before!=after,'one original source or image literal')
        derived=derived.replace(before,after,1);edits.append(dict(before=old,after=new,count=1))
    inverse=derived
    for item in reversed(edits):
        require(inverse.count(item['after'].encode())==1,'unambiguous current source identity inverse')
        inverse=inverse.replace(item['after'].encode(),item['before'].encode(),1)
    require(inverse==original,'entire v2 source including all compare and cleanup bodies must invert')
    ast.parse(derived)
    return derived,dict(original_sha256=sha(original),derived_sha256=sha(derived),edits=edits,
        whole_source_inverse_exact=True,recipes_numerics_metadata_comparison_and_cleanup_unchanged=True,
        original_startup_synchronization_unchanged=True,only_source_image_literals_changed=True)


def comparison_function(inputs):
    pins=source_manifest(inputs);raw=inputs.bind(HERE/'frozen/native_v2.py',pins['original_native_source']['sha256'])
    tree=ast.parse(raw);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    function=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='compare')
    segment=ast.get_source_segment(raw.decode(),function)
    require(sha(segment.encode())==pins['function_sources']['compare']['sha256'],'entire original metadata comparator')
    fields=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='FIELDS' for t in n.targets))
    function.col_offset=0
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace={'FIELDS':fields};exec(compile(module,'<unchanged-v2-comparator>','exec'),namespace)
    return namespace['compare']
