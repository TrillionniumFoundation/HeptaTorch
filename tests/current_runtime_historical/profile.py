"""Exact identity-only 700-reader derivation and disclosed 701 observer hook."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

SOURCE_700 = 'b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4'
IMAGE_700 = 'c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e'
SOURCE = 'fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38'
IMAGE = '5b7aac203d2c913d43781f0da45d40e9bac2037c2bc074ec30f8dd663710a805'
SUPPORT_SHA = '3a6d61b759f45dc62f3d7b8320ce9379904430e76e8e279924d5d0803b73d359'
OWNER_SHA = 'd72283cc8c351607ca3930affe8526a64936ec476436c27d089818cc5aade55d'
INTEGRAL_SHA = '845f38cbe7588bc181009bd40637c2597a84fea5fb036ee52fb0af60af90cd4c'
BOOL_PINS = {
    'cases.py': 'cb489ce173d6cf3984d86c796b516c490b07c2a5d48639a3cdded5cc58187508',
    'common.py': '073bfd1c717587e31dacb71d050ca19b0e128ef456a0696204d0dd079b5c5eba',
    'controls.py': 'cf883730b6ba116f81fc3f8222d76a6b196425717f73756e9caea4d496fba287',
    'native.py': '7c5a2ead8dd9c6a39835991095880a6c61e6ab18fb034172163b2f440c486b2d',
    'owner_adapter.py': '5a32e6995a8e34f81b14e24c271de496e48d62b23f85a3f16fcb1989febe16fa',
    'reference.py': '5ed8900f8cde38b6491db834c3a233e4a7621d9b2ccd808c5dd822cb78b9736c',
}


def require(condition, rule):
    if not condition:
        raise AssertionError(rule)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read_pin(path, pin):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular pinned input: ' + str(path))
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    fields = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
    require(fields(before) == fields(after) and sha(raw) == pin, 'stable pinned input: ' + str(path))
    return raw


def identity_edits(name):
    if name == 'native.py':
        return [(SOURCE_700,SOURCE,1),(IMAGE_700,IMAGE,1),
            ('build["complete_core_bodies_matched"] == 700','build["complete_core_bodies_matched"] == 712',1),
            ('build["functions"] == 700','build["functions"] == 712',1),
            ('aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16',
             '761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e',1),
            ('len(build["units"]) == build["units_planned"] == 14 and len(build["processes"]) == 15',
             'len(build["units"]) == build["units_planned"] == 15 and len(build["processes"]) == 16',1)]
    if name == 'common.py':
        return [('b"functions == 700"','b"functions == 712"',1),
            ('b"sum-bool-source01-UNBOUND-700"','b"sum-bool-source01-UNBOUND-712"',1),
            ('len(functions) == len(matches) == 700','len(functions) == len(matches) == 712',1)]
    if name == 'owner_adapter.py':
        return [('observer.configure(owner, source_sha, image_sha, 700)',
                 'observer.configure(owner, source_sha, image_sha, 712)',1)]
    if name == 'integral24.py':
        return [(SOURCE_700,SOURCE,1),(IMAGE_700,IMAGE,1)]
    return []


def derive(raw, name):
    original = raw.decode(); derived = original; edits = identity_edits(name)
    for before,after,count in edits:
        require(derived.count(before) == count and after not in derived, 'identity edit count: ' + before)
        derived = derived.replace(before,after)
    inverse = derived
    for before,after,count in reversed(edits):
        require(inverse.count(after) == count, 'whole-source inverse count: ' + after)
        inverse = inverse.replace(after,before)
    require(inverse == original, 'complete original source inverse: ' + name)
    old_ast, new_ast = ast.parse(original), ast.parse(derived)
    old_def = {n.name: ast.dump(n, include_attributes=False) for n in old_ast.body
               if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new_def = {n.name: ast.dump(n, include_attributes=False) for n in new_ast.body
               if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    allowed = {'native.py':{'admit_build'}, 'common.py':{'adapted_parent','source_functions'},
               'owner_adapter.py':{'make_observer'}}.get(name,set())
    require(old_def.keys() == new_def.keys(), 'complete class/function inventory')
    changed = {key for key in old_def if old_def[key] != new_def[key]}
    require(changed == allowed, 'only identity-bearing admission bodies may differ: ' + name)
    return derived.encode(), dict(original_sha256=sha(raw), derived_sha256=sha(derived.encode()),
        original_bytes=len(raw), derived_bytes=len(derived.encode()), whole_inverse_exact=True,
        edits=[dict(before=a,after=b,count=n) for a,b,n in edits],
        identical_class_and_function_AST=sorted(set(old_def)-changed),
        changed_identity_admission_AST=sorted(changed), comparison_or_expected_value_edits=[])


def make_profile(repository, destination):
    repository=Path(repository);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False);(destination/'bool').mkdir()
    fixture_module=load_module(repository/'tests/current_runtime_fixtures.py','current_runtime_historical_fixtures')
    frozen=fixture_module.restore_baseline(repository,destination/'frozen700')
    files={}
    for name,pin in BOOL_PINS.items():
        source=frozen/'tests/sum_bool_review'/name;raw=read_pin(source,pin)
        derived,proof=derive(raw,name);target=destination/'bool'/name
        with target.open('xb') as stream:stream.write(derived)
        files['bool/'+name]={**proof,'original_path':str(source),'derived_path':str(target)}
    source=frozen/'tests/current_runtime_integral24/run.py'
    derived,proof=derive(read_pin(source,INTEGRAL_SHA),'integral24.py')
    target=destination/'integral24.py'
    with target.open('xb') as stream:stream.write(derived)
    files['integral24.py']={**proof,'original_path':str(source),'derived_path':str(target)}
    result=dict(schema='heptatorch.712.historical-reader-identity-proof.v1',files=files,
        source_sha256=SOURCE,native_sha256=IMAGE,original_reference_and_case_sources_unchanged=True,
        frozen_source_fixture=dict(archive_sha256=fixture_module.BASELINE_ARCHIVE_SHA,
            manifest_sha256=fixture_module.BASELINE_MANIFEST_SHA,original_modes_restored=True),
        original_all_case_execution_loops_unchanged=True,original_control_expectations_unchanged=True,
        original_700_Integral_classifier_unchanged=True,
        runtime_hooks=['Integral ROOT relocated to explicitly selected unchanged repository',
            'Integral bool_support admits all six original pins, then returns the derived 701 admissions',
            'Integral ReferencePaths permits only exact physical native-package members independently bound to the original fixed manifests; its original paths/read guards remain unchanged',
            'Keep original Bool owner predicates and RNG frame checker; add the separately pinned 701 F32 metadata predicate only for command115, target0, Bool/I64 input'],
        original_FAIL_reports_must_remain_FAIL=True,native_executed=False,Torch_executed=False)
    dump(destination/'source-proof.json',result)
    return result


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def explicit_native_package_paths(original_class, integral_package_sha):
    """Add exact physical native-source members before the original path guard.

The immutable reference path mapper remains responsible for every other path.
This admits no directory prefix: only fixed-manifest members independently
checked for original SHA, byte count and mode enter the physical whitelist.
The complete original read method and both of its metadata ledgers are inherited.
"""
    class NativePackagePaths(original_class):
        def __init__(self, module, binding, args, inputs):
            super().__init__(module, binding, args, inputs)
            self.native_package_paths = set()
            packages = (
                (Path(args.parent_native_package).absolute(),
                 'f13a204f5071c07b34ab35ee1b4a6dc90163137b0762fb57bd8c34abb0c8b277', 23),
                (Path(args.output).absolute()/'frozen', integral_package_sha, 19),
            )
            for directory, manifest_sha, count in packages:
                manifest_path = directory/'source-manifest.json'
                manifest = json.loads(inputs.bind(manifest_path, manifest_sha))
                require(manifest['file_count'] == len(manifest['files']) == count,
                        'complete exact native package for physical path whitelist')
                members = {manifest_path}
                total = 0
                for name, pin in manifest['files'].items():
                    relative = Path(name)
                    require(not relative.is_absolute() and '..' not in relative.parts
                            and relative.as_posix() == name, 'safe exact native-package member')
                    actual = directory/relative
                    raw = inputs.bind(actual, pin['sha256'])
                    row = inputs.rows[str(actual.absolute())]
                    require({key: row[key] for key in ('bytes', 'sha256', 'mode')} == pin
                            and len(raw) == pin['bytes'], 'original native member bytes and mode')
                    total += len(raw)
                    members.add(actual)
                require(total == manifest['payload_bytes'] and len(members) == count + 1,
                        'complete native-package physical whitelist')
                require(not self.native_package_paths.intersection(members),
                        'distinct exact native physical members')
                self.native_package_paths.update(members)
            require(len(self.native_package_paths) == 44, 'exact forty-four native physical paths')

        def paths(self, requested):
            requested = Path(requested).absolute()
            if requested in self.native_package_paths:
                return requested, requested
            return super().paths(requested)

    require(NativePackagePaths.read is original_class.read,
            'complete original file read and metadata assertions inherited unchanged')
    return NativePackagePaths


def execute_worker(config):
    """Execute one complete original main(), preserving its unmodified FAIL exit."""
    require(not sys.flags.optimize, 'all original Python assertions enabled')
    root=Path(config['repository']);folder=Path(config['profile_directory'])
    proof=json.loads(read_pin(folder/'source-proof.json',config['source_proof_sha256']))
    for relative,row in proof['files'].items():
        original=read_pin(Path(row['original_path']),row['original_sha256'])
        expected,again=derive(original,'integral24.py' if relative=='integral24.py' else Path(relative).name)
        require(again['whole_inverse_exact'] is True and read_pin(folder/relative,row['derived_sha256']) == expected,
                'worker rechecks complete source inverse')
    sys.path.insert(0,str(folder/'bool'))
    native=load_module(folder/'bool/native.py','native')
    original_adapter=sys.modules['owner_adapter']
    original_make=original_adapter.make_observer
    sfroot=Path(config['owner_directory'])
    read_pin(sfroot/'native_support.py',SUPPORT_SHA)
    read_pin(sfroot/'owner_adapter.py',OWNER_SHA)
    sf_support=load_module(sfroot/'native_support.py','native_support')
    sf_adapter=load_module(sfroot/'owner_adapter.py','additional_f32_owner_adapter')

    def observer_hook(repository, inputs, source_sha, image_sha):
        require(source_sha == SOURCE and image_sha == IMAGE,'fixed701 observer identity')
        for relative,row in proof['files'].items():inputs.bind(folder/relative,row['derived_sha256'])
        inputs.bind(sfroot/'native_support.py',SUPPORT_SHA)
        inputs.bind(sfroot/'owner_adapter.py',OWNER_SHA)
        source_inverse=sf_support.source_inverse(inputs.bind(Path(config['build_directory'])/'runtime.hepta',SOURCE),inputs,Path(config['build_directory']))
        observer,owner,old_proof=original_make(repository,inputs,source_sha,image_sha)
        unused_observer,new_owner,new_proof=sf_adapter.make_observer(repository,inputs,source_sha,image_sha)
        previous_metadata=owner._command_metadata
        additional_metadata=new_owner._command_metadata

        def metadata(before,after,command,record,body):
            if command[0] == 115 and command[5] == 0:
                source=owner.public_node(before,command[1])
                if owner.w(before['nodes'][str(source)],576) in (1,2):
                    return additional_metadata(before,after,command,record,body)
            return previous_metadata(before,after,command,record,body)

        owner._command_metadata=metadata
        return observer,owner,dict(original_Bool_owner=old_proof,additional_F32_owner=new_proof,
            source_inverse=source_inverse,identity_source_proof=proof,
            original_Bool_and_integral_predicates_unchanged=True,
            added_route='command115 target F32 and source Bool/I64 only',
            original_frozen_control_expectations_unchanged=True)

    native.make_observer=observer_hook
    build=Path(config['build_directory']);output=Path(config['original_output'])
    if config['suite']=='bool27':
        argv=['--repository',str(root),'--binary',str(build/'heptatorch'),'--source',str(build/'runtime.hepta'),
              '--build-report',str(build/'build.json'),'--reference',config['bool_reference'],
              '--expected-source-sha256',SOURCE,'--expected-native-sha256',IMAGE,
              '--expected-build-sha256',config['expected_build_sha256'],
              '--expected-reference-sha256','21dd75aeb86414818f7aa381f8b866d78b16f9bb3f5738e9a60d27ac3d15dd28',
              '--output',str(output)]
        main=native.main
    else:
        integral=load_module(folder/'integral24.py','original_integral24_701_identity')
        integral.ROOT=root
        integral.ReferencePaths=explicit_native_package_paths(integral.ReferencePaths,integral.PACKAGE_SHA)

        def bool_support(inputs):
            require(integral.BOOL_SUPPORT == BOOL_PINS,'all original Bool support pins retained')
            for name,pin in BOOL_PINS.items():inputs.bind(Path(proof['files']['bool/'+name]['original_path']),pin)
            for relative,row in proof['files'].items():inputs.bind(folder/relative,row['derived_sha256'])
            return native,types.SimpleNamespace(make_observer=observer_hook)

        integral.bool_support=bool_support
        refs=Path(config['references_directory'])
        argv=['--build-directory',str(build),'--expected-build-sha256',config['expected_build_sha256'],
            '--reference',str(refs/'reference24/actual.json'),
            '--reference24-source',str(refs/'reference24-source'),
            '--reference32-source',str(refs/'reference32-source'),
            '--integral-source',str(refs/'integral-source'),
            '--parent-native-package',str(refs/'parent-native32-source'),
            '--pip-report',str(refs/'reference-environment/build/pytorch-install.json'),
            '--upstream-lock',str(refs/'reference-environment/compatibility/upstream-lock.json'),
            '--output',str(output)]
        main=integral.main
    if config['prepare_only']:argv.append('--prepare-only')
    sys.argv=[str(folder/('bool/native.py' if config['suite']=='bool27' else 'integral24.py'))]+argv
    return main()
