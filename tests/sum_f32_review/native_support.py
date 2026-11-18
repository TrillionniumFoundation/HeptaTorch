"""Generator712 admission around complete, byte-pinned original701 proofs.

All numerical/reference/owner predicates remain in their original sources.
Only the source census and observer profile identity follow the actual712 image.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import types

HERE = Path(__file__).resolve().parent
FROZEN_SUPPORT_SHA = '5e0a66b1a45c469f6c80ec611ff49d26288230b736b6939f957cba8a7ad21ec1'
FROZEN_NATIVE_SHA = '8599ede3bace8c41a0e62e6672b03b2b885247dcc099defb2227a45869432252'
FROZEN_OWNER_SHA = '4ef9545bb42dbc1ef5da603814a157163bed18418992d9b2489fd39378c3a4aa'
DELTA_SHA = '5fd739f9d58e1a544600c75a7bb9b748bc313523a773db9895337eb8dc51bf7a'
GENERATOR_PROOF_SHA = 'd8522ff168e25bd4730f33fe5449aee687254c0baae0248959e976ae56d2a21f'


def _fixed_bytes(path, pin):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError('regular frozen proof source required')
    raw = path.read_bytes(); after = path.stat()
    fields = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_mtime_ns,s.st_ctime_ns)
    if fields(before) != fields(after) or hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError('frozen proof source identity changed')
    return raw


_FROZEN_RAW = _fixed_bytes(HERE / 'frozen701/native_support.py', FROZEN_SUPPORT_SHA)
_FROZEN = types.ModuleType('generator712_original701_native_support')
# The frozen proof's two logical filenames are served only through the explicit
# inverse-byte bridge below. Every other path binds a real physical input.
_FROZEN.__file__ = str(HERE / 'native_support.py')
sys.modules[_FROZEN.__name__] = _FROZEN
exec(compile(_FROZEN_RAW, str(HERE / 'frozen701/native_support.py'), 'exec'), _FROZEN.__dict__)
Inputs, dump, require, sha = _FROZEN.Inputs, _FROZEN.dump, _FROZEN.require, _FROZEN.sha
PROTOCOL_SHA, CLIENT_SHA = _FROZEN.PROTOCOL_SHA, _FROZEN.CLIENT_SHA
source_functions, bind_sources = _FROZEN.source_functions, _FROZEN.bind_sources

# Derive the whole original observer-admission function with exactly two
# identity literals. Its complete geometry, source and inverse assertions stay.
_AST = ast.parse(_FROZEN_RAW)
_NODE = next(n for n in _AST.body if isinstance(n,ast.FunctionDef) and n.name == 'adapted_parent')
_ORIGINAL_PARENT = ast.get_source_segment(_FROZEN_RAW.decode(), _NODE).encode()
_DERIVED_PARENT = _ORIGINAL_PARENT
_PARENT_EDITS = ((b'functions == 701', b'functions == 712'),
                 (b'sum-integral-f32-source01-UNBOUND-701', b'sum-integral-f32-source01-UNBOUND-712'))
for _before, _after in _PARENT_EDITS:
    require(_DERIVED_PARENT.count(_before) == 1 and _after not in _DERIVED_PARENT,
            'exact two observer profile identity literals')
    _DERIVED_PARENT = _DERIVED_PARENT.replace(_before,_after,1)
_INVERSE_PARENT = _DERIVED_PARENT
for _before,_after in reversed(_PARENT_EDITS):
    require(_INVERSE_PARENT.count(_after) == 1, 'exact observer identity inverse count')
    _INVERSE_PARENT = _INVERSE_PARENT.replace(_after,_before,1)
require(_INVERSE_PARENT == _ORIGINAL_PARENT, 'whole original observer admission function inverse')
_PARENT_GLOBALS = dict(_FROZEN.__dict__)
exec(compile(_DERIVED_PARENT, str(HERE / 'native_support.py') + '#observer-identity', 'exec'), _PARENT_GLOBALS)
adapted_parent = _PARENT_GLOBALS['adapted_parent']


def source_inverse(source, inputs, build_directory):
    inputs.bind(HERE / 'native_support.py')
    support = inputs.bind(HERE / 'frozen701/native_support.py', FROZEN_SUPPORT_SHA)
    original_native = inputs.bind(HERE / 'frozen701/native.py', FROZEN_NATIVE_SHA)
    original_owner = inputs.bind(HERE / 'frozen701/owner_adapter.py', FROZEN_OWNER_SHA)
    require(support == _FROZEN_RAW, 'frozen support did not change after import')
    delta = json.loads(inputs.bind(HERE / 'generator-reader-delta.json', DELTA_SHA))
    require(delta['schema'] == 'heptatorch.generator712.frozen701-reader-inverse.v1'
            and set(delta['files']) == {'native.py','owner_adapter.py'}
            and delta['original_comparisons_controls_and_reference_sources_unchanged'] is True,
            'complete two-file reader identity delta')
    inverse_files = {}
    proofs = {}
    for name, original in (('native.py',original_native),('owner_adapter.py',original_owner)):
        row = delta['files'][name]
        actual = inputs.bind(HERE / name, row['derived_sha256'])
        require(len(actual) == row['derived_bytes'] and len(original) == row['original_bytes']
                and sha(original) == row['original_sha256']
                and row['comparison_predicate_or_expected_value_edits'] == [],
                'actual and frozen reader byte identities')
        inverse = actual.decode()
        for edit in reversed(row['edits']):
            require(edit['count'] == 1 and inverse.count(edit['after']) == 1,
                    'exact whole-reader inverse count')
            inverse = inverse.replace(edit['after'],edit['before'],1)
        require(inverse.encode() == original and row['whole_source_inverse_exact'] is True,
                'whole original reader and owner source inverse')
        inverse_files[(HERE / name).resolve()] = original
        proofs[name] = dict(original_sha256=sha(original), actual_sha256=sha(actual),
                           whole_inverse_exact=True, edits=row['edits'],
                           comparison_predicate_or_expected_value_edits=[])
    inverse_files[(HERE / 'native_support.py').resolve()] = support
    # The original owner file is verified above but never substituted into a
    # numerical execution. Only the frozen proof's two filenames can be mapped.
    inverse_files.pop((HERE / 'owner_adapter.py').resolve())
    proof_raw = inputs.bind(HERE / 'generator_profile.py', GENERATOR_PROOF_SHA)
    spec = importlib.util.spec_from_file_location('generator712_source_proof',HERE / 'generator_profile.py')
    generator = importlib.util.module_from_spec(spec)
    exec(compile(proof_raw,str(HERE / 'generator_profile.py'),'exec'),generator.__dict__)
    restored, generator_proof = generator.prove(source,build_directory,HERE,inputs)

    class VerifiedOriginalProofInputs:
        def bind(self, path, expected=None):
            key = Path(path).resolve()
            if key in inverse_files:
                raw = inverse_files[key]
                require(expected is None or sha(raw) == expected,
                        'original proof requested exact independently verified inverse bytes')
                return raw
            return inputs.bind(path,expected)

    original_proof = _FROZEN.source_inverse(restored,VerifiedOriginalProofInputs())
    require(original_proof['source_sha256'] == generator.PARENT_SOURCE
            and original_proof['current_functions'] == 701
            and original_proof['whole_inverse_exact'] is True
            and original_proof['unchanged_functions'] == 696,
            'complete original three-layer701 source and reader proof actually executed')
    return dict(schema='heptatorch.generator712.complete-source-and-reader-proof.v1',
        source_sha256=generator.SOURCE,native_sha256=generator.IMAGE,
        current_functions=712,parent_source_sha256=generator.PARENT_SOURCE,
        whole_inverse_exact=True,generator_to_701=generator_proof,
        original701_source_proof=original_proof,
        actual_reader_identity=proofs,
        original_support_sha256=FROZEN_SUPPORT_SHA,
        observer_admission_whole_function_inverse_exact=True,
        observer_identity_edits=[dict(before=a.decode(),after=b.decode(),count=1) for a,b in _PARENT_EDITS],
        original_physical_files_bound_before_inverse=True,
        original_proof_inverse_file_map=[str(p) for p in sorted(inverse_files)],
        physical_input_records_modified=False,
        all_original_numerical_comparisons_and_controls_unchanged=True,
        original_failure_reports_modified=False,runtime_qualification_transferred=False)
