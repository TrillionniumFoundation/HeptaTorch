"""Bind original-gate view successors to the shared strict188 reference/build."""
import importlib
import importlib.util
from pathlib import Path
import sys
import types

from .legacy_successors import Successor, EntryOverlay, require


class StrictInputs:
    """Keep both original inventories and their complete recheck predicates.

    The original87 Inputs raises on changed bytes and returns True from
    unchanged(); original192 recheck() returns an error list. The shared view
    reader needs an exception on any failed recheck, never a truthy error list.
    """
    def __init__(self, original):
        self.original = original
        self.bind = original.bind

    def recheck(self):
        if hasattr(self.original, 'unchanged'):
            require(self.original.unchanged() is True,
                    'complete original87 input bytes, extent, mode and identity remain unchanged')
        else:
            result = self.original.recheck()
            require(result is None or result == [],
                    'complete original192/shared input recheck has no errors')


def strict_successor(repository, build, build_sha, reference, execution, inputs):
    repository, build = Path(repository).resolve(strict=True), Path(build).resolve(strict=True)
    profile_path = repository / 'tools/current_runtime/qualification.py'
    profile_raw = inputs.bind(profile_path)
    spec = importlib.util.spec_from_file_location('legacy_view_complete_current_build', profile_path)
    profile = importlib.util.module_from_spec(spec)
    exec(compile(profile_raw, str(profile_path), 'exec', dont_inherit=True, optimize=0), profile.__dict__)
    built = profile.admit_build(build, inputs.bind, build_sha)
    for path in sorted((repository / 'tests/f64_view').glob('*.py')):
        inputs.bind(path)
    evidence = importlib.import_module('f64_view.evidence')
    require(Path(evidence.__file__).resolve() == repository / 'tests/f64_view/evidence.py',
            'shared strict evidence reader belongs to the actual repository')
    observed = evidence.read_reference(Path(reference), Path(execution), StrictInputs(inputs))
    require(observed.get('strict_reference_qualified') is True,
            'strict raw wheel/install/interpreter/intent reference custody is mandatory')
    result = Successor(observed, built, inputs, Path(__file__).with_name('legacy_successors.py'))
    result.proof.update(actual_build_sha256=build_sha, reference_directory=str(Path(reference).resolve(strict=True)),
                        reference_execution=str(Path(execution).resolve(strict=True)))
    return result


def from_arguments(args, inputs):
    return strict_successor(args.repository, args.build_directory, args.expected_build_sha256,
                            args.f64_view_reference, args.f64_view_reference_execution, inputs)


def adapter_type(original_type, repository, reader_directory, binding):
    """Compose with existing arithmetic/negation classes without changing them."""
    class Adapter(original_type):
        def bind_package(self, inputs):
            digest = super().bind_package(inputs)
            inputs.bind(Path(__file__).resolve())
            self.public_f64_view_successor = strict_successor(repository, binding['build'], binding['build_sha'],
                binding['root'], binding['execution'], inputs)
            self.public_f64_arithmetic_successor = EntryOverlay(self.public_f64_arithmetic_successor,
                                                               self.public_f64_view_successor)
            active = types.ModuleType('current_f64_explicit_view_successor_controls')
            active.__dict__.update(self.current_controls.__dict__)
            active.entry_controls = self.public_f64_arithmetic_successor.wrap_entry(self.current_controls.entry_controls)
            sys.modules['controls'] = active
            return digest

        def candidate_admission(self, directory, inputs):
            result = super().candidate_admission(directory, inputs)
            result.update(public_f64_view_successor=self.public_f64_view_successor.proof,
                arithmetic24_is_preserved_predecessor_scoped_contract=True,
                retained_refusals=61, actual_numeric_successors=24, actual_view_successors=2,
                unchanged_other_refusal_expectations=59)
            return result
    return Adapter
