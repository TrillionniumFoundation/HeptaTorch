"""Run the frozen native-only producer and verify its exact profile identity."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    profile = json.loads((HERE/'profile.json').read_bytes())
    for name, expected in profile['authoring_inputs_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected, name
    if '--compactor' not in sys.argv:
        sys.argv.extend(['--compactor', str(HERE/'compact_functions.py')])
    from build_effective_momentum import main as produce
    produce()
    output = Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    actual = json.loads((output/'build.json').read_bytes())
    for name in ('source_sha256', 'native_sha256', 'compiler_sha256', 'linker_sha256',
                 'functions', 'blocks', 'instructions', 'descriptor_bytes',
                 'state_format', 'backing_format', 'units', 'authoring_inputs_sha256'):
        assert actual[name] == profile[name], ('profile differs', name)
    assert actual['status'] == 'PASS'
    assert hashlib.sha256((output/'runtime.hepta').read_bytes()).hexdigest() == profile['source_sha256']
    assert hashlib.sha256((output/'tensor-dtype-dropout-session').read_bytes()).hexdigest() == profile['native_sha256']
    receipt = {'schema':'heptatorch.native-dropout-profile-rebuild.v1', 'status':'PASS',
               'source_sha256':profile['source_sha256'], 'native_sha256':profile['native_sha256'],
               'units_reproduced_exactly':len(actual['units']),
               'runtime_execution_claimed_by_build':False,
               'full_pytorch_rewrite_complete':False, 'pytorch_api_denominator':2590}
    (output/'profile-rebuild.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
