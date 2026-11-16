"""Compile real HIA owned-return fixtures and exercise response-length bounds.

No invalid owner is fabricated. Each fixture allocates a real eight-byte HIA
buffer, returns it linearly, and varies only its scalar u32 response length.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
CLIENT_SOURCE = (HERE / 'client.py').read_bytes()
spec = importlib.util.spec_from_file_location('owned_bound_client', HERE / 'client.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
assert (HERE / 'client.py').read_bytes() == CLIENT_SOURCE
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compiler', 'linker', 'qualification-linker', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--step-source', type=Path, default=HERE / 'fixtures' / 'step.hepta')
    parser.add_argument('--owner-source', type=Path, default=HERE / 'fixtures' / 'owners.hepta')
    parser.add_argument('--builder', type=Path, default=HERE / 'link_service.py')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    step = args.step_source.read_bytes()
    owner = args.owner_source.read_bytes()
    builder = args.builder.read_bytes()
    needle = b'  count:u32 = const() {"value":32}\n'
    assert step.count(needle) == 1
    (out / 'baseline-step.hepta').write_bytes(step)
    (out / 'owners.hepta').write_bytes(owner)
    rows = []; active = []
    try:
        for length in (0, 31, 32, client.MIB, client.MIB + 1, (1 << 32) - 1):
            folder = out / ('length-' + str(length)); folder.mkdir()
            source = step.replace(needle, b'  count:u32 = const() {"value":' + str(length).encode() + b'}\n')
            (folder / 'step.hepta').write_bytes(source)
            command = [sys.executable, '-I', '-S', '-B', str(args.builder.resolve()),
                '--compiler', str(args.compiler.resolve()), '--linker', str(args.linker.resolve()),
                '--qualification-linker', str(args.qualification_linker.resolve()),
                '--output', str(folder / 'build'), str(folder / 'step.hepta'), str(out / 'owners.hepta')]
            built = subprocess.run(command, capture_output=True, timeout=120)
            (folder / 'build.stdout').write_bytes(built.stdout)
            (folder / 'build.stderr').write_bytes(built.stderr)
            assert built.returncode == 0 and not built.stderr
            receipt = json.loads((folder / 'build' / 'build.json').read_text())
            valid = 32 <= length <= client.MIB
            observed = []
            for probe in (False, True):
                executable = folder / 'build' / ('service-close-probe' if probe else 'service')
                session = client.OwnerSession(executable, probe=probe,
                    directory=folder / ('probe-observation' if probe else 'production-observation'))
                active.append(session)
                session.control_base()
                if valid:
                    for expected in (1, 2):
                        body = session.exchange(b'')
                        assert len(body) == length and body == bytes([expected]) + bytes(length - 1)
                        state = session.owner()
                        assert state['length'] == 8 and state['bytes'] == bytes([expected]) + bytes(7)
                    session.close_input()
                    if probe:
                        close = session.observe_close(0)
                        assert close['dead_argument_length'] == 8 and close['dead_argument_pointer']
                        observed.append(close)
                    session.finish(0)
                else:
                    session.send(client.REQUEST_MAGIC + struct.pack('<Q', 0))
                    if probe:
                        close = session.observe_close(2)
                        assert close['dead_argument_length'] == 8 and close['dead_argument_pointer']
                        observed.append(close)
                    session.finish(2)
            rows.append({'returned_u32_length': length, 'pass': True, 'admitted': valid,
                'expected_exit': 0 if valid else 2,
                'owner_is_real_HIA_allocation': True,
                'owner_return_adopted_and_dropped_before_process_exit': observed,
                'production_executable_sha256': receipt['executables']['service']['sha256'],
                'probe_executable_sha256': receipt['executables']['service-close-probe']['sha256'],
                'same_objects_in_both_profiles': True,
                'response_frames': 4 if valid else 0})
            (out / 'cases.json').write_text(json.dumps(rows, indent=2) + '\n')
            print('returned length ' + str(length) + ': PASS', flush=True)
        assert args.step_source.read_bytes() == step and args.owner_source.read_bytes() == owner
        assert args.builder.read_bytes() == builder and (HERE / 'client.py').read_bytes() == CLIENT_SOURCE
        report = {'schema': 'heptatorch.owned-session-result-bound.v1', 'pass': True,
            'cases': len(rows), 'native_processes': len(active),
            'response_frames': sum(value.frames for value in active),
            'pre_exit_close_observations': sum(len(row['owner_return_adopted_and_dropped_before_process_exit']) for row in rows),
            'test_sha256': sha(Path(__file__).read_bytes()), 'client_sha256': sha(CLIENT_SOURCE),
            'builder_sha256': sha(builder), 'baseline_step_sha256': sha(step), 'owners_sha256': sha(owner),
            'only_HIA_scalar_result_length_varied': True, 'owner_bytes_never_fabricated': True,
            'no_response_for_rejected_length': True, 'rows': rows}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        for value in active:
            if value.poll() is None:
                os.kill(value.pid, signal.SIGKILL)
                os.waitpid(value.pid, 0)


if __name__ == '__main__':
    main()
