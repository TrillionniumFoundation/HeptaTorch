#!/usr/bin/env python3
"""Execute the original55 F64 comparisons on a current-object native consumer.

Strict PyTorch mode executes the original eleven installation checks. There is
no local-provenance override. A recorded strict sqrt mismatch remains FAIL.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import runpy
import signal
import struct
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import execute, dump, identity, require, sha
from support import admit, accuracy, comparisons, diagnostic, original


def scripts():
    root = Path(__file__).resolve().parent
    return {name: identity(root/name) for name in ('build.py', 'support.py', 'run.py', 'qualify.py')}


def worker(args):
    repo, current, consumer, out = (p.resolve() for p in (args.repository, args.current_build, args.consumer_build, args.output))
    out.mkdir(parents=True, exist_ok=False)
    before = scripts()
    built = admit(repo, current, consumer)
    context, _, _ = original(repo)
    torch, provenance = None, None
    custody, wheel_state, reference_receipt = None, None, None
    if args.reference == 'pytorch':
        require(args.cpython_directory is not None, 'strict PyTorch needs the actual CPython predecessor execution')
        helper = repo/'tests/pinned_reference_custody.py'
        spec = importlib.util.spec_from_file_location('current_f64_fixed_reference_custody', helper)
        custody = importlib.util.module_from_spec(spec)
        exec(compile(helper.read_bytes(), str(helper), 'exec'), custody.__dict__)
        capsule = out/'reference-custody'; capsule.mkdir()
        capture = custody.begin_torch_capture(capsule)
        support = repo/'tests/pytorch_oracle.py'
        require(sha(support.read_bytes()) == context['REFERENCE_SHA'], 'original locked reference support')
        previous = sys.argv
        try:
            sys.argv = [str(support), str(consumer/'f64-native'), str(out/'reference-provenance.json'),
                        str(repo/'compatibility/pytorch-oracle-lock.json')]
            loaded = runpy.run_path(str(support), run_name='current_object_f64_original_reference')
        finally:
            sys.argv = previous
            custody.finish_torch_capture(capture)
        torch, provenance = loaded['torch'], loaded['PROVENANCE']
        require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
                and all(v is True for v in provenance['checks'].values()), 'all original reference checks')
        reference_receipt = custody.record_reference(repo, capsule, torch, provenance, False)
        require(all(row['bytes'] == 0 for row in reference_receipt['import_streams'].values()), 'silent actual Torch import')
        wheel_state = custody.capture_wheel_entities(repo, capsule, reference_receipt['actual_process'])
    reference_inputs = {'python': identity(Path(sys.executable).resolve()),
                        'python_version': sys.version, 'python_implementation': sys.implementation.name,
                        'original_oracle': identity(repo/'tests/f64_oracle.py'),
                        'reference_support': identity(repo/'tests/pytorch_oracle.py'),
                        'reference_mode': args.reference, 'reference_pid': os.getpid()}
    if custody is not None:
        reference_inputs['custody_helper'] = identity(repo/'tests/pinned_reference_custody.py')
    dump(out/'reference-inputs.json', reference_inputs)
    native_rows, references = [], []
    stage = 'strict55'

    def invoke(raw):
        name = 'native-'+str(len(native_rows)).zfill(4)
        response, receipt = execute(consumer/'f64-native', raw, out, name, timeout=30)
        native_rows.append(receipt)
        return response

    def reference(mode, pairs):
        if args.reference == 'cpython':
            values = context['host_expected'](mode, pairs)
        elif not pairs:
            values = []
        elif mode == 5:
            raw = b''.join(struct.pack('<I', x) for x, _ in pairs)
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).to(torch.float64)
            values = [v & 0xffffffffffffffff for v in tensor.view(torch.int64).tolist()]
        else:
            left = torch.frombuffer(bytearray(b''.join(struct.pack('<Q', x) for x, _ in pairs)), dtype=torch.float64)
            if mode < 4:
                right = torch.frombuffer(bytearray(b''.join(struct.pack('<Q', y) for _, y in pairs)), dtype=torch.float64)
                tensor = (torch.add, torch.sub, torch.mul, torch.div)[mode](left, right)
            elif mode == 4:
                tensor = torch.sqrt(left)
            else:
                tensor = left.to(torch.float32)
            values = [v & (0xffffffff if mode == 6 else 0xffffffffffffffff)
                      for v in tensor.view(torch.int32 if mode == 6 else torch.int64).tolist()]
        require(native_rows, 'reference follows actual native request')
        expected_request = struct.pack('<II', mode, len(pairs))+b''.join(
            struct.pack('<QQ', x, y) if mode < 4 else struct.pack('<Q', x) for x, y in pairs)
        native_name = native_rows[-1]['name'] if native_rows[-1]['request_sha256'] == sha(expected_request) else None
        require(native_name is not None or stage == 'sqrt_diagnostic', 'actual reference corresponds to actual native request')
        packed = b''.join(struct.pack('<Q', v) for v in values)
        name = 'reference-'+str(len(references)).zfill(4)
        (out/(name+'.bin')).write_bytes(packed)
        references.append({'name': name, 'mode': mode, 'values': len(values),
                           'stage': stage, 'native_name': native_name, 'request_sha256': sha(expected_request),
                           'bytes': len(packed), 'sha256': sha(packed), 'actual_reference_pid': os.getpid()})
        return values

    rows, proof = comparisons(repo, invoke, reference)
    stage = 'sqrt_diagnostic'
    diagnostic_result = diagnostic(repo, consumer/'f64-native', invoke, reference)
    require(diagnostic_result['status'] != 'FAIL', 'original rational sqrt diagnostic failed')
    cp_report = None
    if args.reference == 'pytorch':
        cp_path = args.cpython_directory.resolve()/'execution/report.json'
        cp_report = json.loads(cp_path.read_bytes())
        require(cp_report['reference'] == 'cpython' and cp_report['status'] == 'PASS'
                and cp_report['native_sha256'] == built['native_sha256']
                and cp_report['consumer_build_sha256'] == sha((consumer/'build.json').read_bytes()), 'same actual native CPython precursor')
    stage = 'rne55'
    accuracy_result = accuracy(repo, rows, cp_report['rows'] if cp_report else rows, invoke, reference)
    if custody is not None:
        custody.finish_wheel_entities(wheel_state, custody.actual_process(Path(__file__).resolve()))
    require(len(native_rows) == len({r['pid'] for r in native_rows}), 'distinct native process per invocation')
    require(scripts() == before and identity(Path(sys.executable).resolve()) == reference_inputs['python'], 'reference implementation unchanged')
    admitted_after = admit(repo, current, consumer)
    require(admitted_after == built, 'current consumer and all original object inputs unchanged')
    result = {'schema': 'heptatorch.current-object-f64-run.v1',
              'status': 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
              'reference': args.reference, 'oracle_provenance': provenance,
              'reference_inputs': reference_inputs, 'implementation_inputs': before,
              'current_source_sha256': built['current_source_sha256'], 'native_sha256': built['native_sha256'],
              'consumer_build_sha256': sha((consumer/'build.json').read_bytes()), 'original_comparison': proof,
              'cases': len(rows), 'failed_cases': sum(r['status'] != 'PASS' for r in rows),
              'exact_non_nan': sum(r.get('exact_non_nan', 0) for r in rows),
              'nan_classification': sum(r.get('nan_classification', 0) for r in rows),
              'rows': rows, 'native_processes': native_rows, 'references': references,
              'sqrt_diagnostic': diagnostic_result, 'rne_accuracy': accuracy_result,
              'cpython_directory': str(args.cpython_directory.resolve()) if args.cpython_directory else None,
              'cpython_report_sha256': sha(cp_path.read_bytes()) if cp_report else None,
              'reference_custody': sha((out/'reference-custody/reference-inputs.json').read_bytes()) if custody is not None else None,
              'strict_failures_reclassified': False, 'legacy143_restored': False,
              'sqrt_public_tensor_support': False, 'whole_pytorch_rewrite_complete': False}
    dump(out/'report.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'reference', 'cases', 'failed_cases', 'exact_non_nan', 'nan_classification')}))
    return 0 if result['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--current-build', type=Path, required=True)
    parser.add_argument('--consumer-build', type=Path, required=True)
    parser.add_argument('--reference', choices=('cpython', 'pytorch'), required=True)
    parser.add_argument('--reference-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--cpython-directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    require(__debug__, 'original assertions required')
    if args.worker:
        return worker(args)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    # Preserve the selected venv executable path in argv; resolve only the byte
    # identity. Launching a venv symlink's base executable would lose its site.
    interpreter = args.reference_python.absolute()
    interpreter_image = interpreter.resolve(strict=True)
    implementation = Path(__file__).resolve()
    argv = [str(interpreter), '-B', str(implementation), '--worker',
            '--repository', str(args.repository.resolve()), '--current-build', str(args.current_build.resolve()),
            '--consumer-build', str(args.consumer_build.resolve()), '--reference', args.reference,
            '--output', str(out/'execution')]
    if args.cpython_directory:
        argv += ['--cpython-directory', str(args.cpython_directory.resolve())]
    before = scripts()
    intent = {'argv': argv, 'cwd': str(args.repository.resolve()), 'environment': dict(os.environ),
              'interpreter': identity(interpreter_image), 'interpreter_realpath': str(interpreter_image),
              'implementation': before,
              'supervisor_pid': os.getpid(), 'timeout_seconds': 900, 'started': False}
    # Preserve only environment names plus values that affect this numerical/reference process.
    environment = {'PATH': '/bin:/usr/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
                   'TORCHINDUCTOR_CACHE_DIR': str(out/'execution/reference-custody/torchinductor-cache')}
    intent['environment'] = environment
    dump(out/'reference-process.intent.json', intent)
    process = subprocess.Popen(argv, cwd=args.repository.resolve(), env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    started = time.monotonic(); timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        timed_out = True; os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    (out/'reference-process.stdout').write_bytes(stdout)
    (out/'reference-process.stderr').write_bytes(stderr)
    receipt = dict(intent, started=True, pid=process.pid, exit_code=process.returncode, timed_out=timed_out,
                   stdout_EOF=True, stderr_EOF=True, seconds=time.monotonic()-started,
                   stdout_bytes=len(stdout), stdout_sha256=sha(stdout), stderr_bytes=len(stderr), stderr_sha256=sha(stderr))
    dump(out/'reference-process.json', receipt)
    require(scripts() == before and interpreter.resolve(strict=True) == interpreter_image
            and identity(interpreter_image) == intent['interpreter'], 'supervised implementation changed')
    sys.stdout.buffer.write(stdout); sys.stderr.buffer.write(stderr)
    require(not timed_out, 'reference process timeout')
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
