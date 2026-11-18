#!/usr/bin/env python3
"""Observe fixed raw F16/BF16 logical inputs with the pinned actual Torch wheel.

Only Torch supplies result bytes. No HIA/native image or child process is run.
The independent launcher supplies a pre-execution Python/argv intent and keeps
the physical stdout, stderr, exit and EOF receipt.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import runpy
import stat
import sys
import traceback


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SUPPORT = {
    'tests/pytorch_oracle.py': 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae',
    'compatibility/pytorch-oracle-lock.json': 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944',
    'compatibility/upstream-lock.json': 'd204c5a8268a0d0f0a1dd162b5e0ab184beee83310bcad643b084deb691f3f49',
}
ATTEMPTS = []


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique)


class Inputs:
    def __init__(self):
        self.rows = {}

    def bind(self, path, expected=None, size=None, mode=None):
        path = Path(path).absolute()
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'ordinary unique input: ' + str(path))
        raw = path.read_bytes()
        after = path.lstat()
        require(all(getattr(before, key) == getattr(after, key) for key in
            ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'input changed while reading')
        row = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row['sha256'] == expected, 'input SHA differs: ' + str(path))
        require(size is None or row['bytes'] == size, 'input extent differs: ' + str(path))
        require(mode is None or row['mode'] == mode, 'input mode differs: ' + str(path))
        require(str(path) not in self.rows or self.rows[str(path)] == row, 'bound input changed')
        self.rows[str(path)] = row
        return raw

    def recheck(self):
        for path, row in list(self.rows.items()):
            self.bind(path, row['sha256'], row['bytes'], row['mode'])


def safe_member(name):
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and '..' not in path.parts and '\\' not in name
            and str(path) == name, 'canonical producer member name')
    return path


def verify_package(root, expected, inputs):
    root = Path(root).resolve(strict=True)
    manifest = parse(inputs.bind(root / 'manifest.json', expected))
    require(manifest['schema'] == 'heptatorch.half-logical.reference-producer.v1'
            and manifest['expected_outputs_included'] is False
            and manifest['expected_observations'] == 264, 'fixed input-only producer package')
    names = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    require(names == set(manifest['files']) | {'manifest.json'}, 'complete producer file census')
    require(all(not path.is_symlink() and (path.is_dir() or path.is_file()) for path in root.rglob('*')),
            'ordinary producer files and directories')
    for name, row in manifest['files'].items():
        inputs.bind(root.joinpath(*safe_member(name).parts), row['sha256'], row['bytes'], row['mode'])
    cases = parse(inputs.bind(root / 'cases.json', manifest['cases_sha256']))
    require(cases['schema'] == 'heptatorch.half-logical.literal-inputs-and-calls.v1'
            and cases['expected_observations'] == len(cases['cases']) == 264
            and cases['expected_outputs_included'] is False
            and cases['requires_grad_or_general_views_in_compatibility_scope'] is False,
            'exact literal call census and explicit no-grad dense scope')
    require(len({row['name'] for row in cases['cases']}) == 264, 'unique reference case names')
    for name, row in cases['input_blobs'].items():
        require(name.startswith('inputs/') and manifest['files'][name] == row, 'all fixed raw inputs are manifest-bound')
    for spec in cases['cases']:
        item = spec['input']
        require(spec['native_dtype'] in (5, 6) and spec['requires_grad'] is False
                and spec['torch_dtype'] == ('torch.float16' if spec['native_dtype'] == 5 else 'torch.bfloat16')
                and type(spec['keepdim']) is bool
                and spec['host_expected_output_present'] is False
                and spec['numel'] == math.prod(spec['shape']) and len(spec['shape']) <= 4
                and all(type(size) is int and size >= 0 for size in spec['shape'])
                and spec['operation'] in ('any', 'all', 'sum_bool')
                and spec['opcode'] == {'any':118, 'all':119, 'sum_bool':115}[spec['operation']],
                'literal native dtype, operation and shape')
        require(item['path'] in cases['input_blobs']
                and {key: cases['input_blobs'][item['path']][key] for key in ('bytes','sha256')} ==
                    {key: item[key] for key in ('bytes','sha256')}
                and item['bytes'] == 2 * spec['numel'], 'exact physical u16 input extent and identity')
    return manifest, cases


class Blobs:
    def __init__(self, root):
        self.root, self.rows = root, {}
        (root / 'blobs').mkdir()

    def add(self, raw):
        digest = sha(raw)
        name = 'blobs/' + digest + '.bin'
        path = self.root / name
        if name not in self.rows:
            with path.open('xb') as stream:
                stream.write(raw)
            path.chmod(0o644)
            self.rows[name] = dict(path=name, bytes=len(raw), sha256=digest, mode=0o644)
        require(path.read_bytes() == raw, 'complete observed blob identity')
        return self.rows[name]

    def recheck(self):
        for name, row in self.rows.items():
            path = self.root / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
                    and path.stat().st_size == row['bytes'] and sha(path.read_bytes()) == row['sha256']
                    and stat.S_IMODE(path.stat().st_mode) == row['mode'], 'observed blob changed')


def snapshot(torch, value, blobs):
    logical = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
    storage = value.untyped_storage()
    return dict(dtype=str(value.dtype), shape=list(value.shape), strides=list(value.stride()),
        numel=value.numel(), element_size=value.element_size(), storage_bytes=storage.nbytes(),
        storage_offset=value.storage_offset(), version=value._version, requires_grad=value.requires_grad,
        is_leaf=value.is_leaf, is_view=value._is_view(), retains_grad=value.retains_grad,
        grad_fn=None if value.grad_fn is None else type(value.grad_fn).__name__, grad_is_none=value.grad is None,
        tensor_identity=int(value._cdata), storage_identity=int(storage._cdata),
        base_tensor_identity=None if value._base is None else int(value._base._cdata),
        logical=blobs.add(logical), storage=blobs.add(bytes(storage)))


def dense(torch, spec, raw):
    dtype = torch.float16 if spec['native_dtype'] == 5 else torch.bfloat16
    if raw:
        result = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(spec['shape']).clone()
    else:
        result = torch.empty(spec['shape'], dtype=dtype)
    require(result.dtype == dtype and list(result.shape) == spec['shape'] and not result.requires_grad
            and result.is_leaf and not result._is_view() and result.grad_fn is None and result._version == 0,
            'ordinary target-shaped owned no-grad Half Tensor')
    require(bytes(result.detach().reshape(-1).view(torch.uint8).tolist()) == raw,
            'every supplied half bit pattern preserved by construction')
    return result


def operation(torch, source, spec):
    if spec['operation'] == 'sum_bool':
        return torch.ops.aten.sum.dim_IntList(source, spec['axes'], keepdim=spec['keepdim'], dtype=torch.bool)
    return getattr(torch.ops.aten, spec['operation']).dims(source, spec['axes'], keepdim=spec['keepdim'])


def observe(torch, spec, inputs, blobs):
    raw = inputs.bind(HERE / spec['input']['path'], spec['input']['sha256'], spec['input']['bytes'], 0o644)
    rng_before = bytes(torch.get_rng_state().tolist())
    source = dense(torch, spec, raw)
    before = snapshot(torch, source, blobs)
    require(before['logical']['sha256'] == before['storage']['sha256'] == spec['input']['sha256'],
            'Torch actual input and complete storage equal the literal input')
    try:
        result = operation(torch, source, spec)
    except (IndexError, RuntimeError) as error:
        row = dict(accepted=False, result=None, error=dict(type=type(error).__name__, message=str(error)))
    else:
        row = dict(accepted=True, result=snapshot(torch, result, blobs), error=None,
            fresh_tensor=result._cdata != source._cdata,
            fresh_storage=result.untyped_storage()._cdata != source.untyped_storage()._cdata)
    after = snapshot(torch, source, blobs)
    rng_after = bytes(torch.get_rng_state().tolist())
    require(before == after and rng_before == rng_after, 'input and complete RNG unchanged across actual operation')
    return dict(spec=spec, input=before, input_after=after, input_unchanged=True, rng_unchanged=True,
                rng=blobs.add(rng_before), **row)


def forbid_children(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec', 'os.fork', 'os.forkpty'}:
        ATTEMPTS.append(event)
        raise RuntimeError('reference-only observation forbids child process: ' + event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-producer-sha256', required=True)
    parser.add_argument('--execution-intent', type=Path, required=True)
    args = parser.parse_args()
    require(not sys.flags.optimize, 'reference assertions must remain enabled')
    repo, out = args.repository.resolve(strict=True), args.output.absolute()
    require(Path.cwd().resolve() == repo and not out.exists(), 'fixed repository cwd and fresh output')
    out.mkdir(parents=True, exist_ok=False)
    inputs, blobs = Inputs(), Blobs(out)
    report = dict(schema='heptatorch.half-logical.fixed-Torch-reference.v1', status='RUNNING',
        rows=[], failures=[], inputs=inputs.rows, expected_observations=264,
        Torch_executed=False, native_executed=False, compiler_executed=False,
        original47_original12_or_previous_reference_modified=False,
        host_expected_values_computed=False, full_project_status='HOLD')
    code = 1
    try:
        package, cases = verify_package(HERE, args.expected_producer_sha256, inputs)
        intent = parse(inputs.bind(args.execution_intent))
        python = Path(sys.executable).resolve(strict=True)
        inputs.bind(python)
        require(set(intent) == {'argv','observer_python'} and intent['argv'] == sys.orig_argv
                and intent['observer_python'] == dict(path=str(python), **inputs.rows[str(python)]),
                'exact pre-execution interpreter entity and real original argv')
        require(Path('/proc/self/exe').resolve() == python, 'running process executable matches prebound Python')
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        report['actual_process'] = dict(pid=os.getpid(), parent_pid=os.getppid(), start_ticks=int(fields[19]),
            argv=list(sys.orig_argv), python=intent['observer_python'], requested_python=sys.executable,
            intent_path=str(args.execution_intent.absolute()),
            intent_sha256=inputs.rows[str(args.execution_intent.absolute())]['sha256'])
        for name, expected in SUPPORT.items():
            inputs.bind(repo / name, expected)
        inputs.bind(repo / 'build/pytorch-install.json')
        report.update(producer_manifest_sha256=args.expected_producer_sha256,
                      case_sha256=package['cases_sha256'], exhaustive_encoding_observations=6, matrix_observations=258)
        sys.addaudithook(forbid_children)
        old_argv = sys.argv
        try:
            sys.argv = [str(repo / 'tests/pytorch_oracle.py'), str(out / 'NO_NATIVE_EXECUTION'),
                        str(out / 'oracle-provenance-failure.json'), str(repo / 'compatibility/pytorch-oracle-lock.json')]
            context = runpy.run_path(str(repo / 'tests/pytorch_oracle.py'), run_name='half_logical_actual_reference_only')
        finally:
            sys.argv = old_argv
        torch, provenance = context['torch'], context['PROVENANCE']
        require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
                and all(value is True for value in provenance['checks'].values())
                and str(torch.__version__) == '2.15.0+cpu'
                and torch.version.git_version == 'd0f2af8b66b20da4baa50644c0cd3f887a2eae2e'
                and torch.get_num_threads() == 1, 'all11 fixed-wheel provenance checks and original thread policy')
        for path in (Path(torch.__file__), Path(torch._C.__file__)):
            inputs.bind(path.resolve(strict=True))
        distribution = importlib.metadata.distribution('torch')
        for relative in distribution.files or []:
            if relative.name in ('direct_url.json', 'METADATA') and '.dist-info/' in str(relative):
                inputs.bind(Path(distribution.locate_file(relative)).resolve(strict=True))
        report.update(Torch_executed=True, oracle_provenance=provenance,
            torch_version=str(torch.__version__), torch_git_version=torch.version.git_version,
            num_threads=torch.get_num_threads(), aten_cpu_capability_override=os.environ.get('ATEN_CPU_CAPABILITY'))
        rng = bytes(torch.get_rng_state().tolist())
        require(len(rng) == 5056, 'complete actual CPU RNG state')
        report['rng'] = blobs.add(rng)
        for index, spec in enumerate(cases['cases']):
            row = dict(index=index, **observe(torch, spec, inputs, blobs))
            with (out / ('case-%03d.json' % index)).open('x') as stream:
                stream.write(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + '\n')
            report['rows'].append(row)
        require(len(report['rows']) == 264 and rng == bytes(torch.get_rng_state().tolist()),
                'complete real call denominator and unchanged final RNG')
        inputs.recheck()
        blobs.recheck()
        require(not ATTEMPTS, 'no native/compiler/other child process attempted')
        report.update(status='OBSERVED_HALF_LOGICAL_REFERENCE', cases=264,
            accepted=sum(row['accepted'] for row in report['rows']),
            observed_errors=sum(not row['accepted'] for row in report['rows']),
            inputs_unchanged=True, rng_unchanged=True, all_observed_blobs_unchanged=True)
        code = 0
    except BaseException as error:
        report.update(status='FAIL', Torch_executed='torch' in sys.modules,
            error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    report.update(blob_manifest=blobs.rows, forbidden_child_process_attempts=ATTEMPTS)
    with (out / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({key:report.get(key) for key in ('status','cases','accepted','observed_errors','error')}, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
