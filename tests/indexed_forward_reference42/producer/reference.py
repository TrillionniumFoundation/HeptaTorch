#!/usr/bin/env python3
"""Observe fixed F32 scatter/index additions and one old cat recipe with the actual pinned Torch wheel.

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
import struct
import warnings


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
    require(manifest['schema'] == 'heptatorch.indexed-forward.reference-producer.v1'
            and manifest['expected_outputs_included'] is False
            and manifest['expected_observations'] == 42, 'fixed input-only indexed producer package')
    names = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    require(names == set(manifest['files']) | {'manifest.json'}, 'complete producer file census')
    require(all(not path.is_symlink() and (path.is_dir() or path.is_file()) for path in root.rglob('*')),
            'ordinary producer files and directories')
    for name, row in manifest['files'].items():
        inputs.bind(root.joinpath(*safe_member(name).parts), row['sha256'], row['bytes'], row['mode'])
    cases = parse(inputs.bind(root / 'cases.json', manifest['cases_sha256']))
    require(cases['schema'] == 'heptatorch.indexed-forward.literal-inputs-and-calls.v1'
            and cases['expected_observations'] == len(cases['cases']) == 42
            and cases['expected_outputs_included'] is False
            and cases['raw_opcode11_Torch_mapping_included'] is False, 'exact independent call scope')
    require(len({row['name'] for row in cases['cases']}) == 42, 'unique fixed reference names')
    require(set(cases['input_blobs']) == {name for name in manifest['files'] if name.startswith('inputs/')},
            'complete fixed blob census')
    for name, row in cases['input_blobs'].items():
        require(manifest['files'][name] == row, 'all raw inputs manifest-bound')
    categories = {}
    for spec in cases['cases']:
        require(spec['operation'] in ('scatter_add', 'index_add', 'cat')
                and spec['opcode'] == {'scatter_add':36, 'index_add':37, 'cat':34}[spec['operation']]
                and type(spec['axis']) is int and spec['host_expected_output_present'] is False
                and type(spec['backward']) is bool, 'literal operation contract')
        require(spec['setup'] in ('leaves', 'transpose_both', 'cloned_both', 'mutated_both', 'rebased_views', 'cat255'),
                'fixed construction vocabulary')
        categories[spec['category']] = categories.get(spec['category'], 0) + 1
        for name, item in spec['inputs'].items():
            if 'alias' in item:
                require(name == 'source' and item == {'alias':'left'}, 'only explicit self/source alias')
                continue
            require(item['dtype'] in ('float32', 'int64') and item['path'] in cases['input_blobs']
                    and type(item['requires_grad']) is bool and len(item['shape']) <= 4
                    and all(type(size) is int and size >= 0 for size in item['shape'])
                    and item['numel'] == math.prod(item['shape'])
                    and item['bytes'] == item['numel'] * (4 if item['dtype'] == 'float32' else 8)
                    and {key:cases['input_blobs'][item['path']][key] for key in ('bytes','sha256')} ==
                        {key:item[key] for key in ('bytes','sha256')}, 'complete typed literal input extent')
            require(item['dtype'] == 'float32' or item['requires_grad'] is False, 'integer indices have no gradient')
        require((spec['operation'] == 'index_add') == ('alpha_bits' in spec), 'alpha belongs only to index_add')
        if 'alpha_bits' in spec:
            require(type(spec['alpha_bits']) is int and 0 <= spec['alpha_bits'] <= 0xffffffff,
                    'literal binary32 alpha input')
        if spec['backward']:
            require('seed' in spec['inputs'] and spec['inputs']['seed']['dtype'] == 'float32'
                    and spec['inputs']['seed']['shape'] == spec['left_shape'], 'literal target-shaped backward seed')
    require(categories == cases['category_counts'] == manifest['category_counts'], 'fixed complete case denominator')
    verify_original_recipes(root, cases, inputs)
    return manifest, cases



def verify_original_recipes(root, cases, inputs):
    by_name = {row['name']:row for row in cases['cases']}
    for role, spec in cases['original_inputs'].items():
        raw = bytes.fromhex(inputs.bind(root / spec['path'], spec['sha256'], spec['bytes'], 0o644).decode())
        require(sha(raw) == spec['decoded_sha256'] and len(raw) == spec['decoded_bytes']
                and raw[:8] == b'HTORCH01', 'unchanged original legacy input extent and version')
        count, extent, reserved = struct.unpack_from('<3Q', raw, 8)
        require(extent == len(raw) and reserved == 0, 'original packet envelope')
        commands = [struct.unpack_from('<8Q', raw, 32 + 64 * i) for i in range(count)]
        if role == 'large_index':
            require(count == 3 and commands == [(1,2,0,2,0,0,0,224), (1,2,0,125000,0,0,0,224),
                    (37,1,2,0,224,1065353216,0,0)] and struct.unpack_from('<2Q',raw,224)==(1,125000),
                    'original complete large-index command recipe')
            actual = by_name['original-large-index125000']
            require(actual['left_shape']==[2,0] and actual['source_shape']==[125000,0] and actual['axis']==0
                    and actual['alpha_bits']==1065353216 and actual['inputs']['index']['shape']==[125000],
                    'literal Torch recipe maps original scalar/shape fields')
            item = actual['inputs']['index']
            require(inputs.bind(root/item['path'],item['sha256']) == raw[240:], 'all original index bytes identical')
        else:
            require(role=='cat_capacity' and count==256 and
                    all(c==(1,1,0,0,0,0,0,16416) for c in commands[:-1]) and
                    commands[-1]==(34,0,16416,0,0,0,0,0) and struct.unpack_from('<2Q',raw,16416)==(1,1),
                    'original255 empty factories and one-input cat')
            actual = by_name['original-cat-after255-empty-leaves']
            require(actual['setup']=='cat255' and actual['axis']==0 and actual['left_shape']==[0],
                    'literal cat recipe preserves all original prefix inputs')


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


def tensor_snapshot(torch, value, blobs):
    logical = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
    storage = value.untyped_storage()
    return dict(dtype=str(value.dtype), shape=list(value.shape), strides=list(value.stride()),
        numel=value.numel(), element_size=value.element_size(), storage_bytes=storage.nbytes(),
        storage_offset=value.storage_offset(), version=value._version, requires_grad=value.requires_grad,
        is_leaf=value.is_leaf, is_view=value._is_view(), retains_grad=value.retains_grad,
        grad_fn=None if value.grad_fn is None else type(value.grad_fn).__name__,
        tensor_identity=int(value._cdata), storage_identity=int(storage._cdata),
        base_tensor_identity=None if value._base is None else int(value._base._cdata),
        logical=blobs.add(logical), storage=blobs.add(bytes(storage)))



def snapshot(torch, value, blobs):
    result = tensor_snapshot(torch, value, blobs)
    # This reads the real property even for non-leaves. Its expected warning
    # is not a numerical result; all unexpected observer failures propagate.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        gradient = value.grad
    result['gradient'] = None if gradient is None else tensor_snapshot(torch, gradient, blobs)
    result['grad_is_none'] = gradient is None
    return result



def dense(torch, item, raw):
    dtype = {'float32':torch.float32, 'int64':torch.int64}[item['dtype']]
    if raw:
        value = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(item['shape']).clone()
    else:
        value = torch.empty(item['shape'], dtype=dtype)
    if item['requires_grad']:
        value.requires_grad_(True)
    require(value.dtype == dtype and list(value.shape) == item['shape'] and value.is_leaf
            and not value._is_view() and value.grad_fn is None and value._version == 0
            and value.requires_grad == item['requires_grad']
            and bytes(value.detach().reshape(-1).view(torch.uint8).tolist()) == raw,
            'ordinary owned literal Tensor with every original bit preserved')
    return value



def prepare_inputs(torch, spec, inputs):
    tracked, constructed = {}, {}
    for name, item in spec['inputs'].items():
        if 'alias' in item:
            continue
        raw = inputs.bind(HERE/item['path'],item['sha256'],item['bytes'],0o644)
        value = dense(torch,item,raw)
        tracked[name+'_literal'] = value
        constructed[name] = value
    if spec['setup']=='cat255':
        item=spec['inputs']['left']
        values=[constructed['left']]
        for i in range(1,255):
            values.append(dense(torch,item,b''))
            tracked['cat_prefix_%03d'%i]=values[-1]
        tracked['left']=values[0]
        return dict(left=values[0],cat_prefix=values),tracked
    left, source = constructed['left'], constructed.get('source')
    if spec['setup']=='transpose_both':
        left,source=left.transpose(0,1),source.transpose(0,1)
    elif spec['setup'] in ('cloned_both','mutated_both','rebased_views'):
        left,source=left.clone(),source.clone()
        tracked['left_nonleaf_base'],tracked['source_nonleaf_base']=left,source
        if spec['setup']=='rebased_views':
            left,source=left.transpose(0,1),source.transpose(0,1)
        if spec['setup'] in ('mutated_both','rebased_views'):
            # Retain an actual earlier graph branch before mutating each
            # nonleaf base. Forward and backward are observed on the current
            # inputs while the old history remains alive and byte-stable.
            tracked['left_earlier_history']=left * 2.0
            tracked['source_earlier_history']=source * 3.0
            tracked['left_nonleaf_base'].add_(constructed['delta_left'])
            tracked['source_nonleaf_base'].add_(constructed['delta_source'])
    if spec['inputs'].get('source') == {'alias':'left'}:
        source=left
    require(list(left.shape)==spec['left_shape'] and list(source.shape)==spec['source_shape'],
            'actual constructed operation shapes match fixed source-only recipe')
    tracked['left'],tracked['source']=left,source
    return dict(left=left,source=source,index=constructed['index'],seed=constructed.get('seed')),tracked



def operation(torch, prepared, spec):
    if spec['operation']=='cat':
        return torch.ops.aten.cat.default([prepared['left']],spec['axis'])
    if spec['operation']=='index_add':
        alpha=struct.unpack('<f',struct.pack('<I',spec['alpha_bits']))[0]
        return torch.ops.aten.index_add.default(prepared['left'],spec['axis'],prepared['index'],prepared['source'],alpha=alpha)
    return torch.ops.aten.scatter_add.default(prepared['left'],spec['axis'],prepared['index'],prepared['source'])



def observe(torch, spec, inputs, blobs):
    rng_before=bytes(torch.get_rng_state().tolist())
    prepared,tracked=prepare_inputs(torch,spec,inputs)
    before={name:snapshot(torch,value,blobs) for name,value in tracked.items()}
    require(bytes(torch.get_rng_state().tolist())==rng_before,'literal construction does not change RNG')
    try:
        result=operation(torch,prepared,spec)
    except (IndexError,RuntimeError) as error:
        row=dict(accepted=False,result=None,error=dict(type=type(error).__name__,message=str(error)))
    else:
        row=dict(accepted=True,result=snapshot(torch,result,blobs),error=None,
            fresh_tensor=all(result._cdata!=value._cdata for value in tracked.values()),
            fresh_storage=all(result.untyped_storage()._cdata!=value.untyped_storage()._cdata for value in tracked.values()))
    after={name:snapshot(torch,value,blobs) for name,value in tracked.items()}
    require(before==after and bytes(torch.get_rng_state().tolist())==rng_before,
            'out-of-place indexed/cat operation preserves complete inputs and RNG')
    row['backward']=None
    if row['accepted'] and spec['backward']:
        require(prepared['seed'] is not None,'actual literal seed present')
        try:
            torch.autograd.backward(result,prepared['seed'])
        except (IndexError,RuntimeError) as error:
            backward=dict(accepted=False,error=dict(type=type(error).__name__,message=str(error)))
        else:
            backward=dict(accepted=True,error=None)
        gradients={name:snapshot(torch,value,blobs) for name,value in tracked.items()}
        primal_keys=set(next(iter(before.values()))) - {'gradient','grad_is_none'}
        require(all({k:value[k] for k in primal_keys}=={k:gradients[name][k] for k in primal_keys}
                    for name,value in before.items()), 'backward preserves every tracked primal value and metadata')
        require(tensor_snapshot(torch,result,blobs)=={k:v for k,v in row['result'].items() if k not in ('gradient','grad_is_none')},
                'backward preserves the observed forward result')
        backward.update(inputs_after=gradients,primal_inputs_unchanged=True,seed=snapshot(torch,prepared['seed'],blobs))
        row['backward']=backward
    rng_after=bytes(torch.get_rng_state().tolist())
    require(rng_after==rng_before and len(rng_before)==5056,'complete CPU RNG unchanged across all actual calls')
    return dict(spec=spec,inputs_before=before,inputs_after_forward=after,input_unchanged=True,
        input_aliases={name:[other for other,value2 in tracked.items()
                        if value._cdata==value2._cdata] for name,value in tracked.items()},
        storage_aliases={name:[other for other,value2 in tracked.items()
                        if value.untyped_storage()._cdata==value2.untyped_storage()._cdata] for name,value in tracked.items()},
        rng_unchanged=True,rng=blobs.add(rng_before),**row)



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
    report = dict(schema='heptatorch.indexed-forward.fixed-Torch-reference.v1', status='RUNNING',
        rows=[], failures=[], inputs=inputs.rows, expected_observations=42,
        Torch_executed=False, native_executed=False, compiler_executed=False,
        original_legacy_reports_or_expected_fixtures_modified=False,
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
                      case_sha256=package['cases_sha256'], category_counts=cases['category_counts'])
        sys.addaudithook(forbid_children)
        old_argv = sys.argv
        try:
            sys.argv = [str(repo / 'tests/pytorch_oracle.py'), str(out / 'NO_NATIVE_EXECUTION'),
                        str(out / 'oracle-provenance-failure.json'), str(repo / 'compatibility/pytorch-oracle-lock.json')]
            context = runpy.run_path(str(repo / 'tests/pytorch_oracle.py'), run_name='indexed_forward_actual_reference_only')
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
        require(len(report['rows']) == 42 and rng == bytes(torch.get_rng_state().tolist()),
                'complete real call denominator and unchanged final RNG')
        inputs.recheck()
        blobs.recheck()
        require(not ATTEMPTS, 'no native/compiler/other child process attempted')
        report.update(status='OBSERVED_INDEXED_FORWARD_REFERENCE', cases=42,
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
