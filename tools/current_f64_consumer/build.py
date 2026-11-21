#!/usr/bin/env python3
"""Link the original F64 packet protocol to the actual current HIA objects.

The only compiled adapter bodies are the original entry, byte writer and
sqrt64, with collision-free names. All eleven other original bodies are
required byte-identical in the admitted canonical source. Six operations use
the actual current objects; sqrt remains the original separate RNE component.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import stat
import struct
import subprocess
import time
import types

SOURCE = '44a6ff00413829d3cf8f7fbd91fb96a6baf090c3c754882d4361f49ed85a2e56'
PROFILE = '8865de92df8a2e6f55ef1b3d7a28d56e05dd228b21a1c53b3f485151f4027dd3'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
INSPECTOR = '71b72260014417db72ffcebd80c86543e257c80d76cdd6c878b41feaa117cf92'
RENAMES = {'entry': 'f64_packet_entry', 'putw': 'f64_packet_putw', 'sqrt64': 'f64_packet_sqrt64'}
IMPORTS = {'getr', 'add64', 'sub64', 'mul64', 'div64', 'widen32', 'narrow64', 'pack64'}
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def identity(path):
    path = Path(path).absolute()
    require(not path.is_symlink() and path.is_file(), 'regular input: '+str(path))
    a = path.stat(); raw = path.read_bytes(); b = path.stat()
    stable = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stable(a) == stable(b), 'input changed while read: '+str(path))
    return {'bytes': len(raw), 'sha256': sha(raw), 'mode': stat.S_IMODE(b.st_mode)}


class Inputs:
    def __init__(self):
        self.records = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute(); row = identity(path); raw = path.read_bytes()
        require(sha(raw) == row['sha256'], 'input changed after stat')
        require(expected is None or row['sha256'] == expected, 'input identity: '+str(path))
        require(str(path) not in self.records or self.records[str(path)] == row, 'bound input changed')
        self.records[str(path)] = row
        return raw

    def recheck(self):
        for path, row in list(self.records.items()):
            require(identity(path) == row, 'bound input changed: '+path)


def functions(raw):
    rows = [(m[2].decode(), m[1]) for m in re.finditer(
        rb'^(fn (\w+)\([^\n]* entry\([^\n]*\)\n.*?^end\n)', raw, re.M | re.S)]
    require(len(rows) == len(dict(rows)), 'duplicate function')
    return dict(rows)


def rename(body, mapping):
    for old, new in mapping.items():
        body = body.replace(b'fn '+old.encode()+b'(', b'fn '+new.encode()+b'(')
        body = body.replace(b'{"function":"'+old.encode()+b'"}', b'{"function":"'+new.encode()+b'"}')
    return body


def adapter(old, current, profile):
    require(sha(old) == SOURCE and sha(profile) == PROFILE, 'complete original component identity')
    original, present = functions(old), functions(current)
    expected = json.loads(profile)
    require({n: sha(b) for n, b in original.items()} == expected['function_body_sha256'], 'all original14 bodies')
    require(len(original) == 14 and set(RENAMES) <= set(original), 'exact original function inventory')
    reused = set(original)-set(RENAMES)
    require(len(reused) == 11 and all(present.get(n) == original[n] for n in reused), 'all eleven canonical bodies match')
    local = {RENAMES[n]: rename(original[n], RENAMES) for n in original if n in RENAMES}
    inverse = {new: old for old, new in RENAMES.items()}
    require(all(rename(local[new], inverse) == original[old] for old, new in RENAMES.items()), 'complete symbol-only inverse')
    calls = {x.decode() for body in local.values() for x in re.findall(rb'\{"function":"(\w+)"\}', body)}
    require(calls-set(local) == IMPORTS and not (set(local) & set(present)), 'exact adapter dependencies and unique symbols')
    imports = []
    for name in sorted(IMPORTS):
        head = present[name].splitlines(keepends=True)[0]
        decl, count = re.subn(rb'entry\([^)]*\)\n$', b'external()\n', head)
        require(count == 1, 'external signature')
        imports.append(decl)
    raw = b'hia-asm.v1 hepta.hia.program.v8 current_f64_packet_adapter\nexports(f64_packet_entry)\n'
    raw += b''.join(imports)+b''.join(local.values())
    require(functions(raw) == local, 'only three local adapter bodies')
    proof = {'original_source_sha256': SOURCE, 'original_profile_sha256': PROFILE,
             'current_source_sha256': sha(current), 'adapter_source_sha256': sha(raw),
             'symbol_renames': RENAMES, 'complete_inverse': True,
             'original_function_count': 14, 'compiled_adapter_function_count': 3,
             'reused_current_functions': {n: sha(original[n]) for n in sorted(reused)},
             'preserved_adapter_bodies': {n: sha(original[n]) for n in RENAMES},
             'external_calls': sorted(IMPORTS), 'old_runtime143_required': False,
             'old14_function_image_compiled': False, 'new_numerical_body_count': 0,
             'sqrt_uses_current_public_tensor_operation': False}
    return raw, proof


def current_admission(repo, current, inputs):
    path = repo/'tools/current_runtime/qualification.py'
    raw = inputs.bind(path)
    spec = importlib.util.spec_from_file_location('f64_consumer_current_admission', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module.admit_build(current, inputs.bind)


def execute(executable, payload, directory, name, timeout=120):
    directory = Path(directory); executable = Path(executable).absolute()
    before = identity(executable)
    intent = {'name': name, 'argv': [str(executable)], 'cwd': str(directory.absolute()),
              'environment': {'PATH': ''}, 'timeout_seconds': timeout, 'executable': before,
              'supervisor_pid': os.getpid(), 'started': False}
    dump(directory/(name+'.intent.json'), intent)
    (directory/(name+'.request.bin')).write_bytes(payload)
    started = time.monotonic()
    process = subprocess.Popen(intent['argv'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, cwd=directory, env={'PATH': ''}, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(input=payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    (directory/(name+'.stdout.bin')).write_bytes(stdout)
    (directory/(name+'.stderr.bin')).write_bytes(stderr)
    result = dict(intent, started=True, pid=process.pid, exit_code=process.returncode,
                  timed_out=timed_out, stdout_EOF=True, stderr_EOF=True,
                  seconds=time.monotonic()-started,
                  request_bytes=len(payload), request_sha256=sha(payload),
                  stdout_bytes=len(stdout), stdout_sha256=sha(stdout),
                  stderr_bytes=len(stderr), stderr_sha256=sha(stderr))
    dump(directory/(name+'.process.json'), result)
    require(identity(executable) == before, 'executed image changed')
    require(not timed_out and process.returncode == 0 and not stderr, 'native process failed: '+name)
    return stdout, result


def build(repo, current, linker_path, out):
    require(__debug__, 'assertions required')
    require(not (repo/'src/heptatorch.hepta').exists(), 'legacy143 must remain absent')
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs(); inputs.bind(Path(__file__).resolve())
    built = current_admission(repo, current, inputs)
    raw, proof = adapter(inputs.bind(repo/'tools/dtype-f64/f64.hepta', SOURCE),
                         inputs.bind(current/'runtime.hepta', built['source_sha256']),
                         inputs.bind(repo/'tools/dtype-f64/profile.json', PROFILE))
    (out/'packet_adapter.hepta').write_bytes(raw)
    compiler = out/'compiler'; compiler.write_bytes(inputs.bind(current/'toolchain/stack-module-compiler', COMPILER)); compiler.chmod(0o500)
    linker = out/'linker'; linker.write_bytes(inputs.bind(linker_path, LINKER)); linker.chmod(0o500)
    inspector = types.ModuleType('f64_consumer_fixed_stack_inspector')
    exec(compile(inputs.bind(repo/'tools/stack_budget/inspect_stack.py', INSPECTOR), '<fixed-stack-inspector>', 'exec'), inspector.__dict__)
    profile = json.loads(inputs.bind(repo/'tools/stack_budget/profile.json'))
    require(profile['compiler']['executable_sha256'] == COMPILER and profile['linkers']['runtime']['executable_sha256'] == LINKER, 'native tool profile')
    objects, views, units = [], [], []
    for unit in built['units']:
        name = unit['name']; source = inputs.bind(current/(name+'.hepta'), unit['source_sha256'])
        obj = inputs.bind(current/(name+'.o'), unit['object_sha256'])
        objects.append(obj); views.append(inspector.inspect(source, obj))
        units.append({'name': name, 'source_sha256': sha(source), 'object_sha256': sha(obj), 'bytes': len(obj)})
    compiled = [execute(compiler, raw, out, 'compile-'+str(i)) for i in range(2)]
    require(compiled[0][0] == compiled[1][0], 'deterministic adapter compilation')
    obj = compiled[0][0]; (out/'packet_adapter.o').write_bytes(obj)
    views.append(inspector.inspect(raw, obj)); objects.append(obj)
    stack = inspector.longest(views)
    require(stack['maximum_path_bytes'] <= profile['linkers']['runtime']['stack_budget'], 'complete native stack bound')
    entry = b'f64_packet_entry'
    request = b'HIASTKR1'+struct.pack('<3Q', len(objects), len(entry), 0)+entry
    request += b''.join(struct.pack('<Q', len(o))+o for o in objects)
    linked = [execute(linker, request, out, 'link-'+str(i)) for i in range(2)]
    require(linked[0][0] == linked[1][0], 'deterministic current-object link')
    image = linked[0][0]; require(image[:6] == b'\x7fELF\x02\x01', 'native ELF output')
    executable = out/'f64-native'; executable.write_bytes(image); executable.chmod(0o500)
    entry_offset = 482+struct.unpack_from('<i', image, 478)[0]
    record = next(r for r in views[-1]['records'] if r['name'] == 'f64_packet_entry')
    begin = views[-1]['text_at']+record['offset']
    require(image[477] == 0xe8 and image[entry_offset:entry_offset+15] == obj[begin:begin+15], 'actual adapter startup')
    inputs.recheck()
    result = {'schema': 'heptatorch.current-object-f64-build.v1', 'status': 'PASS_NATIVE_BUILD_ONLY',
              'producer_paths': {'repository': str(repo), 'current_build': str(current),
                                 'implementation': str(Path(__file__).resolve()), 'flat_linker': str(linker_path),
                                 'output': str(out)},
              'source_transition': proof, 'current_build_sha256': sha(inputs.bind(current/'build.json')),
              'current_source_sha256': built['source_sha256'], 'current_native_sha256': built['native_sha256'],
              'current_function_count': built['functions'], 'current_units': units,
              'adapter_object_sha256': sha(obj), 'native_sha256': sha(image), 'native_bytes': len(image),
              'entry_offset': entry_offset, 'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
              'stack': stack, 'processes': [p for _, p in compiled+linked], 'inputs': inputs.records,
              'actual_current_objects_reused': True, 'legacy143_restored': False,
              'full_current_image_executed': False, 'runtime_qualified': False,
              'whole_pytorch_rewrite_complete': False}
    dump(out/'build.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--current-build', type=Path, required=True)
    parser.add_argument('--flat-linker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repository.resolve(), args.current_build.resolve(), args.flat_linker.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ('status', 'native_sha256', 'native_bytes', 'entry_offset')}))


if __name__ == '__main__':
    main()
