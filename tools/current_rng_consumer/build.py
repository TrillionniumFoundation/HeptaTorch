#!/usr/bin/env python3
"""Build the complete historical RNG packet consumer on the current RNG object.

Only entry and four packet IO helpers are compiled. Every RNG/rounding call
resolves to the actual, byte-bound rng03.o used by the admitted current build.
No previous runtime/compiler or host numerical implementation is executed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import time
import sys
import os
import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_evidence import identity, record_actual_process

OLD_SOURCE = 'c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032'
RNG_SOURCE = '7c32981c916bc277e16221cce1c3dd2381ce0018642e21b20836d861cf585865'
RNG_OBJECT = 'd462d241e7c3082167b61ccb7b4e6a60eeb94c242dbb69e242fd57b822b9f2a1'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
INSPECTOR = '71b72260014417db72ffcebd80c86543e257c80d76cdd6c878b41feaa117cf92'
LOCAL = {'entry', 'getr', 'getw', 'putw', 'rng_error'}
IMPORTS = {'rng_seed', 'rng_restore', 'rng_next', 'rng_next64', 'rng_unit32', 'rng_unit64', 'round_f16', 'round_bf16'}
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def functions(raw):
    pairs = [(m[2].decode(), m[1]) for m in re.finditer(rb'^(fn (\w+)\([^\n]* entry\([^\n]*\)\n.*?^end\n)', raw, re.M | re.S)]
    require(len(dict(pairs)) == len(pairs), 'duplicate function name')
    return dict(pairs)


def adapter(old, current):
    require(sha(old) == OLD_SOURCE and sha(current) == RNG_SOURCE, 'source identity')
    before, after = functions(old), functions(current)
    require(len(before) == len(after) == 18, 'full original and current function inventories')
    require(set(before) - set(after) == {'entry'} and set(after) - set(before) == {'cg_putw'}, 'function inventory delta')
    unchanged = set(before) & set(after)
    require(len(unchanged) == 17 and all(before[n] == after[n] for n in unchanged), 'all 17 original non-entry bodies unchanged')
    require(LOCAL <= set(before), 'missing packet body')
    calls = {x.decode() for n in LOCAL for x in re.findall(rb'\{"function":"(\w+)"\}', before[n])}
    require(calls - LOCAL == IMPORTS, 'exact complete external dependency closure')
    exported = set(re.search(rb'^exports\(([^\n]*)\)', current, re.M)[1].decode().split(','))
    require(IMPORTS <= exported, 'current RNG object exports all required kernels')
    imports = []
    for name in sorted(IMPORTS):
        header = before[name].splitlines(keepends=True)[0]
        external, n = re.subn(rb'entry\([^)]*\)\n$', b'external()\n', header)
        require(n == 1, 'external signature extraction')
        imports.append(external)
    raw = b'hia-asm.v1 hepta.hia.program.v8 current_rng_packet_adapter\nexports(entry)\n' + b''.join(imports)
    raw += b''.join(body for name, body in before.items() if name in LOCAL)
    require(functions(raw) == {n: b for n, b in before.items() if n in LOCAL}, 'all five packet bodies remain byte-identical')
    proof = {'old_source_sha256': OLD_SOURCE, 'current_rng_source_sha256': RNG_SOURCE,
             'adapter_source_sha256': sha(raw), 'original_functions': 18,
             'unchanged_current_kernel_and_helper_bodies': {n: sha(before[n]) for n in sorted(unchanged)},
             'unchanged_adapter_bodies': {n: sha(before[n]) for n in sorted(LOCAL)},
             'external_kernel_calls': sorted(IMPORTS),
             'new_numerical_bodies': 0, 'historical_runtime143_source_required': False,
             'historical_18_function_runtime_compiled': False}
    return raw, proof


def current_admission(repository, directory, bind):
    require(not sys.flags.optimize, 'original source, object and test assertions must remain enabled')
    path = repository/'tools/current_runtime/qualification.py'
    raw = bind(path)
    spec = importlib.util.spec_from_file_location('rng_consumer_current_build_admission', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module.admit_build(directory, bind)


def input_roles(inputs, repository, current, implementation, flat_linker):
    """Portable input names; absolute producer filesystem paths are not inputs."""
    result = {}
    for path, raw in inputs.items():
        if path.parent == implementation.parent and path.name in ('build.py', 'process_evidence.py'):
            name = 'implementation/'+path.name
        elif path == flat_linker:
            name = 'toolchain/flat-linker'
        elif path.is_relative_to(repository):
            name = 'repository/'+path.relative_to(repository).as_posix()
        elif path.is_relative_to(current):
            name = 'current-build/'+path.relative_to(current).as_posix()
        else:
            raise ValueError('unclassified bound input: '+str(path))
        require(name not in result, 'duplicate input role')
        result[name] = {'bytes':len(raw), 'sha256':sha(raw)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--current-build', type=Path, required=True)
    p.add_argument('--flat-linker', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo, current, out = a.repository.resolve(), a.current_build.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs, processes = {}, []
    actual = record_actual_process(__file__)
    require(actual is not None, 'native packet build requires the supervised production runner')

    def bind(path, expected=None):
        raw = path.read_bytes()
        require(expected is None or sha(raw) == expected, 'input identity: ' + str(path))
        require(path not in inputs or inputs[path] == raw, 'input changed')
        inputs[path] = raw
        return raw

    implementation = Path(__file__).resolve()
    bind(implementation)
    bind(implementation.with_name('process_evidence.py'))
    build = current_admission(repo, current, bind)
    build_raw = bind(current/'build.json')
    source = bind(repo/'tools/rng-cpu/rng.hepta', OLD_SOURCE)
    rng_source = bind(current/'rng03.hepta', RNG_SOURCE)
    require(rng_source == bind(repo/'tools/current_runtime/source/units/rng03.hepta', RNG_SOURCE), 'canonical current RNG source')
    rng_object = bind(current/'rng03.o', RNG_OBJECT)
    require(build['units'][0]['object_sha256'] == RNG_OBJECT and build['units'][0]['source_sha256'] == RNG_SOURCE, 'current object build binding')
    spec_raw = bind(repo/'tools/stack_budget/profile.json')
    spec = json.loads(spec_raw)
    require(spec['compiler']['executable_sha256'] == COMPILER
            and spec['linkers']['runtime']['executable_sha256'] == LINKER, 'canonical tool profile')
    tools = []
    for src, name, pin in [(current/'toolchain/stack-module-compiler', 'compiler', COMPILER),
                            (a.flat_linker.resolve(), 'linker', LINKER)]:
        raw = bind(src, pin)
        path = out/name
        path.write_bytes(raw); path.chmod(0o500); tools.append(path)
    compiler, linker = tools
    inspector_raw = bind(repo/'tools/stack_budget/inspect_stack.py', INSPECTOR)
    namespace = {}
    exec(compile(inspector_raw, '<fixed-inspector>', 'exec'), namespace)
    raw, proof = adapter(source, rng_source)
    (out/'packet_adapter.hepta').write_bytes(raw)
    (out/'rng03.hepta').write_bytes(rng_source)
    (out/'rng03.o').write_bytes(rng_object)

    def run(executable, payload, name):
        (out/(name+'.request.bin')).write_bytes(payload)
        intent = {'name': name, 'argv': [str(executable)], 'cwd': str(out),
                  'environment': {'PATH': ''}, 'timeout_seconds': 120,
                  'executable': identity(executable), 'supervisor_pid': os.getpid(),
                  'started': False, 'stdout_EOF': False, 'stderr_EOF': False}
        (out/(name+'.intent.json')).write_text(json.dumps(intent, indent=2, sort_keys=True)+'\n')
        start = time.monotonic()
        proc = subprocess.Popen(intent['argv'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=out, env=intent['environment'], start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = proc.communicate(input=payload, timeout=120)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        (out/(name+'.stdout.bin')).write_bytes(stdout)
        (out/(name+'.stderr.bin')).write_bytes(stderr)
        record = dict(intent, started=True, pid=proc.pid, timed_out=timed_out,
                      executable_sha256=sha(executable.read_bytes()), request_sha256=sha(payload),
                      request_bytes=len(payload), stdout_sha256=sha(stdout), stdout_bytes=len(stdout),
                      stderr_sha256=sha(stderr), stderr_bytes=len(stderr), exit_code=proc.returncode,
                      stdout_EOF=True, stderr_EOF=True, seconds=time.monotonic()-start)
        processes.append(record)
        require(identity(executable) == intent['executable'], 'actual native tool changed during execution')
        require(not timed_out and proc.returncode == 0 and not stderr and stdout[:4] == b'\x7fELF', 'native tool refused: '+name)
        return stdout

    objects = [run(compiler, raw, 'compile-'+str(i)) for i in range(2)]
    require(objects[0] == objects[1], 'adapter compilation deterministic')
    obj = objects[0]
    (out/'packet_adapter.o').write_bytes(obj)
    views = [namespace['inspect'](rng_source, rng_object), namespace['inspect'](raw, obj)]
    budget = namespace['longest'](views)
    require(budget['maximum_path_bytes'] <= spec['linkers']['runtime']['stack_budget'], 'complete native call graph stack budget')
    request = b'HIASTKR1'+struct.pack('<3Q', 2, 5, 0)+b'entry'
    request += b''.join(struct.pack('<Q', len(x))+x for x in [rng_object, obj])
    images = [run(linker, request, 'link-'+str(i)) for i in range(2)]
    require(images[0] == images[1], 'native link deterministic')
    image = images[0]
    native = out/'rng-native'; native.write_bytes(image); native.chmod(0o500)
    call_hole = spec['linkers']['runtime']['call_holes'][0][1]
    require(call_hole == 478 and image[call_hole-1] == 0xe8, 'fixed native startup entry call')
    entry_offset = call_hole+4+struct.unpack_from('<i', image, call_hole)[0]
    entry_record = next(r for r in views[1]['records'] if r['name'] == 'entry')
    entry_bytes = obj[views[1]['text_at']+entry_record['offset']:views[1]['text_at']+entry_record['offset']+entry_record['size']]
    # Relocations can change call operands, but the complete fixed stack prologue
    # and entry body size are independently decoded from the actual object.
    require(image[entry_offset:entry_offset+15] == entry_bytes[:15], 'linked entry frame/prologue')
    require(image[entry_offset:entry_offset+4] == bytes.fromhex('554889e5'), 'native ABI3 entry')
    for path, before in inputs.items():
        require(path.read_bytes() == before, 'bound input changed after build: '+str(path))
    report = {'schema': 'heptatorch.current-object-rng-packet-build.v1', 'status': 'PASS_NATIVE_BUILD_ONLY',
              'source_transition': proof, 'current_build_sha256': sha(build_raw),
              'current_source_sha256': build['source_sha256'], 'current_native_sha256': build['native_sha256'],
              'current_functions':build['functions'], 'current_units':build['units_planned'],
              'current_rng_object_sha256': RNG_OBJECT, 'adapter_object_sha256': sha(obj),
              'native_sha256': sha(image), 'native_bytes': len(image),
              'compiler_sha256': COMPILER, 'flat_linker_sha256': LINKER,
              'native_interface_abi': 3, 'native_entry_offset': entry_offset,
              'native_entry_prologue_hex': image[entry_offset:entry_offset+15].hex(),
              'original_flat_contract_preserved': 'three borrowed spans, same input/output/workspace bytes and u32 return; native module ABI3 marshals these in arrays',
              'stack': budget, 'processes': processes,
              'actual_process': actual,
              'producer_paths': {'repository': str(repo), 'current_build': str(current), 'output': str(out)},
              'all_inputs_unchanged': True,
              'input_bindings': input_roles(inputs, repo, current, implementation, a.flat_linker.resolve()),
              'kernel_object_is_current_actual_object': True, 'full_current_image_executed': False,
              'legacy143_restored': False, 'old_native_rng_image_executed': False,
              'host_numerical_codegen': False, 'qualified': False, 'whole_pytorch_rewrite_complete': False}
    (out/'build.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: report[k] for k in ('status','native_sha256','native_bytes','native_entry_offset','current_rng_object_sha256')}))


if __name__ == '__main__':
    main()
