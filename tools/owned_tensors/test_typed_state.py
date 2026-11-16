"""Pass real, differently shaped HIA owners through the tensor state boundary.

These are separately compiled native qualification modules. No user wire bytes
are reinterpreted as owner pointers, and production tensor functions are kept.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('typed_tensor_wire', HERE / 'protocol.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compiler', 'linker', 'build-directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    profile = json.loads((HERE / 'profile.json').read_text())
    compiler = args.compiler.resolve(); linker = args.linker.resolve()
    assert sha(compiler.read_bytes()) == profile['compiler_sha256']
    assert sha(linker.read_bytes()) == profile['linker_sha256']
    core_paths = [args.build_directory / (name + '.o') for name in ('ht_kernel', 'ht_runtime')]
    core = [path.read_bytes() for path in core_paths]
    for name, raw in zip(('ht_kernel', 'ht_runtime'), core):
        assert sha(raw) == profile['units'][name]['object_sha256']
    original = (HERE / 'owners.hepta').read_bytes()
    renamed = original.replace(b'exports(step,close)', b'exports(tensor_step,tensor_close)', 1)
    renamed = renamed.replace(b'fn step(', b'fn tensor_step(', 1).replace(b'fn close(', b'fn tensor_close(', 1)
    assert renamed.replace(b'exports(tensor_step,tensor_close)', b'exports(step,close)', 1).replace(
        b'fn tensor_step(', b'fn step(', 1).replace(b'fn tensor_close(', b'fn close(', 1) == original
    (out / 'tensor-owners.hepta').write_bytes(renamed)

    def native(executable, source, label):
        result = subprocess.run([str(executable)], input=source, capture_output=True,
            env={'PATH': ''}, timeout=60)
        (out / (label + '.stdout')).write_bytes(result.stdout)
        (out / (label + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        return result.stdout

    owner_object = native(compiler, renamed, 'owners-compile')
    rows = []
    for name, size, initialize, mutation, expected_status in (
        ('live-empty', 0, False, None, 0),
        ('valid-state', 65536, True, None, 0),
        ('one-byte-owner', 1, False, None, 7),
        ('wrong-format-owner', 65536, False, None, 7),
        ('invalid-descriptor-page', 65536, True, (4096, 65520), 7),
        ('invalid-next-slot', 65536, True, (8, 65537), 7),
    ):
        setup = ''
        if initialize:
            setup += '  initialized:u64 = call(loan) {"function":"init"}\n'
        if mutation:
            at, value = mutation
            setup += f'  at:u64 = const() {{"value":{at}}}\n  value:u64 = const() {{"value":{value}}}\n'
            setup += '  changed:u64 = call(loan,at,value) {"function":"putw"}\n'
        source = f'''hia-asm.v1 hepta.hia.program.v8 typed_tensor_probe
exports(step,close)
fn tensor_step(s:readbuf,o:mutbuf,state:buffer) -> (buffer,u32) effects(alloc,read,write,trap) external()
fn tensor_close(state:buffer) -> () effects(alloc,trap) external()
fn init(w:mutbuf) -> (u64) effects(read,write,trap) external()
fn putw(w:mutbuf,at:u64,value:u64) -> (u64) effects(read,write,trap) external()
fn step(s:readbuf,o:mutbuf,state:buffer) -> (buffer,u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,state:buffer)
  _ = owner.drop(state)
  amount:usize = const() {{"value":{size}}}
  foreign:buffer = buffer.new(amount)
  loan:mutbuf = borrow.mut(foreign)
{setup}  _ = loan.end(loan)
  returned:buffer,count:u32 = call(s,o,foreign) {{"function":"tensor_step"}}
  ret(returned,count)
end
fn close(state:buffer) -> () effects(alloc,trap) entry(b0)
block b0(state:buffer)
  _ = call(state) {{"function":"tensor_close"}}
  ret()
end
'''.encode()
        (out / (name + '.hepta')).write_bytes(source)
        probe_object = native(compiler, source, name + '-compile')
        objects = core + [owner_object, probe_object]
        request = b'HIAOWNS1' + struct.pack('<QQQ', len(objects), 4, 0) + b'step'
        request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
        image = native(linker, request, name + '-link')
        executable = out / name; executable.write_bytes(image); executable.chmod(0o500)
        session = p.Session(executable, directory=out / (name + '-observation'))
        actual, records = p.parse(session.exchange(p.packet([])))
        assert actual == expected_status and not records
        expected = bytearray(65536 if expected_status == 0 else size)
        if initialize or expected_status == 0:
            for offset, value in ((8, 1), (16, 57344), (64, 3622394114853786696), (4096, 8192)):
                struct.pack_into('<Q', expected, offset, value)
        if mutation: struct.pack_into('<Q', expected, mutation[0], mutation[1])
        owner = session.owner()
        assert owner['length'] == len(expected) and owner['bytes'] == bytes(expected)
        session.finish()
        rows.append({'name': name, 'pass': True, 'input_owner_bytes': size,
            'returned_status': actual, 'compared_owner_bytes': len(expected),
            'source_sha256': sha(source), 'executable_sha256': sha(image),
            'actual_HIA_owner_allocation': True, 'response_before_input_EOF': True})
        print(name + ': PASS', flush=True)
    assert sha(compiler.read_bytes()) == profile['compiler_sha256']
    assert sha(linker.read_bytes()) == profile['linker_sha256']
    assert (HERE / 'owners.hepta').read_bytes() == original
    for path, raw in zip(core_paths, core): assert path.read_bytes() == raw
    report = {'schema': 'heptatorch.dynamic-tensor-typed-state.v1', 'pass': True,
        'cases': len(rows), 'test_sha256': sha(Path(__file__).read_bytes()),
        'production_owner_source_sha256': sha(original), 'qualification_owner_source_sha256': sha(renamed),
        'owner_module_changed_only_by_export_renaming': True,
        'numerical_objects_byte_identical': True, 'user_bytes_never_fabricate_an_owner': True,
        'rows': rows}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
