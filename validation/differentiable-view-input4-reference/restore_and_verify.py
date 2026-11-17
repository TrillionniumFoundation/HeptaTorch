#!/usr/bin/env python3
"""Restore original packet delivery modes, verify exact bytes, optionally extract."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile

PINS = {
 'complete.tar.xz': (2860876, '2b8225a7d7a2b0f56fe2caa84911da121f3fa5811e734edfcd990405b4f9d18a'),
 'manifest.json': (31399, '5ea2b8d14384b169985bfa0075144daede07bca23039900b6d846d6f2a672b1c'),
 'verification.json': (764, '43ea5201eab209731946221f54e80308e6822e22f4f02399b8cc67ea74218784'),
 'outer-custody.json': (1242, '90f8c8c4af3314330a02e7d7a3c47c842cbf4fed305237b7be42f40fd4ebb891'),
 'verify_packet.py': (14607, '8d8879e5c23c832dee9fe9be5d3de475ce0801a29e167bef06a680ab07a6d573'),
 'README.md': (3103, '7c46c062839eb753dac5532f5ca0e0048c9744b8d1de4a67e88da4cbd032b12d'),
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def info(path):
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and not path.is_symlink()
    return {'bytes': st.st_size, 'mode': stat.S_IMODE(st.st_mode), 'sha256':sha(path.read_bytes())}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--delivery', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--extract', action='store_true')
    args = ap.parse_args()
    os.umask(0o077)
    assert not args.output.exists()
    delivery = json.loads((args.delivery / 'delivery-manifest.json').read_bytes())
    assert delivery['schema'] == 'heptatorch.input4-reference-repository-delivery.v1'
    assert set(delivery['files']) == set(PINS)
    before = {}
    for name, (size, digest) in PINS.items():
        record = delivery['files'][name]
        assert record['original_mode'] == 0o600 and record['installed_mode'] == 0o644
        assert record['bytes'] == size and record['sha256'] == digest
        before[name] = info(args.delivery/name)
        assert before[name] == {'bytes':size,'mode':0o644,'sha256':digest}
    check_self = info(Path(__file__).resolve())
    assert check_self == delivery['restore_helper']['installed']
    args.output.mkdir(mode=0o700)
    packet = args.output / 'packet'
    packet.mkdir(mode=0o700)
    for name, (size, digest) in PINS.items():
        raw = (args.delivery/name).read_bytes()
        with (packet/name).open('xb') as out:
            out.write(raw)
        os.chmod(packet/name, 0o600)
        assert info(packet/name) == {'bytes':size,'mode':0o600,'sha256':digest}
    command = [sys.executable,'-I','-S','-B',str(packet/'verify_packet.py'),'--verify',str(packet)]
    (args.output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (args.output/'stdout.txt').open('xb') as so, (args.output/'stderr.txt').open('xb') as se:
        process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=so, stderr=se, timeout=120)
    result = {'schema':'heptatorch.input4-reference-restored-delivery.v1','status':'FAIL',
              'native_executed':False,'torch_executed':False,'command':command,'exit_code':process.returncode}
    try:
        assert process.returncode == 0
        assert (args.output/'stderr.txt').read_bytes() == b''
        checked = json.loads((args.output/'stdout.txt').read_bytes())
        assert checked == json.loads((packet/'verification.json').read_bytes())
        assert checked['status'] == 'PASS' and checked['member_count'] == 42
        result['original_verification'] = checked
        extracted = {}
        if args.extract:
            destination = args.output/'payload'
            destination.mkdir(mode=0o700)
            manifest = json.loads((packet/'manifest.json').read_bytes())
            with tarfile.open(packet/'complete.tar.xz','r|xz') as tar:
                for member in tar:
                    rel = PurePosixPath(member.name)
                    assert not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() == member.name
                    assert member.isfile() and member.name not in extracted and member.name in manifest['files']
                    raw = tar.extractfile(member).read()
                    expected = manifest['files'][member.name]
                    assert {'bytes':len(raw),'mode':member.mode,'sha256':sha(raw)} == expected
                    target = destination / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as out:
                        out.write(raw)
                    os.chmod(target,member.mode)
                    assert info(target) == expected
                    extracted[member.name] = expected
            assert set(extracted) == set(manifest['files'])
        result['extracted_files'] = extracted
        assert {name:info(args.delivery/name) for name in PINS} == before
        result.update(status='PASS',all_delivery_bytes_unchanged=True,
                      all_original_modes_explicitly_restored=True)
    except BaseException:
        import traceback
        result['primary_error'] = traceback.format_exc()
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'report':str(args.output/'report.json'),
                      'sha256':sha((args.output/'report.json').read_bytes())}))
    return 0 if result['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
