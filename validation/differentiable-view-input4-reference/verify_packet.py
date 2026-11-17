#!/usr/bin/env python3
"""Seal/verify exact input4 reference evidence; no Torch/native execution."""
import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile

BASE = Path('/home/qian-qi/heptatorch-continuation-20261116/runs')
SOURCE = BASE / 'differentiable-view-input-boundaries-04-20261116'
CUSTODY = BASE / 'differentiable-view-input4-actual-custody-01'
ADMISSION = BASE / 'differentiable-view-input4-independent-admission-01'
PACKET = BASE / 'differentiable-view-input4-reference-handoff-01'
PIN_ACTUAL = 'f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3'
PIN_SOURCE = 'cf569335f48624c1edf2f97a649200f2e792ebcef5b6f1df6bac73293ce02efc'
PIN_EXEC = '78ea18d6ab862d5d289e45b7b12e590b30ac5c6c816056477b23879b5e6aa6bb'
PIN_ADMISSION = 'ffe90f023a412c7ac7afc7eab1d2dc8ca2c11a1f1f83ec6103c736d37906aca4'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fileinfo(path):
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and not path.is_symlink(), str(path)
    return {'bytes': st.st_size, 'mode': stat.S_IMODE(st.st_mode),
            'sha256': sha(path.read_bytes())}

def write_json(path, item):
    with path.open('xb') as out:
        out.write((json.dumps(item, indent=2, allow_nan=False) + '\n').encode())

def safe(name):
    rel = PurePosixPath(name)
    return not rel.is_absolute() and rel.as_posix() == name and '..' not in rel.parts and bool(rel.parts)

def verify(packet):
    manifest_raw = (packet / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest['schema'] == 'heptatorch.input4-original-reference-packet.v1'
    assert manifest['status'] == 'OBSERVED_ONLY'
    assert manifest['native_executed'] is False and manifest['full_pytorch_rewrite_complete'] is False
    assert manifest['closed_schema_count'] == 0 and manifest['pytorch_api_denominator'] == 2590
    archive = packet / 'complete.tar.xz'
    archive_info = fileinfo(archive)
    assert archive_info == manifest['archive']
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    digest, total = hashlib.sha256(), 0
    with archive.open('rb') as inp:
        while True:
            chunk = inp.read(65536)
            if not chunk:
                break
            assert not dec.eof, 'trailing or concatenated XZ stream'
            unpacked = dec.decompress(chunk)
            digest.update(unpacked)
            total += len(unpacked)
            assert not dec.unused_data, 'unused compressed bytes'
    assert dec.eof and dec.unused_data == b''
    assert total == manifest['decompressed_tar']['bytes']
    assert digest.hexdigest() == manifest['decompressed_tar']['sha256']
    expected = manifest['files']
    assert expected and len(expected) == manifest['member_count']
    actual = {}
    with tarfile.open(archive, 'r|xz') as tar:
        for member in tar:
            assert safe(member.name) and member.isfile() and member.name not in actual
            assert member.name in expected
            assert not member.pax_headers, 'unexpected extended metadata'
            stream = tar.extractfile(member)
            raw = stream.read()
            item = {'bytes': len(raw), 'mode': member.mode, 'sha256': sha(raw)}
            assert len(raw) == member.size and item == expected[member.name], member.name
            actual[member.name] = item
    assert set(actual) == set(expected)
    assert sum(v['bytes'] for v in actual.values()) == manifest['expanded_bytes']
    assert actual['source/pytorch_differentiable_view_input_boundaries_probe.py']['sha256'] == PIN_SOURCE
    assert actual['actual/actual.json']['sha256'] == PIN_ACTUAL
    assert actual['execution/execution.json']['sha256'] == PIN_EXEC
    assert actual['admission/report.json']['sha256'] == PIN_ADMISSION
    assert fileinfo(packet / 'verify_packet.py') == actual['tools/verify_packet.py']
    bindings = manifest['actual_input_map']
    assert len(bindings) == 25 and len({v['archive_path'] for v in bindings.values()}) == 25
    for name, item in bindings.items():
        assert item['archive_path'] in actual, name
        assert actual[item['archive_path']] == {k:item[k] for k in ('bytes','mode','sha256')}, name
    return {'schema': 'heptatorch.input4-original-reference-packet-verification.v1',
            'status': 'PASS', 'native_executions': 0, 'torch_executions': 0,
            'manifest_sha256': sha(manifest_raw), 'archive': archive_info,
            'member_count': len(actual), 'expanded_bytes': sum(v['bytes'] for v in actual.values()),
            'decompressed_tar': {'bytes': total, 'sha256': digest.hexdigest()},
            'all_members_bytes_sizes_modes_verified': True,
            'all_25_original_actual_input_bindings_verified': True,
            'single_xz_stream_verified_through_eof': True,
            'original_actual_report_unchanged': True}

def seal():
    os.umask(0o077)
    PACKET.mkdir(mode=0o700, exist_ok=False)
    members = {}
    source_members = {}
    def add(name, path):
        assert safe(name) and name not in members
        fileinfo(path)
        members[name] = path
        source_members.setdefault(str(path), name)
    manifest = json.loads((SOURCE / 'source-manifest.json').read_bytes())
    assert sha((SOURCE / 'source-manifest.json').read_bytes()) == '7056b24c58d5e0589cef7d40709e38e63017b794cf9a58f6af8687a80023f240'
    assert len(manifest['files']) == 15
    for item in manifest['files']:
        path = SOURCE / item['path']
        assert fileinfo(path) == {k:item[k] for k in ('bytes','mode','sha256')}
        add('source/' + item['path'], path)
    add('source/source-manifest.json', SOURCE / 'source-manifest.json')
    add('actual/actual.json', SOURCE / 'actual.json')
    for path in sorted(CUSTODY.iterdir()):
        assert path.is_file() and not path.is_symlink()
        add('execution/' + path.name, path)
    add('admission/report.json', ADMISSION / 'report.json')
    add('tools/actual_wrapper.py', BASE / 'dview-input-boundaries4-actual-wrapper-01.py')
    add('tools/admit_original.py', BASE / 'dview-input4-admit-original-01.py')
    add('tools/verify_packet.py', Path(__file__).resolve())
    execution = json.loads((CUSTODY / 'execution.json').read_bytes())
    admission = json.loads((ADMISSION / 'report.json').read_bytes())
    assert sha((CUSTODY / 'execution.json').read_bytes()) == PIN_EXEC
    assert sha((ADMISSION / 'report.json').read_bytes()) == PIN_ADMISSION
    assert execution['status'] == 'OBSERVED' and execution['exit_code'] == 0
    assert execution['inputs_before'] == execution['inputs_after']
    assert admission['status'] == 'OBSERVATION_BYTE_INTEGRITY_VERIFIED'
    assert admission['counts']['checks'] == 64059
    bindings = {}
    for name, item in execution['inputs_before'].items():
        path = Path(item['path'])
        assert fileinfo(path) == {k:item[k] for k in ('bytes','mode','sha256')}
        if str(path) not in source_members:
            rel = 'external-inputs/' + name.replace(':','--') + '/' + path.name
            add(rel, path)
        bindings[name] = {**item, 'archive_path': source_members[str(path)]}
    description = """# Direct differentiable view: four new input boundaries

This packet preserves the sole actual Torch invocation for the fixed cf569335 probe.
It contains original actual JSON, all source/plan/prepare files, both upstream source
files, the actual interpreter and installed metadata inputs, exact child stdout and
stderr, unchanged MCP response text, original durable job identity, source review,
and a pure JSON integrity admission. It contains no new native execution.

Actual job: 1a1e68172b2-392-3b67, succeeded, exit 0.
Key: heptatorch-d266-direct-view-input-boundaries4-actual-20261116-v1.
Pure admission job: 1a1e6844048-392-3b70, succeeded, exit 0.
Actual report SHA256: f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3.
Probe SHA256: cf569335f48624c1edf2f97a649200f2e792ebcef5b6f1df6bac73293ce02efc.

Observed: 4 recipes, 103 successful operations, 7 successful backward operations,
zero probe errors, all eleven source/install provenance checks. The exact original
Version, physical/logical F32 words, T/S relations, Saved observations, warnings,
getter order, complete 5056-byte RNG states and weak lifetime observations remain
in actual/actual.json. All final Storage weak observations expired in these cases.
The 371-byte original child stderr is the optional NumPy-unavailable import warning.

Pure byte admission: 64,059 checks, 2,019 Tensor records, 210 snapshots,
10,120 logical words. It independently gathers logical data from each physical
Storage record; it does not predict correct gradients or compare a native image.
Input-overlap, holes and sibling conclusions are only actual observations. Original
reference14, earlier72, native R1 failure and native R2 results are separate evidence.

The probe itself binds self/support/lock only. The outer actual wrapper reads and
hashes all fifteen prepared source members, including both upstream originals,
the actual repository pip/inventory records, interpreter, virtualenv configuration
and installed Torch metadata. All 25 input bindings are exact before and after.

Restore/verify:
    python -I -S -B verify_packet.py --verify /absolute/path/to/this/packet

The verifier checks every member's complete bytes, size and POSIX mode, the full
decompressed TAR hash, a single XZ stream through EOF, and the 25 input mappings.
Archive members are regular files with safe relative names; there are no links.
Restoring modes requires using the member mode; delivery-file modes are recorded
separately. The packet's scripts refer to the historical ROG paths for execution,
but --verify only reads the chosen packet directory and does not require ROG.

Original durable inventory query's server-side entry-bound refusal is retained in
execution/tool-responses-original.json. It did not launch a process. The one actual
start used the original unique key and returned reused=false. No Torch rerun was
used for source review, byte admission, packaging or verification.

This is observation evidence, not complete API qualification. Closed schemas: 0;
upstream schema denominator: 2590. Values and errors were never substituted.
"""
    readme_path = PACKET / 'README.md'
    with readme_path.open('xb') as out:
        out.write(description.encode())
    add('README.md', readme_path)
    (PACKET / 'verify_packet.py').write_bytes(Path(__file__).read_bytes())
    # Preserve the source mode for the verifier's standalone delivery as well.
    os.chmod(PACKET / 'verify_packet.py', stat.S_IMODE(Path(__file__).stat().st_mode))
    file_records = {name:fileinfo(path) for name,path in sorted(members.items())}
    archive = PACKET / 'complete.tar.xz'
    with archive.open('xb') as raw:
        with lzma.LZMAFile(raw, 'wb', format=lzma.FORMAT_XZ, preset=6) as xz:
            with tarfile.open(fileobj=xz, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                for name,path in sorted(members.items()):
                    data = path.read_bytes()
                    record = file_records[name]
                    assert sha(data) == record['sha256'] and len(data) == record['bytes']
                    item = tarfile.TarInfo(name)
                    item.size = len(data)
                    item.mode = record['mode']
                    item.mtime = 0
                    item.uid = item.gid = 0
                    item.uname = item.gname = ''
                    tar.addfile(item, io.BytesIO(data))
    td, tb = hashlib.sha256(), 0
    with lzma.open(archive, 'rb') as inp:
        while True:
            chunk = inp.read(65536)
            if not chunk:
                break
            td.update(chunk)
            tb += len(chunk)
    original_file_map = {name: {'original_path': str(path), **file_records[name]} for name,path in sorted(members.items())}
    packet_manifest = {
        'schema': 'heptatorch.input4-original-reference-packet.v1',
        'status':'OBSERVED_ONLY', 'native_executed':False,
        'full_pytorch_rewrite_complete':False, 'closed_schema_count':0,
        'pytorch_api_denominator':2590, 'actual_job_id':'1a1e68172b2-392-3b67',
        'pure_admission_job_id':'1a1e6844048-392-3b70',
        'member_count':len(members), 'expanded_bytes':sum(r['bytes'] for r in file_records.values()),
        'archive':fileinfo(archive), 'decompressed_tar':{'bytes':tb,'sha256':td.hexdigest()},
        'files':file_records, 'original_file_map':original_file_map,
        'actual_input_map':bindings, 'original_actual_report_sha256':PIN_ACTUAL,
        'original_source_sha256':PIN_SOURCE,
        'original_execution_sha256':PIN_EXEC,
        'independent_actual_byte_admission_sha256':PIN_ADMISSION,
        'verification_cli':['python','-I','-S','-B','verify_packet.py','--verify','PACKET'],
        'delivery_mode_policy':'Original member modes preserved. Standalone packet files have explicit fileinfo modes.'
    }
    write_json(PACKET / 'manifest.json', packet_manifest)
    verified = verify(PACKET)
    write_json(PACKET / 'verification.json', verified)
    # Verify all original member bytes/modes again after archive verification.
    for name,path in members.items():
        assert fileinfo(path) == file_records[name], name
    outer = {
        'schema':'heptatorch.input4-original-reference-outer-custody.v1',
        'status':'VERIFIED_OBSERVATION_PACKET', 'native_executions':0,'torch_executions':0,
        'all_original_member_sources_unchanged_after_packaging':True,
        'files':{name:fileinfo(PACKET/name) for name in (
            'complete.tar.xz','manifest.json','verification.json','verify_packet.py','README.md')},
        'actual_job_id':'1a1e68172b2-392-3b67',
        'actual_report_sha256':PIN_ACTUAL,
        'native_replay_role':'pytorch_scope; not part of this actual-reference packet'
    }
    write_json(PACKET / 'outer-custody.json', outer)
    print(json.dumps({'status':'PASS','packet':str(PACKET),'member_count':len(members),
                      'expanded_bytes':packet_manifest['expanded_bytes'],
                      'files':{name:fileinfo(PACKET/name) for name in (
                          'complete.tar.xz','manifest.json','verification.json','outer-custody.json','verify_packet.py','README.md')}}))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', type=Path)
    args = ap.parse_args()
    if args.verify is not None:
        print(json.dumps(verify(args.verify)))
    else:
        seal()
