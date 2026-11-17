#!/usr/bin/env python3
"""Verify frozen evidence bytes/modes and single-XZ/TAR EOF; never execute tests."""
import argparse, hashlib, json, lzma, stat, tarfile
from pathlib import Path, PurePosixPath

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def safe(base, name):
    q=PurePosixPath(name)
    assert name and not q.is_absolute() and ".." not in q.parts and "\\" not in name
    p=base.joinpath(*q.parts)
    assert p.is_file() and not p.is_symlink(), str(p)
    return p

def verify_outer(directory, expected_manifest, expected_schema):
    manifest_path=safe(directory,"package-manifest.json")
    raw=manifest_path.read_bytes()
    assert digest(raw)==expected_manifest
    assert stat.S_IMODE(manifest_path.stat().st_mode)==0o644
    package=json.loads(raw)
    assert package["schema"]==expected_schema
    rows=package["payload"]
    assert len(rows)==package["payload_count"] and len({r["path"] for r in rows})==len(rows)
    assert sum(r["bytes"] for r in rows)==package["payload_bytes"]
    for row in rows:
        p=safe(directory,row["path"]); b=p.read_bytes()
        assert len(b)==row["bytes"] and digest(b)==row["sha256"],row["path"]
        assert stat.S_IMODE(p.stat().st_mode)==row["mode"],row["path"]
    return package

def verify_archive(directory):
    mraw=safe(directory,"file-manifest.json").read_bytes()
    manifest=json.loads(mraw)
    rows=manifest["members"]
    assert len(rows)==manifest["member_count"]
    assert sum(r["bytes"] for r in rows)==manifest["payload_bytes"]
    expected={r["path"]:r for r in rows}
    assert len(expected)==len(rows) and "file-manifest.json" not in expected
    path=safe(directory,"complete.tar.xz")
    decoder=lzma.LZMADecompressor()
    with path.open("rb") as f:
        while not decoder.eof:
            chunk=f.read(65536) if decoder.needs_input else b""
            assert chunk or not decoder.needs_input,"truncated XZ"
            decoder.decompress(chunk,max_length=1048576)
        assert not decoder.unused_data and not f.read(1),"trailing or concatenated XZ stream"
    with tarfile.open(path,"r:xz") as tf:
        members=tf.getmembers()
        assert len(members)==len(rows)+1
        assert len({m.name for m in members})==len(members)
        assert {m.name for m in members}==set(expected)|{"file-manifest.json"}
        for member in members:
            q=PurePosixPath(member.name)
            assert member.isfile() and not q.is_absolute() and ".." not in q.parts
            raw=tf.extractfile(member).read()
            if member.name=="file-manifest.json":
                assert raw==mraw and member.mode==0o644
            else:
                row=expected[member.name]
                assert len(raw)==row["bytes"] and digest(raw)==row["sha256"],member.name
                assert member.mode==row["mode"],member.name
        tail_start=max(m.offset_data+((m.size+511)//512)*512 for m in members)
        tf.fileobj.seek(tail_start)
        tail=tf.fileobj.read()
        assert len(tail)>=1024 and not any(tail),"nonzero or missing TAR end padding"
    return {"archive_sha256":digest(path.read_bytes()),"archive_bytes":path.stat().st_size,
            "file_manifest_sha256":digest(mraw),"archive_members":len(rows)+1,
            "payload_bytes":manifest["payload_bytes"],"all_member_sizes_hashes_modes":True,
            "single_XZ_complete_EOF":True,"TAR_end_padding_zero":True}

def verify(directory, expected_manifest, native_package=None):
    directory=Path(directory)
    package=verify_outer(directory,expected_manifest,"heptatorch.scalar07b.pure-raw-admission-handoff.v1")
    archive=verify_archive(directory)
    report=json.loads(safe(directory,"qualification-report.json").read_bytes())
    fail=json.loads(safe(directory,"first-failure-report.json").read_bytes())
    summary=json.loads(safe(directory,"summary.json").read_bytes())
    assert report["status"]=="PASS_PURE_RECORDED_EVIDENCE" and len(report["cases"])==76
    assert fail["status"]=="FAIL_PURE_ADMISSION" and fail["error"]["type"]=="TypeError" and fail["cases"]==[]
    assert report["all_inputs_unchanged"] and report["error"] is None
    assert report["audit_guard"]=={"process_attempts":0,"proc_memory_attempts":0}
    assert summary["qualification_report_sha256"]==digest(safe(directory,"qualification-report.json").read_bytes())
    assert summary["first_FAILURE_report_sha256"]==digest(safe(directory,"first-failure-report.json").read_bytes())
    native_result=None
    if native_package is not None:
        dependency=json.loads(safe(directory,"external-native-package.json").read_bytes())
        native_package=Path(native_package)
        verify_outer(native_package,dependency["package_manifest_sha256"],
                     "heptatorch.scalar07b.native-four-suite-handoff.v1")
        native_result=verify_archive(native_package)
        assert native_result["archive_sha256"]==dependency["archive_sha256"]
        assert native_result["archive_bytes"]==dependency["archive_bytes"]
        assert native_result["file_manifest_sha256"]==dependency["member_manifest_sha256"]
    return {"schema":"heptatorch.scalar07b.pure-admission-package-verification.v1",
            "status":"PASS_BYTE_MODE_SINGLE_STREAM_VERIFICATION",
            "package_manifest_sha256":expected_manifest,"outer_payload_count":package["payload_count"],
            **archive,"original_native_package_verified_here":native_result,
            "case_admissions_executed_here":0,"native_executions":0,"Torch_executions":0,
            "compiler_executions":0,"oracle_computations":0}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,default=Path(__file__).parent)
    parser.add_argument("--expected-manifest-sha256",required=True)
    parser.add_argument("--native-package",type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.package,args.expected_manifest_sha256,args.native_package),indent=2))
