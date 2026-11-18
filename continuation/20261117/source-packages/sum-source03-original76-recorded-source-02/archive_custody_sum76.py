"""Full byte/mode custody for original696 four-suite and pure32 supplemental evidence."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import lzma
import os
import stat
import tarfile
from admission_common import SHA, need, read, data, write, safe_path, same, digest_file

MEMBER_SCHEMA = "heptatorch.sum03.original76.complete-supplement-members.v1"
BINDING_SCHEMA = "heptatorch.sum03.original76.supplement-binding.v1"

def inventory(directory):
    root = Path(directory).resolve()
    need(root.is_dir() and not root.is_symlink(), "real-original-evidence-directory")
    files, directories = {}, {"": stat.S_IMODE(root.stat().st_mode)}
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        names.sort()
        parent = Path(base)
        for name in dirs:
            p = parent/name
            need(p.is_dir() and not p.is_symlink(), "no-symlink-or-special-evidence-directory", path=str(p))
            directories[p.relative_to(root).as_posix()] = stat.S_IMODE(p.stat().st_mode)
        for name in names:
            p = parent/name
            need(not p.is_symlink() and stat.S_ISREG(p.stat().st_mode),
                 "only-regular-original-evidence-files", path=str(p))
            rel = p.relative_to(root).as_posix()
            files[rel] = {**digest_file(p), "source_path": str(p)}
    return files, directories

def full_xz(path):
    digest_file(path)
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    raw_hash, raw_bytes = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while not decoder.eof:
            block = stream.read(65536) if decoder.needs_input else b""
            need(block or not decoder.needs_input, "one-XZ-stream-not-truncated")
            raw = decoder.decompress(block, max_length=1 << 20)
            raw_hash.update(raw)
            raw_bytes += len(raw)
        need(not decoder.unused_data and stream.read(1) == b"", "one-XZ-stream-complete-physical-EOF")
    return {"uncompressed_tar_bytes": raw_bytes, "uncompressed_tar_sha256": raw_hash.hexdigest(),
            "single_XZ_stream": True}

def create_archive(output, manifest):
    path = Path(output)
    need(not path.exists(), "create-only-complete-archive")
    with path.open("xb") as target:
        with lzma.LZMAFile(target, "w", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64) as packed:
            with tarfile.open(fileobj=packed, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
                for name, mode in sorted(manifest["directories"].items()):
                    safe_path(path.parent, name)
                    info = tarfile.TarInfo(name+"/")
                    info.type, info.size, info.mode = tarfile.DIRTYPE, 0, mode
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    tar.addfile(info)
                for name, entry in sorted(manifest["files"].items()):
                    safe_path(path.parent, name)
                    p = Path(entry["source_path"])
                    digest_file(p, entry["sha256"], entry["bytes"], entry["mode"])
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = entry["bytes"], entry["mode"]
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    with p.open("rb") as source:
                        tar.addfile(info, source)
                    digest_file(p, entry["sha256"], entry["bytes"], entry["mode"])
    return {**digest_file(path), **full_xz(path)}

def verify_archive(path, manifest, expected):
    need(manifest["schema"] == MEMBER_SCHEMA, "dedicated-original-raw-member-schema")
    need(set(manifest) == {"schema", "native_output", "pure_output", "native_report_pins",
         "native_original_statuses", "pure_report_sha256", "files", "directories",
         "synthetic_container_directories", "file_count", "directory_count", "payload_bytes",
         "input_bindings", "execution_evidence", "source_profile_sha256", "mode_policy",
         "manifest_is_outside_archive", "native_Torch_compiler_or_recorded_admission_executed",
         "shared_carrier"}, "exact-whole-supplement-member-manifest-fields")
    need(manifest["manifest_is_outside_archive"] is True
         and manifest["native_Torch_compiler_or_recorded_admission_executed"] is False,
         "member-manifest-custody-only-scope")
    files, directories = manifest["files"], manifest["directories"]
    input_members = {}
    for index, (input_path, row) in enumerate(sorted(manifest["input_bindings"].items())):
        need(set(row) == {"path", "bytes", "sha256", "mode"} and row["path"] == input_path,
             "full-original-declared-input-record")
        name = "inputs/%04d-%s" % (index, Path(input_path).name)
        input_members[name] = {key: row[key] for key in ("bytes", "sha256", "mode")}
        input_members[name]["source_path"] = input_path
    same({name: row for name, row in files.items() if name.startswith("inputs/")},
         input_members, "every-original-input-is-an-exact-archive-member")
    receipts = {}
    for index, row in enumerate(manifest["execution_evidence"]):
        name = "execution/%04d-%s" % (index, Path(row["path"]).name)
        receipts[name] = {key: row[key] for key in ("bytes", "sha256", "mode")}
        receipts[name]["source_path"] = row["path"]
    same({name: row for name, row in files.items()
          if name.startswith("execution/") and name != "execution/seal-plan.json"},
         receipts, "every-original-execution-evidence-file-is-an-exact-member")
    need("execution/seal-plan.json" in files, "original-seal-plan-member")
    need(type(files) is dict and type(directories) is dict and set(files).isdisjoint(directories),
         "distinct-exact-file-directory-sets")
    need(manifest["file_count"] == len(files) and manifest["directory_count"] == len(directories)
         and manifest["payload_bytes"] == sum(entry["bytes"] for entry in files.values()),
         "whole-original-member-counts")
    for name, entry in files.items():
        safe_path(Path(path).parent, name)
        need(set(entry) == {"bytes", "sha256", "mode", "source_path"}
             and type(entry["bytes"]) is int and entry["bytes"] >= 0
             and type(entry["mode"]) is int and 0 <= entry["mode"] <= 0o7777,
             "exact-original-member-metadata")
    for name, mode in directories.items():
        safe_path(Path(path).parent, name)
        need(type(mode) is int and 0 <= mode <= 0o7777, "original-directory-mode")
    digest_file(path, expected["sha256"], expected["bytes"], expected["mode"])
    same(full_xz(path), {key: expected[key] for key in (
        "uncompressed_tar_bytes", "uncompressed_tar_sha256", "single_XZ_stream")},
        "complete-XZ-and-uncompressed-tar-digest")
    seen_files, seen_dirs = set(), set()
    with tarfile.open(path, "r:xz") as tar:
        for member in tar:
            name = member.name.rstrip("/") if member.isdir() else member.name
            need(not member.pax_headers and not member.issym() and not member.islnk()
                 and not member.issparse(), "no-unbound-archive-extension-link-or-sparse-member")
            if member.isdir():
                need(name in directories and name not in seen_dirs
                     and member.size == 0 and member.mode == directories[name],
                     "exact-original-directory-member", path=name)
                seen_dirs.add(name)
                continue
            need(member.isfile() and name in files and name not in seen_files,
                 "unique-declared-regular-tar-member", path=name)
            seen_files.add(name)
            entry = files[name]
            need(member.size == entry["bytes"] and member.mode == entry["mode"],
                 "complete-member-length-and-original-mode", path=name)
            source = tar.extractfile(member)
            need(source is not None, "whole-member-stream")
            h, count = hashlib.sha256(), 0
            for block in iter(lambda: source.read(1 << 20), b""):
                h.update(block)
                count += len(block)
            need(count == entry["bytes"] and h.hexdigest() == entry["sha256"],
                 "whole-member-SHA256", path=name)
        # The LZMA trailer was already consumed above. TAR may stop at its first
        # zero header; all remaining uncompressed bytes must also be padding.
        tar.fileobj.seek(tar.offset)
        for block in iter(lambda: tar.fileobj.read(1 << 20), b""):
            need(not any(block), "no-hidden-members-or-data-after-first-tar-EOF")
    need(seen_files == set(files) and seen_dirs == set(directories), "exact-complete-tar-member-set")
    return {"file_count": len(files), "directory_count": len(directories),
            "payload_bytes": manifest["payload_bytes"],
            "archive_sha256": expected["sha256"],
            "all_file_bytes_modes_and_XZ_TAR_EOF_verified": True}

def verify_originals(manifest):
    for name, entry in manifest["files"].items():
        digest_file(entry["source_path"], entry["sha256"], entry["bytes"], entry["mode"])
    checked = {}
    for prefix, field in (("native-output", "native_output"), ("pure32-output", "pure_output")):
        root = Path(manifest[field])
        current_files, current_dirs = inventory(root)
        expected_files = {name[len(prefix)+1:]: entry for name, entry in manifest["files"].items()
                          if name.startswith(prefix+"/")}
        expected_dirs = {}
        for name, mode in manifest["directories"].items():
            if name == prefix:
                expected_dirs[""] = mode
            elif name.startswith(prefix+"/"):
                expected_dirs[name[len(prefix)+1:]] = mode
        same(current_files, expected_files, "all-original-tree-files-still-exact/"+prefix)
        same(current_dirs, expected_dirs, "all-original-tree-directories-still-exact/"+prefix)
        checked[prefix] = {"files": len(current_files), "directories": len(current_dirs)}
    return {"original_roots": checked, "all_archive_source_files_unchanged": True}

# Binding identity and both shared/supplemental carriers are admitted by supplement.py.
