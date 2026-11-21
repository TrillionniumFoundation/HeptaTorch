"""Source-owned full-contract admission; evidence never chooses executable code.

This module does not qualify a PyTorch contract by itself. A reviewed verifier
must be registered for the exact declaration/subsystem after its implementation
and complete independent qualification exist. There are no such verifiers yet.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import NamedTuple


UPSTREAM = "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457"
MANIFEST = "tools/current_runtime/source/manifest.json"
BASE_ROLES = frozenset(("native_image", "build", "execution", "reference", "readback"))


class Snapshot(NamedTuple):
    kind: str
    name: str
    runtime_manifest_sha256: str
    native_sha256: str
    source_files: object
    files: object


class Verdict(NamedTuple):
    passed: bool
    complete_contract: bool
    checked_roles: frozenset


class Verifier(NamedTuple):
    identifier: str
    required_roles: frozenset
    source_file: Path
    verify: object
    max_file_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024


# Keys are ("native", exact_schema), ("derivative", exact_name), or
# ("subsystem", exact_gate_id). Entries and callables are source-reviewed code,
# never paths/imports supplied by a ledger, command-line flag, or evidence file.
VERIFIERS = MappingProxyType({})


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_object(data):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    def constant(value):
        raise ValueError("non-finite JSON value: " + value)

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    need(isinstance(value, dict), "expected JSON object")
    return value


def sha256_string(value):
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value)


def file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_regular(root, relative, limit, identities=None):
    """Read only the opened regular file, with no-follow traversal and custody.

    All absolute path components are opened relative to held directory FDs.
    Post-read fstat and namespace checks reject replacement during the read.
    A shared identities table also rejects inode aliases and later replacements.
    """
    need(isinstance(relative, str) and relative and "\\" not in relative,
         "canonical relative file path required")
    path = PurePosixPath(relative)
    need(path.parts and not path.is_absolute() and ".." not in path.parts
         and str(path) == relative, "path escapes repository or is not canonical")
    need(root.is_absolute(), "absolute read root required")
    absolute = PurePosixPath(root.as_posix()) / path
    directories = []
    descriptors = []
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    state = identities if identities is not None else {}
    try:
        parent = os.open("/", directory_flags)
        descriptors.append(parent)
        spelling = ""
        for component in absolute.parts[1:-1]:
            child = os.open(component, directory_flags, dir_fd=parent)
            descriptors.append(child)
            identity = file_identity(os.fstat(child))[:2]
            spelling += "/" + component
            old = state.setdefault("directories", {}).setdefault(spelling, identity)
            need(old == identity, "source/evidence directory changed during admission")
            directories.append((parent, component, identity))
            parent = child
        name = absolute.parts[-1]
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                             dir_fd=parent)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit,
             "missing, nonregular or oversized evidence/source file")
        chunks, size = [], 0
        while True:
            chunk = os.read(descriptor, min(65536, limit - size + 1))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            need(size <= limit, "file grew beyond verifier read budget")
        after = os.fstat(descriptor)
        need(file_identity(before) == file_identity(after) and size == after.st_size,
             "source/evidence file changed during descriptor read")
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        need(file_identity(named) == file_identity(after),
             "source/evidence file name changed during descriptor read")
        for directory, component, identity in reversed(directories):
            named_directory = os.stat(component, dir_fd=directory, follow_symlinks=False)
            need(stat.S_ISDIR(named_directory.st_mode)
                 and file_identity(named_directory)[:2] == identity,
                 "source/evidence directory name changed during descriptor read")
        key, identity = str(absolute), file_identity(after)
        files = state.setdefault("files", {})
        need(all(other == key or observed[:2] != identity[:2]
                 for other, observed in files.items()), "duplicate source/evidence inode alias")
        need(key not in files or files[key] == identity,
             "source/evidence object changed during admission")
        files[key] = identity
        return b"".join(chunks)
    except OSError as error:
        raise ValueError("unreadable, replaced or symlink evidence/source path") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def read_pinned(root, record, limit, identities=None):
    need(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"},
         "exact path/bytes/sha256 binding required")
    need(type(record["bytes"]) is int and record["bytes"] >= 0
         and sha256_string(record["sha256"]), "invalid byte/hash binding")
    raw = read_regular(root, record["path"], limit, identities)
    need(len(raw) == record["bytes"] and digest(raw) == record["sha256"],
         "bound evidence bytes changed")
    return raw


def admit(root, kind, name, qualification, registry):
    """Verify one explicit claim; never execute code named by that claim.

    The generic layer checks complete byte custody, current source/image identity
    and the source-owned verifier. That verifier, not JSON status fields, owns
    complete behavioral, backend, resource and independent-reference predicates.
    """
    key = (kind, name)
    need(key in registry, "no reviewed full-contract verifier for this target")
    spec = registry[key]
    need(isinstance(spec, Verifier) and callable(spec.verify)
         and isinstance(spec.identifier, str) and bool(spec.identifier)
         and type(spec.required_roles) is frozenset
         and BASE_ROLES <= spec.required_roles
         and all(isinstance(role, str) and role for role in spec.required_roles),
         "invalid source-owned verifier registration")
    need(type(spec.max_file_bytes) is int and spec.max_file_bytes > 0
         and type(spec.max_total_bytes) is int
         and spec.max_total_bytes >= spec.max_file_bytes, "invalid verifier read budget")
    declared_verifier = Path(spec.source_file).resolve(strict=True)
    actual_verifier = inspect.getsourcefile(spec.verify)
    need(actual_verifier is not None
         and Path(actual_verifier).resolve(strict=True) == declared_verifier,
         "registered verifier source does not own the callback")
    root = root.resolve(strict=True)
    identities = {}
    raw_envelope = read_pinned(root, qualification, spec.max_file_bytes, identities)
    envelope = json_object(raw_envelope)
    need(set(envelope) == {"schema", "upstream_commit", "contract", "verifier_id",
                          "verifier_source_sha256", "admission_engine_sha256",
                          "runtime_manifest_sha256", "files"},
         "exact full-contract evidence envelope required")
    need(envelope["schema"] == "heptatorch.full-contract-evidence.v1"
         and envelope["upstream_commit"] == UPSTREAM
         and envelope["contract"] == {"kind": kind, "name": name},
         "evidence belongs to another contract or upstream target")
    code_identities = {}
    engine_path = Path(__file__).resolve(strict=True)
    verifier_source = read_regular(declared_verifier.parent, declared_verifier.name,
                                   spec.max_file_bytes, code_identities)
    admission_engine = read_regular(engine_path.parent, engine_path.name,
                                    spec.max_file_bytes, code_identities)
    need(envelope["verifier_id"] == spec.identifier
         and envelope["verifier_source_sha256"] == digest(verifier_source),
         "evidence belongs to another verifier revision")
    need(envelope["admission_engine_sha256"] == digest(admission_engine),
         "evidence belongs to another admission engine revision")
    manifest_raw = read_regular(root, MANIFEST, spec.max_file_bytes, identities)
    need(envelope["runtime_manifest_sha256"] == digest(manifest_raw),
         "evidence does not bind the current runtime source manifest")
    manifest = json_object(manifest_raw)
    need(manifest.get("schema") == "heptatorch.complete-runtime-source.v1"
         and sha256_string(manifest.get("expected_native_sha256"))
         and isinstance(manifest.get("runtime"), dict)
         and isinstance(manifest.get("units"), list) and manifest["units"],
         "complete current source manifest required")
    source_files = {MANIFEST: manifest_raw}
    unit_names = set()
    for index, record in enumerate([manifest["runtime"]] + manifest["units"]):
        need(isinstance(record, dict) and sha256_string(record.get("sha256")),
             "invalid current source binding")
        relative = record.get("path")
        need(isinstance(relative, str), "missing current source path")
        full = "tools/current_runtime/source/" + relative
        need(full not in source_files, "duplicate runtime/unit source path")
        if index:
            need(isinstance(record.get("name"), str) and record["name"]
                 and record["name"] not in unit_names
                 and sha256_string(record.get("expected_object_sha256")),
                 "unique named source unit and object identity required")
            unit_names.add(record["name"])
        raw = read_regular(root, full, spec.max_file_bytes, identities)
        need(digest(raw) == record["sha256"], "current runtime/unit source changed")
        source_files[full] = raw
        need(sum(map(len, source_files.values())) + len(raw_envelope) <= spec.max_total_bytes,
             "total verifier source budget exceeded")
    records = envelope["files"]
    need(isinstance(records, dict) and set(records) == spec.required_roles,
         "complete verifier-defined evidence roles required")
    files = {}
    seen_paths = set(source_files) | {qualification["path"]}
    for role, record in records.items():
        raw = read_pinned(root, record, spec.max_file_bytes, identities)
        need(record["path"] not in seen_paths, "evidence roles must have distinct files")
        seen_paths.add(record["path"])
        files[role] = raw
        need(sum(map(len, source_files.values())) + sum(map(len, files.values()))
             + len(raw_envelope) <= spec.max_total_bytes, "total verifier read budget exceeded")
    need(sum(map(len, source_files.values())) + sum(map(len, files.values()))
         + len(raw_envelope) <= spec.max_total_bytes, "total verifier read budget exceeded")
    need(digest(files["native_image"]) == manifest["expected_native_sha256"],
         "evidence executable differs from current native image")
    snapshot = Snapshot(kind, name, digest(manifest_raw), manifest["expected_native_sha256"],
                        MappingProxyType(source_files), MappingProxyType(files))
    verdict = spec.verify(snapshot)
    need(isinstance(verdict, Verdict) and verdict.passed is True
         and verdict.complete_contract is True
         and type(verdict.checked_roles) is frozenset
         and verdict.checked_roles == spec.required_roles,
         "complete contract predicates were not verified")
    # Callbacks see immutable original bytes. Recheck all consumed paths after
    # admission so concurrent edits cannot make a stale snapshot current.
    need(read_regular(declared_verifier.parent, declared_verifier.name,
                      spec.max_file_bytes, code_identities) == verifier_source,
         "verifier changed during admission")
    need(read_regular(engine_path.parent, engine_path.name,
                      spec.max_file_bytes, code_identities) == admission_engine,
         "admission engine changed during admission")
    need(read_pinned(root, qualification, spec.max_file_bytes, identities) == raw_envelope,
         "evidence envelope changed during admission")
    for path, raw in source_files.items():
        need(read_regular(root, path, spec.max_file_bytes, identities) == raw,
             "source changed during admission")
    for role, record in records.items():
        need(read_pinned(root, record, spec.max_file_bytes, identities) == files[role],
             "evidence changed during admission")
    return {"kind": kind, "name": name, "verifier_id": spec.identifier,
            "verifier_source_sha256": digest(verifier_source),
            "admission_engine_sha256": digest(admission_engine),
            "evidence_sha256": digest(raw_envelope),
            "runtime_manifest_sha256": snapshot.runtime_manifest_sha256,
            "native_sha256": snapshot.native_sha256,
            "checked_roles": sorted(verdict.checked_roles)}
