#!/usr/bin/env python3
"""Copy original observer reports and inventory raw evidence without judging suites.

The manifest describes byte custody only. COMPLETE is never a test PASS.
Run after all writers have stopped; the artifact consumer must recheck every
recorded size and SHA256 after downloading the separately uploaded artifact.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid

SCHEMA = "heptatorch.private-node-journal-custody.v1"
CHUNK = 1024 * 1024
SHA_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class Refusal(RuntimeError):
    pass


def absolute(value):
    value = os.fspath(value)
    if ".." in Path(value).parts:
        raise Refusal("Parent traversal is not admitted: " + value)
    return Path(os.path.abspath(value))


def below(path, root):
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        raise Refusal("Path escapes the declared build root: " + str(path))
    if any(p in ("", ".", "..") for p in parts):
        raise Refusal("Invalid path component: " + str(path))
    return parts


def directory_at(parent, parts, create=False):
    fd = os.dup(parent)
    try:
        for part in parts:
            if part in ("", ".", "..") or "/" in part or "\0" in part:
                raise Refusal("Invalid relative directory component")
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=fd)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextlib.contextmanager
def absolute_directory(path, create=False):
    root = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = directory_at(root, path.parts[1:], create=create)
    finally:
        os.close(root)
    try:
        yield fd
    finally:
        os.close(fd)


def regular_at(parent, name):
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise Refusal("Expected a nonlinked ordinary file: " + name)
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
        os.close(fd)
        raise Refusal("File identity changed while opening: " + name)
    return fd


def fingerprint(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_nlink,
            s.st_mtime_ns, s.st_ctime_ns)


def digest_fd(fd, copy_to=None):
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise Refusal("Expected a nonlinked ordinary file descriptor")
    h, size = hashlib.sha256(), 0
    while True:
        block = os.read(fd, CHUNK)
        if not block:
            break
        h.update(block)
        size += len(block)
        if copy_to is not None:
            view = memoryview(block)
            while view:
                wrote = os.write(copy_to, view)
                if wrote <= 0:
                    raise OSError("Short report-copy write made no progress")
                view = view[wrote:]
    after = os.fstat(fd)
    if size != before.st_size or fingerprint(before) != fingerprint(after):
        raise Refusal("File changed while its bytes were read")
    return {"size": size, "sha256": h.hexdigest(),
            "mode": stat.S_IMODE(before.st_mode)}


def digest_path(build_fd, parts):
    parent = directory_at(build_fd, parts[:-1])
    try:
        fd = regular_at(parent, parts[-1])
        try:
            return digest_fd(fd)
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def mirror_report(build_fd, source_parts, mirror_parts):
    source_parent = directory_at(build_fd, source_parts[:-1])
    source = None
    destination_parent = None
    temporary_fd = None
    temporary_name = None
    try:
        source = regular_at(source_parent, source_parts[-1])
        destination_parent = directory_at(build_fd, mirror_parts[:-1], create=True)
        try:
            existing = regular_at(destination_parent, mirror_parts[-1])
        except FileNotFoundError:
            existing = None
        if existing is not None:
            try:
                observed = digest_fd(existing)
            finally:
                os.close(existing)
            original = digest_fd(source)
            if any(original[k] != observed[k] for k in ("size", "sha256")):
                raise Refusal("Existing report mirror differs; refusing overwrite")
            return {**original, "mirror_mode": observed["mode"], "reused_identical_mirror": True}
        temporary_name = ".report-copy-" + uuid.uuid4().hex
        temporary_fd = os.open(temporary_name,
                               os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                               0o600, dir_fd=destination_parent)
        original = digest_fd(source, temporary_fd)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        # Hard-link publication refuses an existing destination; the temporary
        # name is removed before any inventory, leaving exactly one file link.
        os.link(temporary_name, mirror_parts[-1],
                src_dir_fd=destination_parent, dst_dir_fd=destination_parent,
                follow_symlinks=False)
        os.unlink(temporary_name, dir_fd=destination_parent)
        temporary_name = None
        copied = regular_at(destination_parent, mirror_parts[-1])
        try:
            observed = digest_fd(copied)
        finally:
            os.close(copied)
        if any(original[k] != observed[k] for k in ("size", "sha256")):
            raise Refusal("Published report mirror does not match original bytes")
        return {**original, "mirror_mode": observed["mode"], "reused_identical_mirror": False}
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        # A failed partial temporary copy remains as evidence and is inventoried.
        if source is not None:
            os.close(source)
        if destination_parent is not None:
            os.close(destination_parent)
        os.close(source_parent)


def inventory(root, prefix, manifest_path, errors):
    rows = []

    def walk(fd, components):
        try:
            names = sorted(os.listdir(fd))
        except OSError as error:
            errors.append({"phase": "inventory", "path": "/".join(components),
                           "error": type(error).__name__ + ": " + str(error)})
            return
        for name in names:
            path = "/".join((*components, name))
            if path == manifest_path:
                continue
            try:
                entry = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(entry.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=fd)
                    try:
                        current = os.fstat(child)
                        if (current.st_dev, current.st_ino) != (entry.st_dev, entry.st_ino):
                            raise Refusal("Directory changed while opening")
                        walk(child, (*components, name))
                    finally:
                        os.close(child)
                elif stat.S_ISREG(entry.st_mode):
                    source = regular_at(fd, name)
                    try:
                        rows.append({"path": path, **digest_fd(source)})
                    finally:
                        os.close(source)
                else:
                    raise Refusal("Symlinks and special files are refused")
            except (OSError, Refusal) as error:
                errors.append({"phase": "inventory", "path": path,
                               "error": type(error).__name__ + ": " + str(error)})
    walk(root, tuple(prefix))
    return sorted(rows, key=lambda row: row["path"])


def git_snapshot():
    def git(*args):
        result = subprocess.run(["git", "--no-optional-locks", "-c", "core.fsmonitor=false",
                                 *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False, timeout=30)
        if result.returncode:
            raise Refusal("Read-only git command failed: " +
                          result.stderr.decode("utf-8", "replace")[:2000])
        return result.stdout
    head = git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
    tree = git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    if not SHA_PATTERN.fullmatch(head) or not SHA_PATTERN.fullmatch(tree):
        raise Refusal("Unexpected Git object identity")
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    return {"head": head, "tree": tree, "clean": not status,
            "status_bytes": len(status), "status_sha256": hashlib.sha256(status).hexdigest()}


def context(before, errors):
    keys = ("GITHUB_ACTIONS", "GITHUB_REPOSITORY", "GITHUB_SERVER_URL", "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "GITHUB_SHA", "GITHUB_EVENT_NAME",
            "GITHUB_WORKFLOW", "GITHUB_REF", "GITHUB_HEAD_REF", "GITHUB_BASE_REF",
            "RUNNER_OS", "RUNNER_ARCH")
    github = {key: os.environ.get(key) for key in keys}
    source_head = os.environ.get("HEPTATORCH_SOURCE_HEAD_SHA")
    source_basis = "HEPTATORCH_SOURCE_HEAD_SHA supplied by the workflow"
    if source_head is None and github["GITHUB_ACTIONS"] != "true":
        source_head = before.get("head") if before else None
        source_basis = "local checkout HEAD; not a hosted GitHub source-head observation"
    if source_head is None or not SHA_PATTERN.fullmatch(source_head):
        errors.append({"phase": "identity", "error": "Missing or malformed source-head identity"})
    name = os.environ.get("HEPTATORCH_OBSERVER_ARTIFACT")
    if github["GITHUB_ACTIONS"] == "true":
        for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB"):
            if not github[key]:
                errors.append({"phase": "identity", "error": "Missing " + key})
        if not name:
            errors.append({"phase": "identity", "error": "Missing artifact name"})
    return {"github": github, "source_head": source_head, "source_head_basis": source_basis,
            "artifact": {"name": name, "numeric_id": None, "upload_observed": False,
                         "identity_boundary": "The collector runs before upload. The actual artifact ID and archive SHA256 must be recorded by a later upload/download receipt."}}


def publish_json(parent, name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = ".custody-copy-" + uuid.uuid4().hex
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=parent)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("Manifest write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
    os.unlink(temporary, dir_fd=parent)
    fd = regular_at(parent, name)
    try:
        result = digest_fd(fd)
    finally:
        os.close(fd)
    if result["size"] != len(raw) or result["sha256"] != hashlib.sha256(raw).hexdigest():
        raise Refusal("Manifest publication readback differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--journal-root", type=Path, required=True)
    parser.add_argument("--wire-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--job-status", choices=("success", "failure", "cancelled"), required=True)
    args = parser.parse_args()
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise Refusal("Descriptor-safe POSIX file access is required")
    build = absolute(args.build)
    journal, wire = absolute(args.journal_root), absolute(args.wire_root)
    roots = [below(journal, build), below(wire, build)]
    if not all(roots) or journal == wire or journal in wire.parents or wire in journal.parents:
        raise Refusal("Journal and wire roots must be disjoint strict build descendants")
    reports = [absolute(p) for p in args.reports]
    if len(set(reports)) != len(reports):
        raise Refusal("Duplicate report paths")
    report_parts = []
    for path in reports:
        parts = below(path, build)
        if not parts or any(path == root or root in path.parents for root in (journal, wire)):
            raise Refusal("Original reports must be outside both collected roots")
        report_parts.append(parts)
    errors, missing, copies = [], [], []
    try:
        before = git_snapshot()
    except (OSError, Refusal, subprocess.SubprocessError) as error:
        before = None
        errors.append({"phase": "git-before", "error": type(error).__name__ + ": " + str(error)})
    identity = context(before, errors)
    manifest_path = "/".join((*roots[0], "custody.json"))
    with absolute_directory(build, create=True) as build_fd:
        root_fds = []
        try:
            for label, parts in zip(("journal", "wire"), roots):
                try:
                    fd = directory_at(build_fd, parts)
                except FileNotFoundError:
                    missing.append({"kind": "root", "path": "/".join(parts), "role": label})
                    fd = directory_at(build_fd, parts, create=True)
                root_fds.append(fd)
            journal_fd, wire_fd = root_fds
            try:
                os.stat("custody.json", dir_fd=journal_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise Refusal("Existing custody.json is never overwritten")
            for parts in report_parts:
                original = "/".join(parts)
                mirror_parts = (*roots[0], "report-bytes", *parts)
                record = {"original": original, "mirror": "/".join(mirror_parts)}
                try:
                    result = mirror_report(build_fd, parts, mirror_parts)
                    copies.append({**record, **result, "status": "COPIED_BYTES_VERIFIED"})
                except FileNotFoundError:
                    missing.append({"kind": "report", "path": original})
                    copies.append({**record, "status": "MISSING"})
                except (OSError, Refusal) as error:
                    record.update(status="REFUSED", error=type(error).__name__ + ": " + str(error))
                    copies.append(record)
                    errors.append({"phase": "report-copy", **record})
            rows = inventory(journal_fd, roots[0], manifest_path, errors)
            rows += inventory(wire_fd, roots[1], manifest_path, errors)
            rows.sort(key=lambda row: row["path"])
            again_errors = []
            again = inventory(journal_fd, roots[0], manifest_path, again_errors)
            again += inventory(wire_fd, roots[1], manifest_path, again_errors)
            again.sort(key=lambda row: row["path"])
            if rows != again or again_errors:
                errors.append({"phase": "recheck", "error": "Raw evidence inventory changed or remained unreadable",
                               "details": again_errors})
            for row in copies:
                if row["status"] != "COPIED_BYTES_VERIFIED":
                    continue
                try:
                    current = digest_path(build_fd, tuple(row["original"].split("/")))
                    if any(current[key] != row[key] for key in ("size", "sha256")):
                        raise Refusal("Original report changed after it was mirrored")
                except (OSError, Refusal) as error:
                    errors.append({"phase": "original-report-recheck", "path": row["original"],
                                   "error": type(error).__name__ + ": " + str(error)})
            try:
                after = git_snapshot()
            except (OSError, Refusal, subprocess.SubprocessError) as error:
                after = None
                errors.append({"phase": "git-after", "error": type(error).__name__ + ": " + str(error)})
            if before != after:
                errors.append({"phase": "checkout", "error": "Checkout identity or status changed during collection"})
            if args.job_status == "success" and (after is None or not after["clean"]):
                errors.append({"phase": "checkout", "error": "Successful job requires a clean checkout"})
            complete = not errors and not missing and len(copies) == len(reports)
            result = {"schema": SCHEMA, "status": "COMPLETE" if complete else "PARTIAL",
                      "status_scope": "Byte custody only. Neither status defines or changes any suite result.",
                      "job_status_received": args.job_status,
                      "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "build_argument": str(args.build), "build_absolute": str(build),
                      "journal_root": "/".join(roots[0]), "wire_root": "/".join(roots[1]),
                      "artifact_common_parent_relative_to_build":
                          str(Path(os.path.commonpath((str(journal), str(wire)))).relative_to(build)),
                      "custody_path": manifest_path, "custody_excluded_from_files": True,
                      "identity": identity, "checkout_before": before, "checkout_after": after,
                      "requested_report_count": len(reports), "report_copies": copies,
                      "file_count": len(rows), "total_file_bytes": sum(row["size"] for row in rows),
                      "files": rows, "missing": missing, "errors": errors,
                      "report_contents_parsed_or_rewritten": False,
                      "native_executed": False, "torch_executed": False,
                      "full_pytorch_rewrite_complete": False}
            manifest_identity = publish_json(journal_fd, "custody.json", result)
            print(json.dumps({"status": result["status"], "status_scope": result["status_scope"],
                              "custody": manifest_path, **manifest_identity,
                              "file_count": len(rows), "requested_report_count": len(reports),
                              "missing_count": len(missing), "error_count": len(errors)}, sort_keys=True))
            return 0 if complete else 1
        finally:
            for fd in root_fds:
                os.close(fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, Refusal, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "PARTIAL", "status_scope": "Collector configuration or publication refusal; no suite result is inferred.",
                          "error": type(error).__name__ + ": " + str(error),
                          "native_executed": False, "torch_executed": False}), file=sys.stderr)
        raise SystemExit(2)
