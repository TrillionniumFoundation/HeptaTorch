"""Stream and verify exact regular-file evidence without changing originals."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import lzma
import stat
import tarfile


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include", action="append", required=True, help="archive-prefix=existing file or directory")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    assert not a.archive.exists() and not a.manifest.exists()
    files, roots = {}, []
    for arg in a.include:
        prefix, path = arg.split("=", 1)
        prefix = PurePosixPath(prefix)
        assert str(prefix) not in ("", ".") and not prefix.is_absolute() and ".." not in prefix.parts
        path = Path(path)
        assert not path.is_symlink()
        path = path.resolve(); assert path.exists()
        roots.append({"prefix": str(prefix), "source": str(path)})
        found = [path] if path.is_file() else sorted(path.rglob("*"))
        for source in found:
            assert not source.is_symlink(), str(source)
            if source.is_dir():
                continue
            assert source.is_file() and stat.S_ISREG(source.stat().st_mode)
            name = str(prefix if source == path else prefix / source.relative_to(path).as_posix())
            assert name not in files
            files[name] = {"path": name, "source": str(source), "bytes": source.stat().st_size,
                           "mode": stat.S_IMODE(source.stat().st_mode), "sha256": digest_file(source)}
    assert files
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.archive.with_name(a.archive.name + ".partial")
    assert not temporary.exists()
    with lzma.open(temporary, "wb", format=lzma.FORMAT_XZ, preset=6) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, record in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = record["bytes"], record["mode"], 0
                with Path(record["source"]).open("rb") as stream:
                    archive.addfile(info, stream)
    observed = set()
    with tarfile.open(temporary, "r|xz") as archive:
        for member in archive:
            assert member.isfile() and member.name in files and member.name not in observed
            record = files[member.name]
            assert member.size == record["bytes"] and member.mode == record["mode"]
            stream = archive.extractfile(member)
            digest, size = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block); size += len(block)
            assert size == record["bytes"] and digest.hexdigest() == record["sha256"]
            observed.add(member.name)
    assert observed == set(files)
    # Verify the XZ stream footer as well as tar member contents.
    with lzma.open(temporary, "rb") as stream:
        for _ in iter(lambda: stream.read(1024 * 1024), b""):
            pass
    for record in files.values():
        path = Path(record["source"])
        assert path.stat().st_size == record["bytes"]
        assert stat.S_IMODE(path.stat().st_mode) == record["mode"] and digest_file(path) == record["sha256"]
    temporary.rename(a.archive)
    result = {"schema": "heptatorch.exact-evidence-archive.v1", "status": "PASS",
              "archive": {"name": a.archive.name, "bytes": a.archive.stat().st_size,
                          "sha256": digest_file(a.archive)}, "roots": roots,
              "file_count": len(files), "uncompressed_bytes": sum(x["bytes"] for x in files.values()),
              "member_names_exact": True, "all_member_bytes_hashes_modes_verified": True,
              "xz_footer_verified": True, "source_bytes_modes_unchanged_after": True,
              "originals_deleted": False, "files": [files[name] for name in sorted(files)]}
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("roots", "files")}))


if __name__ == "__main__":
    main()
