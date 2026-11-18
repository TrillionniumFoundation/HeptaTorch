"""Stable input bindings for the isolated F32 sum reference producer."""
import hashlib
import json
import stat
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, rule, **facts):
    if not condition:
        raise AssertionError(dict(rule=rule, **facts))


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


class Inputs:
    def __init__(self):
        self.rows = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        require(path.is_file() and not path.is_symlink(), "regular-input", path=str(path))
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)
        require(fields(before) == fields(after) and stat.S_ISREG(after.st_mode), "stable-input", path=str(path))
        digest = sha(raw)
        require(expected is None or digest == expected, "input-hash", path=str(path), actual=digest, expected=expected)
        row = dict(path=str(path), bytes=len(raw), sha256=digest, mode=stat.S_IMODE(after.st_mode))
        require(str(path) not in self.rows or self.rows[str(path)] == row, "input-rebinding")
        self.rows[str(path)] = row
        return raw

    def unchanged(self):
        old = dict(self.rows)
        for path, item in old.items():
            self.bind(path, item["sha256"])
        require(old == self.rows, "all-inputs-unchanged")
        return True


def bind_sources(inputs, folder):
    for name in ("reference.py", "cases.py", "common.py"):
        inputs.bind(Path(folder) / name)
