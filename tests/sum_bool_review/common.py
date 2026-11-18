"""File bindings and frozen observer reuse for the new Bool test revision."""
import hashlib
import json
import os
import re
import stat
import sys
import types
from pathlib import Path

INTENT = "continuation/20261117/evidence/root-sum-integral-native24-reviewed-intent-v2.json"
INTENT_SHA = "4050bf41236f33080d590d0b65bbf1359f9834d4001eb64883a62cbef2153efc"
PARENT_SHA = {
    "owner_integral.py": "c39600e7c37dea390f88cf9e74d86dfa924301456e0f32e127d82c98ad6778b0",
    "native_observer_integral.py": "787f05d1bc1018bacfb0e21b70dd356b56d776c437decabb728ef7f19d261273",
}
PROTOCOL_SHA = "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
CLIENT_SHA = "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b"


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
    for path in sorted(Path(folder).glob("*.py")):
        inputs.bind(path)


def adapted_parent(repository, filename, inputs, *, owner=False):
    """Construct a new observer revision with an exact, reversible tag/profile delta.

    No original file is edited. Both raw and adapted SHA values and the complete
    edit list enter the run report. The unchanged I64 eight-byte result predicate
    remains in the inherited source and is never relaxed to an arbitrary width.
    """
    raw = inputs.bind(Path(repository) / INTENT, INTENT_SHA)
    row = json.loads(raw)["source_payloads"][filename]
    original = row["text"].encode()
    require(sha(original) == row["sha256"] == PARENT_SHA[filename]
            and len(original) == row["size"] and row["eof"], "frozen-parent-payload", name=filename)
    edits = [(b"HTSINT01", b"HTSINT02", 2 if owner else 1),
             (b"HTOSIN01", b"HTOSIN02", 2 if owner else 1)]
    if owner:
        edits += [(b"functions == 699", b"functions == 700", 1),
                  (b"sum-integral-source01-UNBOUND-699", b"sum-bool-source01-UNBOUND-700", 1),
                  (b"sum-integral-source01-%d-preserved47-48", b"sum-bool-source01-%d-preserved47-48", 1),
                  (b"heptatorch.sum-integral-owner-source01.v1", b"heptatorch.sum-bool-owner.v1", 1)]
    derived = original
    for before, after, count in edits:
        require(derived.count(before) == count and after not in derived, "exact-observer-edit", before=before.decode())
        derived = derived.replace(before, after)
    inverse = derived
    for before, after, count in reversed(edits):
        require(inverse.count(after) == count, "observer-inverse-count")
        inverse = inverse.replace(after, before)
    require(inverse == original, "whole-original-observer-inverse")
    name = "sum_bool_" + filename.replace(".py", "")
    module = types.ModuleType(name)
    module.__file__ = str(Path(repository) / INTENT) + "#" + filename
    sys.modules[name] = module
    exec(compile(derived, module.__file__, "exec"), module.__dict__)
    proof = dict(parent_name=filename, original_sha256=sha(original), derived_sha256=sha(derived),
                 original_bytes=len(original), derived_bytes=len(derived), whole_inverse_exact=True,
                 edits=[dict(before=a.decode(), after=b.decode(), count=n) for a, b, n in edits])
    return module, proof


def source_functions(raw):
    matches = list(re.finditer(rb"^fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n", raw, re.M | re.S))
    require(matches and raw[:matches[0].start()] + b"".join(x[0] for x in matches) == raw,
            "whole-hia-source-function-coverage")
    functions = {m[1].decode(): m[0] for m in matches}
    require(len(functions) == len(matches) == 700, "actual-seven-hundred-functions")
    require("sb_sum_bool" in functions, "actual-bool-kernel")
    return functions
