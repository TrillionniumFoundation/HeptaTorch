"""Bind inherited regressions to one actual complete native build."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat


ROOT = Path(__file__).resolve().parents[2]
FUNCTION = re.compile(rb"^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)", re.M | re.S)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class Inputs:
    def __init__(self):
        self.records = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        require(not path.is_symlink(), "symlinked input: " + str(path))
        before = path.stat()
        require(stat.S_ISREG(before.st_mode), "nonregular input: " + str(path))
        raw = path.read_bytes()
        after = path.stat()
        fields = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(fields(before) == fields(after), "input changed while read: " + str(path))
        row = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row["sha256"] == expected, "input SHA mismatch: " + str(path))
        require(str(path) not in self.records or self.records[str(path)] == row,
                "bound input changed: " + str(path))
        self.records[str(path)] = row
        return raw

    def recheck(self):
        errors = []
        for path, row in list(self.records.items()):
            try:
                self.bind(path, row["sha256"])
            except BaseException as error:
                errors.append(dict(path=path, error=repr(error)))
        return errors


def bind_build(directory, expected, inputs):
    """Use the common current-source admission; execute regressions separately."""
    directory = Path(directory).resolve()
    helper = ROOT / "tools/current_runtime/qualification.py"
    inputs.bind(helper)
    spec = importlib.util.spec_from_file_location("inherited_current_build_admission", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = module.admit_build(directory, inputs.bind, expected)
    source = inputs.bind(directory / "runtime.hepta", built["source_sha256"])
    bodies = {name.decode(): body for body, name in FUNCTION.findall(source)}
    require(len(bodies) == built["functions"] == built["complete_core_bodies_matched"],
            "complete core function count")
    return built, bodies
