"""Pure filesystem and exact-byte primitives for recorded Sum admission."""
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import lzma
import os
import stat
import struct
import sys

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
BOUND = {}
AUDIT = {"process_attempts": 0, "proc_memory_attempts": 0, "forbidden_import_attempts": 0}
FORBIDDEN = {"subprocess.Popen", "os.system", "os.exec", "os.fork", "os.forkpty",
             "os.posix_spawn", "os.posix_spawnp", "os.kill", "os.killpg",
             "socket.connect", "socket.bind", "ctypes.dlopen", "ctypes.dlsym"}

def no_execution(event, args):
    if event in FORBIDDEN:
        AUDIT["process_attempts"] += 1
        raise AssertionError(("recorded-admission-forbids-process-network-native-load", event))
    if event == "import" and args and str(args[0]).split(".")[0] in {"torch", "ctypes", "subprocess"}:
        AUDIT["forbidden_import_attempts"] += 1
        raise AssertionError(("recorded-admission-forbids-import", args[0]))
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if path.startswith("/proc/"):
            AUDIT["proc_memory_attempts"] += 1
            raise AssertionError(("no-live-or-invented-process-observation", path))

sys.dont_write_bytecode = True
sys.addaudithook(no_execution)

def need(condition, assertion_label, **facts):
    if not condition:
        raise AssertionError({"raw_admission_rule": assertion_label, **facts})

def regular(path):
    p = Path(path).absolute()
    need(not p.is_symlink() and p.resolve() == p, "canonical-no-symlink-input", path=str(p))
    st = p.stat()
    need(stat.S_ISREG(st.st_mode), "regular-input-file", path=str(p))
    return p, st

def stable_stat(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, stat.S_IMODE(st.st_mode))

def remember(p, row):
    key = str(p)
    if key in BOUND:
        need(BOUND[key] == row, "input-byte-mode-stable-on-reread", path=key)
    BOUND[key] = row

def read(path, pin=None, size=None, mode=None):
    p, before = regular(path)
    raw = p.read_bytes()
    after = p.stat()
    need(stable_stat(before) == stable_stat(after), "stable-whole-input-read", path=str(p))
    row = {"bytes": len(raw), "sha256": SHA(raw), "mode": stat.S_IMODE(after.st_mode)}
    if pin is not None:
        need(row["sha256"] == pin, "input-SHA256", path=str(p), got=row["sha256"], expected=pin)
    if size is not None:
        need(row["bytes"] == size, "input-byte-length", path=str(p))
    if mode is not None:
        need(row["mode"] == mode, "input-original-mode", path=str(p))
    remember(p, row)
    return raw

def digest_file(path, pin=None, size=None, mode=None):
    p, before = regular(path)
    h, length = hashlib.sha256(), 0
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
            length += len(block)
    after = p.stat()
    need(stable_stat(before) == stable_stat(after), "stable-streamed-input-read", path=str(p))
    row = {"bytes": length, "sha256": h.hexdigest(), "mode": stat.S_IMODE(after.st_mode)}
    if pin is not None:
        need(row["sha256"] == pin, "streamed-input-SHA256", path=str(p))
    if size is not None:
        need(row["bytes"] == size, "streamed-input-byte-length", path=str(p))
    if mode is not None:
        need(row["mode"] == mode, "streamed-input-mode", path=str(p))
    remember(p, row)
    return row

def data(path, pin=None, size=None, mode=None):
    return json.loads(read(path, pin, size, mode))

def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with p.open("xb") as stream:
        stream.write(raw)
    return {"bytes": len(raw), "sha256": SHA(raw), "mode": stat.S_IMODE(p.stat().st_mode)}

def safe_path(root, rel):
    q = PurePosixPath(rel)
    need(type(rel) is str and rel and not q.is_absolute() and ".." not in q.parts
         and "." not in q.parts and str(q) == rel, "canonical-relative-input-path", path=rel)
    p = Path(root) / rel
    need(p.is_relative_to(Path(root)) and p.resolve() == p.absolute(),
         "contained-no-symlink-input-path", path=str(p))
    return p

def same(a, b, label):
    # Python json retains exact u64 integers; no JS Number or float conversion.
    need(json.loads(json.dumps(a)) == json.loads(json.dumps(b)), label)

def load(path, name, pin):
    read(path, pin)
    need(name not in sys.modules, "unique-frozen-module-name", name=name)
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "explicit-source-module-spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def single_xz(path, expected):
    packed = read(path, expected["compressed_sha256"], expected["compressed_bytes"])
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    raw = decoder.decompress(packed)
    need(decoder.eof and not decoder.unused_data, "one-complete-owner-XZ", path=str(path))
    need(len(raw) == expected["bytes"] and SHA(raw) == expected["sha256"],
         "decompressed-whole-owner", path=str(path))
    return raw

def unpack_frames(raw, magic):
    rows, at = [], 0
    while at < len(raw):
        need(at + 16 <= len(raw), "whole-wire-frame-header")
        tag, size = struct.unpack_from("<8sQ", raw, at)
        at += 16
        need(tag == magic and 32 <= size <= 1 << 20 and at + size <= len(raw),
             "whole-wire-frame-bound")
        rows.append(raw[at:at+size])
        at += size
    need(at == len(raw), "wire-complete-EOF")
    return rows

def recheck_bound():
    errors = []
    for path, expected in list(BOUND.items()):
        try:
            current = digest_file(path, expected["sha256"], expected["bytes"], expected["mode"])
            same(current, expected, "final-input-record-unchanged")
        except BaseException as error:
            errors.append({"path": path, "error": repr(error)})
    return errors
