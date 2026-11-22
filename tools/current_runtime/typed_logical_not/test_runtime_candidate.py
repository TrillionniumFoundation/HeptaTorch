"""Focused production-protocol test on an opt-in native logical-not image."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import struct
import tempfile

from test_component import WIDTHS, known_words, oracle

NATIVE_SHA256 = "0cd10504717c5e5b01114455c55030e63a22b62bb1ba43f188355d399d6c158a"
BASE_SHA256 = "083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5"
PROTOCOL_SHA256 = "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
OWNER_CLIENT_SHA256 = "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wire(path):
    # The original protocol imports its sibling by file path. Copy both checked
    # byte strings into a fresh directory so neither can load an old .pyc.
    with tempfile.TemporaryDirectory(prefix="heptatorch-logical-not-wire-") as directory:
        isolated = Path(directory)
        for name, expected in (("protocol.py", PROTOCOL_SHA256),
                               ("owner_client.py", OWNER_CLIENT_SHA256)):
            source = path if name == "protocol.py" else path.with_name(name)
            raw = source.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == expected, "wrong exact production input: " + name)
            (isolated / name).write_bytes(raw)
        module_spec = importlib.util.spec_from_file_location("logical_not_fixed_wire", isolated / "protocol.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module


def factory(dtype, shape, words, command_count=1):
    """Use literal public typed factory bytes, optionally in a same-frame transaction."""
    raw = b"".join(word.to_bytes(WIDTHS[dtype], "little") for word in words)
    end = 32 + 64 * command_count
    cfg = struct.pack("<8Q", len(shape), 0, *shape, *([0] * (4 - len(shape))), end + 64, len(raw))
    return (b"HTORCH03" + struct.pack("<3Q", command_count, end + 64 + len(raw), 0),
            struct.pack("<8Q", 97, dtype, end, 0, 0, 0, 0, 0), cfg + raw)


def run_native(w, binary, baseline, evidence_dir=None):
    require(__debug__, "run with assertions enabled for the fixed protocol")
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=False)

    def start_session(image, name):
        return w.Session(image, directory=evidence_dir / name if evidence_dir else None)
    require(digest(binary) == NATIVE_SHA256, "wrong exact opt-in candidate image")
    require(digest(baseline) == BASE_SHA256, "wrong exact #57 original image")

    def exchange(session, request, expected, opcodes=()):
        status, rows = w.parse(session.exchange(request))
        require(status == expected, ("unexpected status", expected, status))
        require([r["opcode"] for r in rows] == list(opcodes), "missing or mismatched command records")
        return rows

    def one(session, *command):
        return exchange(session, w.packet([command]), 0, [command[0]])[0]

    def create(session, dtype, shape, words):
        return exchange(session, b"".join(factory(dtype, shape, words)), 0, [97])[0]["value"]

    def inspect(session, handle):
        status, rows = w.parse(session.exchange(w.packet([(13, handle), (96, handle, 0)])))
        require(status == 0 and [r["opcode"] for r in rows] == [13, 96], "missing tensor metadata")
        meta = w.meta(rows[0])
        dtype, width, _, _ = struct.unpack("<4Q", rows[1]["payload"])
        body = one(session, 98, handle, 0, meta["numel"])
        require(body["value"] == meta["numel"], "logical readback extent")
        return meta, dtype, width, body["payload"]

    def view(session, owner, shape, strides, offset=0):
        cfg = struct.pack("<10Q", len(shape), *shape, *([0]*(4-len(shape))), *strides, *([0]*(4-len(strides))), offset)
        return one(session, 111, owner, w.Payload(cfg), len(cfg))["value"]

    def check_output(session, source, result, dtype, words, original, strides=None):
        result_meta, result_dtype, result_width, body = inspect(session, result)
        rank = original[0]["rank"]
        require((result_dtype, result_width, body) == (1, 1, bytes(oracle(dtype, words))), "typed Boolean bytes")
        require((result_meta["rank"], result_meta["shape"]) == (rank, original[0]["shape"]), "rank and shape")
        require(result_meta["requires_grad"] == 0 and result_meta["storage_offset"] == 0, "no gradient/offset in result")
        require(result != source and result_meta["storage_id"] != original[0]["storage_id"]
                and result_meta["version_id"] != original[0]["version_id"], "fresh independent owner identities")
        if strides is not None:
            require(result_meta["strides"][:rank] == strides, "dense input permutation preserved")
        require(inspect(session, source) == original, "source metadata and numerical bytes unchanged")

    candidate = start_session(binary, "candidate")
    parent = start_session(baseline, "baseline")
    dense = 0
    parent_checks = 0
    controls = 0
    try:
        for dtype in WIDTHS:
            if dtype == 3:
                continue  # wrapped F64 is constructed separately, not through the public factory.
            examples = known_words(dtype)
            if dtype in (5,6):
                examples = examples[::1021] + [0, 0x8000, 0x7c00, 0xfc00, 0xffff]
            if dtype in (0,4):
                examples = examples[:101]
            for shape, words in [((), [examples[0]]), ((len(examples),), examples), ((0,3), [])]:
                source = create(candidate, dtype, shape, words)
                old_source = create(parent, dtype, shape, words)
                if dense == 0:
                    # Reuse the exact same valid handle on both cold builds.
                    require(source == old_source, "cold factory handles differ")
                    exchange(parent, w.packet([(127, old_source)]), 4)
                    controls += 1
                original = inspect(candidate, source)
                result = one(candidate, 127, source)["value"]
                check_output(candidate, source, result, dtype, words, original)
                one(candidate, 41, result)
                if words:
                    for op in (120,121,122,123,124,126):
                        a = one(candidate, op, source)["value"]
                        b = one(parent, op, old_source)["value"]
                        x, y = inspect(candidate, a), inspect(parent, b)
                        owners = {"storage_id", "version_id"}
                        require(({k:v for k,v in x[0].items() if k not in owners}, *x[1:])
                             == ({k:v for k,v in y[0].items() if k not in owners}, *y[1:]),
                                f"preexisting opcode {op} result changed")
                        one(candidate, 41, a)
                        one(parent, 41, b)
                        parent_checks += 1
                one(candidate, 41, source)
                one(parent, 41, old_source)
                dense += 1

        source = create(candidate, 4, (3,2), [0,1,0x8000000000000000,2,3,0])
        transposed = view(candidate, source, (2,3), (1,2))
        original = inspect(candidate, transposed)
        result = one(candidate, 127, transposed)["value"]
        expected_order = [0,0x8000000000000000,3,1,2,0]
        check_output(candidate, transposed, result, 4, expected_order, original, strides=[1,2])
        one(candidate, 41, result)
        one(candidate, 41, transposed)
        one(candidate, 41, source)

        raw_words = [0,0x80000000,1,0x7fc00000]
        head, raw_factory, extra = factory(0, (4,), raw_words, command_count=2)
        command = struct.pack("<8Q",127,w.reference(1),0,0,0,0,0,0)
        same_rows = exchange(candidate, head+raw_factory+command+extra, 0, [97,127])
        source,result = (r["value"] for r in same_rows)
        check_output(candidate, source, result, 0, raw_words, inspect(candidate, source))
        one(candidate, 41, result)
        frozen = candidate.owner()
        for word in range(2,8):
            malformed = [127,source,0,0,0,0,0,0]
            malformed[word] = 1
            exchange(candidate,w.packet([malformed]),7)
            require(candidate.owner() == frozen, "bad reserved-word request did not restore all owner bytes")
            controls += 1
        exchange(candidate,w.packet([(127,source),(127,source,1)]),7)
        require(candidate.owner() == frozen, "successful predicate prefix was not rolled back")
        controls += 1

        for miss in (0,1):
            exchange(candidate,w.packet([(127,miss)]),1)
            require(candidate.owner() == frozen, "bad handle changed owner bytes")
            controls += 1
        exchange(candidate,w.packet([(128,source)]),4)
        require(candidate.owner() == frozen, "unknown opcode changed owner bytes")
        controls += 1
        one(candidate, 41, source)
        frozen = candidate.owner()
        exchange(candidate,w.packet([(127,source)]),1)
        require(candidate.owner() == frozen, "stale handle changed owner bytes")
        controls += 1
        require(one(candidate,53)["value"] == 1, "final owner audit failed")
        require(one(parent,53)["value"] == 1, "parent owner audit failed")
    finally:
        candidate.finish()
        parent.finish()

    oom = start_session(binary, "candidate-oom")
    try:
        dtype, words = 5, list(range(65536))
        source = create(oom,dtype,(len(words),),words)
        original = inspect(oom,source)
        generator = one(oom,60,1729)["value"]
        old_rng = one(oom,63,generator)["payload"]
        owner = oom.owner()
        limit = resource.prlimit(oom.pid,resource.RLIMIT_AS)
        page = os.sysconf("SC_PAGE_SIZE")
        up = lambda n: ((n + page - 1) // page) * page
        vmem = int(Path(f"/proc/{oom.pid}/statm").read_bytes().split()[0]) * page
        cap = vmem + up(owner["length"] + 16) + up(len(old_rng) + 16) + 65536
        require(limit[1] == resource.RLIM_INFINITY or cap <= limit[1], "hard address limit too small")
        resource.prlimit(oom.pid,resource.RLIMIT_AS,(cap,limit[1]))
        require(resource.prlimit(oom.pid,resource.RLIMIT_AS) == (cap, limit[1]), "actual bounded probe limit")
        alive=[]
        request=w.packet([(127,source)])
        try:
            for _ in range(32):
                owner_at=oom.owner()
                status,rows=w.parse(oom.exchange(request))
                if status:
                    require(status == 3 and not rows, "wrong allocation refusal")
                    require(oom.owner() == owner_at, "real OOM did not restore complete owner bytes")
                    break
                require(len(rows) == 1 and rows[0]["opcode"] == 127, "invalid successful bounded reply")
                alive.append(rows[0]["value"])
            else:
                raise ValueError("no real OOM within bounded 32 attempts")
        finally:
            resource.prlimit(oom.pid,resource.RLIMIT_AS,limit)
        require(resource.prlimit(oom.pid,resource.RLIMIT_AS) == limit, "actual address limit not restored")
        retry=exchange(oom,request,0,[127])[0]["value"]
        check_output(oom,source,retry,dtype,words,original)
        require(one(oom,63,generator)["payload"] == old_rng, "complete Generator state changed")
        for handle in [*alive,retry,source]:
            one(oom,41,handle)
        one(oom,66,generator)
        require(one(oom,53)["value"] == 1, "post-OOM owner audit failed")
    finally:
        oom.finish()
    return {"native_image_sha256":NATIVE_SHA256,"baseline_image_sha256":BASE_SHA256,
            "dense_cases":dense,"noncontiguous_cases":1,"same_frame_cases":1,
            "previous_predicate_comparisons":parent_checks,"refusal_controls":controls,
            "actual_oom_status":3,"actual_oom_attempt":len(alive)+1,"exact_same_packet_retry":True,
            "oom_retry_values":len(words), "production_frames":candidate.frames+parent.frames+oom.frames,
            "strict_pytorch_qualification":False,"scope":"scoped actual native experimental image"}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--binary",required=True,type=Path)
    p.add_argument("--baseline",required=True,type=Path)
    p.add_argument("--evidence-dir",type=Path)
    a=p.parse_args()
    print(json.dumps(run_native(load_wire(a.protocol),a.binary,a.baseline,a.evidence_dir),indent=2,sort_keys=True))
