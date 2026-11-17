"""Independent input-overlap normalization helper.
Source-only authoring: no native or Torch execution. The producer owns integration.
The helper is general for nonnegative coverage; count zero delegates unchanged.
"""
import hashlib
from pathlib import Path

ORIGINAL_BACKWARD_SHA256 = "0675d40a2fc2f7d82820c68c432a55f193bda1314d2ce4da322e5ce813f2e4d5"
HELPER_SOURCE = "fn dview_coverage_divide(v0:u64,v1:u64) -> (u64) effects(read,write,trap) entry(b0)\nblock b0(v0:u64,v1:u64)\n  v2:u64 = const() {\"value\":4294967295}\n  v3:u64 = bit.and(v0,v2)\n  v4:u64 = const() {\"value\":0}\n  v5:bool = compare.gt(v1,v4)\n  branch v5 b1(v3,v1) b3(v3,v1)\nblock b1(v6:u64,v7:u64)\n  v8:u64 = const() {\"value\":2147483647}\n  v9:u64 = bit.and(v6,v8)\n  v10:u64 = const() {\"value\":2139095040}\n  v11:bool = compare.gt(v9,v10)\n  branch v11 b2(v6) b3(v6,v7)\nblock b2(v12:u64)\n  v13:u64 = const() {\"value\":4194304}\n  v14:u64 = bit.or(v12,v13)\n  ret(v14)\nblock b3(v15:u64,v16:u64)\n  v17:u64 = const() {\"value\":2}\n  v18:u64 = call(v16,v17) {\"function\":\"integer_as_f32\"}\n  v19:u64 = call(v15,v18) {\"function\":\"divf\"}\n  ret(v19)\nend\n"
OLD_CALL = "  v249:u64 = const() {\"value\":2}\n  v250:u64 = call(v237,v249) {\"function\":\"integer_as_f32\"}\n  v251:u64 = call(v235,v250) {\"function\":\"divf\"}\n"
NEW_CALL = "  v251:u64 = call(v235,v237) {\"function\":\"dview_coverage_divide\"}\n"

def helper_source():
    return HELPER_SOURCE

def patch_original_backward(body):
    if not isinstance(body, str):
        raise TypeError("expected exact UTF-8 function source")
    if hashlib.sha256(body.encode()).hexdigest() != ORIGINAL_BACKWARD_SHA256:
        raise ValueError("original backward body pin mismatch")
    if body.count(OLD_CALL) != 1 or "dview_coverage_divide" in body:
        raise ValueError("ambiguous or already-patched backward")
    candidate = body.replace(OLD_CALL, NEW_CALL, 1)
    if candidate.count(NEW_CALL) != 1:
        raise ValueError("new call not unique")
    if candidate.replace(NEW_CALL, OLD_CALL, 1) != body:
        raise ValueError("whole-function inverse mismatch")
    return candidate

def inverse_original_backward(body):
    if body.count(NEW_CALL) != 1:
        raise ValueError("new call not unique")
    original = body.replace(NEW_CALL, OLD_CALL, 1)
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_BACKWARD_SHA256:
        raise ValueError("inverse source pin mismatch")
    return original

def write_sources(output, original_body_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = Path(original_body_path).read_text(encoding="utf-8")
    patched = patch_original_backward(original)
    for name, value in (
        ("dview_coverage_divide.hepta", HELPER_SOURCE),
        ("original-to_as_strided_backward.hepta", original),
        ("patched-to_as_strided_backward.hepta", patched),
    ):
        p = output / name
        with p.open("x", encoding="utf-8", newline="") as f:
            f.write(value)
        p.chmod(0o600)
    return patched

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--original-body", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    write_sources(a.output, a.original_body)
