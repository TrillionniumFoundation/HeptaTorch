"""Full fixed35 accounting for the separately pinned 640-byte owner.

The frozen 576-byte observer remains a source input. Its pure fixed35 checks,
exact transition ledger, Journal and Session overlay are reused under an
explicit AST proof. The only changed legacy predicate is descriptor width 72 -> 80.
Every actual full 80-word snapshot additionally passes the general 640-byte structural verifier with one explicitly source-proved
Storage tombstone correction (the original6ea file stays frozen). No old snapshot is fabricated or truncated.

This observer supports only the original35 command domain. NativeDropout99,
detach_101, retained hooks and owned Optimizers are outside that domain and
are not explained by a residual allocation count. Their separate 4+24 suite
does not qualify the full35 workload.
"""
from pathlib import Path
from collections import Counter
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import re
import struct
import sys

HERE = Path(__file__).resolve().parent
LEGACY = HERE / "legacy_owner_rules.py"
PROFILE_FILE = HERE / "profile.json"
ORIGINAL_STRUCTURAL = HERE.parent / "dtype_owner_accounting" / "owner640.py"
STRUCTURAL = HERE / "owner640_fixed35.py"
LEGACY_SHA = "281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64"
ORIGINAL_STRUCTURAL_SHA = "6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833"
STRUCTURAL_SHA = "d40fb59a74edfaf87113e7eac1e13587c7e8d33aa6ca1edea36a7e9da7bbde89"

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

_legacy_raw, _structural_raw = LEGACY.read_bytes(), STRUCTURAL.read_bytes()
assert _sha(ORIGINAL_STRUCTURAL.read_bytes()) == ORIGINAL_STRUCTURAL_SHA
assert _sha(_legacy_raw) == LEGACY_SHA
assert _sha(_structural_raw) == STRUCTURAL_SHA
_spec = importlib.util.spec_from_file_location("frozen35_structural_owner640", STRUCTURAL)
structural = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = structural
_spec.loader.exec_module(structural)
assert STRUCTURAL.read_bytes() == _structural_raw
CONFIG = json.loads(PROFILE_FILE.read_bytes())
assert CONFIG["legacy_owner_rules_sha256"] == LEGACY_SHA
assert CONFIG["structural_observer_sha256"] == STRUCTURAL_SHA
assert CONFIG["source_sha256"] == structural.SOURCE
assert CONFIG["native_sha256"] == structural.IMAGE
assert CONFIG["descriptor_bytes"] == structural.DESCRIPTOR == 640
SCHEMA, BASE, DESC_BYTES, TAG = structural.SCHEMA, structural.BASE, 640, structural.TAG
SESSIONS = []

_tree = ast.parse(_legacy_raw, filename=str(LEGACY))
_selected = ("sha", "canonical", "need", "word", "identity", "semantic_snapshot",
             "verify_snapshot", "verify_transition", "Journal", "install")
_nodes = {node.name: node for node in _tree.body
          if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _selected}
assert set(_nodes) == set(_selected)
SOURCE_PINS = next(ast.literal_eval(node.value) for node in _tree.body
    if isinstance(node, ast.Assign) and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SOURCE_PINS")
_proof = {"legacy_source_sha256": LEGACY_SHA, "structural_source_sha256": STRUCTURAL_SHA,
          "full_640_snapshot_no_projection": True, "functions": {}}
for _name in _selected:
    _original = _nodes[_name]
    _node = copy.deepcopy(_original)
    _changes = 0
    if _name == "verify_snapshot":
        for _item in ast.walk(_node):
            if (isinstance(_item, ast.Compare) and ast.unparse(_item.left) == "len(node)"
                and len(_item.ops) == 1 and isinstance(_item.ops[0], ast.Eq)
                and len(_item.comparators) == 1 and isinstance(_item.comparators[0], ast.Constant)
                and _item.comparators[0].value == 72):
                _item.comparators[0].value = 80
                _changes += 1
        assert _changes == 1
        _node.name = "_fixed35_snapshot"
    if _name == "verify_transition":
        _node.name = "_fixed35_transition"
    _inverse = copy.deepcopy(_node)
    _inverse.name = _original.name
    if _name == "verify_snapshot":
        for _item in ast.walk(_inverse):
            if (isinstance(_item, ast.Compare) and ast.unparse(_item.left) == "len(node)"
                and len(_item.comparators) == 1 and isinstance(_item.comparators[0], ast.Constant)
                and _item.comparators[0].value == 80):
                _item.comparators[0].value = 72
    _dump = lambda n: ast.dump(n, include_attributes=False)
    assert _dump(_inverse) == _dump(_original), _name
    _proof["functions"][_name] = {
        "original_ast_sha256": _sha(_dump(_original).encode()),
        "executed_ast_sha256": _sha(_dump(_node).encode()),
        "width_predicates_changed": _changes,
        "name_binding_changed": _node.name != _original.name,
        "inverse_entire_function_AST_equals_original": True}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_node], type_ignores=[])),
                 str(LEGACY) + "::<explicit-640-binding>", "exec"), globals())

# These are source-derived successful constructor counts, independent of the
# legacy client's next_id arithmetic. Optional parameters are tested explicitly.
TENSOR_RESULTS = {1,2,3,4,5,6,15,16,17,18,19,20,21,22,23,26,27,28,30,31,32,
                  34,35,36,37,38,55,56}
FIXED35_OPCODES = set(range(1, 57))

def constructor_count(command):
    code = command[0]
    if code == 15:  # transpose(weight), matmul, optional bias add
        return 2 + int(command[3] != 0)
    if code == 16:  # subtract, multiply, reduce
        return 3
    if code == 26:  # eight base ops, optional weight and bias
        return 8 + int(command[2] != 0) + int(command[3] != 0)
    if code == 27:  # logsoftmax, multiply, two reductions, negate
        return 5
    if code == 28:  # seven base ops; each affine argument adds view + binary
        return 7 + 2 * int(command[4] != 0) + 2 * int(command[5] != 0)
    return int(code in TENSOR_RESULTS)

def verify_snapshot(snapshot):
    # Keep original18 negative-control failure precedence and every original
    # invariant. The complete 640 structural check is additional, never waived; its one
    # tombstone correction is independently source-bound and reversible.
    old = _fixed35_snapshot(snapshot)
    for key, node in snapshot["nodes"].items():
        need(not any(word(node, off) for off in range(576, 640, 8)),
             "fixed35-unrequested-dtype-history-role", node=int(key))
    new = structural.verify_snapshot(snapshot)
    need(new["stats"] == old["stats"] and new["live_nodes"] == old["live_nodes"]
         and new["public_nodes"] == old["public_nodes"], "dual-structural-check-agreement")
    need(not new.get("unrooted_strong_cycles"), "fixed35-unrequested-strong-cycle")
    return old

def verify_transition(before, after, tickets, commands, status=0):
    result = _fixed35_transition(before, after, tickets, commands, status)
    if status == 0:
        need(all(len(command) == 8 and command[0] in FIXED35_OPCODES for command in commands),
             "fixed35-command-domain")
        expected = {i: constructor_count(command) for i, command in enumerate(commands)
                    if command[0] in TENSOR_RESULTS}
        actual = {ticket["record"]: ticket["count"] for ticket in tickets}
        need(actual == expected, "source-constructor-ticket-count", expected=expected, actual=actual)
        # Successful destruction keeps requested bytes until the final Weak is
        # released. A same-generation tombstone never revives its payload.
        if before["length"] and after["length"]:
            for prior, current in zip(before["tables"]["2"], after["tables"]["2"]):
                if prior[1] == current[1] == 1 and prior[0] == current[0]:
                    if current[5] == 0:
                        need(current[3] == prior[3], "expired-Storage-requested-preserved")
                    if prior[5] == 0:
                        need(current[5] == 0, "expired-Storage-payload-revived")
    return result

class OwnerReader:
    def __init__(self, process, profile):
        need(profile["native_sha256"] == structural.IMAGE
             and profile["source_sha256"] == structural.SOURCE
             and profile["name"] == structural.PROFILE, "reader-profile")
        self.reader = structural.OwnerReader(process)

    def snapshot(self, full_digest=False):
        # Frozen640 reads all bytes and always supplies the whole-owner digest.
        # No descriptor truncation, pointer adjustment, or snapshot rewriting.
        value = self.reader.snapshot()
        verify_snapshot(value)
        return value

def dependency_paths():
    return {"legacy_owner_rules": LEGACY, "structural_owner640": STRUCTURAL,
            "original_structural_owner640": ORIGINAL_STRUCTURAL,
            "storage_tombstone_correction": HERE / "storage-tombstone-correction.json",
            "accounting_profile": PROFILE_FILE,
            "structural_profile_adaptation": HERE / "structural-profile-adaptation.json"}

def inverse_proof():
    need(LEGACY.read_bytes() == _legacy_raw and STRUCTURAL.read_bytes() == _structural_raw
         and _sha(ORIGINAL_STRUCTURAL.read_bytes()) == ORIGINAL_STRUCTURAL_SHA,
         "observer-dependency-changed")
    return copy.deepcopy(_proof)

def pin_profile(binary, source):
    profile = structural.pin_profile(binary, source)
    raw = Path(source).read_bytes()
    bodies = {name.decode(): body for body, name in re.findall(
        rb"^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)",
        raw, re.M | re.S)}
    need(len(bodies) == 544, "native-function-count")
    for name, expected in CONFIG["function_pins"].items():
        need(name in bodies and sha(bodies[name]) == expected, "native-function-pin", function=name)
    for target, expected in CONFIG["capture_callers"].items():
        callers = [name for name, body in bodies.items()
                   if ('"function":"' + target + '"').encode() in body]
        need(sorted(callers) == sorted(expected), "capture-call-chain", target=target, callers=callers)
    profile.update(function_pins=CONFIG["function_pins"], capture_callers=CONFIG["capture_callers"],
                   accounting_domain=CONFIG["fixed35_domain"], legacy_source_sha256=LEGACY_SHA,
                   structural_observer_sha256=STRUCTURAL_SHA,
                   inverse_proof=inverse_proof(), source_constructor_costs=True)
    return profile
