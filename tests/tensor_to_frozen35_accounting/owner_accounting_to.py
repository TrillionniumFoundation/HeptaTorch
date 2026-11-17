"""Full fixed35 accounting for the separately pinned 672-byte owner.

The frozen 576-byte observer remains a source input. Its pure fixed35 checks,
exact transition ledger, Journal and Session overlay are reused under an
explicit AST proof. Legacy deltas are explicit width and original-Saved-T reference/closure equations; all old numeric, transition and refusal checks remain.
Every actual full 84-word snapshot additionally passes the general 672-byte structural verifier with one explicitly source-proved
Storage tombstone correction (the original6ea file stays frozen). No old snapshot is fabricated or truncated.

This observer supports only the original35 command domain. NativeDropout99,
detach_101, retained hooks and owned Optimizers are outside that domain; ordinary F32 mul/div really hold original Saved T pairs and
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
ORIGINAL_STRUCTURAL = HERE / "owner672_general.py"
STRUCTURAL = HERE / "owner672_fixed35.py"
LEGACY_SHA = "281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64"
ORIGINAL_STRUCTURAL_SHA = "6d230e2df25e04e32f76fb10fd0936ec9d220b60acda37c40b52a45633372312"
STRUCTURAL_SHA = "e58adb23bb65cdc81ec36e3e2fcbd178c0d5198f3b10c5d274afa94da7761a6f"

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

_legacy_raw, _structural_raw = LEGACY.read_bytes(), STRUCTURAL.read_bytes()
assert _sha(ORIGINAL_STRUCTURAL.read_bytes()) == ORIGINAL_STRUCTURAL_SHA
assert _sha(_legacy_raw) == LEGACY_SHA
assert _sha(_structural_raw) == STRUCTURAL_SHA
_spec = importlib.util.spec_from_file_location("frozen35_structural_owner672", STRUCTURAL)
structural = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = structural
_spec.loader.exec_module(structural)
assert STRUCTURAL.read_bytes() == _structural_raw
CONFIG = json.loads(PROFILE_FILE.read_bytes())
assert CONFIG["legacy_owner_rules_sha256"] == LEGACY_SHA
assert CONFIG["structural_observer_sha256"] == STRUCTURAL_SHA
assert CONFIG["source_sha256"] == structural.SOURCE
assert CONFIG["native_sha256"] == structural.IMAGE
assert CONFIG["descriptor_bytes"] == structural.DESCRIPTOR == 672
SCHEMA, BASE, DESC_BYTES, TAG = structural.SCHEMA, structural.BASE, 672, structural.TAG
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
          "full_672_snapshot_no_projection": True, "functions": {}}
for _name in _selected:
    _original = _nodes[_name]
    _text = ast.get_source_segment(_legacy_raw.decode(), _original)
    _edits = []
    def _edit(label, before, after, count=1):
        global _text
        assert _text.count(before) == count, (label, _text.count(before), count)
        _text = _text.replace(before, after)
        _edits.append({"label":label,"before":before,"after":after,"count":count})
    if _name == "verify_snapshot":
        _edit("width", "len(node) == 72", "len(node) == 84")
        _edit("actual-original-T-pair-counter", "    gradients = set()\n",
              "    gradients = set()\n    saved_T_refs = _saved_tensor_references(snapshot)\n")
        _edit("dead-node-no-Saved-root", "(200, 208, 240, 88, 312, 384, 392, 400)",
              "(200, 208, 240, 88, 312, 384, 392, 400, 600)")
        _edit("rowless-no-owned-T-root", "(88, 312, 384, 392, 400)", "(88, 312, 384, 392, 400, 600)")
        _edit("reference-sum-from-pairs", "graph_refs[index] + word(node, 312) + grad_refs[index]",
              "graph_refs[index] + word(node, 312) + grad_refs[index] + saved_T_refs[index]", 2)
        _edit("exact600-from-pairs", "        need(word(node, 88) == leaf_refs[index],",
              "        need(word(node, 600) == saved_T_refs[index], \"fixed35-original-Saved-T-reference-count\", node=index)\n"
              "        need(word(node, 88) == leaf_refs[index],")
        _edit("reachable-original-T-roots",
              '        pending.extend(parent for parent, _ in snapshot["edges"][str(index)] if parent)\n',
              '        pending.extend(parent for parent, _ in snapshot["edges"][str(index)] if parent)\n'
              '        pending.extend(_saved_tensor_targets(snapshot, index))\n')
    if _name == "verify_transition":
        _edit("reset-original-T-root-closure",
              '                pending.extend(p for p, _ in before["edges"][str(index)] if p)\n',
              '                pending.extend(p for p, _ in before["edges"][str(index)] if p)\n'
              '                pending.extend(_saved_tensor_targets(before, index))\n')
    _reverse = _text
    for _change in reversed(_edits):
        assert _reverse.count(_change["after"]) == _change["count"]
        _reverse = _reverse.replace(_change["after"], _change["before"])
    _dump = lambda n: ast.dump(n, include_attributes=False)
    assert _dump(ast.parse(_reverse).body[0]) == _dump(_original)
    _node = ast.parse(_text).body[0]
    if _name == "verify_snapshot": _node.name = "_fixed35_snapshot"
    if _name == "verify_transition": _node.name = "_fixed35_transition"
    _proof["functions"][_name] = {
        "original_ast_sha256": _sha(_dump(_original).encode()),
        "executed_ast_sha256": _sha(_dump(_node).encode()),
        "width_predicates_changed": int(_name == "verify_snapshot"),
        "name_binding_changed": _node.name != _original.name,
        "inverse_entire_function_AST_equals_original": True,
        "explicit_role_edits": _edits}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_node], type_ignores=[])),
                 str(LEGACY) + "::<explicit-672-Saved-binding>", "exec"), globals())

def _saved_tensor_targets(snapshot, index):
    values = snapshot["binary_saved_roots"].get(str(index), [0,0,0,0])
    need(len(values) == 4 and all(type(v) is int and 0 <= v < 1 << 64 for v in values),
         "fixed35-original-Saved-pair-shape")
    return [values[slot] for slot in (0,2) if values[slot]]

def _saved_tensor_references(snapshot):
    references = Counter()
    for index in snapshot["binary_saved_roots"]:
        references.update(_saved_tensor_targets(snapshot,index))
    return references

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
    # invariant. The complete 672 structural check is additional, never waived; its one
    # tombstone correction is independently source-bound and reversible.
    old = _fixed35_snapshot(snapshot)
    for key, node in snapshot["nodes"].items():
        need(word(node,576) == 0 and not any(word(node,off) for off in (608,616,624,632,640,648,656,664)),
             "fixed35-unrequested-dtype-history-role", node=int(key))
        role = word(node,584)
        need(role == 0 or role > 65535, "fixed35-unrequested-mask-or-I64-view-role", node=int(key))
        if role == 0:
            need(word(node,592) == 0, "fixed35-unowned-Saved-relative-address", node=int(key))
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
        # Frozen672 reads all bytes and always supplies the whole-owner digest.
        # No descriptor truncation, pointer adjustment, or snapshot rewriting.
        value = self.reader.snapshot()
        verify_snapshot(value)
        return value

def dependency_paths():
    return {"legacy_owner_rules": LEGACY, "structural_owner672": STRUCTURAL,
            "original_structural_owner672": ORIGINAL_STRUCTURAL,
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
    need(len(bodies) == 614, "native-function-count")
    need(set(bodies) == set(CONFIG["function_pins"]), "native-function-pin-coverage")
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
