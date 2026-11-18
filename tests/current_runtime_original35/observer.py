"""Current-image identity and explicit ViewInfo adapter for original35.

Image/source/profile identities and the two owner format tags follow the build.
The original fixed35 equations additionally count each actual ViewInfo base
reference. Their allocation, gradient, rollback and transition predicates and
the general full-width structural verifier remain intact. No snapshot is projected.
"""
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys

from binding import ROOT, require, sha


PARENT = ROOT / "tests/tensor_to_frozen35_accounting/owner_accounting_to.py"
PARENT_SHA = "a496aae3ca0b9f02b7f0e8c26c7177c3ce5b5a8c4198354c22059434b1491914"
LAUNCHER = ROOT / "tests/tensor_to_frozen35_accounting/run_frozen35_accounted.py"
LAUNCHER_SHA = "45b635448e7c4d2db48cbc3a9b31e722eaf3ec93be7e252ef0c51c740d04d949"
CHECKER = ROOT / "tests/tensor_to_frozen35_accounting/check_frozen35_accounted.py"
CHECKER_SHA = "a86cde9c035e622932560441ad4d496452bef6afd648c836f297240e0b0136bf"
CONTROLS = ROOT / "tests/tensor_to_frozen35_accounting/check_owner_snapshots_672.py"
CONTROLS_SHA = "b276b3239d254c52b7877919fe1401e95121014975fdce64f5adc40a264a523e"
OLD_NAME = "tensor-to-copy-614-fixed35-F32-Saved"
NAME = "current-runtime-672-original35-F32-ViewInfo-Saved"

# These exact bodies produce the public and compound constructor counts used by
# the original fixed35 ledger. New dtype/reduction commands do not execute them
# differently. Actual per-frame reference and allocation equations still run.
CONSTRUCTORS = (
    "tnew", "descriptor", "tensor_initialize", "publish_tensor", "finish_command",
    "linear", "mse", "layernorm", "cross_entropy_probs", "batchnorm_eval",
    "gradient_new", "incoming_adopt_gradient", "graph_reset", "descriptor_page",
    "so_state_size", "object_size", "getw", "putw", "so_uget", "so_uput",
)

# Already published body pins independently describe the exact owned reference:
# retain records base/incarnation/version at 640/648/656, marks 664, increments
# base.600 and node_retain's base.208. Release clears all four fields, decrements
# base.600 and calls node_release. graph_reset's original closure is unchanged.
VIEWINFO_ROOT_FUNCTIONS = (
    "to_view_root_retain", "to_view_base_node", "to_view_base_release",
    "node_retain", "node_release", "graph_reset",
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_tag(bodies, producer, checker):
    """Read a unique printable eight-byte tag from both HIA bodies."""
    candidates = []
    for value in re.findall(rb'const\(\) \{"value":([0-9]+)\}', bodies[producer]):
        number = int(value)
        if not 0 <= number < 1 << 64:
            continue
        raw = number.to_bytes(8, "little")
        if raw.startswith(b"HT") and all(32 <= byte < 127 for byte in raw):
            candidates.append(raw)
    require(len(candidates) == 1, "unique native owner format tag: " + producer)
    value = str(int.from_bytes(candidates[0], "little")).encode()
    require(b'{"value":' + value + b'}' in bodies[checker], "native format checker/producer agreement")
    return candidates[0]


def extend_viewinfo(module, original_raw):
    """Add exact ViewInfo edges to existing equations with reversible edits.

    The original general verifier already validates these four fields, the live
    base/incarnation, Storage and Version equality, effective requires_grad and
    both incoming reference counts. The fixed35 layer previously prohibited
    this role even though current transpose/view commands now produce it.
    """
    target_before = '    return [values[slot] for slot in (0,2) if values[slot]]'
    target_after = '''    targets = [values[slot] for slot in (0,2) if values[slot]]
    node = snapshot["nodes"][str(index)]
    need(len(node) == 84, "descriptor-width")
    marker = word(node, 664)
    need(marker in (0, 1, 3), "current35-ViewInfo-marker")
    if marker:
        need(word(node, 640) > 0, "current35-ViewInfo-base")
        targets.append(word(node, 640))
    return targets'''
    changes = {
        "_saved_tensor_targets": [("actual-ViewInfo-base-edge", target_before, target_after)],
        "_saved_tensor_references": [(
            "all-live-root-providers", '    for index in snapshot["binary_saved_roots"]:\n',
            '    for index, node in snapshot["nodes"].items():\n'
            '        if not word(node, 304):\n'
            '            continue\n')],
        "verify_snapshot": [(
            "four-ViewInfo-fields-checked-by-general-verifier",
            "(608,616,624,632,640,648,656,664)", "(608,616,624,632)")],
    }
    tree = ast.parse(original_raw)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    proof = {}
    for name, edits in changes.items():
        original = functions[name]
        text = ast.get_source_segment(original_raw.decode(), original)
        for label, before, after in edits:
            require(text.count(before) == 1, "unique ViewInfo accounting edit: " + label)
            text = text.replace(before, after, 1)
        reverse = text
        for label, before, after in reversed(edits):
            require(reverse.count(after) == 1, "unique inverse ViewInfo edit: " + label)
            reverse = reverse.replace(after, before, 1)
        original_ast = ast.dump(original, include_attributes=False)
        executed_ast = ast.dump(ast.parse(text).body[0], include_attributes=False)
        require(ast.dump(ast.parse(reverse).body[0], include_attributes=False) == original_ast,
                "entire ViewInfo accounting inverse AST: " + name)
        exec(compile(text, str(PARENT) + "::<explicit-ViewInfo-base>", "exec"), module.__dict__)
        proof[name] = dict(original_ast_sha256=sha(original_ast.encode()),
                           executed_ast_sha256=sha(executed_ast.encode()),
                           entire_inverse_AST_equals_original=True,
                           edits=[dict(label=label, before=before, after=after, count=1)
                                  for label, before, after in edits])
    return dict(
        functions=proof,
        exact_reference_equation="node.600 = binary Saved Tensor targets + live ViewInfo base targets; "
                                 "node.208 = graph edges + optimizer roots + gradient roots + node.600",
        reachable_and_graph_reset_closure_include_actual_ViewInfo_bases=True,
        original_fixed35_equation_and_allocation_bodies_unchanged=True,
        general_84_word_structural_predicates_unchanged=True,
        still_zero_history_fields=[608, 616, 624, 632],
        residual_allocation_or_reference_counts_used=False,
    )


def configure(build, bodies, directory, inputs, module=None):
    original_raw = inputs.bind(PARENT, PARENT_SHA)
    inputs.bind(LAUNCHER, LAUNCHER_SHA)
    module = module or load(PARENT, "current_runtime_original35_owner")
    original_dependencies = module.dependency_paths()
    for path in original_dependencies.values():
        inputs.bind(path)
    config = json.loads(inputs.bind(module.PROFILE_FILE))
    constructor_pins = {}
    for name in CONSTRUCTORS:
        expected = config["function_pins"][name]
        require(name in bodies and sha(bodies[name]) == expected, "constructor or layout contract changed: " + name)
        constructor_pins[name] = expected
    viewinfo_pins = {}
    for name in VIEWINFO_ROOT_FUNCTIONS:
        expected = config["function_pins"][name]
        require(name in bodies and sha(bodies[name]) == expected, "ViewInfo root contract changed: " + name)
        viewinfo_pins[name] = expected
    viewinfo = extend_viewinfo(module, original_raw)
    viewinfo["published_HIA_root_body_pins"] = viewinfo_pins
    state = source_tag(bodies, "init_state", "state_header")
    backing = source_tag(bodies, "so_state_init", "so_state_header")
    structural = module.structural
    structural_raw = inputs.bind(module.STRUCTURAL, module.STRUCTURAL_SHA)
    tree = ast.parse(structural_raw)
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_owner")
    text = ast.get_source_segment(structural_raw.decode(), original)
    edits = []
    for old, new in (("b\"HTTOCP08\"", repr(state)), ("b\"HTOTOCP8\"", repr(backing))):
        require(text.count(old) == 1, "unique owner decoder tag literal")
        text = text.replace(old, new, 1)
        edits.append(dict(before=old, after=new, count=1))
    reverse = text
    for edit in reversed(edits):
        require(reverse.count(edit["after"]) == 1, "unique inverse owner decoder tag literal")
        reverse = reverse.replace(edit["after"], edit["before"], 1)
    original_ast = ast.dump(original, include_attributes=False)
    require(ast.dump(ast.parse(reverse).body[0], include_attributes=False) == original_ast,
            "entire owner decoder inverse AST differs")
    # The original functions retain their module globals and every assertion.
    # Compile only the explicitly inverse-checked tag adapter in that namespace.
    structural.SOURCE = build["source_sha256"]
    structural.IMAGE = build["native_sha256"]
    structural.PROFILE = NAME
    exec(compile(text, str(module.STRUCTURAL) + "::<current-owner-tags>", "exec"), structural.__dict__)
    profile = dict(name=NAME, source_sha256=build["source_sha256"], native_sha256=build["native_sha256"],
                   state=state.decode(), backing=backing.decode(), descriptor_bytes=672,
                   source_constructor_costs=True, function_pins=constructor_pins,
                   accounting_domain="Original35 F32 commands 1..56; exact binary Saved and ViewInfo base roots. "
                                     "Fields 608/616/624/632 remain zero; no residual allocation allowance.",
                   original_accounting_domain=config["fixed35_domain"],
                   private_Tensor_root_roles=["ordinary-binary-Saved-Tensor", "F32-ViewInfo-ultimate-base"],
                   current_viewinfo_accounting=viewinfo, inverse_proof=module.inverse_proof(),
                   legacy_source_sha256=module.LEGACY_SHA, structural_observer_sha256=module.STRUCTURAL_SHA)
    original_pin_profile = module.pin_profile
    original_dependency_paths = module.dependency_paths

    def pin_profile(binary, source):
        require(Path(binary).resolve() == directory / "heptatorch"
                and Path(source).resolve() == directory / "runtime.hepta", "current build input paths")
        inputs.bind(binary, build["native_sha256"])
        inputs.bind(source, build["source_sha256"])
        return json.loads(json.dumps(profile))

    def dependency_paths():
        return {**original_dependency_paths(), "current_identity_adapter": Path(__file__).resolve(),
                "current_build_binding": Path(__file__).with_name("binding.py").resolve(),
                "current_actual_build": directory / "build.json"}

    module.pin_profile = pin_profile
    module.dependency_paths = dependency_paths
    adaptation = dict(
        schema="heptatorch.current-runtime.original35-identity-adapter.v1",
        original_observer_sha256=PARENT_SHA, original_structural_sha256=module.STRUCTURAL_SHA,
        source_sha256=build["source_sha256"], native_sha256=build["native_sha256"],
        source_function_count=build["functions"], native_units=build["units_planned"],
        profile_name=NAME, source_format_tags=dict(state=state.decode(), backing=backing.decode()),
        decoder_edits=edits, decoder_original_ast_sha256=sha(original_ast.encode()),
        decoder_entire_inverse_AST_equals_original=True,
        runtime_rebindings=["structural.SOURCE", "structural.IMAGE", "structural.PROFILE",
                            "structural.decode_owner tags", "pin_profile", "dependency_paths",
                            "_saved_tensor_targets explicit ViewInfo base", "_saved_tensor_references live providers",
                            "verify_snapshot four ViewInfo fields checked by unchanged general verifier"],
        constructor_bodies_checked=constructor_pins,
        viewinfo_accounting=viewinfo,
        original_fixed35_equation_and_allocation_bodies_unchanged=True,
        general_snapshot_and_transition_predicates_unchanged=True, full_84_word_snapshots=True,
        reference_or_numeric_assertions_changed=False, old_pass_transferred=False,
    )
    # Hold originals for explicit restoration at the process boundary.
    module._current_identity_adaptation = adaptation
    module._current_original_pin_profile = original_pin_profile
    return module, adaptation
