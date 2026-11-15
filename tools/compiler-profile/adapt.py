#!/usr/bin/env python3
"""Author a separately identified local HIA compiler capacity profile.

This script is a build-time authoring tool. It reads the exact pinned upstream
source, proves that its unchanged generator reproduces that source, changes only
the enumerated capacity/layout/private-counter sites in a private namespace,
and runs the unchanged upstream Core verifier and assembly parser. It does not
change the upstream checkout, monkeypatch verifier limits, compile native code,
or qualify the resulting compiler. Native bootstrap/qualification is separate.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import FunctionType

sys.dont_write_bytecode = True

UPSTREAM_COMMIT = "f33d8da316de8fdf27b504371f1a5310997ff548"
UPSTREAM_SOURCE_SHA256 = "388b182e42675f11435867dd99520e9452a4ff48e23c0766a3952b662acc7627"
UPSTREAM_CONTRACT_SHA256 = "8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c"
PROFILE = "heptatorch.hia.compiler.f256-b2560-w2m.v1"
MIB = 1 << 20
LIMITS = {
    "functions": 256,
    "blocks_total": 2560,
    "blocks_per_function": 256,
    "instructions": 8192,
    "core_structural_nodes": 262144,
    "ssa_names_per_function": 2048,
    "parameters_or_arguments": 16,
    "function_results": 3,
    "acyclic_call_depth_max": 256,
    "tail_self_call_additional_frames": 0,
    "explicit_exports": 256,
    "relative_relocations": 14336,
    "source_bytes": MIB,
    "output_capacity_bytes": MIB,
    "workspace_required_bytes": 2 * MIB,
}
AUTHORING_FILES = tuple("""
hepta/__init__.py
hepta/artifact_publication.py
hepta/hia_agent_assembly.py
hepta/hia_capability_flow.py
hepta/hia_core_admission_native.py
hepta/hia_function_shape_native.py
hepta/hia_instruction_policy_native.py
hepta/hia_json_canonical_native.py
hepta/hia_json_decode.py
hepta/hia_json_decode_native.py
hepta/hia_json_preflight.py
hepta/hia_json_preflight_native.py
hepta/hia_machine_x64.py
hepta/hia_opcodes.py
hepta/hia_program.py
hepta/hia_program_shape_native.py
hepta/hia_program_tools.py
hepta/hia_schema.py
hepta/hia_sha256_native.py
tools/generate_hia_selfhost.py
tools/hia_selfhost_chain_elf.py
tools/hia_selfhost_cli.py
tools/hia_selfhost_effects.py
tools/hia_selfhost_exports.py
tools/hia_selfhost_keywords.py
tools/hia_selfhost_limits.py
tools/hia_selfhost_multi_results.py
tools/hia_selfhost_names.py
tools/hia_selfhost_owned_elf.py
tools/hia_selfhost_returned_loans.py
tools/hia_selfhost_signed.py
tools/hia_selfhost_traps.py
""".split())


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def layout(functions, blocks, relocations, exports):
    """All extents are exclusive and all non-byte tables are 8-byte aligned."""
    specifications = [
        ("function_records", functions, 320),
        ("block_records", blocks, 320),
        ("ssa_type_records", 2048, 16),
        ("argument_vector", 1, 1024),
        ("relative_relocations", relocations, 24),
        ("call_graph", functions * functions, 1),
        ("call_readiness", functions, 8),
        ("capability_records", 2048, 24),
        ("block_reachability", blocks, 1),
        ("capability_temporary", 16, 1),
        ("resource_graph_queue", 256, 8),
        ("ssa_formal_origins", 2048, 1),
        ("block_formal_origins", blocks, 16),
        ("ssa_names", 2048, 24),
        ("block_names", 256, 24),
        ("explicit_exports", exports, 48),
    ]
    at = 4096
    regions = []
    for name, count, stride in specifications:
        require(at % 8 == 0, "misaligned region: " + name)
        size = count * stride
        regions.append({"name": name, "start": at, "end_exclusive": at + size,
                        "count": count, "stride": stride, "bytes": size})
        at += size
    return regions


def structure(doc, assembly):
    pending, nodes = [doc], 0
    while pending:
        current = pending.pop()
        nodes += 1
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    functions = doc["functions"]
    return {
        "functions": len(functions),
        "blocks": sum(len(fn["blocks"]) for fn in functions),
        "instructions": sum(len(block["ops"]) for fn in functions for block in fn["blocks"]),
        "max_blocks_per_function": max(len(fn["blocks"]) for fn in functions),
        "core_structural_nodes": nodes,
        "source_bytes": len(assembly),
        "source_sha256": digest(assembly),
        "canonical_json_bytes": len(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()),
    }


def adapt(source_root: Path, output: Path):
    source_root = source_root.resolve()
    contract_path = source_root / "docs/HIA_SELFHOST_CORE_SUPPORT.json"
    original_path = source_root / "examples/hia_bootstrap/selfhost.hepta"
    original = original_path.read_bytes()
    contract_raw = contract_path.read_bytes()
    require(digest(original) == UPSTREAM_SOURCE_SHA256, "pinned upstream HIA source identity mismatch")
    require(digest(contract_raw) == UPSTREAM_CONTRACT_SHA256, "pinned upstream support contract identity mismatch")
    upstream_files = json.loads(contract_raw)["source_identity"]["files"]
    preverified_files = {}
    for relative in AUTHORING_FILES:
        actual = digest((source_root / relative).read_bytes())
        require(upstream_files.get(relative) == actual, "upstream authoring input identity mismatch: " + relative)
        preverified_files[relative] = actual

    sys.path.insert(0, str(source_root))
    from tools import generate_hia_selfhost as generator
    from hepta.hia_agent_assembly import assemble_program, render_assembly

    original_doc, original_names = generator.document()
    regenerated = render_assembly(original_doc).encode()
    require(regenerated == original, "unchanged upstream generator does not reproduce exact pinned HIA source")

    original_layout = layout(128, 1536, 12000, 128)
    expanded_layout = layout(256, 2560, 14336, 256)
    old = {row["name"]: row for row in original_layout}
    new = {row["name"]: row for row in expanded_layout}
    require(old["explicit_exports"]["end_exclusive"] == 1016592, "unexpected original layout")
    require(new["explicit_exports"]["end_exclusive"] == 1515024, "unexpected expanded layout")
    require(new["explicit_exports"]["end_exclusive"] <= LIMITS["workspace_required_bytes"], "workspace overflow")

    macro_regions = {
        "FBASE": "function_records", "BBASE": "block_records", "VBASE": "ssa_type_records",
        "ARGS": "argument_vector", "FIXBASE": "relative_relocations", "CBASE": "call_graph",
        "CSTATE": "call_readiness", "CAPBASE": "capability_records", "KBASE": "block_reachability",
        "CAPTMP": "capability_temporary", "QBASE": "resource_graph_queue",
        "ORIGINBASE": "ssa_formal_origins", "BLOCKORIGIN": "block_formal_origins",
    }
    macro_changes = {}
    for key, region in macro_regions.items():
        require(getattr(generator, key) == old[region]["start"], "upstream macro/layout mismatch: " + key)
        macro_changes[key] = {"old": old[region]["start"], "new": new[region]["start"]}
    require(generator.FSTRIDE == generator.BSTRIDE == 320, "upstream record stride changed")
    require(generator.MAX_INSTRUCTIONS == 8192 and generator.MAX_CORE_NODES == 262144,
            "unchanged instruction/Core limits differ")
    require(generator.MAX_RELOCATIONS == 12000, "unexpected original relocation capacity")

    edits = []
    text = generator.SOURCE

    def replace(old_text, new_text, expected, reason):
        nonlocal text
        actual = text.count(old_text)
        require(actual == expected, "unexpected transformation match count: " + old_text)
        text = text.replace(old_text, new_text)
        edits.append({"old": old_text, "new": new_text, "matches": actual, "reason": reason})

    replace("if n < 1048576:", "if n < 2097152:", 1, "private compiler workspace minimum only")
    replace("fill_range(w, CBASE, 17408, 0)", "fill_range(w, CBASE, 67584, 0)", 1,
            "clear all 256x256 call edges and 256 eight-byte readiness marks")
    replace("fill_range(w, KBASE, 1536, 0)", "fill_range(w, KBASE, 2560, 0)", 1,
            "clear every admitted block reachability byte, including dirty workspace")
    replace("f != get(w, 64) or f >= 128", "f != get(w, 64) or f >= 256", 1, "function declaration guard")
    replace("if f >= 128:", "if f >= 256:", 1, "function name discovery guard")
    replace("if x >= 128:", "if x >= 256:", 1, "callee ordinal guard")
    replace("CBASE + get(w, 72) * 128 + x", "CBASE + get(w, 72) * 256 + x", 1, "call graph row stride")
    replace("while target < 128:", "while target < 256:", 1, "complete call graph validation")
    replace("CBASE + f * 128 + target", "CBASE + f * 256 + target", 1, "call graph validation stride")
    replace("get(w, 80) >= 1536", "get(w, 80) >= 2560", 1, "total block declaration guard")
    replace("bid >= 1536", "bid >= 2560", 1, "total block edge guard")
    replace("if n >= 128 or get(w, 32) != 0:", "if n >= 256 or get(w, 32) != 0:", 1, "scalar export count guard")
    replace("955152", str(new["ssa_names"]["start"]), 2, "move SSA-name clear and probe addresses together")
    replace("1004304", str(new["block_names"]["start"]), 4, "move block-name lookup and all three record writes")
    replace("1010448", str(new["explicit_exports"]["start"]), 6, "move every explicit-export table consumer")

    # The upstream generator substitutes these macros into the final HIA.
    # A private function namespace avoids changing imported module globals.
    namespace = dict(generator.source_program.__globals__)
    namespace.update({key: change["new"] for key, change in macro_changes.items()})
    namespace.update(SOURCE=text, MAX_RELOCATIONS=14336)
    expanded_program = FunctionType(generator.source_program.__code__, namespace)()

    # The original byte count can represent 128, but not the new exact count256.
    # Use existing 32-bit header helpers; full pass initialization already zeroes
    # the high words. No public value or generic get/put operation is narrowed.
    text = expanded_program
    for at, reads in ((64, 7), (72, 18)):
        replace(f"byte(w, {at})", f"state_get32(w, {at})", reads, "widen private function count/ordinal")
        replace(f"store(w, {at},", f"state_put32(w, {at},", 1, "widen private function count/ordinal")
    expanded_program = text
    document_namespace = dict(generator.document.__globals__)
    document_namespace["source_program"] = lambda: expanded_program
    expanded_doc, expanded_names = FunctionType(generator.document.__code__, document_namespace)(compact=True)
    candidate = render_assembly(expanded_doc).encode()
    checked = assemble_program(candidate)
    require(checked.as_dict() == expanded_doc, "generated assembly and independently parsed Core disagree")
    require(expanded_names == original_names, "compiler function ordering changed")

    original_structure = structure(original_doc, original)
    candidate_structure = structure(expanded_doc, candidate)
    for key, cap in (("functions", 128), ("blocks", 1536), ("instructions", 8192),
                     ("source_bytes", MIB), ("core_structural_nodes", 262144)):
        require(candidate_structure[key] <= cap, "candidate cannot be bootstrapped by pinned old profile: " + key)

    old_functions = {fn["name"]: fn for fn in original_doc["functions"]}
    new_functions = {fn["name"]: fn for fn in expanded_doc["functions"]}
    invariant_functions = {}
    for name in ("cli_begin", "cli_finish", "wrapper"):
        identifier = original_names[name]
        before, after = old_functions[identifier], new_functions[identifier]
        require(before == after, "ordinary application wrapper changed: " + name)
        invariant_functions[name] = {
            "function": identifier, "core_body_identical": True,
            "core_body_sha256": digest(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()),
        }

    # Bind every imported upstream source to the exact pinned contract; the
    # sparse export is sufficient for authoring but is not a full cold suite.
    loaded_files = {}
    for module_name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(source_root).as_posix()
        except ValueError:
            continue
        actual = digest(path.read_bytes())
        require(preverified_files.get(relative) == actual, "loaded upstream source outside preverified closure: " + relative)
        loaded_files[relative] = actual
    require(set(loaded_files) == set(preverified_files), "upstream authoring import closure changed")

    reloc_bound = 2 * LIMITS["functions"] + LIMITS["instructions"] + 2 * LIMITS["blocks_total"] + 2
    core_byte_bound = MIB + 1024 + LIMITS["functions"] * 1024 + LIMITS["blocks_total"] * 1024 + LIMITS["instructions"] * 512
    require(reloc_bound <= LIMITS["relative_relocations"], "relocation derivation exceeds table")
    require(core_byte_bound <= 8 * MIB, "canonical Core derivation exceeds unchanged Core byte gate")
    receipt = {
        "schema": "heptatorch.compiler-capacity-adaptation.v1",
        "profile": PROFILE,
        "adaptation_script_sha256": digest(Path(__file__).read_bytes()),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "upstream_contract_sha256": UPSTREAM_CONTRACT_SHA256,
        "unchanged_generator_exact_source_reproduction": True,
        "upstream_authoring_sources_verified_before_import": True,
        "upstream_authoring_sources_verified_after_generation": True,
        "upstream_source_files_loaded": dict(sorted(loaded_files.items())),
        "source_changes": edits,
        "layout_macros": macro_changes,
        "relocation_capacity_macro": {"old": 12000, "new": 14336},
        "limits": LIMITS,
        "original_layout": original_layout,
        "expanded_layout": expanded_layout,
        "workspace_end_exclusive": new["explicit_exports"]["end_exclusive"],
        "workspace_unoccupied_tail_bytes": 2 * MIB - new["explicit_exports"]["end_exclusive"],
        "relative_relocations_upper_bound": reloc_bound,
        "canonical_core_bytes_upper_bound": core_byte_bound,
        "unchanged_general_core_byte_budget": 8 * MIB,
        "original_source_structure": original_structure,
        "candidate_source_structure": candidate_structure,
        "candidate_core_sha256": checked.sha256,
        "application_wrapper_invariants": invariant_functions,
        "native_bootstrap_performed": False,
        "native_qualification_performed": False,
        "upstream_qualification_claimed": False,
        "authoring_host_boundary": "Python upstream AST authoring and unchanged Core verification only",
    }
    # Only the explicitly selected isolated output directory is written.
    output.mkdir(parents=True, exist_ok=True)
    (output / "compiler.hepta").write_bytes(candidate)
    (output / "adaptation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"profile": PROFILE, "source": str(output / "compiler.hepta"),
                      "structure": candidate_structure, "workspace_end": 1515024,
                      "unchanged_application_wrappers": sorted(invariant_functions)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    adapt(options.source_root, options.output)
