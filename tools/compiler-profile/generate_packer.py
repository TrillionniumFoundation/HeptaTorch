#!/usr/bin/env python3
"""Author the pinned HIA packer for the distinct larger compiler workspace.

Only build-time authoring uses Python. The generated HIA executable performs
image copying, ELF fields, and call displacement when the native build runs.
No upstream source is changed, and no ordinary application wrapper is changed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import FunctionType

sys.dont_write_bytecode = True
PROFILE = "heptatorch.hia.compiler.f256-b2560-w2m.v1"
UPSTREAM_COMMIT = "f33d8da316de8fdf27b504371f1a5310997ff548"
CONTRACT_SHA256 = "8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c"
ORIGINAL_SHA256 = "63ec799f8b427e089c2a66bc345c1ca1ed6b022c12e6af5bfebd6953d66b4c2d"
EXPECTED_SHA256 = "388337bd72acb13847a894601a9c43057c86caaf33787aa0c5c706916b94a0f1"
MIB = 1 << 20
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
tools/generate_hia_elf_linker.py
tools/generate_hia_native_filter.py
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


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def private_function(original, **replacements):
    namespace = dict(original.__globals__)
    namespace.update(replacements)
    result = FunctionType(original.__code__, namespace, original.__name__,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def generate(source_root: Path, output: Path):
    source_root = source_root.resolve()
    contract_raw = (source_root / "docs/HIA_SELFHOST_CORE_SUPPORT.json").read_bytes()
    require(digest(contract_raw) == CONTRACT_SHA256, "upstream contract hash mismatch")
    source_hashes = json.loads(contract_raw)["source_identity"]["files"]
    original_raw = (source_root / "examples/hia_bootstrap/native_filter_packer.hepta").read_bytes()
    require(digest(original_raw) == ORIGINAL_SHA256, "upstream packer HIA hash mismatch")
    preverified = {}
    for relative in AUTHORING_FILES:
        actual = digest((source_root / relative).read_bytes())
        require(actual == source_hashes.get(relative),
                "upstream authoring input hash mismatch: " + relative)
        preverified[relative] = actual

    sys.path.insert(0, str(source_root))
    from tools import generate_hia_native_filter as generator
    from hepta.hia_agent_assembly import assemble_program, render_assembly

    require(render_assembly(generator.document()).encode() == original_raw,
            "unchanged upstream packer generator does not reproduce original HIA")
    require(generator.WIRE_BYTES == 8 * MIB and generator.OUTPUT_OFFSET == 8 * MIB + 4096,
            "unexpected upstream wire layout")
    template = private_function(generator.template, WIRE_BYTES=MIB, OUTPUT_OFFSET=MIB + 4096)
    source_program = private_function(generator.source_program, template=template)
    document = private_function(generator.document, source_program=source_program)
    core = document(output_bytes=MIB, workspace_bytes=2 * MIB, module="selfhost_exec")
    raw = render_assembly(core).encode()
    require(digest(raw) == EXPECTED_SHA256, "adapted HIA packer source hash mismatch")
    checked = assemble_program(raw)
    require(checked.as_dict() == core, "adapted packer assembly/Core mismatch")

    loaded = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(source_root).as_posix()
        except ValueError:
            continue
        actual = digest(path.read_bytes())
        require(preverified.get(relative) == actual,
                "loaded source outside preverified upstream closure: " + relative)
        loaded[relative] = actual
    require(set(loaded) == set(preverified), "upstream authoring import closure changed")

    receipt = {
        "schema": "heptatorch.compiler-native-packer-authoring.v1",
        "profile": PROFILE,
        "upstream_commit": UPSTREAM_COMMIT,
        "adaptation_script_sha256": digest(Path(__file__).read_bytes()),
        "upstream_source_files_preverified": preverified,
        "upstream_contract_sha256": CONTRACT_SHA256,
        "upstream_packer_source_sha256": ORIGINAL_SHA256,
        "unchanged_generator_exact_source_reproduction": True,
        "source_sha256": digest(raw),
        "source_bytes": len(raw),
        "core_sha256": checked.sha256,
        "upstream_source_files_loaded": dict(sorted(loaded.items())),
        "settings": {
            "WIRE_BYTES": {"old": 8 * MIB, "new": MIB},
            "OUTPUT_OFFSET": {"old": 8 * MIB + 4096, "new": MIB + 4096},
            "output_bytes": MIB,
            "workspace_bytes": 2 * MIB,
            "module": "selfhost_exec",
        },
        "compiler_executable_mapping_bytes": 4 * MIB + 4096,
        "native_packing_performed": False,
        "authoring_host_boundary": "Python upstream AST authoring and unchanged Core verification only",
        "application_wrapper_unchanged": "The compiler's ordinary cli_begin/cli_finish/wrapper functions are unchanged; this packer only wraps the compiler flat image.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "compiler-packer.hepta").write_bytes(raw)
    (output / "packer-authoring.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"profile": PROFILE, "source_sha256": digest(raw), "source_bytes": len(raw)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source_root, args.output)
