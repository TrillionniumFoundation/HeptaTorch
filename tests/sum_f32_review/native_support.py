"""Exact frozen observer adaptation for the distinct 701-function image."""
import json
import re
import sys
import types
from pathlib import Path

from common import Inputs, dump, require, sha

INTENT = "continuation/20261117/evidence/root-sum-integral-native24-reviewed-intent-v2.json"
INTENT_SHA = "4050bf41236f33080d590d0b65bbf1359f9834d4001eb64883a62cbef2153efc"
PARENT_SHA = {
    "owner_integral.py": "c39600e7c37dea390f88cf9e74d86dfa924301456e0f32e127d82c98ad6778b0",
    "native_observer_integral.py": "787f05d1bc1018bacfb0e21b70dd356b56d776c437decabb728ef7f19d261273",
}
PROTOCOL_SHA = "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
CLIENT_SHA = "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b"
PARENT_SOURCE = "b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4"


def bind_sources(inputs, folder):
    for name in ("native.py", "native_support.py", "owner_adapter.py", "controls.py",
                 "allocation_controls.py", "reference.py", "cases.py", "common.py"):
        inputs.bind(Path(folder) / name)


def adapted_parent(repository, filename, inputs, *, owner=False):
    raw = inputs.bind(Path(repository) / INTENT, INTENT_SHA)
    row = json.loads(raw)["source_payloads"][filename]
    original = row["text"].encode()
    require(sha(original) == row["sha256"] == PARENT_SHA[filename]
            and len(original) == row["size"] and row["eof"], "frozen-parent-payload", name=filename)
    edits = [(b"HTSINT01", b"HTSINT02", 2 if owner else 1),
             (b"HTOSIN01", b"HTOSIN02", 2 if owner else 1)]
    if owner:
        edits += [(b"functions == 699", b"functions == 701", 1),
                  (b"sum-integral-source01-UNBOUND-699", b"sum-integral-f32-source01-UNBOUND-701", 1),
                  (b"sum-integral-source01-%d-preserved47-48", b"sum-integral-f32-source01-%d-preserved47-48", 1),
                  (b"heptatorch.sum-integral-owner-source01.v1", b"heptatorch.sum-integral-f32-owner.v1", 1)]
    derived = original
    for before, after, count in edits:
        require(derived.count(before) == count and after not in derived, "exact-observer-edit", before=before.decode())
        derived = derived.replace(before, after)
    inverse = derived
    for before, after, count in reversed(edits):
        require(inverse.count(after) == count, "observer-inverse-count")
        inverse = inverse.replace(after, before)
    require(inverse == original, "whole-original-observer-inverse")
    name = "sum_integral_f32_" + filename.replace(".py", "")
    module = types.ModuleType(name)
    module.__file__ = str(Path(repository) / INTENT) + "#" + filename
    sys.modules[name] = module
    exec(compile(derived, module.__file__, "exec"), module.__dict__)
    proof = dict(parent_name=filename, original_sha256=sha(original), derived_sha256=sha(derived),
                 original_bytes=len(original), derived_bytes=len(derived), whole_inverse_exact=True,
                 edits=[dict(before=a.decode(), after=b.decode(), count=n) for a, b, n in edits])
    return module, proof


def source_functions(raw, expected_count=701):
    matches = list(re.finditer(rb"^fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n", raw, re.M | re.S))
    require(matches and raw[:matches[0].start()] + b"".join(x[0] for x in matches) == raw,
            "whole-hia-source-function-coverage")
    functions = {m[1].decode(): m[0] for m in matches}
    require(len(functions) == len(matches) == expected_count, "exact-complete-function-count")
    return functions


def _first_source_inverse(source, inputs):
    here = Path(__file__).resolve().parent
    parent = inputs.bind(here / "baseline/runtime.hepta", PARENT_SOURCE)
    old = source_functions(parent, 700)
    current = source_functions(source)
    addition = inputs.bind(here / "sf_sum_f32.hepta", "da9e19a49edfa40679d732f3873a805703bdf4ec1aebae6aa963c5aa1d585b4e")
    require(current["sf_sum_f32"] == addition, "complete-authored-F32-function")
    old_call = b'  v23:u64 = call(v9,v10,v11,v12,v13,v14,v15) {"function":"sd_sum_tensor"}\n'
    new_call = b'  v23:u64 = call(v9,v10,v11,v12,v13,v14,v15) {"function":"sf_sum_f32"}\n'
    require(current["si_dispatch_sum"].count(new_call) == 1 and
            current["si_dispatch_sum"].replace(new_call, old_call) == old["si_dispatch_sum"],
            "exact-dispatch-inverse")
    require(set(current) == set(old) | {"sf_sum_f32"}, "exact-added-function-set")
    unchanged = [name for name in old if current[name] == old[name]]
    require(len(unchanged) == 699, "all-699-other-functions-unchanged")
    inverse = dict(current)
    del inverse["sf_sum_f32"]
    inverse["si_dispatch_sum"] = old["si_dispatch_sum"]
    restored = parent.split(b"\n", 1)[0] + b"\n" + b"".join(inverse.values())
    require(restored == parent, "complete-700-parent-source-inverse")
    return dict(whole_inverse_exact=True, parent_source_sha256=sha(parent),
                source_sha256=sha(source), original_functions=700, current_functions=701,
                unchanged_functions=699, modified_functions=["si_dispatch_sum"], added_functions=["sf_sum_f32"])


# Explicit AVX2 source identity and two-layer whole-source inverse.
FIRST701_SOURCE = "de4fdf964a74be11acfeddcf5fe16656c6cfa671fc7917585a72364918387a92"
AVX2_SOURCE = "706673d47858a47fafeefe9f40aed6c52e5181caa4183ec12f72151385ca84e3"
FIRST701_IMAGE = "75e29e13c0a7f14395fd21acb82686279dc962615ae6bcde3176b72083cac971"
AVX2_IMAGE = "589367e5db70847c727e8d1005680dc5e8f73d536d887c9bb2c30c41db4ae9c4"
TRANSITION_SHA = "fc102e38302708bbee94f7021071ee11875aeb28563dc07c26b890c68d09c840"
FIRST701_READER_SHA = "6b3e61e1ab9bbe3679687c797bc15eaa249d0eba0438da541216ce42c1978be2"
FIRST701_SUPPORT_SHA = "f8e80daceae2b184d7fd5fc75f4c3ab014be2d32fc9cf39484daf5087bcfed63"
PROFILE_MARKER = b"\n\n# Explicit AVX2 source identity and two-layer whole-source inverse.\n"
EXACT_EDITS = {
    "sd_vector_inner": (("v3", 16, 8), ("v41", 16, 8), ("v60", 64, 32)),
    "sd_loop2d": (("v169", 16, 8), ("v195", 16, 8), ("v343", 64, 32)),
}


def reader_inverse(inputs):
    here = Path(__file__).resolve().parent
    current = inputs.bind(here / "native.py")
    inverse = current
    for new, old in ((AVX2_SOURCE, FIRST701_SOURCE), (AVX2_IMAGE, FIRST701_IMAGE)):
        new, old = new.encode(), old.encode()
        require(inverse.count(new) == 1 and old not in inverse, "exact-reader-identity-only-edit")
        inverse = inverse.replace(new, old)
    require(sha(inverse) == FIRST701_READER_SHA and len(inverse) == 35016,
            "whole-frozen-reader-inverse-all-comparators-unchanged")
    support = inputs.bind(Path(__file__).resolve())
    require(support.count(PROFILE_MARKER) == 1, "exact-support-extension-boundary")
    prefix = support.split(PROFILE_MARKER, 1)[0]
    renamed = b"def _first_source_inverse(source, inputs):"
    original = b"def source_inverse(source, inputs):"
    require(prefix.count(renamed) == 1 and original not in prefix, "exact-preserved-original-source-proof")
    restored = prefix.replace(renamed, original)
    require(sha(restored) == FIRST701_SUPPORT_SHA and len(restored) == 5149,
            "whole-frozen-support-prefix-inverse")
    return dict(reader_sha256=sha(current), reader_original_sha256=sha(inverse),
                reader_whole_inverse_exact=True, reader_identity_edits_only=2,
                support_sha256=sha(support), support_original_sha256=sha(restored),
                support_whole_original_prefix_inverse_exact=True,
                original_owner_comparators_controls_and_expectations_modified=False)


def source_inverse(source, inputs):
    require(sha(source) == AVX2_SOURCE, "exact-AVX2-corrected-complete-source")
    here = Path(__file__).resolve().parent
    transition = json.loads(inputs.bind(here / "transition.json", TRANSITION_SHA))
    require(transition["schema"] == "heptatorch.sum-f32.dispatch-width-source-transition.v1"
            and transition["status"] == "SOURCE_ONLY_NOT_RUN"
            and transition["candidate_source_sha256"] == AVX2_SOURCE
            and transition["original_source_sha256"] == FIRST701_SOURCE
            and transition["modified_constants"] == 6 and transition["functions"] == 701,
            "exact-audited-six-constant-source-transition")
    require(set(transition["changed_functions"]) == set(EXACT_EDITS)
            and set(transition["bodies"]) == set(EXACT_EDITS), "only-two-reduction-functions")
    functions = source_functions(source)
    inverse = source
    body_proofs = {}
    for name, expected_edits in EXACT_EDITS.items():
        row = transition["bodies"][name]
        current, original = row["candidate_body"].encode(), row["original_body"].encode()
        require(current == functions[name] and sha(current) == row["candidate_sha256"]
                and sha(original) == row["original_sha256"], "complete-original-and-candidate-function", name=name)
        require(tuple((edit["variable"], edit["before"], edit["after"]) for edit in row["edits"])
                == expected_edits, "exact-six-constant-edit-set", name=name)
        reconstructed = current
        for variable, before, after in expected_edits:
            old_line = ('  %s:u64 = const() {"value":%d}\n' % (variable, before)).encode()
            new_line = ('  %s:u64 = const() {"value":%d}\n' % (variable, after)).encode()
            require(reconstructed.count(new_line) == 1 and old_line not in reconstructed,
                    "exact-one-constant-line", function=name, variable=variable)
            reconstructed = reconstructed.replace(new_line, old_line)
        require(reconstructed == original, "complete-function-inverse-no-other-change", name=name)
        require(inverse.count(current) == 1, "single-complete-function-in-source", name=name)
        inverse = inverse.replace(current, original)
        body_proofs[name] = dict(original_sha256=sha(original), current_sha256=sha(current),
                                changed_constants=len(expected_edits), whole_inverse_exact=True)
    require(sha(inverse) == FIRST701_SOURCE, "whole-current-to-first701-source-inverse")
    first_functions = source_functions(inverse)
    require(set(first_functions) == set(functions)
            and {name for name in functions if functions[name] != first_functions[name]} == set(EXACT_EDITS),
            "all-other-699-first701-function-bodies-unchanged")
    # The original whole-file proof executes on the exactly reconstructed first
    # 701 source. Its original 699-unchanged claim applies to that first source,
    # not to this corrected source.
    first_to_700 = _first_source_inverse(inverse, inputs)
    parent = source_functions(inputs.bind(here / "baseline/runtime.hepta", PARENT_SOURCE), 700)
    changed_700 = sorted(name for name in parent if functions[name] != parent[name])
    unchanged_700 = sum(functions[name] == parent[name] for name in parent)
    require(changed_700 == ["sd_loop2d", "sd_vector_inner", "si_dispatch_sum"]
            and unchanged_700 == 697 and set(functions) - set(parent) == {"sf_sum_f32"},
            "exact-current-source-difference-from-original700")
    reader_proof = reader_inverse(inputs)
    return dict(whole_inverse_exact=True, source_sha256=sha(source),
                parent_source_sha256=PARENT_SOURCE, original_functions=700, current_functions=701,
                unchanged_functions=697, modified_functions=changed_700, added_functions=["sf_sum_f32"],
                current_to_first701=dict(source_sha256=sha(source), inverse_source_sha256=sha(inverse),
                    whole_inverse_exact=True, unchanged_functions=699,
                    modified_functions=sorted(EXACT_EDITS), modified_constants=6,
                    transition_sha256=TRANSITION_SHA, function_proofs=body_proofs),
                first701_to_700=first_to_700, reader_identity_transition=reader_proof,
                sum_kernel_dispatch=dict(advertised_cpu_capability="AVX512",
                    actual_fixed_sum_kernel="AVX2", float32_vector_lanes=8, float32_vector_bytes=32,
                    evidence=transition["registration_evidence"]),
                original_first701_failure_modified=False, runtime_qualification_transferred=False)


# Explicit skipped-empty cat source identity and complete third inverse layer.
CAT_SOURCE = "82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313"
CAT_IMAGE = "80445a8c433f20314d3bc7b00ebb68f6a70fa200ee2e6aba193cff21aabfbc17"
CAT_DELTA_SHA = "cf90860307f8b4dac2d369371a0c23176f5e8d8fc1d050a7daf03eb4e6a8726d"
CAT_NATIVE_SHA = "8599ede3bace8c41a0e62e6672b03b2b885247dcc099defb2227a45869432252"
CAT_EMITTER_SHA = "2b4e162ca8cd1c8c4b27184ddf72ef7e0a35ec903ee918ce4d9223546d740248"
AVX2_READER_SHA = "f0a1da59c90c83006cc1eafb6b914c658512f57a76eabd6ca6d12df7e7c2db52"
AVX2_SUPPORT_SHA = "3998704a444987f85aaac8a0914ddb8d0bc41b4a816d39aad0df49d350a401b5"
CAT_LAYER_MARKER = b"\n\n# Explicit skipped-empty cat source identity and complete third inverse layer.\n"
_avx2_source_inverse = source_inverse
_avx2_reader_inverse = reader_inverse


def reader_inverse(inputs):
    """Run the complete original reader proof on explicitly inverted bytes.

    Actual file custody always binds the new physical files first. The closed
    two-file bridge is used only by the frozen proof, and its returned data is
    labelled as inverse bytes rather than as a physical file observation.
    """
    here = Path(__file__).resolve().parent
    current = inputs.bind(here / "native.py", CAT_NATIVE_SHA)
    inverse_reader = current
    for candidate, previous in ((CAT_SOURCE, AVX2_SOURCE), (CAT_IMAGE, AVX2_IMAGE)):
        candidate, previous = candidate.encode(), previous.encode()
        require(inverse_reader.count(candidate) == 1 and previous not in inverse_reader,
                "cat-reader-exact-identity-edit")
        inverse_reader = inverse_reader.replace(candidate, previous)
    require(sha(inverse_reader) == AVX2_READER_SHA and len(inverse_reader) == 35016,
            "whole-cat-reader-to-original-avx2-reader-inverse")
    support = inputs.bind(Path(__file__).resolve())
    require(support.count(CAT_LAYER_MARKER) == 1, "single-cat-support-extension")
    inverse_support = support.split(CAT_LAYER_MARKER, 1)[0]
    require(sha(inverse_support) == AVX2_SUPPORT_SHA and len(inverse_support) == 12249,
            "complete-original-avx2-support-prefix-unchanged")
    inverse_files = {(here / "native.py").resolve(): inverse_reader,
                     Path(__file__).resolve(): inverse_support}

    class VerifiedInverseBytes:
        def bind(self, path, expected=None):
            path = Path(path).resolve()
            require(path in inverse_files, "frozen-reader-proof-only-two-inverted-inputs")
            raw = inverse_files[path]
            require(expected is None or sha(raw) == expected,
                    "frozen-reader-proof-inverse-digest")
            return raw

    original_proof = _avx2_reader_inverse(VerifiedInverseBytes())
    require(original_proof["reader_sha256"] == AVX2_READER_SHA
            and original_proof["support_sha256"] == AVX2_SUPPORT_SHA
            and original_proof["reader_whole_inverse_exact"] is True
            and original_proof["support_whole_original_prefix_inverse_exact"] is True,
            "complete-original-avx2-reader-proof-executed")
    return dict(reader_sha256=sha(current), support_sha256=sha(support),
                current_to_avx2=dict(reader_sha256=AVX2_READER_SHA,
                    support_sha256=AVX2_SUPPORT_SHA, reader_identity_edits_only=2,
                    reader_whole_inverse_exact=True,
                    complete_support_prefix_unchanged=True),
                avx2_to_first701=original_proof,
                frozen_proof_executed_on_verified_inverse_bytes=True,
                actual_physical_files_bound_before_inverse=True,
                physical_input_records_modified=False,
                original_owner_comparators_controls_and_expectations_modified=False)


def source_inverse(source, inputs):
    require(CAT_IMAGE != "0" * 64, "cat-image-identity-must-be-resolved-before-admission")
    require(sha(source) == CAT_SOURCE, "exact-cat-corrected-complete-source")
    here = Path(__file__).resolve().parent
    inputs.bind(here / "emit_cat.py", CAT_EMITTER_SHA)
    delta = json.loads(inputs.bind(here / "cat-delta.json", CAT_DELTA_SHA))
    require(delta["schema"] == "heptatorch.current701.cat-empty-source-delta.v1"
            and delta["status"] == "SOURCE_ONLY_NOT_RUN"
            and delta["parent_source_sha256"] == AVX2_SOURCE
            and delta["candidate_source_sha256"] == CAT_SOURCE
            and delta["function"] == "incoming_describe_cat"
            and delta["functions"] == 701,
            "exact-audited-cat-source-delta")
    before, after = delta["before_body"].encode(), delta["after_body"].encode()
    require(sha(before) == delta["before_sha256"] ==
            "6fec0a6aec5be686824fb0e78b4ebb93e5584c1962732dc8039784e2fdee1383"
            and sha(after) == delta["after_sha256"] ==
            "2440c01cae1280eb2a02305333a2babc82d19915e8cffc7692742ba7df4122fa",
            "complete-original-and-cat-corrected-function-identities")
    before_block = b'''block b3(v35:mutbuf)
  v51:u64 = const() {"value":3848}
  v52:u64 = const() {"value":1}
  v53:u64 = call(v35,v51,v52) {"function":"putw"}
  ret(v52)
'''
    after_block = b'''block b3(v35:mutbuf)
  v51:u64 = const() {"value":3848}
  v52:u64 = const() {"value":1}
  v251:u64 = const() {"value":2}
  v53:u64 = call(v35,v51,v251) {"function":"putw"}
  ret(v52)
'''
    require(before.count(before_block) == 1 and after.count(after_block) == 1
            and after.replace(after_block, before_block) == before,
            "complete-cat-function-inverse-only-mode-assignment-ret1-unchanged")
    current = source_functions(source)
    require(current["incoming_describe_cat"] == after and source.count(after) == 1,
            "complete-cat-function-bound-to-actual-source")
    restored = source.replace(after, before)
    require(sha(restored) == AVX2_SOURCE, "whole-cat-to-avx2-source-inverse")
    avx2 = source_functions(restored)
    require(list(avx2) == list(current)
            and [name for name in current if current[name] != avx2[name]] == ["incoming_describe_cat"],
            "all-other-700-avx2-function-bodies-unchanged")
    # This calls the entire frozen r2 source proof, which itself calls the entire
    # original first701-to-700 proof. All original assertions remain in the
    # unmodified support prefix above this explicit extension.
    original_proof = _avx2_source_inverse(restored, inputs)
    require(original_proof["source_sha256"] == AVX2_SOURCE
            and original_proof["whole_inverse_exact"] is True
            and original_proof["unchanged_functions"] == 697,
            "complete-original-avx2-source-proof-executed")
    parent = source_functions(inputs.bind(here / "baseline/runtime.hepta", PARENT_SOURCE), 700)
    changed = sorted(name for name in parent if current[name] != parent[name])
    unchanged = sum(current[name] == parent[name] for name in parent)
    require(changed == ["incoming_describe_cat", "sd_loop2d", "sd_vector_inner", "si_dispatch_sum"]
            and unchanged == 696 and set(current) - set(parent) == {"sf_sum_f32"},
            "exact-cat-corrected-source-difference-from-original700")
    return dict(whole_inverse_exact=True, source_sha256=sha(source),
                parent_source_sha256=PARENT_SOURCE, original_functions=700, current_functions=701,
                unchanged_functions=696, modified_functions=changed, added_functions=["sf_sum_f32"],
                current_to_avx2=dict(source_sha256=sha(source), inverse_source_sha256=sha(restored),
                    whole_inverse_exact=True, unchanged_functions=700,
                    modified_functions=["incoming_describe_cat"],
                    delta_sha256=CAT_DELTA_SHA, original_success_return_preserved=True,
                    shared_cache_and_inplace_guards_unchanged=True),
                avx2_to_700=original_proof,
                original_first701_failure_modified=False,
                original_avx2_original270_failure_modified=False,
                runtime_qualification_transferred=False)
