"""Exact frozen source/build/reference admission used by both pure entry points."""
from pathlib import Path
import json
from admission_common import SHA, need, read, data, same, safe_path, load

CORE_FILES = {
    "README.md",
    "admission-request-template.json",
    "admission_common.py",
    "admit_recorded_integral24.py",
    "archive_custody.py",
    "authoring-notes.json",
    "complete_owners.py",
    "control_owners.py",
    "design-checkpoint.json",
    "profile.json",
    "provenance/native24-source01-manifest.json",
    "provenance/native32-source03-manifest.json",
    "provenance/pure32-source02-manifest.json",
    "provenance/r4-pure-admission20b23.py",
    "provenance/source02-archive-inverse.json",
    "provenance/source02-author-review.json",
    "recorded-wire-inverse.json",
    "recorded_controls.py",
    "recorded_observer.py",
    "recorded_wire.py",
    "seal-plan-template.json",
    "seal-request-template.json",
    "seal_raw.py",
    "source-adaptation.json",
    "source-reuse.json",
    "source_custody.py",
}
SOURCE_SCHEMA = "heptatorch.sum-integral24.pure-recorded-source.v1"

def self_package(root, expected_sha256):
    root = Path(root).resolve()
    manifest = data(root/"source-manifest.json", expected_sha256)
    need(manifest["schema"] == SOURCE_SCHEMA and manifest["status"] == "SOURCE_ONLY_NOT_RUN"
         and set(manifest["files"]) == CORE_FILES and manifest["file_count"] == len(CORE_FILES),
         "complete-dedicated-source-only-reader-package")
    total = 0
    for name, entry in manifest["files"].items():
        need(set(entry) == {"bytes", "sha256", "mode"}, "source-member-metadata-fields")
        raw = read(safe_path(root, name), entry["sha256"], entry["bytes"], entry["mode"])
        total += len(raw)
    need(total == manifest["payload_bytes"], "entire-source-package-byte-count")
    profile_raw = read(root/"profile.json", manifest["files"]["profile.json"]["sha256"])
    profile = json.loads(profile_raw)
    need(profile["schema"] == "heptatorch.sum-integral24.recorded-reader-profile.v1"
         and profile["status"] == "SOURCE_ONLY_NOT_RUN"
         and profile["source_revision"] == "sum-integral-source01",
         "frozen-actual-integral699-profile")
    parent_raw = read(root/"provenance/r4-pure-admission20b23.py")
    need(SHA(parent_raw) == "20b23badd9393a8f8ed8bdc23dc7bfbcc991ebabe007257297fd5e9b40ee3529",
         "complete-original-R4-pure-reader-source")
    inverse = data(root/"recorded-wire-inverse.json")
    text = read(root/"recorded_wire.py").decode()
    block = text[text.index("class RecordedSession:"):]
    need(SHA(text.encode()) == inverse["generated_file_sha256"]
         and len(text.encode()) == inverse["generated_file_bytes"]
         and SHA(block.encode()) == inverse["generated_block_sha256"]
         and len(block.encode()) == inverse["generated_block_bytes"],
         "recorded-wire-entire-derived-source-bound")
    restored = block
    for edit in reversed(inverse["edits"]):
        need(restored.count(edit["new"]) == 1, "wire-inverse-exact-occurrence", edit=edit["label"])
        restored = restored.replace(edit["new"], edit["old"])
    original = parent_raw.decode()
    begin = original.index("class RecordedSession:")
    end = original.index("\nclass Owners:", begin)
    need(restored == original[begin:end] and SHA(restored.encode()) == inverse["parent_block_sha256"],
         "whole-recorded-transport-inverse-to-original-R4")
    forward = restored
    for edit in inverse["edits"]:
        need(forward.count(edit["old"]) == 1, "wire-forward-exact-occurrence", edit=edit["label"])
        forward = forward.replace(edit["old"], edit["new"])
    need(forward == block and len(inverse["edits"]) == 4, "all-four-explicit-transport-edits")
    proof = data(root/"source-reuse.json")
    common = read(root/"admission_common.py").decode()
    for name, item in proof["exact_common_functions"].items():
        need(SHA(item["body"].encode()) == item["sha256"]
             and len(item["body"].encode()) == item["bytes"]
             and original.count(item["body"]) == common.count(item["body"]) == 1,
             "original-pure-byte-helper-exact-body-reuse", function=name)
    copied_manifest = read(root/"provenance/native32-source03-manifest.json")
    need(SHA(copied_manifest) == profile["native_parent_package"]["manifest"]["sha256"],
         "original-native32-parent-source-manifest-byte-copy")
    current_manifest = read(root/"provenance/native24-source01-manifest.json")
    need(SHA(current_manifest) == profile["native_package"]["manifest"]["sha256"],
         "original-native24-source-manifest-byte-copy")
    parent_raw = read(root/"provenance/pure32-source02-manifest.json",
                      "a0968c1dfdf00d9f5a2c67d181d15dc9b9b3ea6bfd671ac1c672dd2badcd2dd3")
    parent_manifest = json.loads(parent_raw)
    adaptation = data(root/"source-adaptation.json")
    need(adaptation["parent_manifest_sha256"] == SHA(parent_raw)
         and set(adaptation["files"]) == set(parent_manifest["files"]),
         "every-parent-pure32-source-file-has-complete-lineage")
    restored_files = {}
    for old_name, old_pin in parent_manifest["files"].items():
        item = adaptation["files"][old_name]
        current = read(safe_path(root, item["current_path"]))
        need(SHA(current) == item["after_sha256"]
             and len(current) == item["after_bytes"]
             and item["before_sha256"] == old_pin["sha256"]
             and item["before_bytes"] == old_pin["bytes"],
             "whole-current-and-original-source-lineage-pins", member=old_name)
        restored = current.decode()
        for change in reversed(item["edits"]):
            need(type(change["count"]) is int and change["count"] > 0
                 and change["before"] and change["after"]
                 and restored.count(change["after"]) == change["count"],
                 "whole-source-inverse-exact-occurrence", member=old_name)
            restored = restored.replace(change["after"], change["before"])
        original_bytes = restored.encode()
        need(SHA(original_bytes) == old_pin["sha256"] and len(original_bytes) == old_pin["bytes"],
             "whole-original-pure32-file-restored", member=old_name)
        forward = restored
        for change in item["edits"]:
            need(forward.count(change["before"]) == change["count"],
                 "whole-source-forward-exact-occurrence", member=old_name)
            forward = forward.replace(change["before"], change["after"])
        need(forward.encode() == current, "whole-source-forward-is-current-file", member=old_name)
        restored_files[old_name] = restored
    need(read(root/"complete_owners.py").decode() == restored_files["complete_owners.py"]
         and read(root/"admission_common.py").decode() == restored_files["admission_common.py"],
         "complete-original-owner-Case-reader-and-audit-primitives-unchanged")
    owner_text = restored_files["complete_owners.py"]
    begin = owner_text.index("    def __init__(self, directory, report, session, owner_module):")
    end = owner_text.index('        self.case_records = report["owner_records"]', begin)
    prefix = owner_text[begin:end]
    need(read(root/"control_owners.py").decode().count(prefix) == 1
         and SHA(prefix.encode()) == adaptation["control_frame_prefix"]["sha256"]
         and len(prefix.encode()) == adaptation["control_frame_prefix"]["bytes"],
         "all-control-frame-owner-predicates-reuse-entire-original-prefix")
    need({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} ==
         CORE_FILES | {"source-manifest.json"}
         and not any(p.is_symlink() for p in root.rglob("*")),
         "exact-current-source-directory-no-unbound-files")
    return manifest, profile, {"sha256": SHA(profile_raw), "bytes": len(profile_raw)}

def native_sources(profile):
    native_root = Path(profile["native_package"]["root"]).resolve()
    parent_root = Path(profile["native_parent_package"]["root"]).resolve()
    item = profile["native_package"]["manifest"]
    raw = read(item["path"], item["sha256"], item["bytes"], item["mode"])
    manifest = json.loads(raw)
    parent_item = profile["native_parent_package"]["manifest"]
    read(parent_item["path"], parent_item["sha256"], parent_item["bytes"], parent_item["mode"])
    need(Path(item["path"]).resolve() == native_root/"source-manifest.json"
         and Path(parent_item["path"]).resolve() == parent_root/"source-manifest.json",
         "explicit-original-native-and-parent-package-roots")
    # Exactly the original native24 sibling imports; no transformed code is run.
    reference_module = load(native_root/"admit_reference24.py", "admit_reference24",
                            manifest["files"]["admit_reference24.py"]["sha256"])
    build_module = load(native_root/"admit_integral_build.py", "admit_integral_build",
                        manifest["files"]["admit_integral_build.py"]["sha256"])
    replay = load(native_root/"replay_integral24.py", "integral24_original_replay01",
                  manifest["files"]["replay_integral24.py"]["sha256"])
    inputs = {}
    replay.admit_integral_package(native_root, item["sha256"], parent_root, inputs)
    same(manifest, json.loads(raw), "native24-source-package-unmodified")
    case = profile["original_case"]
    source = read(native_root/case["path"]).decode()
    begin, end = source.index(case["begin"]), source.index(case["end"])
    whole_case = source[begin:end].encode()
    need(SHA(whole_case) == case["sha256"] and len(whole_case) == case["bytes"],
         "original-entire-IntegralCase-unchanged")
    parent = load(parent_root/case["parent_case_path"], "integral24_original_parent_Case52c3",
                  case["parent_case_sha256"])
    owner = load(native_root/"owner_integral.py", "integral24_original_owner",
                 manifest["files"]["owner_integral.py"]["sha256"])
    observer = load(native_root/"native_observer_integral.py", "integral24_original_observer",
                    manifest["files"]["native_observer_integral.py"]["sha256"])
    entry = profile["native_exchange_constructor"]
    observer_source = read(native_root/entry["path"]).decode()
    begin = observer_source.index(entry["begin"], observer_source.index(entry["begin_after"]))
    end = observer_source.index(entry["end"], begin)
    constructor = observer_source[begin:end].encode()
    need(SHA(constructor) == entry["sha256"] and len(constructor) == entry["bytes"],
         "original-Native-exchange-wrapper-unchanged")
    control_pin = profile["original_controls"]
    control_raw = read(native_root/control_pin["path"], control_pin["sha256"], control_pin["bytes"])
    control_text = control_raw.decode()
    control_body = control_text[control_text.index(control_pin["begin"]):].encode()
    need(SHA(control_body) == control_pin["function_sha256"]
         and len(control_body) == control_pin["function_bytes"],
         "entire-original-five-control-driver-unchanged")
    controls = load(native_root/control_pin["path"], "integral24_original_bounded_controls",
                    control_pin["sha256"])
    return {"native_root": native_root, "native_manifest": manifest, "replay": replay,
            "reference_module": reference_module, "build_module": build_module,
            "parent": parent, "owner": owner, "observer": observer, "inputs": inputs,
            "controls": controls, "protocol_path": parent_root/"vendor"/"protocol.py"}

def admit_original_inputs(profile, context):
    ref = profile["reference"]
    reference_binding = profile["reference_binding"]
    reference, contract, ref_summary = context["reference_module"].admit_reference(
        Path(reference_binding["path"]), reference_binding["sha256"], context["inputs"])
    read(reference_binding["path"], reference_binding["sha256"],
         reference_binding["bytes"], reference_binding["mode"])
    actual = profile["actual_build_binding"]
    binding, build, build_summary = context["build_module"].admit_build(
        Path(actual["path"]), actual["sha256"], context["inputs"])
    read(actual["path"], actual["sha256"], actual["bytes"], actual["mode"])
    for key in ("runtime", "binary", "build_report"):
        same(binding[key], profile[key], "entire-actual-build-file-binding-"+key)
    same({key: binding["profile"][key] for key in profile["profile"]},
         profile["profile"], "actual-profile-fields")
    need(len(reference["rows"]) == len(contract["cases"]) == ref["cases"] == 24
         and sum(len(row["events"]) for row in reference["rows"]) == ref["events"] == 260,
         "all-original24-and260-present")
    for path, value in context["inputs"].items():
        record = value["record"]
        need(record["path"] == path, "original-input-key-is-canonical-original-path")
        read(path, record["sha256"], record["bytes"], record["mode"])
    context.update(reference=reference, contract=contract, reference_admission=ref_summary,
                   binding=binding, build=build, build_admission=build_summary)
    return context
