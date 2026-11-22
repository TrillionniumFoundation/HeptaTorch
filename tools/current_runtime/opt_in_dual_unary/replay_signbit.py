#!/usr/bin/env python3
"""Re-run retained signbit checks with a separate exact dual-image successor."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import types

BASE = "083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5"
DUAL = "5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72"
RUNNER = "tests/signbit/run.py"
NATIVE = "tests/signbit/native.py"
ADMIT = "tools/current_runtime/opt_in_admission/admit.py"
CHECK = b"        n.check('unassigned opcode remains rejected', status, 4)\n"
CHECK_NEXT = b"        observe_assigned127(n, opcode, source, row, status, before, after, records)\n"
ISSUE = b"        status, _, _, _ = n.issue(n.wire.packet([(opcode, source)]))\n"
ISSUE_NEXT = b"        status, records, before, after = n.issue(n.wire.packet([(opcode, source)]))\n"
LABEL = b"original_unknown125_rejected=True, unknown127_rejected=True)"
LABEL_NEXT = b"original_unknown125_rejected=True, unknown127_rejected=False, accepted127_successor=True)"
PROFILE_ERROR = "build used another source manifest"
REF_FIRST = b"        supervisor=module(ROOT/'tests/current_runtime_process.py'"
REF_LAST = b"        report.update(Torch_executed=True,reference_directory="
PINS = {
    'tests/current_runtime_process.py': '36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523',
    'tests/differentiable_view_native_review/owner_client.py': 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
    'tests/differentiable_view_native_review/protocol.py': '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9',
    'tests/half_logical/__init__.py': '4c5fbdf0c7f8da54986ba8c0aef475abc92284941bcf215c0545a3528dc8b8fb',
    'tests/half_logical/native.py': '9c2cd19a5bb04eaa037f5905a52bce2dcc31042101f72d07e71205745160a14f',
    'tests/half_logical/reference.py': '5bb6a3804717968dba37f3174809a7c1eee5b225f0a3085b44fcec7656155528',
    'tests/numeric_predicates/__init__.py': '7d9b42b2f3b86a533773fdb030f2cb539632f68ffdb22874a5071d0a4cc1b64f',
    'tests/numeric_predicates/cases.py': 'e69bebed091db88ef81eda2ea2052af6b2d88c74840909ee27e2cbdf6a9ec694',
    'tests/numeric_predicates/evidence.py': 'a3d39ee66d5511d49bb484bc9b95a131504085679546d25266b281e26b879c84',
    'tests/numeric_predicates/native.py': 'c0e9d1ba6f0dac5499f091333611a5bea0fb6f4a25e74faa39f35486fd404d2d',
    'tests/numeric_predicates/reference.py': '3ec85934110f2d8f273bc1e209a9c9ce13331cc4ff915dc9d7058d81913d0abd',
    'tests/pinned_reference_custody.py': 'fa91a644b8743493ed6ed8f29da966494983cf5eed7300faa715ee4e08c7212e',
    'tests/pytorch_oracle.py': 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae',
    'tests/signbit/__init__.py': '28b2f009de665a7122a5bb49ffef1173092abc53a9018ba2d01dc732a664d8e4',
    'tests/signbit/cases.py': '634059245c835c8620806975c8b153973058a0deef18c9c8503470a2e4e2c06c',
    'tests/signbit/evidence.py': '814161b56b25a43bc7e44cecc830165faf1aec76eead0dee120079a46847fd15',
    'tests/signbit/native.py': 'c754b4b4d3049fb3de9a51802f0ad2fd6d47fa92bbf4d76e7ce1be7053243924',
    'tests/signbit/reference.py': '4af95d575ce786bddaa877ce82cd814ca368b7eb71a1c2b00d9cb18b8d62e437',
    'tests/signbit/run.py': 'fb500a7d6f34e4748c389e0b582dfc7b3bc4063333dab029468fc0317abc7714',
    'tools/current_runtime/opt_in_admission/admit.py': '51b32b1835fe8a294f1f42b98ca3b87b10a5096dec08d34840ac7acab61f672d',
    'tools/current_runtime/qualification.py': 'f4be6d41c29659c3132f1f64014ad9d8ebcf612db3a9cb556ee5ddd178a540b6',
}

ORACLE = r'''
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "tests"))
from numeric_predicates.reference import setup
import torch
assert torch.__version__ == "2.15.0+cpu"
assert torch.version.git_version == "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
assert torch.backends.cpu.get_cpu_capability() == "AVX2"
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
spec = json.load(sys.stdin)
source, retained = setup(torch, spec)
before = bytes(source.untyped_storage())
a = torch.logical_not(source)
logical = bytes(a.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8).tolist())
assert bytes(source.untyped_storage()) == before
assert a._cdata != source._cdata and a.untyped_storage()._cdata != source.untyped_storage()._cdata
extension = Path(torch._C.__file__).resolve()
print(json.dumps(dict(
    torch_version=torch.__version__, torch_git=torch.version.git_version,
    extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
    spec=spec, input_storage=before.hex(),
    output=dict(dtype=str(a.dtype), element_size=a.element_size(), shape=list(a.shape),
        strides=list(a.stride()), numel=a.numel(), storage_bytes=a.untyped_storage().nbytes(),
        storage_offset=a.storage_offset(), version=a._version, requires_grad=a.requires_grad,
        is_leaf=a.is_leaf, is_view=a._is_view(), history_present=a.grad_fn is not None,
        retains_grad=a.retains_grad, logical=logical.hex(), storage=bytes(a.untyped_storage()).hex()))))
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def need(value, why):
    if not value:
        raise ValueError(why)


def read(path, digest=None):
    need(path.resolve() == path and path.is_file() and not path.is_symlink(), "ordinary physical file required")
    raw = path.read_bytes()
    need(digest is None or sha(raw) == digest, "wrong exact input: " + str(path))
    return raw


def load(name, path, raw, package=None, **extras):
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), package
    module.__dict__.update(extras)
    exec(compile(raw, str(path), "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def change(raw, old, new):
    need(raw.count(old) == 1 and new not in raw, "unexpected successor anchor")
    changed = raw.replace(old, new, 1)
    need(changed.replace(new, old, 1) == raw, "successor inverse differs")
    return changed


def json_output(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def suite(args):
    repo, base, cand, output = args.repository, args.baseline_build, args.dual_build, args.output
    for path in {(repo / p).parent for p in PINS}:
        need(not (path / "__pycache__").exists(), "use a clean source checkout")
    source = {p: read(repo / p, pin) for p, pin in PINS.items()}
    admit = load("signbit_opt_in_admit", repo / ADMIT, source[ADMIT]).admit
    manifest = repo / "tools/current_runtime/source/manifest.json"
    baseline_id = admit("baseline", manifest, base, {})
    candidate_id = admit("dual", manifest, cand, {"dual_scoped": args.dual_scoped})
    need((baseline_id["native_image_sha256"], candidate_id["native_image_sha256"]) == (BASE, DUAL), "wrong pair")
    historical = args.baseline_run
    original_raw = read(historical / "report.json")
    original_report = json.loads(original_raw)
    physical = json.loads(read(historical / "physical-qualification.json"))
    need(original_report["status"] == "PASS" and original_report["failures"] == []
         and original_report["native_sha256"] == BASE and original_report["source_sha256"] == baseline_id["source_sha256"]
         and original_report["actual_build_sha256"] == baseline_id["build_report_sha256"]
         and original_report["all_inputs_unchanged"] is True
         and original_report["full_installation_provenance_verified"] is True
         and physical["physical_replay"] is True and physical["cases"] == 97, "baseline did not actually qualify")
    by_name = {row["name"]:row for row in original_report["results"]}
    baseline_row = by_name["packet-controls"]
    need(baseline_row["status"] == "PASS" and baseline_row["result"]["unknown127_rejected"] is True
         and baseline_row["result"]["original_unknown125_rejected"] is True, "historical refusal missing")

    reference = historical / "reference"
    execution = historical / "reference-execution"
    latest = {}
    def read_ref(inputs):
        from signbit import evidence
        checked = evidence.read(repo, reference, execution, inputs.bind, False)
        need(checked["report_sha256"] == original_report["reference_sha256"], "wrong original reference")
        return checked

    original = source[RUNNER]
    begin = original.index(REF_FIRST)
    end = original.index(REF_LAST, begin)
    derived = change(original, original[begin:end], b"        fixed=read_bound_original_reference(inputs)\n"
                                                    b"        execution=fixed['execution']\n")
    runner = load("dual_signbit_existing_runner", repo / RUNNER, derived,
                  read_bound_original_reference=read_ref)
    # Verify the exact original baseline and exact original rejection of the dual
    # build. Only then bind the candidate build with independent admission.
    old_profile = runner.profile
    def candidate_profile(repository, build, build_sha, inputs):
        need((repository, build, build_sha) == (repo, cand, candidate_id["build_report_sha256"]), "wrong candidate")
        legacy_base = old_profile(repo, base, baseline_id["build_report_sha256"], inputs)
        need(legacy_base["native_sha256"] == BASE, "original baseline profile differs")
        try:
            old_profile(repo, cand, build_sha, inputs)
        except ValueError as err:
            need(str(err) == PROFILE_ERROR, "unexpected original profile rejection")
        else:
            raise ValueError("original profile unexpectedly admitted the candidate")
        again = admit("dual", manifest, cand, {"dual_scoped": args.dual_scoped})
        need(again == candidate_id, "candidate changed")
        inputs.bind(cand / "heptatorch", DUAL)
        inputs.bind(cand / "runtime.hepta", again["source_sha256"])
        return json.loads(inputs.bind(cand / "build.json", build_sha))
    runner.profile = candidate_profile
    def packet(raw_root, image):
        from half_logical.native import split_frames
        wire = runner.native.load_wire(repo, runner.Inputs().bind)
        raw_root = raw_root / "packet-controls/raw"
        r, a = read(raw_root / "requests.frames.bin"), read(raw_root / "responses.frames.bin")
        requests, answers = split_frames(r, b"HTOWNR01"), split_frames(a, b"HTOWNS01")
        transport = json.loads(read(raw_root / "transport.json"))
        need(len(requests) == len(answers) == transport["frames"] and transport["executable_sha256"] == image,
             "physical control transport differs")
        found = {}
        for q, reply in zip(requests, answers, strict=True):
            if q[:8] != b"HTORCH03" or len(q) != 96 or int.from_bytes(q[8:16], "little") != 1:
                continue
            op = struct.unpack_from("<Q", q, 32)[0]
            if op not in (125, 127):
                continue
            need(op not in found, "duplicate old opcode control")
            status, rows = wire.parse(reply)
            found[op] = dict(status=status, response_opcodes=[item["opcode"] for item in rows],
                             request_sha256=sha(q), response_sha256=sha(reply))
        need(set(found) == {125, 127}, "exact packet control not found")
        return found

    baseline_packets = packet(historical, BASE)
    need(all(row["status"] == 4 and row["response_opcodes"] == [] for row in baseline_packets.values()),
         "unchanged baseline opcode refusal not observed from wire")
    if args.mode == "successor":
        retained = args.retained_audit
        need(retained is not None, "first retain the actual old candidate refusal result")
        fail_raw = read(retained / "observer/report.json")
        fail_report = json.loads(fail_raw)
        old_candidate_packets = packet(retained / "observer", DUAL)
        need(fail_report["status"] == "FAIL" and fail_report["failures"] == ["packet-controls"]
             and fail_report["actual_build_sha256"] == candidate_id["build_report_sha256"]
             and fail_report["reference_sha256"] == original_report["reference_sha256"]
             and fail_report["native_sha256"] == DUAL
             and old_candidate_packets[125]["status"] == 4 and old_candidate_packets[127]["status"] == 0,
             "missing actual historical candidate failure for this exact build")
        latest["legacy_candidate_failure_sha256"] = sha(fail_raw)
        need(str(args.torch_python) == json.loads(read(reference / "report.json"))["actual_process"]["argv"][0],
             "supplemental Torch must use exact supervised baseline interpreter")
        untouched = runner.native
        torch_spec = next(row["spec"] for row in json.loads(read(reference / "report.json"))["rows"]
                          if row["spec"]["name"].endswith("/special-bits"))
        args_run = [str(args.torch_python), "-I", "-B", "-c", ORACLE, str(repo)]
        oracle_process = subprocess.run(args_run, input=json.dumps(torch_spec).encode("utf8"), timeout=90,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                env={"PATH":"/bin:/usr/bin", "ATEN_CPU_CAPABILITY":"avx2", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"})
        need(oracle_process.stderr == b"", "supplemental Torch reference emitted stderr")
        oracle = json.loads(oracle_process.stdout)
        extension = reference / "reference-custody/reference-inputs/torch-extension.bin"
        need(oracle["extension_sha256"] == sha(read(extension)), "different supplemental Torch extension")
        expected = dict(oracle["output"])
        for k in ("logical", "storage"):
            expected[k] = bytes.fromhex(expected[k])

        def check_next(n, op, src, row, status, before, after, records):
            if op == 125:
                n.check('unassigned opcode remains rejected', status, 4)
                return
            n.check("only the explicitly assigned opcode has a successor", op, 127)
            n.check("fixed original source fixture", row["spec"], oracle["spec"])
            n.check("actual Torch original fixture bytes", n.expected(row["input"])["storage"], bytes.fromhex(oracle["input_storage"]))
            n.check("assigned logical_not opcode accepts the same original request", status, 0)
            n.check("assigned logical_not reports opcode 127", [r["opcode"] for r in records], [127])
            handle = records[0]["value"]
            n.check("assigned logical_not preserves all 84 input words", untouched.Memory(before["bytes"]).node(src),
                   untouched.Memory(after["bytes"]).node(src))
            value, node = n.tensor(handle)
            n.check("assigned logical_not result matches actual supplemental Torch", value, expected)
            owner = untouched.Memory(before["bytes"]).node(src)
            n.check("assigned logical_not result has fresh Tensor, Storage and Version",
                    [handle != src, node[27] != owner[27], node[28] != owner[28]], [True] * 3)
            n.one(41, handle)
            latest.update({"expected_torch": oracle, "oracle_program_sha256": sha(ORACLE.encode()),
                           "candidate_127_status": status, "candidate_127_checks_preserved": True})

        rebased = change(change(change(source[NATIVE], ISSUE, ISSUE_NEXT), CHECK, CHECK_NEXT), LABEL, LABEL_NEXT)
        runner.native = load("signbit.dual_successor_native", repo / NATIVE, rebased,
                             package="signbit", observe_assigned127=check_next)
        latest["reversible_candidate_native_sha256"] = sha(rebased)

    argv = sys.argv[:]
    try:
        sys.argv = [str(repo / RUNNER), "--build-directory", str(cand),
                    "--expected-build-sha256", candidate_id["build_report_sha256"],
                    "--torch-python", str(args.torch_python), "--output", str(output / "observer")]
        code = runner.main()
    finally:
        sys.argv = argv
    raw = read(output / "observer/report.json")
    report = json.loads(raw)
    results = {row["name"]:row for row in report["results"]}
    need(set(results) == set(by_name) and len(results) == 106
         and report["source_sha256"] == candidate_id["source_sha256"] and report["native_sha256"] == DUAL
         and report["full_installation_provenance_verified"] is True and report["all_inputs_unchanged"] is True
         and report["reference_sha256"] == original_report["reference_sha256"], "candidate observer roster or identity differs")
    for name, old in by_name.items():
        if name == "packet-controls":
            continue
        row = results[name]
        need(row["status"] == old["status"] == "PASS" and row["result"] == old["result"]
             and row["checks"] == old["checks"] and row["finish"]["frames"] == old["finish"]["frames"],
             "original case or control changed: " + name)
    observed_row = results["packet-controls"]
    observed_packets = packet(output / "observer", DUAL)
    need(observed_packets[125] == baseline_packets[125]
         and observed_packets[127]["request_sha256"] == baseline_packets[127]["request_sha256"]
         and observed_packets[127]["status"] == 0 and observed_packets[127]["response_opcodes"] == [127],
         "same physical packets must retain 125 refusal and separately demonstrate 127 acceptance")
    latest.update(baseline_wire_controls=baseline_packets, candidate_wire_controls=observed_packets)
    if args.mode == "retained":
        need(code == 1 and report["status"] == "FAIL" and report["failures"] == ["packet-controls"] and
             observed_row["error"]["message"] == "unassigned opcode remains rejected",
             "expected exact historical assertion failure not observed")
    else:
        need(code == 0 and report["status"] == "PASS" and report["failures"] == []
             and observed_row["result"]["accepted127_successor"] is True
             and observed_row["result"]["original_unknown125_rejected"] is True
             and observed_row["result"]["controls"] == baseline_row["result"]["controls"] == 13,
             "successor row did not preserve the retained packet scope")
        native_proof = read(output / "observer/physical-qualification.json")
        need(json.loads(native_proof)["physical_replay"] is True, "candidate physical readback did not pass")
        latest["physical_replay_sha256"] = sha(native_proof)

    need(read(historical / "report.json") == original_raw and
         all(read(repo / p, pin) == source[p] for p, pin in PINS.items()), "existing evidence changed")
    return dict(
        status="EXPECTED_RETAINED_OPCODE127_FAILURE" if args.mode == "retained" else "PASS_SIGNBIT_OPCODE127_SUCCESSOR",
        schema="heptatorch.opt-in.signbit-replay.v1", legacy_reader_unchanged=True, full_candidate_suite_qualification=False,
        candidate_image_sha256=DUAL, baseline_image_sha256=BASE, candidate_build_sha256=candidate_id["build_report_sha256"],
        original_baseline_report_sha256=sha(original_raw), native_report_sha256=sha(raw),
        baseline_packet_control=baseline_row, observed_candidate_packet_control=observed_row,
        observer_derived_sha256=sha(derived), reference_sha256=original_report["reference_sha256"],
        observer_exit_code=code, reference_reused_from_verified_baseline=True, unchanged_cases_and_extra_controls=105,
        frames=report["frames"], checks=report["checks"], extra=latest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "baseline-build", "baseline-run", "dual-build", "dual-scoped", "torch-python", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--mode", choices=("retained","successor"), required=True)
    p.add_argument("--retained-audit", type=Path)
    args = p.parse_args()
    need(sys.flags.isolated and not sys.flags.optimize and sys.dont_write_bytecode, "use python3 -I -B")
    for key in vars(args):
        if key not in {"mode","torch_python"} and getattr(args,key) is not None:
            setattr(args,key,getattr(args,key).resolve())
    args.torch_python=args.torch_python.absolute()
    args.output.mkdir()
    try:
        report = suite(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report = {"status":"FAIL_REPLAY", "type":type(error).__name__,"error":str(error)}
    json_output(args.output / "receipt.json", report)
    print(report["status"])
    return {"PASS_SIGNBIT_OPCODE127_SUCCESSOR":0, "EXPECTED_RETAINED_OPCODE127_FAILURE":2}.get(report["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
