#!/usr/bin/env python3
"""Admission-framework fixtures, not native or PyTorch qualification.

The test-only verifier checks a finite, exactly specified byte transcript. It is
never installed in the production registry and does not claim any PyTorch API.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("contract_acceptance_test", ROOT / "tools/check_acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)
C = acceptance.contracts
TARGET = ("native", "fixture/exact-byte-transcript-v1")


def fixture_verifier(snapshot):
    """Check this fixture's exact transcript; no runtime is executed or qualified."""
    C.need((snapshot.kind, snapshot.name) == TARGET, "fixture target mismatch")
    expected = b"input: 2,3\noutput: 5\n"
    C.need(snapshot.files["reference"] == expected, "fixture reference changed")
    C.need(snapshot.files["execution"] == expected, "fixture actual output differs")
    C.need(snapshot.files["readback"] == hashlib.sha256(expected).hexdigest().encode(),
           "fixture physical readback differs")
    build = C.json_object(snapshot.files["build"])
    C.need(build == {"manifest_sha256": snapshot.runtime_manifest_sha256,
                     "native_sha256": snapshot.native_sha256}, "fixture build binding differs")
    return C.Verdict(True, True, frozenset(snapshot.files))


def partial_verifier(snapshot):
    return C.Verdict(True, False, frozenset(snapshot.files))


def unchecked_verifier(snapshot):
    return C.Verdict(True, True, frozenset({"execution"}))


def boolean_verifier(snapshot):
    return True


class ContractAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hia-contract-admission-")
        self.root = Path(self.temporary.name)
        self.verifier = C.Verifier("fixture-only-v1", C.BASE_ROLES, Path(__file__), fixture_verifier)
        self.registry = {TARGET: self.verifier}
        self.source = b"fixture-only-source; no PyTorch implementation\n"
        self.unit = b"fixture-only-unit\n"
        self.image = b"fixture-only-image; not a HIA executable\n"
        self.write("tools/current_runtime/source/runtime.hepta", self.source)
        self.write("tools/current_runtime/source/units/fixture.hepta", self.unit)
        self.manifest = {"schema": "heptatorch.complete-runtime-source.v1",
                         "runtime": {"path": "runtime.hepta", "sha256": C.digest(self.source)},
                         "units": [{"name": "fixture", "path": "units/fixture.hepta",
                                    "sha256": C.digest(self.unit), "expected_object_sha256": "1" * 64}],
                         "expected_native_sha256": C.digest(self.image)}
        self.write_json(C.MANIFEST, self.manifest)
        self.envelope = {"schema": "heptatorch.full-contract-evidence.v1",
                         "upstream_commit": C.UPSTREAM,
                         "contract": {"kind": TARGET[0], "name": TARGET[1]},
                         "verifier_id": self.verifier.identifier,
                         "verifier_source_sha256": C.digest(Path(__file__).read_bytes()),
                         "admission_engine_sha256": C.digest(Path(C.__file__).read_bytes()),
                         "runtime_manifest_sha256": C.digest((self.root / C.MANIFEST).read_bytes()),
                         "files": {}}
        transcript = b"input: 2,3\noutput: 5\n"
        values = {"native_image": self.image,
                  "build": json.dumps({"manifest_sha256": self.envelope["runtime_manifest_sha256"],
                                       "native_sha256": C.digest(self.image)}).encode(),
                  "execution": transcript, "reference": transcript,
                  "readback": C.digest(transcript).encode()}
        for role, raw in values.items():
            path = "evidence/" + role + ".bin"
            self.write(path, raw)
            self.envelope["files"][role] = self.pin(path)
        self.repin_envelope()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, raw):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def write_json(self, relative, value):
        self.write(relative, json.dumps(value, sort_keys=True).encode())

    def pin(self, relative):
        raw = (self.root / relative).read_bytes()
        return {"path": relative, "bytes": len(raw), "sha256": C.digest(raw)}

    def repin_envelope(self):
        self.write_json("evidence/contract.json", self.envelope)
        self.claim = self.pin("evidence/contract.json")

    def admit(self):
        return C.admit(self.root, *TARGET, self.claim, self.registry)

    def test_registered_fixture_admits_exact_bytes(self):
        result = self.admit()
        self.assertEqual(result["name"], TARGET[1])
        self.assertEqual(result["checked_roles"], sorted(C.BASE_ROLES))
        self.assertEqual(result["native_sha256"], C.digest(self.image))

    def test_fixture_record_count_comes_from_verifier(self):
        rows = [{"schema": TARGET[1], "status": "qualified", "qualification": self.claim}]
        qualified, errors = acceptance.qualified_records(
            self.root, rows, "native", "schema", {"unimplemented"}, "qualified", self.registry)
        self.assertEqual(set(qualified), {TARGET[1]})
        self.assertEqual(errors, [])
        self.assertEqual(len(C.VERIFIERS), 0)

    def test_production_registry_cannot_admit_fixture(self):
        with self.assertRaisesRegex(ValueError, "no reviewed"):
            C.admit(self.root, *TARGET, self.claim, C.VERIFIERS)

    def test_status_cannot_replace_verifier(self):
        self.envelope["status"] = "PASS"
        self.repin_envelope()
        with self.assertRaisesRegex(ValueError, "exact full-contract"):
            self.admit()

    def test_wrong_contract_or_upstream_refused(self):
        for key, value in (("contract", {"kind": "native", "name": "different"}),
                           ("upstream_commit", "0" * 40)):
            with self.subTest(key=key):
                old = self.envelope[key]
                self.envelope[key] = value
                self.repin_envelope()
                with self.assertRaises(ValueError):
                    self.admit()
                self.envelope[key] = old
        self.repin_envelope()

    def test_wrong_verifier_revision_refused(self):
        self.envelope["verifier_source_sha256"] = "0" * 64
        self.repin_envelope()
        with self.assertRaisesRegex(ValueError, "verifier revision"):
            self.admit()

    def test_wrong_admission_engine_refused(self):
        self.envelope["admission_engine_sha256"] = "0" * 64
        self.repin_envelope()
        with self.assertRaisesRegex(ValueError, "admission engine revision"):
            self.admit()

    def test_callback_source_must_match_registration(self):
        self.registry[TARGET] = self.verifier._replace(source_file=Path(C.__file__))
        with self.assertRaisesRegex(ValueError, "does not own the callback"):
            self.admit()

    def test_boolean_byte_count_refused(self):
        self.claim["bytes"] = True
        with self.assertRaisesRegex(ValueError, "invalid byte/hash"):
            self.admit()

    def test_missing_or_unexpected_roles_refused(self):
        self.envelope["files"].pop("reference")
        self.repin_envelope()
        with self.assertRaisesRegex(ValueError, "evidence roles"):
            self.admit()

    def test_changed_source_or_unit_refused(self):
        for relative in ("runtime.hepta", "units/fixture.hepta"):
            with self.subTest(relative=relative):
                path = self.root / "tools/current_runtime/source" / relative
                old = path.read_bytes()
                path.write_bytes(old + b"changed")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    self.admit()
                path.write_bytes(old)

    def test_stale_manifest_refused(self):
        self.manifest["note"] = "changed"
        self.write_json(C.MANIFEST, self.manifest)
        with self.assertRaisesRegex(ValueError, "current runtime source manifest"):
            self.admit()

    def test_missing_native_image_identity_refused(self):
        path = self.envelope["files"]["native_image"]["path"]
        self.write(path, b"another image")
        self.envelope["files"]["native_image"] = self.pin(path)
        self.repin_envelope()
        with self.assertRaisesRegex(ValueError, "native image"):
            self.admit()

    def test_rehashed_wrong_actual_output_still_refused(self):
        path = self.envelope["files"]["execution"]["path"]
        self.write(path, b"input: 2,3\noutput: 6\n")
        self.envelope["files"]["execution"] = self.pin(path)
        self.repin_envelope()
        with self.assertRaisesRegex(ValueError, "actual output differs"):
            self.admit()

    def test_altered_file_without_rehash_refused(self):
        self.write(self.envelope["files"]["execution"]["path"], b"modified")
        with self.assertRaisesRegex(ValueError, "bound evidence bytes changed"):
            self.admit()

    def test_shared_file_roles_refused(self):
        self.envelope["files"]["reference"] = dict(self.envelope["files"]["execution"])
        self.repin_envelope()
        with self.assertRaisesRegex(ValueError, "distinct files"):
            self.admit()

    def test_path_escape_and_symlink_refused(self):
        self.envelope["files"]["reference"]["path"] = "../outside"
        self.repin_envelope()
        with self.assertRaisesRegex(ValueError, "path escapes"):
            self.admit()
        self.envelope["files"]["reference"] = self.pin("evidence/reference.bin")
        self.repin_envelope()
        path = self.root / "evidence/reference.bin"
        path.unlink()
        path.symlink_to(self.root / "evidence/execution.bin")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.admit()

    def test_hardlinked_evidence_roles_refused(self):
        reference = self.root / "evidence/reference.bin"
        reference.unlink()
        os.link(self.root / "evidence/execution.bin", reference)
        with self.assertRaisesRegex(ValueError, "inode alias"):
            self.admit()

    def test_parent_symlink_swap_during_descriptor_read_refused(self):
        target = (self.root / "evidence/execution.bin").stat()
        original_read = os.read
        swapped = False

        def swap_parent(descriptor, size):
            nonlocal swapped
            raw = original_read(descriptor, size)
            opened = os.fstat(descriptor)
            if not swapped and (opened.st_dev, opened.st_ino) == (target.st_dev, target.st_ino):
                (self.root / "evidence").rename(self.root / "original-evidence")
                (self.root / "evidence").symlink_to(self.root / "original-evidence", target_is_directory=True)
                swapped = True
            return raw

        with mock.patch.object(C.os, "read", side_effect=swap_parent):
            with self.assertRaisesRegex(ValueError, "directory name changed during descriptor read"):
                self.admit()
        self.assertTrue(swapped)

    def test_duplicate_json_keys_refused(self):
        raw = (self.root / self.claim["path"]).read_bytes()
        raw = b'{"schema":"untrusted",' + raw[1:]
        self.write(self.claim["path"], raw)
        self.claim = self.pin(self.claim["path"])
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.admit()

    def test_partial_unchecked_or_boolean_verdict_refused(self):
        for callback in (partial_verifier, unchecked_verifier, boolean_verifier):
            with self.subTest(callback=callback.__name__):
                self.registry[TARGET] = self.verifier._replace(verify=callback)
                with self.assertRaisesRegex(ValueError, "complete contract predicates"):
                    self.admit()

    def test_inflight_source_mutation_refused(self):
        def mutate(snapshot):
            verdict = fixture_verifier(snapshot)
            self.write("tools/current_runtime/source/runtime.hepta", self.source + b"mutated")
            return verdict
        self.registry[TARGET] = self.verifier._replace(verify=mutate)
        with self.assertRaisesRegex(ValueError, "changed during"):
            self.admit()

    def test_immutable_snapshot(self):
        def attempt_mutation(snapshot):
            snapshot.files["execution"] = b"replaced"
        self.registry[TARGET] = self.verifier._replace(verify=attempt_mutation)
        rows = [{"schema": TARGET[1], "status": "qualified", "qualification": self.claim}]
        qualified, errors = acceptance.qualified_records(
            self.root, rows, "native", "schema", {"unimplemented"}, "qualified", self.registry)
        self.assertEqual(qualified, {})
        self.assertIn("TypeError", errors[0]["error"])

    def test_claim_requires_explicit_qualified_status(self):
        rows = [{"schema": TARGET[1], "status": "unimplemented", "qualification": self.claim}]
        qualified, errors = acceptance.qualified_records(
            self.root, rows, "native", "schema", {"unimplemented"}, "qualified", self.registry)
        self.assertEqual(qualified, {})
        self.assertIn("must not carry", errors[0]["error"])

    def test_current_real_target_remains_hold(self):
        report = acceptance.check(ROOT)
        self.assertEqual(report["accounting_status"], "PASS")
        self.assertEqual(report["whole_project_status"], "HOLD")
        self.assertEqual(report["native_schema_inventory_count"], 2590)
        self.assertEqual(report["derivative_declaration_inventory_count"], 688)
        self.assertEqual(report["full_upstream_schema_parity_count"], 0)
        self.assertEqual(report["full_derivative_declaration_parity_count"], 0)
        self.assertEqual(report["open_full_subsystem_gates"], 17)
        child = subprocess.run([sys.executable, "-I", "-S", "-B",
                                str(ROOT / "tools/check_acceptance.py"), "--require-complete"],
                               capture_output=True, check=False)
        self.assertEqual(child.returncode, 2)
        self.assertEqual(json.loads(child.stdout)["whole_pytorch_rewrite_complete"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
