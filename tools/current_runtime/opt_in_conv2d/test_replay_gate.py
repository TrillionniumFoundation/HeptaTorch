"""Failure and non-portable provenance controls for the replay gate."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import replay_gate as gate


class ReplayGateTests(unittest.TestCase):
    def setUp(self):
        self.expected = json.loads((Path(__file__).parent / "qualification.json").read_bytes())

    def test_only_install_report_digest_is_machine_specific(self):
        actual = copy.deepcopy(self.expected)
        actual["reference"]["install_report_sha256"] = "0" * 64
        gate.compare_report(actual, self.expected)
        actual["reference"]["installed_component_sha256"]["torch/__init__.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differential report differs"):
            gate.compare_report(actual, self.expected)

    def test_rejects_changed_fixture_result_or_missing_negative(self):
        for change in (
            lambda report: report["cases"][0].update(forward_sha256="0" * 64),
            lambda report: report["channel_refusals"].pop(),
            lambda report: report["no_grad_chain"].update(frames=0),
        ):
            actual = copy.deepcopy(self.expected)
            change(actual)
            with self.assertRaisesRegex(ValueError, "differential report differs"):
                gate.compare_report(actual, self.expected)

    def test_lock_rejects_mutated_receipt(self):
        with TemporaryDirectory() as tmp:
            changed = Path(tmp) / "changed.json"
            changed.write_text(json.dumps(self.expected))
            with self.assertRaisesRegex(ValueError, "conv2d differential receipt SHA-256 mismatch"):
                gate.read_locked(changed, gate.CONV_REPORT, "conv2d differential receipt")


if __name__ == "__main__":
    unittest.main()
