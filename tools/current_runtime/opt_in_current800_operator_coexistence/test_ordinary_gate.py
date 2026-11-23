"""Stdlib-only parser and count checks; no archive or runtime required."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "ordinary_gate", Path(__file__).with_name("ordinary_gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class EvidenceTests(unittest.TestCase):
    def good(self):
        return {"status": "PASS", "scope": "reviewed ordinary v3 numerical fixture only",
                "metrics": {lane: dict(zip(gate.FIELDS, values, strict=True), image=gate.IMAGES[lane])
                            for lane, values in gate.LANES.items()}}

    def test_accept_expected_counts(self):
        report = self.good()
        self.assertEqual(gate.verify_summary(json.dumps(report)), report)

    def test_reject_malformed_json(self):
        for raw in ('{"x"', '{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{} {}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                gate.load_json(raw)

    def test_reject_wrong_scope(self):
        report = self.good()
        report["scope"] = "hosted"
        with self.assertRaises(ValueError):
            gate.verify_summary(json.dumps(report))

    def test_reject_wrong_lane_and_count(self):
        report = self.good()
        del report["metrics"]["add"]
        with self.assertRaises(ValueError):
            gate.verify_summary(json.dumps(report))
        for count in (170, True, 171.0):
            report = self.good()
            report["metrics"]["combined"]["ordinary_outputs"] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                gate.verify_summary(json.dumps(report))

    def test_reject_wrong_image(self):
        report = self.good()
        report["metrics"]["combined"]["image"] = gate.IMAGES["where"]
        with self.assertRaises(ValueError):
            gate.verify_summary(json.dumps(report))


if __name__ == "__main__":
    unittest.main()
