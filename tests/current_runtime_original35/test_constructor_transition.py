"""Reversible admission tests over exact checked-in owner800 source and profile."""
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binding import FUNCTION, Inputs, ROOT, sha
from constructor_transition import EDITS, HELPERS, OWNER800, checked_transition, observe_frozen_domain
from observer import CONSTRUCTORS


class ConstructorTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (ROOT / "tools/current_runtime/source/runtime.hepta").read_bytes()
        if sha(raw) != OWNER800[0]:
            raise AssertionError("tests require the exact reviewed owner800 source")
        cls.bodies = {name.decode(): body for body, name in FUNCTION.findall(raw)}
        cls.expected = json.loads(
            (ROOT / "tests/tensor_to_frozen35_accounting/profile.json").read_bytes()
        )["function_pins"]
        cls.build = dict(source_sha256=OWNER800[0], native_sha256=OWNER800[1])

    def accept(self, build=None, bodies=None):
        return checked_transition(
            self.build if build is None else build,
            self.bodies if bodies is None else bodies,
            self.expected,
            CONSTRUCTORS,
        )

    def test_exact_source_and_complete_inverse(self):
        pins, receipt = self.accept()
        self.assertEqual(set(pins), set(CONSTRUCTORS))
        self.assertEqual(set(receipt["constructor_reverse_edits"]), set(EDITS))
        self.assertEqual(receipt["observed_helper_pins"], HELPERS)
        for name, item in receipt["constructor_reverse_edits"].items():
            self.assertEqual(item["inverse_sha256"], self.expected[name])
            self.assertTrue(item["exact_whole_body_inverse"])
        self.assertFalse(receipt["previously_qualified_runtime_result_transferred"])

    def test_no_transfer_to_another_exact_image_or_source(self):
        for identity_key in ("source_sha256", "native_sha256"):
            with self.subTest(identity_key=identity_key), self.assertRaisesRegex(
                AssertionError, "constructor or layout contract changed: linear"
            ):
                self.accept(build={**self.build, identity_key: "0" * 64})

    def test_reject_unknown_changes_inside_and_outside_splices(self):
        for name in CONSTRUCTORS:
            with self.subTest(name=name), self.assertRaisesRegex(
                AssertionError, "constructor or layout contract changed: " + name
            ):
                self.accept(bodies={**self.bodies, name: self.bodies[name] + b"\n"})
        for name, splice in EDITS.items():
            valid = self.bodies[name]
            before, after = splice[-1]
            self.assertIn(after, valid)
            with self.subTest(splice=name), self.assertRaisesRegex(
                AssertionError, "constructor or layout contract changed: " + name
            ):
                self.accept(bodies={**self.bodies, name: valid.replace(after, after + before, 1)})

    def test_reject_unknown_helper_body_and_missing_helper(self):
        for name in HELPERS:
            for edited in (self.bodies[name] + b"\n", None):
                with self.subTest(name=name, missing=edited is None), self.assertRaisesRegex(
                    AssertionError, "changed constructor helper: " + name
                ):
                    changed = self.bodies.copy()
                    if edited is None:
                        changed.pop(name)
                    else:
                        changed[name] = edited
                    self.accept(bodies=changed)

    def test_frozen_workload_guard_rejects_new_normalization_or_missing_linear(self):
        with TemporaryDirectory() as temp:
            journals = Path(temp)
            rows = []
            for session in range(35):
                path = journals / str(session)
                path.mkdir()
                count = 6344 - 34 if session == 0 else 1
                first = dict(commands=[[15]] * 48 if session == 0 else [])
                (path / "events.json").write_text(json.dumps([first] + [dict(commands=[])] * (count - 1)))
                rows.append(dict(private_node_observer=dict(
                    snapshot_directory=str(path), frames_checked=count)))

            check = observe_frozen_domain(rows, journals, Inputs())
            self.assertEqual(check, dict(journals=35, frames=6344, linear_commands=48,
                layernorm_commands=0, batchnorm_eval_commands=0,
                normalization_allocation_qualified=False))
            last = journals / "34" / "events.json"
            for opcode in (15, 26, 28):
                last.write_text(json.dumps([dict(commands=[[opcode]])]))
                with self.subTest(opcode=opcode), self.assertRaisesRegex(
                    AssertionError, "owner800 reviewed original35 command scope changed"
                ):
                    observe_frozen_domain(rows, journals, Inputs())


if __name__ == "__main__":
    unittest.main()
