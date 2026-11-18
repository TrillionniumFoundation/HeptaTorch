"""Independent current dtype boundaries and malformed Sum115 entry controls.

Accepted outputs are compared with the two fixed, actually observed Torch
reports. This module computes no reduction or expected tensor values.
"""
import struct

from cases import CASES, MASK
from native_support import require


DTYPES = (
    ("Bool_explicit_Bool", 1, 1, 0),
    ("I64_explicit_Bool", 2, 1, 0),
    ("Bool_explicit_F32", 1, 0, 0),
    ("I64_explicit_F32", 2, 0, 0),
    ("F32_explicit_I64", 0, 2, 7),
    ("F32_explicit_Bool", 0, 1, 7),
    ("wrapped_F64_explicit_Bool", 3, 1, 7),
)


def run_dtype_controls(args, observer, owner, reference, bool_reference, run_one):
    rows = []
    for index, (name, source_dtype, target_dtype, expected) in enumerate(DTYPES):
        def execute(c, source_dtype=source_dtype, target_dtype=target_dtype, expected=expected):
            if source_dtype in (1, 2):
                oracle = bool_reference if target_dtype == 1 else reference
                ref = oracle["rows"][0 if source_dtype == 1 else 2]
                c.setup(ref["spec"])
                command = c.sum_command(ref["spec"], target_dtype)
            elif source_dtype == 0:
                c.live["input"] = c.n.create(0, [2], struct.pack("<2f", 1.0, -1.0), requires=0)
                command = (115, c.live["input"], MASK, 0, 0, target_dtype, 0, 0)
            else:
                c.live["leaf"] = c.n.create(0, [2], struct.pack("<2f", 1.0, 2.0), requires=1)
                c.live["graph"] = c.n.one(114, c.live["leaf"], 0x3ff0000000000000)["value"]
                c.live["input"] = c.n.one(107, c.live["graph"], 1)["value"]
                memory = c.n.memory()
                c.check("genuine-original-Saved-wrapped-F64", memory.node(memory.public_node(c.live["input"]))[72], 3)
                command = (115, c.live["input"], MASK, 0, 0, target_dtype, 0, 0)
            status, record = c.issue(command, "dtype-control")
            c.check("documented-current-dtype-boundary", status, expected)
            if expected == 0:
                c.publish("result", record["value"])
                actual = c.tensor(record["value"])
                wanted = next(x for x in ref["marks"] if x["label"] == "after_sum")["tensors"]["result"]
                c.compare_tensor(wanted, actual, "actual-fixed-reference-output")
                width = 1 if target_dtype == 1 else 4
                c.check("exact-result-element-width", actual["element_size"], width)
                c.check("exact-requested-Storage-bytes", actual["storage_bytes"], width * actual["numel"])
                c.check("result-remains-nondifferentiable-at-creation", actual["requires_grad"], False)
            else:
                c.check("refusal-publishes-nothing", record, None)
            for role in list(c.live)[::-1]:
                c.drop(role)
            c.finish_controls()
        rows.append(run_one(args, observer, owner, args.output / ("dtype-%02d" % index), name, execute))
    require(len(rows) == 7 and [x["name"] for x in rows] == [x[0] for x in DTYPES], "all-seven-current-dtype-controls")
    return rows


def run_entry_controls(args, observer, owner, run_one):
    def execute(c):
        c.setup(CASES[0])
        token = c.live["input"]
        controls = [
            ("invalid_public_capability", (115, 1, MASK, 0, 0, 0, 0, 0), 1),
            ("invalid_keepdim", (115, token, MASK, 0, 2, 0, 0, 0), 7),
            ("unsupported_dtype_code", (115, token, MASK, 0, 0, 255, 0, 0), 7),
            ("reserved_word_six", (115, token, MASK, 0, 0, 0, 1, 0), 7),
            ("reserved_word_seven", (115, token, MASK, 0, 0, 0, 0, 1), 7),
            ("truncated_axis_payload", (115, token, 2, c.n.w.Payload(struct.pack("<q", 0)), 0, 0, 0, 0), 2),
            ("None_nonzero_payload", (115, token, MASK, c.n.w.Payload(struct.pack("<q", 0)), 0, 0, 0, 0), 7),
            ("empty_axes_nonzero_payload", (115, token, 0, c.n.w.Payload(struct.pack("<q", 0)), 0, 0, 0, 0), 7),
            ("axis_payload_before_payload_region", (115, token, 1, 32, 0, 0, 0, 0), 2),
        ]
        for name, command, expected in controls:
            status, record = c.issue(command, name)
            c.check(name + "/cause", status, expected)
            c.check(name + "/no-result", record, None)
        source = c.live["input"]
        c.drop("input")
        status, record = c.issue((115, source, MASK, 0, 0, 0, 0, 0), "stale-public-capability")
        c.check("stale-capability-cause", status, 1)
        c.check("stale-capability-no-result", record, None)
        c.check("all-ten-entry-controls-observed", len(c.operations), 10)
        c.drop("backing")
        c.finish_controls()
    return [run_one(args, observer, owner, args.output / "entry-controls",
                    "integral-F32-command115-ten-entry-and-capability-controls", execute)]
