"""Separate native dtype, packet, capability, and allocation controls."""
import resource
import struct
from pathlib import Path

from cases import CASES, MASK
from common import dump, require


DTYPES = (
    ("Bool_explicit_Bool", 1, 1, 0),
    ("I64_explicit_Bool", 2, 1, 0),
    ("Bool_explicit_F32", 1, 0, 7),
    ("I64_explicit_F32", 2, 0, 7),
    ("F32_explicit_I64", 0, 2, 7),
    ("F32_explicit_Bool", 0, 1, 7),
    ("wrapped_F64_explicit_Bool", 3, 1, 7),
)


def run_dtype_controls(args, observer, owner, reference, run_one):
    rows = []
    for index, (name, source_dtype, target_dtype, expected) in enumerate(DTYPES):
        def execute(c, source_dtype=source_dtype, target_dtype=target_dtype, expected=expected):
            if source_dtype in (1, 2):
                ref = reference["rows"][0 if source_dtype == 1 else 2]
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
            c.check("documented-dtype-boundary", status, expected)
            if expected == 0:
                c.publish("result", record["value"])
                actual = c.tensor(record["value"])
                wanted = next(x for x in ref["marks"] if x["label"] == "after_sum")["tensors"]["result"]
                c.compare_tensor(wanted, actual, "actual-reference-Bool-output")
                c.check("one-byte-width", actual["element_size"], 1)
                c.check("one-byte-owner-size", actual["storage_bytes"], actual["numel"])
            for role in list(c.live)[::-1]:
                c.drop(role)
            c.finish_controls()
        rows.append(run_one(args, observer, owner, args.output / ("dtype-%02d" % index), name, execute))
    require(len(rows) == 7 and [x["name"] for x in rows] == [x[0] for x in DTYPES], "all-seven-dtype-controls")
    return rows


def run_entry_controls(args, observer, owner, run_one):
    def execute(c):
        c.setup(CASES[0])
        token = c.live["input"]
        controls = [
            ("invalid_public_capability", (115, 1, MASK, 0, 0, 1, 0, 0), 1),
            ("invalid_keepdim", (115, token, MASK, 0, 2, 1, 0, 0), 7),
            ("unsupported_dtype_code", (115, token, MASK, 0, 0, 255, 0, 0), 7),
            ("reserved_word_six", (115, token, MASK, 0, 0, 1, 1, 0), 7),
            ("reserved_word_seven", (115, token, MASK, 0, 0, 1, 0, 1), 7),
            ("truncated_axis_payload", (115, token, 2, c.n.w.Payload(struct.pack("<q", 0)), 0, 1, 0, 0), 2),
            ("None_nonzero_payload", (115, token, MASK, c.n.w.Payload(struct.pack("<q", 0)), 0, 1, 0, 0), 7),
            ("empty_axes_nonzero_payload", (115, token, 0, c.n.w.Payload(struct.pack("<q", 0)), 0, 1, 0, 0), 7),
            ("axis_payload_before_payload_region", (115, token, 1, 32, 0, 1, 0, 0), 2),
        ]
        for name, command, expected in controls:
            status, record = c.issue(command, name)
            c.check(name + "/cause", status, expected)
            c.check(name + "/no-result", record, None)
        source = c.live["input"]
        c.drop("input")
        status, record = c.issue((115, source, MASK, 0, 0, 1, 0, 0), "stale-public-capability")
        c.check("stale-capability-cause", status, 1)
        c.check("stale-capability-no-result", record, None)
        c.drop("backing")
        c.finish_controls()
    return [run_one(args, observer, owner, args.output / "entry-controls",
                    "Bool-command115-ten-entry-and-capability-controls", execute)]


def run_allocation_control(args, observer, owner, run_one):
    def execute(c):
        # A legal zero-stride input has 262144 logical elements but four bytes
        # of backing Storage. Reducing its inner axis requires a fresh 65536-B
        # Bool output. Numeric output is not an expected value in this resource
        # control; the separate fixed-Torch cases test numeric semantics.
        c.live["backing"] = c.n.create(1, [4], b"\0\1\0\0", requires=0)
        c.live["input"] = c.view(c.live["backing"], [65536, 4], [0, 1], 0)
        initial, _, _ = c.n.capture_owner("allocation-initial")
        pool = initial["header"]["pool"]
        fillers = []
        for _ in range(24):
            state, _, _ = c.n.capture_owner("allocation-fill-inspection")
            free = sum(n for _, n in state["free_extents"])
            if 16 * 1024 <= free <= 32 * 1024:
                break
            require(free > 32 * 1024, "allocation-setup-free-space", free=free)
            amount = min(262144, (free - 32 * 1024) & ~15)
            require(amount > 0, "allocation-positive-filler")
            fillers.append(c.n.create(1, [amount], b"\0" * amount, requires=0))
            after, _, _ = c.n.capture_owner("allocation-filled")
            c.check("filler-stays-in-original-pool", after["header"]["pool"], pool)
        state, _, _ = c.n.capture_owner("allocation-ready")
        free = sum(n for _, n in state["free_extents"])
        require(16 * 1024 <= free <= 32 * 1024, "output-cannot-fit-current-free-pool", free=free)
        # The result descriptor fits an already present page; reusable T/S/V
        # and temporary Hold slots exist. Thus the tested new allocation is
        # the result Storage, not a new descriptor page or a full handle table.
        next_node = state["header"]["next"]
        require(next_node < 240 and any(p["page"] == (next_node >> 8) for p in state["pages"]),
                "result-descriptor-page-already-present")
        available = {kind: sum(row[1] == 0 and row[0] < 0xffffffff for row in state["tables"][str(kind)])
                     for kind in (1, 2, 3, 6)}
        require(min(available.values()) >= 2, "result-handle-slots-already-present", available=available)
        command = (115, c.live["input"], 1, c.n.w.Payload(struct.pack("<q", 1)), 0, 1, 0, 0)
        pid = c.n.s.pid
        old_limit = resource.prlimit(pid, resource.RLIMIT_AS)
        status_raw = Path("/proc/%d/status" % pid).read_bytes()
        (c.directory / "before-limit.status").write_bytes(status_raw)
        vm = int(next(x.split()[1] for x in status_raw.decode().splitlines() if x.startswith("VmSize:"))) * 1024
        page = lambda n: (n + 4095) & ~4095
        # Allow the ordinary transaction's same-size owner copy, transport
        # scratch and RNG observation; deny the required pool growth.
        soft = vm + page(state["length"] + 16) + page(5056 + 16) + 65536
        require(old_limit[1] == resource.RLIM_INFINITY or soft < old_limit[1], "test-child-limit-within-hard-limit")
        setup = dict(source_logical_elements=262144, output_requested_bytes=65536,
                     free_payload_bytes=free, original_pool=pool, next_node=next_node,
                     available_slots=available, filler_count=len(fillers), current_vm=vm,
                     old_limit=list(old_limit), applied_soft_limit=soft)
        dump(c.directory / "allocation-setup.json", setup)
        resource.prlimit(pid, resource.RLIMIT_AS, (soft, old_limit[1]))
        try:
            c.check("same-capacity-query-under-limit", c.n.one(94, c.live["input"])["value"], 0)
            maps_before = Path("/proc/%d/maps" % pid).read_bytes()
            result_status, record = c.issue(command, "Bool-result-Storage-allocation-failure")
            maps_after = Path("/proc/%d/maps" % pid).read_bytes()
            (c.directory / "failure.maps-before").write_bytes(maps_before)
            (c.directory / "failure.maps-after").write_bytes(maps_after)
            c.check("output-growth-allocation-cause", result_status, 3)
            c.check("output-growth-publishes-nothing", record, None)
            c.check("all-child-mappings-restored", maps_after == maps_before, True)
        finally:
            resource.prlimit(pid, resource.RLIMIT_AS, old_limit)
            c.check("child-original-resource-limit-restored", list(resource.prlimit(pid, resource.RLIMIT_AS)), list(old_limit))
        status, record = c.issue(command, "same-Bool-operation-after-resource-recovery")
        c.check("same-command-recovers", status, 0)
        c.live["result"] = record["value"]
        observed = c.n.tensor(record["value"])
        c.check("recovered-output-dtype", observed["dtype"], "torch.bool")
        c.check("recovered-output-width", observed["element_size"], 1)
        c.check("recovered-output-requested-bytes", observed["storage_bytes"], 65536)
        c.check("recovered-output-shape", observed["shape"], [65536])
        c.check("recovered-output-no-grad", observed["requires_grad"], False)
        after, _, _ = c.n.capture_owner("after-resource-recovery")
        c.check("real-pool-growth-after-limit-restored", after["header"]["pool"] > pool, True)
        for token in reversed(fillers):
            c.n.one(41, token)
        for role in list(c.live)[::-1]:
            c.drop(role)
        c.finish_controls()
    return [run_one(args, observer, owner, args.output / "allocation-control",
                    "Bool-result-Storage-OOM-rollback-and-recovery", execute)]
