"""Bind runtime tests to the one published source and its actual cold build.

Runtime tests keep their independent expected values. This module only checks
source partitions, native tool identities, objects, link input and process EOFs.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import types


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "tools/current_runtime/source/manifest.json"
PROFILE_BYTES = PROFILE_PATH.read_bytes()
PROFILE = json.loads(PROFILE_BYTES)
SOURCE = PROFILE["runtime"]["sha256"]
IMAGE = PROFILE["expected_native_sha256"]
FUNCTIONS = PROFILE["runtime"]["functions"]
UNITS = len(PROFILE["units"])
COMPILER = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER = "aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def admit_build(directory, bind, expected=None):
    """Validate a complete build against the current checked-in manifest."""
    directory = Path(directory).resolve()
    bind(Path(__file__).resolve())
    require(bind(PROFILE_PATH) == PROFILE_BYTES, "current source manifest changed")
    builder_path = ROOT / "tools/current_runtime/build.py"
    builder_raw = bind(builder_path)
    spec = importlib.util.spec_from_file_location("current_runtime_source_checks", builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source, source_units, summary = builder.check_sources(PROFILE, PROFILE_PATH.parent, bind)
    raw = bind(directory / "build.json", expected)
    built = json.loads(raw)
    require(built["schema"] == "heptatorch.complete-runtime-build.v1"
            and built["status"] == "PASS_NATIVE_BUILD_ONLY", "a successful cold build is required")
    require(built["manifest_sha256"] == sha(PROFILE_BYTES), "build used another source manifest")
    require(all(built[key] == value for key, value in summary.items()), "complete source partition changed")
    require(built["source_sha256"] == SOURCE and built["native_sha256"] == IMAGE
            and built["compiler_sha256"] == COMPILER and built["linker_sha256"] == LINKER,
            "source, image or native tool identity changed")
    require(built["objects_reused"] is False and built["all_inputs_unchanged"] is True
            and built["runtime_qualified"] is False and built["full_pytorch_rewrite_complete"] is False
            and built["historical_runtime_executed"] is False
            and built["result_executable_executed"] is False and built["torch_executed"] is False
            and built["link_exit"] == 0, "cold-build provenance or scope changed")
    require(built["limits"] == builder.LIMITS and built["stack_inspector_sha256"] == builder.INSPECTOR,
            "native capacity or stack inspector changed")
    inspector_path = ROOT / "tools/stack_budget/inspect_stack.py"
    inspector_raw = bind(inspector_path, builder.INSPECTOR)
    inspector = types.ModuleType("current_runtime_fixed_stack_inspector")
    exec(compile(inspector_raw, "<fixed-stack-inspector>", "exec"), inspector.__dict__)
    require(any(Path(path).name == "build.py" and row["sha256"] == sha(builder_raw)
                for path, row in built["inputs"].items()), "build did not use the current builder")
    require(any(Path(path).name == "inspect_stack.py" and row["sha256"] == builder.INSPECTOR
                and row["bytes"] == len(inspector_raw)
                for path, row in built["inputs"].items()),
            "build did not bind the fixed stack inspector as an input")
    require(bind(directory / "runtime.hepta", SOURCE) == source, "built source differs from current source")
    require(bind(directory / "manifest.json") == PROFILE_BYTES, "built manifest differs from current source")
    for name, pin in (("stack-module-compiler", COMPILER), ("stack-linker-session", LINKER)):
        bind(directory / "toolchain" / name, pin)
    require(len(built["units"]) == UNITS and len(built["processes"]) == UNITS + 1,
            "incomplete compiler/linker process set")
    require([row["name"] for row in built["processes"]] ==
            [row["name"] for row in PROFILE["units"]] + ["link"], "unexpected compilation order")
    objects, views = [], []
    for (entry, unit_source, parsed), unit, process in zip(source_units, built["units"], built["processes"]):
        name = entry["name"]
        require(unit["name"] == name and unit["source_sha256"] == entry["sha256"]
                and unit["compiler_exit"] == 0 and unit["compiler_stderr_bytes"] == 0,
                "unit source or compiler result changed: " + name)
        require(bind(directory / (name + ".hepta"), entry["sha256"]) == unit_source,
                "compiled source differs from current unit: " + name)
        obj = bind(directory / (name + ".o"), unit["object_sha256"])
        require(len(obj) == unit["object_bytes"] and obj[:6] == b"\x7fELF\x02\x01"
                and struct.unpack_from("<2H", obj, 16) == (1, 62), "invalid native object: " + name)
        require(len(obj) <= builder.LIMITS["object_bytes"]
                and unit["object_sha256"] == entry["expected_object_sha256"]
                and all(unit[key] == parsed[key] for key in ("functions", "blocks", "instructions")),
                "object identity, limits or source counts changed: " + name)
        require(process["input_sha256"] == entry["sha256"] and process["input_bytes"] == len(unit_source)
                and process["stdout_sha256"] == sha(obj) and process["stdout_bytes"] == len(obj),
                "compiler input/output mismatch: " + name)
        require(bind(directory / (name + ".stderr")) == b"", "compiler stderr is not empty: " + name)
        objects.append(obj)
        views.append(inspector.inspect(unit_source, obj))
    budget = inspector.longest(views)
    require(budget["maximum_path_bytes"] <= builder.LIMITS["stack_path_bytes"]
            and built["stack"] == {k: v for k, v in budget.items() if k != "per_function_path_bytes"},
            "actual source/object native stack demand differs from build receipt")
    for process in built["processes"]:
        require(process["started"] is True and process["exit_code"] == 0
                and process["stdout_EOF"] is True and process["stderr_EOF"] is True
                and process["stderr_bytes"] == 0 and process["stderr_sha256"] == sha(b"")
                and not process.get("timed_out") and not process.get("forced_cleanup")
                and not process.get("execution_error") and not process.get("error")
                and not process.get("cleanup_errors") and not process.get("running_descendants_after_cleanup"),
                "incomplete native tool execution: " + process["name"])
    request = b"HIASTKS1" + struct.pack("<3Q", UNITS, 4, 0) + b"step"
    request += b"".join(struct.pack("<Q", len(obj)) + obj for obj in objects)
    require(bind(directory / "link.request.bin") == request, "link input differs from newly compiled objects")
    image = bind(directory / "heptatorch", IMAGE)
    require(built["native_basename"] == "heptatorch" and len(image) == built["native_bytes"]
            and image[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<2H", image, 16) == (2, 62),
            "invalid current executable")
    link = built["processes"][-1]
    require(link["input_sha256"] == sha(request) and link["input_bytes"] == len(request)
            and link["stdout_sha256"] == IMAGE and link["stdout_bytes"] == len(image)
            and bind(directory / "link.stderr") == b"", "native linker input/output mismatch")
    return built
