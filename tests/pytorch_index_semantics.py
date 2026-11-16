"""Read-only PyTorch 2.15 boundary probe; run after the locked main oracle.

Usage: python pytorch_semantics_probe.py OUTPUT_JSON [ORACLE_LOCK_JSON]
No HeptaTorch implementation is imported or used as a reference.
"""
import json
import sys
from pathlib import Path

import torch


def describe(value):
    if value is None:
        return None
    return {"shape": list(value.shape), "values": value.detach().reshape(-1).tolist()}


def exception(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def make(shape, requires, offset):
    count = 1
    for dimension in shape:
        count *= dimension
    value = (torch.arange(count, dtype=torch.float32) + offset).reshape(shape)
    return value.requires_grad_(requires)


def run_case(name, op, self_shape, source_shape, index_shape, index_values,
             dim=0, source_requires=True, alpha=0.25):
    a = make(self_shape, True, 1)
    b = make(source_shape, source_requires, 10)
    index = torch.tensor(index_values, dtype=torch.int64).reshape(index_shape)
    row = {"name": name, "operation": op, "dim": dim,
           "self": describe(a), "source": describe(b), "index": describe(index),
           "requires_grad": {"self": True, "source": source_requires}}
    if op == "index_add":
        row["alpha"] = alpha
    try:
        out = (torch.scatter_add(a, dim, index, b) if op == "scatter_add"
               else torch.index_add(a, dim, index, b, alpha=alpha))
        row["forward"] = {"status": "RETURNED", "output": describe(out)}
    except Exception as exc:
        row["forward"] = {"status": "RAISED", "exception": exception(exc)}
        row["backward"] = {"status": "NOT_RUN"}
        return row
    seed = (torch.arange(out.numel(), dtype=torch.float32) + 1).reshape(out.shape)
    row["seed"] = describe(seed)
    try:
        out.backward(seed)
        row["backward"] = {"status": "RETURNED"}
    except Exception as exc:
        row["backward"] = {"status": "RAISED", "exception": exception(exc)}
    row["backward"]["self_grad"] = describe(a.grad)
    row["backward"]["source_grad"] = describe(b.grad)
    return row


def probe():
    cases = [
        ("scatter_smaller_index_backward", "scatter_add", [2, 3], [3, 4], [2, 3], [0, 1, 0, 1, 0, 1]),
        ("scatter_smaller_index_self_only", "scatter_add", [2, 3], [3, 4], [2, 3], [0, 1, 0, 1, 0, 1], 0, False),
        ("scatter_scalar_index_and_source", "scatter_add", [], [], [], [0]),
        ("scatter_scalar_index_vector_source", "scatter_add", [], [1], [], [0]),
        ("scatter_vector_index_scalar_source", "scatter_add", [2], [], [1], [0]),
        ("scatter_empty_broadcast_gradient", "scatter_add", [2, 3], [1, 3], [0, 3], []),
        ("scatter_empty_mismatched_rank", "scatter_add", [2, 3], [4, 5], [0], []),
        ("scatter_empty_rank4_scalar_gradient", "scatter_add", [2, 3], [], [0, 2, 3, 4], []),
        ("index_add_scalar_index_and_source", "index_add", [], [], [], [0]),
        ("index_add_empty_other_axis_bad_indices", "index_add", [2, 0], [3, 0], [3], [99, -1, 9223372036854775807]),
        ("index_add_empty_selected_axis_bad_indices", "index_add", [0, 3], [2, 3], [2], [0, 1]),
        ("index_add_empty_rank2_index", "index_add", [2, 3], [0, 3], [0, 2], []),
    ]
    return [run_case(*case) for case in cases]


def main():
    report_path = Path(sys.argv[1])
    lock_path = Path(sys.argv[2] if len(sys.argv) > 2 else "compatibility/pytorch-oracle-lock.json")
    expected = json.loads(lock_path.read_text())["oracle"]
    observed = {"version": str(torch.__version__), "git_commit": torch.version.git_version}
    if observed != {key: expected[key] for key in observed}:
        raise RuntimeError("Probe requires the existing pinned oracle: " + json.dumps(observed))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = probe()
    report = {"format": "heptatorch.scatter-index-semantics.v1", "status": "OBSERVED",
              "oracle": observed, "cases": len(rows), "rows": rows}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
