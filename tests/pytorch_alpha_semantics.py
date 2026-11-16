"""Determine the pinned CPU scalar/vector alpha rounding path, without HIA."""
import json
import struct
import sys
from pathlib import Path
import torch

expected = json.loads(Path('compatibility/pytorch-oracle-lock.json').read_text())['oracle']
if str(torch.__version__) != expected['version'] or torch.version.git_version != expected['git_commit']:
    raise RuntimeError('Alpha probe requires the pinned PyTorch oracle')
rows = []
for shape, dim in [([], 0), ([1], 0), ([33], 0), ([1, 1], 0), ([1, 33], 0), ([33, 1], 1)]:
    a = torch.full(shape, -2e38, dtype=torch.float32)
    b = torch.full(shape, 2e38, dtype=torch.float32)
    index = torch.arange(shape[dim] if shape else 1, dtype=torch.int64)
    out = torch.index_add(a, dim, index, b, alpha=2.)
    rows.append({"shape": shape, "dim": dim, "output_bits":
                 [struct.unpack("<I", struct.pack("<f", v))[0] for v in out.reshape(-1).tolist()]})
report = {"schema": "heptatorch.pytorch-alpha-semantics.v1", "status": "OBSERVED", "cases": len(rows), "oracle": {"version": str(torch.__version__), "git_commit": torch.version.git_version},
          "operation": "index_add_alpha_overflow_cancellation", "alpha": 2., "rows": rows}
text = json.dumps(report, allow_nan=False)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text + "\n")
print(text)
