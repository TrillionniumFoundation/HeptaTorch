# Independent owned Optimizer replay

Candidate 08 qualifies the exposed dense binary32 SGD and leaf-flag behavior
against separately executed, pinned Torch observations. The 24-case reference
has 18 qualified recipes, one partial retained-nonleaf recipe, and five
unimplemented recipes. The separate six leaf-flag recipes all pass. Parameter
and momentum words, Versions, layouts, Storage sizes, None presence and opaque
identity relationships use strict checks; no floating tolerance is applied.

The private-root control is one additional native trajectory, not a Torch
reference test. It grows the table from 64 to 512 slots, retains a momentum node
beyond descriptor 256 across graph reset, grows the actual owner from 684,176
to 4,616,336 bytes, republishes fresh public tokens without replacing the retained
Tensor incarnations, and proves whole-owner rollback after a later rejected
group addition. Optimizer drop preserves independent public aliases; the final
public drops expire both weak Storage references.

The first candidate's two native failures and the first adapter's reference
schema error remain in `history` and the original trajectories. The corrected
adapter does not change a recipe, numeric policy or reference result. Transient
state getters close their newly published tokens after inspection, so the
observer does not add permanent public ownership roots to the lifetime tests.

Public `.grad` Tensor identity/Version/alias access, retained nonleaf parameters,
Adam/AdamW objects, Python warning objects and arbitrary partial Optimizer errors
are not closed by this receipt. The case-16 fixed literal-write fixture only
reproduces an observed input mutation and one shared Version increment; it is
not a Tensor.add_ implementation. Additional AccumulateGrad lifetime probes are
still separately pending in this frozen package.

Run the portable replay with explicit `--binary`, `--source`, expected hashes,
`--controls`, `--transport-observer`, `--observations`, `--output`, and
`--suite optimizer` or `--suite requires`. Run the private control with the same
binary/source pins plus `--replay`, `--controls`, and `--observer`. Outputs must
be new directories. Complete raw request/response streams and source copies are
included. `qualification-08.json` states exact identities and separate counts.
The whole PyTorch denominator remains 2,590 and completion remains false.
