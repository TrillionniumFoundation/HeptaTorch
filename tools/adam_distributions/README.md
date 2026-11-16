# One owned runtime for Adam, Generator and CPU distributions

This component combines the frozen Adam02 object runtime and the frozen CPU
distributions03/partial01 extension in one native HIA executable. Tensor values,
Generator state, gradients, Saved objects, SGD momentum, Adam step and moments
are held by the same actual owned buffer. The host build scripts emit source
and invoke the native tools; they do not compute production Tensor values.

The current source has 378 functions, 4,971 blocks and 18,539 instructions.
Eight actual compilation units are compiled by the unchanged ABI3 compiler
and linked by its native whole-program stack gate.

| Identity | SHA-256 |
| --- | --- |
| Combined HIA source | `5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873` |
| Combined ELF | `3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22` |
| Adam02 source | `253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6` |
| Partial distribution01 source | `55ed0c6e16317add03974c01ffd85e38684fc535a3e5cb6423ed043f4217f05b` |
| Native compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Native session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

## Reproduce

First build the frozen `tools/owned_adam` and `tools/cpu_distributions_partial`
predecessors using their own recipes. This adapter consumes their complete
source outputs. It also needs the pinned Hepta checkout and the frozen ABI3
tool build. No upstream source is changed or vendored here.

```sh
python tools/adam_distributions/rebuild.py \
  --adam-build build/owned-adam \
  --distribution-build build/cpu-distributions-partial \
  --upstream /path/to/pinned/Hepta \
  --stack-build build/stack-tools \
  --output build/adam-distributions
```

The two predecessor directories must include their original `runtime.hepta`,
`rng03.hepta`, and `owners.hepta`; Adam02 also supplies its generated
`overlay-authoring.py`. Exact input hashes are checked before any authoring.
The 30 imported upstream files are verified against both a frozen file map
and the pinned upstream support manifest before import and after the build.
The final source, every unit, every object and the final executable must match
the profile pins. The native tools run with an empty `PATH`.

## Source inheritance

345 of Adam02's 352 function bodies are byte-identical. The seven changed
names are `entry`, `gradient_direct`, `partial_apply_node`, and four state tag
functions. `partial_apply_node_adam` retains the complete old dispatcher with
only its declaration renamed. Its exact inverse is verified. The wrapper at
the old name handles Bernoulli opcode41 and delegates every previous opcode
to the preserved function.

All 25 distribution function bodies are byte-identical to partial01.
`gradient_direct` is copied exactly from that predecessor. The borrowed RNG
module and the actual owner step/close module are byte-identical in both
predecessors and this composition. Eight-kind migration, private optimizer
roots, graph capture, `record`, parent-slot accounting and graph audit remain
the Adam02 implementations.

The host authoring admission is deliberately bounded to the selected actual
function bodies and interfaces. It is not an aggregate Core-document
admission. The original aggregate document limit is not raised or bypassed;
all emitted production units undergo the unchanged native checker and all
linked imports, exports, resource contracts and stack metadata undergo the
unchanged native linker.

## Qualification and its limits

The frozen 183-row actual PyTorch report is replayed twice on the combined
ELF. The original atomic commands retain 161 mapped passes, four documented
transaction differences and 18 unrepresented cases. Explicit commands86/87
retain 165 mapped passes and 18 unrepresented cases. No expected values,
reference rows, numerical tolerances, or full Generator states are rewritten.

The unchanged old optimizer replay executes its 19 exposed cases: 18 qualify
within their stated scope and one retains its known nonleaf partial behavior.
The separately frozen Adam replay qualifies all four actual Adam/AdamW cases.
Together this is 22 qualified cases, one explicit partial case and one public
gradient object case still open in the original 24-case reference. General
optimizer exception behavior, hooks and state serialization remain open.

One old negative test must stay a recorded failure: atomic03's native control
script treats every opcode86–89 as unknown and expects error4. Partial01
introduced86/87, so their empty arguments now fail typed Tensor admission
with error1. The original script and its actual failure are retained. A
separate derivative changes only these two new opcode expectations;88/89
still expect error4. All three native profile ELFs are executed to verify this
specific difference and full owner rollback. It is not reported as an
unchanged old control-suite pass.

Native qualifications also cover partial probability and dropout controls,
Adam ownership and Version behavior, real allocation failures, output bounds,
and legal owned handoffs to both predecessors. The handoff tests compile
renamed complete native source units, move actual owners through checked HIA
calls, and prove refusal preserves pointer, length and bytes before recovery
and eventual release. They do not import foreign process pointers.

This is a **PARTIAL component**. The whole-project decision remains **HOLD**.
All-RNG, all-optimizer, all-Tensor-dtype and full-PyTorch completion flags remain
false. The fixed PyTorch API inventory remains 2,590 entries. Unsupported
double/int64 probability Tensors, native_dropout's boolean Tensor mask,
normal-gradient-mode in-place nonleaf operations, public `.grad` Tensor
objects, general real-exponent Adam bias correction and the other frozen
predecessor exclusions are not closed by this composition.

## Preserved history

The first authoring attempt failed before native compilation because the
pinned host Builder does not accept a Python conditional expression. The
next attempt uses an ordinary `if`/`return`; no native checker changes were
made. Its source and failure trace remain in the evidence archive.

The old atomic negative-test failure, both frozen reference outcomes, all
native raw requests/responses, compiler outputs, qualification objects and
source binding receipts are preserved with per-member length, SHA-256 and mode
checks. A live tmpfs execution directory is only temporary; a qualification
is not handed off until its complete evidence has been archived and verified.
