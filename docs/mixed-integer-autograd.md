# Native mixed F32 / int64 autograd

This separate CPU profile differentiates the admitted mixed F32/int64 add,
subtract, multiply and true-division operations in native HIA. The graph retains
the actual original inputs and their Storage, Version and Tensor identities.
The complete target remains PyTorch 2.15.0 at
`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. Its 2,590 schemas, 688 derivatives
and seventeen whole-system gates remain the acceptance boundary.

## Exact implementation

| Identity | Frozen value |
| --- | --- |
| HIA source | `5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49` |
| Native ELF | `69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6` |
| Core shape | 556 functions, 7,420 blocks, 26,889 instructions |
| Compilation | Eleven actual native modules |
| Descriptor and owner formats | 640 bytes; `HTMXAD02` / `HTOMXD02` |
| Frozen source package | [Manifest](../tools/mixed_integer_autograd/package-manifest.json) |
| Complete source and failure history | [Handoff](../validation/mixed-integer-autograd-evidence/handoff-manifest.json) |

The [locked profile](../tools/mixed_integer_autograd/profile.json) records all
eleven source/object hashes and the actual compiler and linker identities.
Production arithmetic, conversion, allocation, graph execution and lifetime
run in HIA. Python authors the primitive source, transports requests and
independently compares observations.

Command103 admits the bounded mixed binary paths, with the F32 operand on
either side and an I64 operand that cannot require gradients. Forward promotion
still allocates real temporary F32 Tensors. The graph saves the original inputs;
backward reads the original int64 values and performs the specified conversion.
The fixed observations cover shared-Version mutation errors, independent
`.data` Versions, actual Saved presence and partial-backward behavior.
Command104 continues to reject requires-grad in-place inputs. The
[protocol](../tools/mixed_integer_autograd/PROTOCOL.md) defines the exact scope.

Command107 accesses an original Saved input. It publishes its real public
Tensor token, returns zero for an absent Saved slot, and distinguishes a spent
Saved slot from a shared-Version mismatch. It retains the ordinary full-frame
rollback contract. A successful accessor may create a public reference; an
observer that retains only metadata must release that reference.

## Tensor ownership and two observed repairs

The new private 32-byte Storage and Hold record two original
`(node, incarnation)` pairs. Each populated slot owns its own strong Tensor and
metadata references. Repeated inputs such as `x*x` retain and release twice.
Independent accounting derives these counts from the actual pairs, checks the
Saved Storage/Version/offset relation, and verifies every descriptor word.

Revision08 passed its mixed numerical cases but failed the inherited public
gradient cleanup case: `zero_grad(False)` discarded graph state while leaving
the new Saved Tensor Hold alive. Revision09 releases that Hold after the
original admission guard and before graph teardown. The original failure and
the unchanged legacy expectations remain in the evidence.

The separate public-Saved lifetime reader then found another real failure in
revision09. An I64 view's Tensor row was dead after the final public Saved
reference was released, but graph metadata still retained its base Tensor and
kept Storage alive. Revision10 releases the base edge when the Tensor row dies,
while preserving the remaining graph metadata. A live view Tensor row must
retain its base; rowless graph metadata must not. These two cases have distinct
audited invariants and a new owner-format barrier.

The same independent fourteen-case reference and comparison logic reported
13 passing cases and one failure on revision09, then fourteen passing cases on
revision10. The first diagnostic serialization failure is also retained.
The source package keeps its original `PREPARED_NOT_RUN` history; the separate
actual execution reports record the subsequent passes.

## Executed local evidence

| Independent evidence group | Actual bounded result |
| --- | --- |
| Fresh producer | Five gates, including a new compilation of all eleven modules |
| Fixed mixed autograd reference | 36 cases; 4,728 frames; 21,477 checks; 4,670 exact words |
| Existing int64 behavior | Eight inherited gates, including the unchanged independent inplace40 reader |
| Existing 523 behavior | Twelve gates, including the original sixteen inherited suites |
| Public Saved lifetimes | Fourteen cases; 2,058 frames; 8,842 checks; 2,166 exact words |
| Complete owner accounting | Four cases; 58 frames; 144 full 80-word snapshots; 53 transitions; 105 assertions |
| New-role copied refusals | Seventeen required refusals in the host checker |
| Resource and allocation recovery | Six cases; 536 frames; 162 assertions, including real allocation failure |
| Owned output capacity | Five cases; 46 native probe processes; 33 frames |
| Saved getter output and errors | Six cases; fourteen probe processes; seventeen frames |
| Actual previous/new owner handoff | Two directions; 23 compiled modules; fourteen frames; 1,398,048 rollback bytes |

The producer's [actual cold report](../validation/mixed-integer-autograd-evidence/cold10-reports/report.json)
has SHA-256 `2369c55491f9b1e7714ec3bd1ef197c4967fc5b2695e1a103446d56a607343a2`.
The [independent owner and resource package](../validation/mixed-integer-owner-review/qualification.json)
retains its own native executions and three complete raw archives. The
[public Saved package](../validation/saved-public-lifetime/qualification.json)
retains the original PyTorch observations, both native revisions, all compiler
units and both failed and successful runs. Every archived member's size, mode,
hash and complete single-stream EOF are checked by the supplied verifiers.

These groups overlap. Their counts are not summed into unique APIs or full
schemas. No complete upstream schema is newly closed by these finite checks.
The seven-command integration coordinator has now actually completed the fresh
producer, source custody, owner accounting, resources, owned entry, Saved entry
and typed handoff against one newly built image. The [original report](../validation/mixed-integer-autograd-root/report.json)
is `77ae37fddbce0ad15628c0bf265c9e41649f9614714deaec99b91c15b7c3fbb5`.
It captured 474 inputs before execution and bound 648 files at completion.
Its complete archive retains 2,225 original members and 259,777,728 file bytes.
Root independently rechecked every member and the complete single XZ stream;
that additional byte check does not claim another native execution.

## Full reference and hosted obligations

The original live32 and complete35 workloads have now actually passed on this
new image in the pinned ROG environment. The original35 F32 domain now allocates real original-Saved Tensor
roots even for ordinary F32 multiply and divide. The previous544 observer's
zero-tail assumption therefore does not describe the new owner.

The successor observer adds explicit Saved-pair reference equations and
reachability to the old rules. It retains complete 80-word snapshots and the
original transition checks, numerical oracles, provenance, EOF and parameter
Version conditions. Whole-file and whole-function inverse receipts identify
the changes. Every one of the 6,344 original transitions has been recomputed from the
archived wire and 6,321 unique snapshots by the exact frozen observer. The
[complete remote evidence](../validation/mixed-frozen35-accounting/remote10/)
binds all eleven actual native units and both original eleven-check provenance
records. Live32 retains 2,132 frames, 17,455 exact words and 1,137,600 RNG bytes.
Full35 retains 2,590,274 comparisons, all EOF/post-close Version checks, 283
explicit state Version differences and zero parameter Version differences.
Four positive and eighteen copied refusal controls also actually ran. The
root byte review is independent replay and does not add a native execution.

The new source head must separately pass its actual PR and push jobs and a
complete read of their original artifacts before merge. Previous-profile
passes remain evidence for those previous profiles. CI preserves the parent
workflow's steps and adds new native and observer artifact namespaces.

## Build and remaining scope

See [the build instructions](build.md#native-mixed-integer-autograd).
`make test-mixed-integer-autograd` uses an actual544 parent, constructs a new
eleven-module556 executable, and runs the seven outer gates. All produced
reports and raw evidence remain available when a gate fails.

Public `Tensor.to`, general dtype and layout coverage, higher-order and forward
autograd, hooks, arbitrary Python wrapper reference counts, full exception
frontend behavior, accelerators and all-system parity remain open. Two Python
aliases to one Saved Tensor in the finite lifetime reference are released
together; they do not certify arbitrary wrapper reference counting. The global
acceptance decision remains **HOLD**, with zero complete-schema parity.


## Historical tombstone regression and archive transport

The added [tombstone regression](../validation/mixed-frozen35-accounting/tombstone-legacy-refusal/)
reproduces one actual native case and four copied refusals. Its original host
test accidentally loaded the already-corrected checker while expecting the
old refusal, and its cleanup used a nonexistent `Session.stop`. Both original
failures and the original source19 packet remain unchanged. A separate entry
loads the SHA-pinned historical6ea predicate and changes only four module data
identity bindings; all historical predicate bodies and original assertions
remain preserved. Its six native frames end with a real successful EOF.

The cold7 original XZ is distributed as two opaque parts because its base64
request exceeded the connector size cap. [Transport instructions](../validation/mixed-integer-autograd-root/TRANSPORT.md)
and the exact index permit full verification without extraction. Concatenation
reproduces the original whole archive SHA and every original member. Neither
this representation nor the new regression changes the global acceptance gate.
