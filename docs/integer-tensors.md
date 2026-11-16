# Native int64 Tensor profile

The separately qualified CPU profile adds real eight-byte signed int64 Storage
to the existing binary32 and one-byte bool owner. Tensor arithmetic, conversion,
allocation, mutation and lifetime run in native HIA. Python assembles source,
transports requests and independently checks observations.

The complete target remains PyTorch 2.15.0 at
`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`: all 2,590 schemas, 688 derivatives
and seventeen full-system gates remain in scope. No complete schema or
whole-system gate is closed by this profile.

## Exact executable

| Identity | Frozen value |
| --- | --- |
| HIA source SHA-256 | `adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62` |
| Native ELF SHA-256 | `b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08` |
| Source shape | 544 functions, 7,263 blocks, 26,361 instructions |
| Compilation | Eleven independently compiled native modules |
| Private descriptor | 640 bytes; `HTINT002` / `HTOINT02` state tags |
| Source package | [Exact build and observation inputs](../tools/integer_tensors/package-manifest.json) |
| Complete delivery | [Sources, results and failure history](../validation/integer-tensors/package-manifest.json) |

The [profile](../tools/integer_tensors/profile.json) binds every native source
unit, object, tool and executable. The implementation is assembled through
[the frozen build recipe](../tools/integer_tensors/build.py); the full operation
contract is in [PROTOCOL.md](../tools/integer_tensors/PROTOCOL.md).

## Implemented scope

The typed constructor, raw read and fill accept dtype2 int64. Command103
implements the admitted typed add, subtract, multiply, bitwise and/or/xor and
true division combinations. Command104 provides the qualified int64 Tensor-RHS
in-place forms; command106 is the separately observed Scalar `add_` overload
with alpha1. Scalar and Tensor-RHS behavior remain distinct.

Native int64-to-binary32 conversion uses integer round-to-nearest-even. Mixed
arithmetic materializes actual common-dtype temporary Tensors before selecting
the output layout. Their Tensor, Storage and Version resources are released.
This internal conversion does not implement the public `Tensor.to` API.

Int64 `as_strided`, transpose and view retain a real ultimate-base Tensor.
The checks separately observe base destruction, a view of a view, final-view
release, detached aliases and weak Storage expiration. Detach and `.data` hold
their qualified Storage/Version roles without inventing a base relationship.
The prior F32/bool command69 retains its documented detached-alias behavior;
general cross-dtype view equivalence remains open.

The explicit command105 preserves the observed int64 dropout failure contract:
actual RNG advancement and completed temporary allocation/release commit,
then execution stops before the suffix. Ordinary malformed or inadmissible
operations continue to require full-frame rollback. Mixed F32/int64 operations
that require autograd are explicitly refused; mixed-dtype AD remains open.

## Executed evidence and counting boundaries

The [first portable cold execution](../validation/integer-tensors/evidence/portable-cold07-report.json)
actually rebuilt all eleven units and passed ten top-level gates. A supplied
native executable cannot substitute for this build.

| Separate evidence group | Actual bounded qualification |
| --- | --- |
| Original int64 reference | 54 observations partitioned into disjoint storage12, arithmetic30 and dropout12 |
| Alias mutation | Eight cases, including real mutation through each alias direction |
| Conversion rounding | 1,237 signed integers through conversion and mixed arithmetic |
| CPU promotion layout | Twenty cases and forty original mixed operations |
| Ultimate-base lifetime | Six cases with actual destruction and final release |
| Independent in-place reference | Forty cases; 1,747 frames, 7,178 checks and 13,980,480 rollback bytes |
| Existing runtime behavior | Twelve named gates, including the original sixteen inherited suites and bounded retained12 |
| Independent resource ownership | Six cases; 527 frames, 157 checks and 44,811,376 rollback bytes |
| Owned entry | Five cases; 46 native probe processes and 33 interactive frames |
| Previous/new owner handoff | Two directions; 23 native modules, thirteen frames and 1,398,048 rollback bytes |

These groups overlap and must not be added as unique tests, operators or
schemas. The [resource qualification](../validation/integer-resource-review/qualification.json)
and [typed handoff qualification](../validation/integer-resource-review/typed-handoff-07/qualification.json)
are independent of the producer. The integration coordinator executes fresh
producer, resource, entry and handoff processes against the same cold image.
The [first integrated execution](../validation/integer-tensors/integration-final/report.json)
passed all four gates, verified 219 unchanged inputs and retained 1,772 original
evidence members. Every member's bytes, mode and size and the single XZ stream
EOF were checked. Hosted admission remains a separate source-bound obligation.

The unchanged live32 and complete35 floating-point workloads require their own
execution on this executable. The successor observer consumes all eighty words
of the 640-byte descriptor. Its original F32 command domain requires the new
int64-only fields to remain zero. This domain does not certify every int64
view-root state. General int64 ownership, broader dtypes, devices, Python
objects, exceptions and complete API parity remain open.

## Preserved failures

The independent in-place reader first reported 37/40 passing on candidate06.
The two nondense overlap admissions and overlap-before-cast error priority
were corrected in three HIA function bodies. The same actual PyTorch
observations and unchanged reader then passed on candidate07.
[The original failed run](../validation/int64-inplace-review/first-06/manifest.json)
and [new complete run](../validation/int64-inplace-review/candidate07/report.json)
retain their separate source and executable identities.

The first 64 MiB resource budget did not cause allocation failure and therefore
failed that control. A separate 56 MiB execution actually triggered native
allocation failure, verified unchanged owner/RNG state and completed recovery.
The original attempt remains in the evidence. Earlier alias, promotion,
base-lifetime, build and preparation failures remain in the producer history.

The standalone forty-case reference copy changed from mode0600 to0644 for Git
checkout. Its bytes are identical; the mode0600 original remains in the archive
and the earlier manifest is retained. This explicit packaging change does not
alter expected behavior.

## Build and ongoing acceptance

`make test-integer-tensors` uses an actual523 parent build, creates a new
qualification directory and cold-builds the new eleven-unit image. It copies
the resulting runtime to `build/integer-tensors/` after all four native gates
pass. See [build instructions](build.md#native-int64-tensor-profile).

CI preserves original reports, request/response streams, owner snapshots,
native source units and tool provenance. Failed and cancelled runs retain
their real status and partial evidence. A successful predecessor or a stored
report does not qualify a new source head. The full-project decision remains
**HOLD** until the complete acceptance matrix is independently satisfied.
