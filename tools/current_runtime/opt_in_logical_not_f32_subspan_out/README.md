# logical_not(out=): F32 dense nonzero-offset subspan

Adds only disjoint nonempty F32 positive-offset dense permutation
destinations of direct contiguous roots. A positive offset must be
aligned to the validated four-byte dtype width. Logical root elements,
live storage bytes and tagged address are independently bounded with
subtraction before accessing the offset, whose descriptor already
contains the displaced start. No other offset dtype/layout is admitted.
All four previously qualified zero-offset destination types retain
their guards. No change to default runtime activation.

The exact predecessor constructed and refused four new cases with
byte-identical rollback. The candidate passed three independent index
coordinate masks, with distinct sentinels on either side of the
[2,3]/[1,2]/offset2 interval in [10], and passes the end-of-root [5] /
[2,2]/[1,2]/offset1 case previously refused at frame284 in the F64
bundle. It also passed the 12 retained zero-offset permutation cases.
The native 498-frame observation verified 57 exact rollbacks. Separate
CPU Torch produced immutable raw references first. The independent
reader binds the physical opcode111 config, node metadata, live
Storage/Version, exact kind1 byte offset and tagged displacement,
derived dense byte extent within recorded backing span, all canaries
and logical/physical values. It verifies one shared Version increment
and unchanged disjoint source. No constructor controls were skipped.

Two matching independent cold builds. The unchanged original reader
also passed all 79 earlier cases and 82 rollbacks, with 876 frame bytes
unchanged. The first fixture attempt stopped at the memory guard; the
one retry was launched only after exact supplemental scratch
coalescing and a higher preflight. An offline interpretation correction
did not re-run or alter raw native results. See evidence.json for
exact hashes and scopes; unique raw remains local and temporary.

prepare_build.py consumes the prior projected F64 source, fresh output
dir, existing build.py and optional discovery receipt. project_subspan.py
is the reversible exact source transformation and static fingerprint.
Use pinned-manifest.json for an exact repeat. Native and offline
observer arguments are the same as in the previous F64 package.
