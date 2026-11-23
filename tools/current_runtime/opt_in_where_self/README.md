# Native opt-in CPU where.self

Adds the missing functional ternary selector to an exact HeptaTorch #65 base:
a Bool condition, two same-dtype values, equal shapes, and a fresh result.
The admitted case is ordinary contiguous, zero-offset, no-grad owned CPU
tensors in the six public dtypes, rank at most four and at most 262,144
elements. The runtime actually checks live owner bounds and fresh result
shape/dtype/count before accessing raw storage. It writes only selected bytes,
not floating arithmetic, and keeps the same framing and rollback boundary.
Canonical/default routing is unchanged. Broadcasting, dtype promotion,
views, autograd, out= and full hosted parity remain unqualified.

Two cold compilations independently produced the same ELF. Official locked
CPU PyTorch, a no-Torch native observer, and a third process reading only the
raw artifacts agreed on 23 cases and 308 result elements, including all six
typed empty shapes, two scalar conditions, both lifetime orders and one
same-handle left/right case. Raw output preserved signed zeros, denormals,
infinities, qNaN/sNaN payloads and I64 extrema. Result Tensor, Storage and
Version identities were fresh and initial Version zero, including when empty.
The exact previous ELF refused all 23 valid requests (228 physical frames).
The new ELF passed 40 whole-owner rollback checks across 487 frames, including
an actual process-local allocation refusal, exact-packet recovery of 65,536
additional raw F32 values, valid selector then invalid command in one frame,
and weak Storage expiry. A same-frame local result reference also passed.

All four prior raw request/response streams replayed byte-identically:
50 layout outputs / 175 rollbacks, six physical F16 and six BF16 constructors /
20 rollbacks each, plus 79 prior ordinary positives / 82 rollbacks. An initial
offline reader was mistakenly pointed at the F16 capture with BF16 expectations;
the unchanged F16 reader subsequently passed on identical captures. This was
not an ELF failure. Proofs of current large shape admission and malformed outer
frames were not added for this slice.

Reproduce only on the exact input manifest. First,
`python3 -I -B prepare_where.py --source EXACT_PARENT --checker PINNED_BUILD_PY`
checks the reversible projection without compiling. Add
`--materialize-later NEW_DIR` if creating executable source. Set the projection
manifest aside, copy `pinned-manifest.json` over the output manifest for a
locked second build, and run the usual pinned complete builder independently.
Use the following independent stages:
- `oracle_where.py PINNED_HELPER_DIR ORACLE_LOCK INSTALL_REPORT FIXTURE` with the
  official locked CPU wheel environment only; check fixture bytes against
  the included official fixture.
- `native_where.py PINNED_HELPER_DIR PROTOCOL_PY ELF FIXTURE OUT baseline|candidate`
  runs without Torch; apply the per-process guard on a bounded local test system.
- `pure_where.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE` reads only captured
  bytes; it imports neither the transport nor Torch nor the native observer.
- For inherited checks, replay exact #65 observers with only their candidate ELF
  pin changed, their locked fixtures unmodified, and compare raw bytes.
