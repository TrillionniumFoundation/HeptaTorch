# Original Bool27 comparisons on the current image

`native.py` is the compatibility entry for the complete original Bool27 suite.
The six original reader sources are preserved once, with their original
bytes and modes, in `validation/current-runtime-baseline`. The current
identity adaptation and strict outer admission live in
`tests/current_runtime_historical`.

Every original case and control still executes. The original report retains
`FAIL`: its 27 numeric cases pass, five historical dtype cases pass, and the
unchanged F32-refusal assertions at indices 2 and 3 fail on the newly supported
successes. All ten original entry controls and the original Bool allocation
failure/recovery case must pass. Only those two exact refusal failures may be
admitted, and their actual raw successful commands are checked independently.

The separate F32 suite checks all seven current dtype outcomes, including
the two F32 successes. Run `make test` or `tests/qualify_current_runtime.py`
to require both suites on the same newly built image. This compatibility
entry uses the explicit arguments of `current_runtime_historical/run.py`;
the coordinator prepares the source/reference directories automatically.
