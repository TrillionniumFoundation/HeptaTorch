# Preserved first hosted PR28 failure

Source head: 30ea38d3572a18ede2dda80decc2fcb4a83feb84
Source tree: 80466b159e5636c78d3888aa7b05a58e6a5d9970
Base/main: dcb903ccbb679497f4c7fa3c413dc0f6d19340ef
Generated PR checkout: e4511fe8f8ee7ac33eb63b7cb3b88d1a6cef0a12
Workflow run: 43850393178
Job: 133241099450
Artifact: 14219765370

The actual job completed FAILURE: 74 actual steps, 71 success, one failure,
two skipped. Step 66, the unchanged 355 frozen35 entry, returned 3 PASS and
32 FAIL. The 32 errors are the additional v3 bridge close assertion
stats[11] == self.next_id-1. Actual native public .grad ownership creates
private HIA Tensor nodes, so the physical node high-water is not the old client
creation ordinal.

All original case bodies returned, but the failed close assertion prevented
32 original EOF/exit/empty-pipe checks and their final parameter Version gates.
The observed 6,344 frames, 2,590,274 compared values, zero recorded direct
parameter Version mismatches, and 283 explicit legacy state Version
differences do not replace those unexecuted gates.

The standalone 355 native qualification and unchanged live32 passed. The
common-428 native qualification passed, while its hosted live32 and frozen35
steps 67 and 68 were skipped. A separate ROG 428 live32 later passed; that
result does not change this hosted job.

original-pr28-failure.zip is the complete unmodified downloaded artifact:
3,914,565 bytes, SHA256
ac4f9a5af8cbb8ec58db4ee5a015b8789b159835dd370ea213fe8dd582e0ed8c.
It contains 54 original reports plus ci-report-manifest.json. Every original
report's bytes and SHA256 were checked against that manifest. The manifest is
PARTIAL and the job status is failure; only the two skipped common-428 report
envelopes are absent.

The original frozen35 failure report inside the ZIP is
pytorch-gradient-objects-frozen35.json:
3,022,230 bytes, SHA256
46c861a63c1dae54b257ea78b140f442868e0848f347d6cc10fec38013d08dbe.

final-pr-jobs-response.json and merge-checkout-commit.json are raw GitHub
response content retained without JSON reserialization. A new strict observer
must pass new actual execution; this evidence remains FAILURE.
