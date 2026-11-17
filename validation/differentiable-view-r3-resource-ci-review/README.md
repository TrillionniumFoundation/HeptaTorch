# R3 resource CI incremental review

The unchanged stage5 Makefile and workflow passed 216 ROG source/CLI checks and 36 Mac YAML/new-shell syntax checks. No native image, Torch operation or compilation ran in this review.

The review binds Make f6e103fcc49d0b0a4b8149659d0fd32fa8059e56174ef08716e8dc811478cfd5 and workflow b26279e67b7183d0ec6fa3f47c347f4bd8f28d8f24c8e453bf921d8a3d3c3a21. Removing exactly the added target, step and artifact paths reproduces both reviewed stage4 files byte for byte. The existing stage4 763-check report and original 99-step preservation evidence are bound by SHA; older run blocks were not executed or rechecked.

Actual pure commands: installed driver --help, make -n -B, installed driver --prepare-only on immutable R3/reference, Ruby YAML parsing, and bash -n for the single new resource shell. Parsed totals are 116 steps, 101 run entries, 50 preserved status exits and a 90-minute budget. The resource target uses the default all-three CLI, the correct R3 build/profile/protocol/restored input4 reference, and a fresh output path. Its step retains stdout/stderr, copies any terminal report, emits its envelope, and preserves the Make exit status. All resource raw output, JSON and Make logs are included in the existing always artifact.

Three original pure-review refusals are retained unchanged. They were reviewer schema/format assumptions: stage4 used PASS_SOURCE_AND_CLI_ONLY; ten older YAML run entries are inline scalars; the new redirection uses valid shell continuation. The exact reviewer-only corrections are reversible. No tested production, reference, runner or target assertions changed.

The original finite resource history remains separate: first invocation two PASS plus one setup FAIL, followed by one OOM-only PASS. This package performs no resource rerun and claims no broader PyTorch closure. Whole PyTorch remains HOLD, closed schemas 0.

evidence.tar.xz contains the full original pure reports, streams, copied inputs, durable process requests, job status/read receipts, and Mac transport archive. Inner original modes are preserved; the outer review files are explicitly installed as Git-compatible 0644. package-manifest.json binds every installed payload. The manifest excludes itself; final import receipt is outside this package.
