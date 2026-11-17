# Resource counting addendum

The three distinct finite controls passed across two native invocations. No additional native or Torch execution was performed while writing this note.

- The original source01 invocation remains FAIL: two controls passed, while the OOM control failed during filler setup. Its report contains 96 frames and 199 completed checks. The unsuccessful setup accounts for 13 frames and 17 completed checks; it did not reach temporary prlimit or backward.
- The source02 successor ran only the OOM control: PASS, 30 frames and 61 checks. The two earlier successful controls were not repeated.
- Successful-control totals are therefore 113 frames = 96 - 13 + 30 and 243 checks = 199 - 17 + 61. These totals explicitly exclude the failed setup's 13 frames and 17 checks.
- The complete archive preserves all 126 original native frames, all 260 reported completed checks, and all four EOF exit codes of zero, including the unsuccessful setup.
- The portable source02 default --case all permits CI to run all three controls in one fresh invocation. This combined invocation was not repeated locally. Its eventual CI report has a separate identity.

The first report is reports/first-resource3.json (SHA256 184636ba83de144c6f4543bf4fdfbeb63b83fb633cb1b45ca069c3086f24727f). The successor is reports/successor-oom1.json (SHA256 846bcad6839c542948b603e51d11eb6a833ac35daef9d5bc38d28b74860ae9f0). Neither original report, the original source02 manifest, nor the frozen producer profile was changed. Qualification remains limited to these controls; whole-PyTorch remains HOLD with zero newly closed schemas.
