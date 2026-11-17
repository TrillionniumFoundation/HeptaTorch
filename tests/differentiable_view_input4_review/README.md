# R3 differentiable view reference restoration and replay

This package connects the frozen original14 and input4 reference data to the actual R3 HIA image. It does not perform Tensor computation in Python. The native profile is source 3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6, ELF 7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221, 627 functions, 13 native units, descriptor 672, HTVAD003 / HTOVAD03.

## Reference and reader custody

- prepare_reference.py restores the original input4 single-stream XZ packet and all 42 member bytes and original modes. It admits the original f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3 actual reference and its 25 source bindings. It checks the earlier independent admission report but does not rerun its 64,059 checks or execute Torch/native code. Its own actual restoration report is 86a1759f1a696400183cd80d0548d35e821c0eef797db5dbcc397f34cc743557.
- replay_input_boundaries4.py is the exact reader02 (5b8809eb9e405bf89a3586741d64210483fe3fbc5bc323e2cc16cb712ce4de66). Its original14 support reader remains 52c3b3242a73fb412102c83c5ba47cf8d95a4eb3c01e56f41bfe593066506d05 under tests/differentiable_view_native_review/. All 21 inherited comparison methods are unchanged. The explicit profile arguments bind 627 functions and 13 units.
- replay_direct_view14_profile.py is the R5 profile-binding reader (deb9f6fafc3f91b2749bdfac030e19cb0ccdb832fa238af91ed9feb396e62eb8). Only its main profile bindings and revision label differ from the original reader. All non-main functions, the entire Case class, reference admission, comparisons and error mappings are unchanged. Both exact inverse records and the intermediate R4 source are retained in provenance/; R4 was not executed as a runtime reader.
- native_observer_direct03.py is copied byte for byte (68d0f4657c6cc7062d98cbf033565b16c7b51f3381c6490247e42512c5baf4a2). Only the two format tag literals differ from observer02; the exact inverse record is preserved.

## Actual finite qualification

The independently executed original14 report is c3147b161f32a85a87ccf00da47f3d2fc8bc2a5c47e159cf1f7ad1f8938606a1: 14 PASS, 15,135 frames, 96,461 assertions, 14,126 exact raw words, 2,325,760 full RNG bytes and 459,258,768 rollback bytes. The input4 report is 59851d848c383305d8e827d0ef3f1ef1b5fba6f513f5ecfda088fa9ced544d68: 4 PASS, 12,353 frames, 77,432 assertions, 10,120 exact words, 1,102,208 RNG bytes and 374,676,864 rollback bytes. Every session reached EOF with exit zero and all bound inputs were unchanged. Full original streams, objects, source and process receipts are in validation/differentiable-view-r3-native14/ and validation/differentiable-view-r3-input4/.

These are finite results. They do not close a full PyTorch schema or qualify unexecuted original16, general resources, count-zero NaN behavior, other dtypes/devices or the entire autograd surface. The later multiaxis72 gradient result has its own evidence and its own narrower observation scope. The earlier R2 reader01 failure and reader02 1 PASS / 3 FAIL remain unchanged in validation/differentiable-view-r2-input4-history/.

## Make targets

make test-differentiable-view-reference14 restores the frozen original14 reference. make test-differentiable-view-input4-reference restores input4. make test-differentiable-view-native14 and make test-differentiable-view-native-input4 build the distinct R2 parent followed by all 13 R3 units, then replay the respective actual references. CI preserves Make stdout/stderr and full per-frame evidence on failure as well as success. Installing this reader package does not rerun those business tests.

The full project remains HOLD, with 0 fully closed API schemas out of 2590 and all existing full-acceptance gates retained.
