# Same-dtype public F64 arithmetic on exact #63

This replay is scoped to the original same-dtype public F64 arithmetic gate, not mixed arithmetic, F64_48, Half view admission, #803, or production. The executable and source are the actual current800/15 joint opt-in image. Compare real Torch observations with the unchanged historical numerical, geometry, ownership, failure and rollback predicates. The actual candidate admission is verified before any native worker; the historical source chain runs only on the exactly recovered canonical ancestor source, never on the ancestor executable.

Use the exact canonical checkout, adjacent Resource7 proof files, and the pinned Torch venv. Run the driver with Python isolation and a fresh path:
```sh
python3 -I -S -B tools/current_runtime/opt_in_dual_unary/replay_same_f64_arithmetic.py \
  --repository "$canonical_repository" --dual-build "$physical_build" \
  --dual-source "$candidate_source" --dual-scoped "$candidate_scoped_receipt" \
  --torch-python "$torch_venv/bin/python" --output "$new_run_directory"
```

The driver does not change any existing test file, chooses no fallback oracle, and requires the actual fixed-wheel Torch process with AVX2 and one thread. The unchanged qualifier performs complete native physical evidence readback. The driver independently invokes that raw reader again from a fresh binder, and checks exact original opcode scope on all retained packets. The byte and mode archive verifier shipped with the Live32 evidence can optionally check a compressed copy against the original retained raw tree.

The original 3,288-case prefix is verified byte-for-byte and the gate also checks all 1,800 appended same-F64 singleton/control cases. The retained physical reader checks the exact selected NaN payload cases, original private3 and AD refusals, 88 protocol controls, and four actual allocation failures followed by identical-request recovery. Those finite checks do not stand in for unrelated dtype, view, or general PyTorch coverage.
