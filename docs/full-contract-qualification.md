# Complete-contract qualification

`make acceptance-status` retains the fixed PyTorch target: **2,590 native
declarations, 688 derivative declarations and 17 complete subsystem contracts**.
It checks target accounting and any complete-contract qualifications registered
in source. `make full-acceptance` exits with code 2 while any complete contract
remains open; it can return zero when all required contracts have been independently
qualified and the complete target ledger agrees. Invalid accounting or evidence
returns code 1.

The current production registry is empty. Consequently the qualified counts
remain **0 / 2,590**, **0 / 688**, and **0 / 17**, regardless of passing finite
runtime suites. This change implements the admission mechanism, not additional
PyTorch functionality or a full-contract behavioral verifier.

## Register a real complete-contract verifier

Implement the native contract and its independent qualification first. Add a
reviewed entry to `VERIFIERS` in `tools/acceptance_contracts.py`, keyed by exactly
one of:

- `("native", <complete frozen native declaration>)`
- `("derivative", <complete frozen derivative declaration>)`
- `("subsystem", <one of the 17 fixed subsystem IDs>)`

An entry specifies a stable verifier ID, its actual source file, the exact required
evidence roles and a callable. Only this source-owned registry chooses code. A
ledger, evidence file or command-line argument cannot select an import or command.
Unknown targets and claims without registered verifiers fail closed.

The callable receives immutable original byte snapshots for the current runtime
manifest, all canonical source units, and its evidence files. It must independently
check the entire contract: all applicable overloads, generated variants, shapes,
dtypes, layouts, devices, aliases, gradients, error and resource behavior. It must
also check actual build and process completion, fixed toolchain and independent
oracle identities, original full inputs/outputs, ownership and failure controls.
Any helper code or additional inputs it depends on must have explicit evidence
roles and be checked as part of those predicates. A generic JSON `PASS`, a finite
case count, or matching digests alone is insufficient.

Return `Verdict(True, True, frozenset(required_roles))` only after those predicates
pass. Exceptions, booleans, partial verdicts and omitted checked roles are rejected.
Verifier source and admission-engine revisions are bound to the evidence. All
source/evidence paths are reread after verification to reject concurrent changes.

## Evidence format

A qualified native or derivative record retains its exact declaration and has
`status: "qualified"`. A closed subsystem has `status: "CLOSED"`, its original
required contract, and an empty `remaining` field. Each has one `qualification`
object containing the repository-relative `path`, integer `bytes` and `sha256`
of its evidence envelope. Open records carry no full qualification claim.

The envelope has exactly these fields:

```json
{
  "schema": "heptatorch.full-contract-evidence.v1",
  "upstream_commit": "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457",
  "contract": {"kind": "native", "name": "<exact frozen declaration>"},
  "verifier_id": "<source-registered ID>",
  "verifier_source_sha256": "<actual verifier source SHA-256>",
  "admission_engine_sha256": "<acceptance_contracts.py SHA-256>",
  "runtime_manifest_sha256": "<current canonical runtime manifest SHA-256>",
  "files": {
    "native_image": {"path": "<path>", "bytes": 1, "sha256": "<SHA-256>"},
    "build": {"path": "<path>", "bytes": 1, "sha256": "<SHA-256>"},
    "execution": {"path": "<path>", "bytes": 1, "sha256": "<SHA-256>"},
    "reference": {"path": "<path>", "bytes": 1, "sha256": "<SHA-256>"},
    "readback": {"path": "<path>", "bytes": 1, "sha256": "<SHA-256>"}
  }
}
```

This is a format illustration, not usable qualification evidence. A verifier may
require additional roles. Every role must have a distinct ordinary file; missing
or extra roles, duplicate source/evidence inodes (including hardlinks), symlinks,
traversal paths and duplicate JSON keys are rejected. Reads use held file and
directory descriptors with no-follow traversal of every path component. File
type, size, mode, modification/change times and namespace identity are checked
after each bounded read; later reads must retain the same file and directory
identities. The native image must match the current manifest, and every current
source unit must match its manifest binding. Read budgets are source-owned.

Qualified counts are calculated from accepted verifier results. Capability prose
and manually edited totals cannot increment them. Full acceptance additionally
requires all fixed declarations and all 17 subsystems, with consistent completion
flags. The existing original target-preservation controls remain active.

## Framework verification

`tests/check_acceptance_contracts.py` exercises registered and unregistered
admission, source/image/verifier binding, independent rejection of wrong outputs
even after their hashes are updated, malformed evidence, partial verdicts,
immutable snapshots and concurrent edits. Its finite byte-transcript verifier is
test-only and is never registered for a production PyTorch declaration. These tests
do not qualify a native kernel or change the full-project counts.

Run the framework checks with:

```sh
python3 -I -S -B tests/check_acceptance_negative.py
python3 -I -S -B tests/check_acceptance_contracts.py
```
