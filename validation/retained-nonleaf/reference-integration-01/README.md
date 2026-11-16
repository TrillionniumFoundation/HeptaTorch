# Retained nonleaf reference integration handoff

This is a local proposal for the next reference-only PR, based on commit
`9a20eadbe20a801c8b26727ae369e0ee0832e423` (tree
`5519a9f84b9768e8dcf9ff041ddd1d5969f3bcbc`). Nothing was applied to the
reference or native worktree. No normal Git index, commit or remote changed.

Apply only the six-file `retained-nonleaf-reference.patch`, or copy the six
repo-relative files under `overlay/`, after the existing reference PR is ready.
`git apply --check retained-nonleaf-reference.patch` must succeed on the intended
checkout before application. A private index loaded with the exact base already
accepted the patch, and its resulting blobs matched the six overlay files.
The patch SHA256 is
`90ea530a7407c76dda51a4b44d6bbbbe670ff89900159c2cb5d59ffac910ebb0`.

## What was actually checked

The new verify and prepare shell blocks were extracted from the final parsed
YAML and executed without editing their commands. Preparation produced the
original 2,668 bytes exactly: eight families, twelve planned variants,
`PREPARED_NOT_RUN`, zero executed cases, no Torch or native execution. Changed
oracle bytes and lock bytes were refused before producing a report. Changed
probe bytes were refused by the exact workflow hash check before execution.
Raw stdout/stderr and complete commands are retained in `executions/` and
`integration-report.json`.

The new four steps and one trigger-path insertion reverse exactly to the base
workflow. Its 41 declared steps and 24 envelope outputs become 45 and 26.
The old collector, emit helper, pinned oracle/lock and entire native workflow
remain byte-identical. The two added outputs are the prepared plan and actual
observation; actual JSON file counts may additionally include the existing
installation/provenance receipts and artifact manifest. No observed behavior
or full-project closure is claimed.

The fixed probe remains SHA256
`7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba`.
The peer's original source-only review remains
`08d979da4564aaca19bdeaf9a229e4b56602d3e8d3dbb2634ac61dbc7871ef68`.
`original-preparation/` preserves its entire frozen nine-file package, including
three pinned upstream sources and the unchanged original manifest. Only the
six `overlay/` files are part of the repository patch.

The first local integration-author check stopped because its independent YAML
fragment parsing discarded the last newline from the preserve step. That
checker snapshot and refusal are in `history/`; the frozen probe and intended
workflow did not change. The final checker retains that newline and verifies
full byte equality rather than relaxing the comparison.

## Recheck this proposal

The optional proposal builder requires Python with PyYAML, only for local
integration checking. The shipped probe itself uses the standard library until
its admitted real-PyTorch branch. From this handoff directory:

```sh
python3 prepare_integration.py \
  --repository /path/to/repository-with-base-commit \
  --frozen-preparation original-preparation \
  --output /path/to/new-independent-output
```

The actual hosted run is still pending. Run the workflow's real observation
step only with the existing pinned wheel and all eleven provenance checks.
Do not turn the prepared plan into expected native answers. Implementation of
retained nonleaf gradient ownership/Version behavior must use the actual
observations, while preserving existing numerical assertions and open scope.
