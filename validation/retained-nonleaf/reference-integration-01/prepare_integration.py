#!/usr/bin/env python3
"""Build and check an isolated, reference-only retain_grad integration proposal."""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import yaml


BASE = '9a20eadbe20a801c8b26727ae369e0ee0832e423'
PROBE_SHA = '7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba'
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PREPARED_SHA = 'fff892d371d90f3d64b498a0f8bddb908a153773492d9ce715c0b239dc292bab'
REVIEW_SHA = '08d979da4564aaca19bdeaf9a229e4b56602d3e8d3dbb2634ac61dbc7871ef68'
ORIGINAL_MANIFEST = '4610b7d1412c302e8d87cb184edc7edd8edb12db86c7d6d533a73d09c60faf67'
WORKFLOW = '.github/workflows/backward-observations.yml'
PROBE = 'tests/pytorch_retained_nonleaf_probe.py'
VALIDATION = 'validation/retained-nonleaf-reference'


STEPS = r'''      - name: Verify frozen retained nonleaf source and provenance inputs
        run: |
          echo '7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba  tests/pytorch_retained_nonleaf_probe.py' | sha256sum -c -
          echo 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae  tests/pytorch_oracle.py' | sha256sum -c -
          echo 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944  compatibility/pytorch-oracle-lock.json' | sha256sum -c -
          echo 'fff892d371d90f3d64b498a0f8bddb908a153773492d9ce715c0b239dc292bab  validation/retained-nonleaf-reference/prepared.json' | sha256sum -c -
          echo '08d979da4564aaca19bdeaf9a229e4b56602d3e8d3dbb2634ac61dbc7871ef68  validation/retained-nonleaf-reference/independent-probe-review.json' | sha256sum -c -
          python3 -I -S -B -c "import ast,pathlib; ast.parse(pathlib.Path('tests/pytorch_retained_nonleaf_probe.py').read_bytes())"
      - name: Prepare retained nonleaf gradient and optimizer observation plan
        run: |
          python3 -I -S -B tests/pytorch_retained_nonleaf_probe.py \
            --reference tests/pytorch_oracle.py --lock compatibility/pytorch-oracle-lock.json \
            --repo . --output build/torch-retained-nonleaf-prepared.json --prepare-only
          cmp build/torch-retained-nonleaf-prepared.json validation/retained-nonleaf-reference/prepared.json
          python3 -I -S -B tools/emit_ci_report.py build/torch-retained-nonleaf-prepared.json
      - name: Observe retained nonleaf gradients optimizer state and failure lifetimes
        run: |
          python3 tests/pytorch_retained_nonleaf_probe.py \
            --reference tests/pytorch_oracle.py --lock compatibility/pytorch-oracle-lock.json \
            --repo . --output build/torch-retained-nonleaf.json
      - name: Preserve complete retained nonleaf observations including failures
        if: always()
        run: |
          test -f build/torch-retained-nonleaf.json
          python3 -I -S -B tools/emit_ci_report.py build/torch-retained-nonleaf.json
'''


DOC = '''# Retained nonleaf gradients: pinned reference preparation

This change adds a reference-only observation plan for `Tensor.retain_grad()`.
The probe is frozen at SHA256
`7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba`.
Local preparation has run; real PyTorch execution and native compatibility are
pending. It does not add a native opcode or change a production runtime.

The original 24-case Optimizer observation only demonstrated that construction
accepts a retained nonleaf parameter. It did not execute its retained backward,
optimizer step, or lifetime. The new plan preserves that original evidence and
adds eight families with twelve variants:

| Family | Variants | Observations |
| --- | ---: | --- |
| Admission, leaf no-op, idempotence | 1 | Requires-grad rejection, retaining twice, nonleaf Optimizer admission |
| Gradient replacement and aliases | 1 | First materialization, repeated out-of-place replacement, clear, old Saved gradient |
| Strided incoming and assigned gradient | 1 | Logical words, actual retained layout and identity after accumulation |
| Empty retained gradient | 1 | Defined empty gradient versus None and repeated replacement |
| Retained Optimizer parameter | 3 | SGD, Adam and AdamW lazy state, zero/None skip and resume |
| Own Saved-version failure | 1 | Retained gradient effects before the node's failing backward |
| Own released-Saved failure | 1 | Repeated default backward and retained gradient effects |
| Weak lifetime and Optimizer ownership | 3 | Public drop, held gradient, privately owned parameter and final release |

Every snapshot records exact logical binary32 words, shape, strides, storage
offset/size, Version counter and opaque Tensor/Storage identity. None remains
distinct from defined zero. Python wrapper weak lifetime and Storage weak
lifetime are separate observations; neither is an intrusive reference-count
oracle. The observer returns primitive snapshots and installs no custom hooks.
Held seeds, gradients, and the guard using an old gradient are explicit inputs.

## Reproduce preparation

```sh
python3 -I -S -B tests/pytorch_retained_nonleaf_probe.py \\
  --reference tests/pytorch_oracle.py \\
  --lock compatibility/pytorch-oracle-lock.json \\
  --repo . --output build/torch-retained-nonleaf-prepared.json --prepare-only
cmp build/torch-retained-nonleaf-prepared.json \\
  validation/retained-nonleaf-reference/prepared.json
```

The result is `PREPARED_NOT_RUN`, with eight planned families, twelve variants,
zero executed cases and no Torch or native execution. The probe refuses changed
oracle or lock bytes. CI also verifies the frozen probe, preparation receipt and
independent source review before invoking it.

## Real reference execution and custody

The reference workflow installs the exact wheel from the existing oracle lock.
Its new execution command is the preparation command without `--prepare-only`,
using `--output build/torch-retained-nonleaf.json`. A real observation requires
all eleven original provenance checks, `OBSERVED`, eight rows with their fixed
variant counts, zero probe errors, and unchanged probe/oracle/lock bytes. No
native-derived expected values are used. Future Adam comparisons must retain
the existing declared numerical tolerance; identities, metadata, Versions and
None remain exact.

The four new workflow steps add two original-byte envelope reports:
`torch-retained-nonleaf-prepared.json` and `torch-retained-nonleaf.json`.
Against reference commit `9a20eadbe20a801c8b26727ae369e0ee0832e423`, declared
reference-job steps increase from 41 to 45 and required envelopes from 24 to 26.
The existing collector discovers both paths without modification and exports
all available JSON files with exact lengths and hashes, including failures.
The native workflow and all previous reference steps remain unchanged.

Pinned primary sources are recorded in
`validation/retained-nonleaf-reference/source-custody.json`:
[retain_grad](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp),
[engine execution](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/engine.cpp),
and [Optimizer admission](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/optimizer.py).
Their source-level rationale does not substitute for the pending wheel
observations. The original independent source-only review is preserved at
`validation/retained-nonleaf-reference/independent-probe-review.json` without
claiming that it executed Torch.

This finite CPU binary32 first-order plan does not establish custom hooks,
higher-order differentiation, sparse gradients, other devices/dtypes, graph
rebasing, or full PyTorch compatibility. The full 2,590-schema scope remains.
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


def record(path, prefix):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(prefix)), 'bytes': len(raw), 'sha256': sha(raw)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'frozen-preparation', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    a.repository = a.repository.resolve()
    a.frozen_preparation = a.frozen_preparation.resolve()
    a.output = a.output.resolve()
    assert not a.output.exists(), a.output
    a.output.mkdir(parents=True)
    git = lambda *args, **kw: subprocess.check_output(['git', *args], cwd=a.repository, **kw)
    assert git('cat-file', '-t', BASE).strip() == b'commit'
    original_manifest = (a.frozen_preparation / 'manifest.json').read_bytes()
    assert sha(original_manifest) == ORIGINAL_MANIFEST
    for item in json.loads(original_manifest)['files']:
        raw = (a.frozen_preparation / item['path']).read_bytes()
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    shutil.copytree(a.frozen_preparation, a.output / 'original-preparation')
    overlay = a.output / 'overlay'
    stage = a.output / 'cli-check-worktree'
    before = git('show', BASE + ':' + WORKFLOW)
    assert sha(before) == '8236c182a03116be3ad56e95c0437f7c8a5d3112f8ea553d486f5707b046ded8'
    trigger = '"tests/pytorch_retained_nonleaf_probe.py", '
    old_trigger = '".github/workflows/backward-observations.yml"]'
    text = before.decode()
    assert text.count(old_trigger) == 1
    text = text.replace(old_trigger, trigger + old_trigger)
    anchor = '      - name: Index original reference reports and checkout identity\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, STEPS + anchor)
    after = text.encode()
    assert text.replace(STEPS, '').replace(trigger, '') == before.decode()
    files = {
        WORKFLOW: after,
        PROBE: (a.frozen_preparation / 'pytorch_retained_nonleaf_probe.py').read_bytes(),
        'docs/retained-nonleaf-reference.md': DOC.encode(),
        VALIDATION + '/prepared.json': (a.frozen_preparation / 'prepared-01.json').read_bytes(),
        VALIDATION + '/independent-probe-review.json': (a.frozen_preparation / 'independent-probe-review.json').read_bytes(),
        VALIDATION + '/source-custody.json': (a.frozen_preparation / 'source-custody.json').read_bytes(),
    }
    assert sha(files[PROBE]) == PROBE_SHA
    assert sha(files[VALIDATION + '/prepared.json']) == PREPARED_SHA
    assert sha(files[VALIDATION + '/independent-probe-review.json']) == REVIEW_SHA
    ast.parse(files[PROBE])
    for path, raw in files.items():
        write(overlay / path, raw)
        write(stage / path, raw)
    support_paths = ('tests/pytorch_oracle.py', 'compatibility/pytorch-oracle-lock.json',
                     'tools/emit_ci_report.py', 'tools/collect_ci_reports.py')
    base_inputs = {WORKFLOW: {'bytes': len(before), 'sha256': sha(before)}}
    for path in support_paths:
        raw = git('show', BASE + ':' + path)
        write(stage / path, raw)
        base_inputs[path] = {'bytes': len(raw), 'sha256': sha(raw)}
    assert sha((stage / 'tests/pytorch_oracle.py').read_bytes()) == SUPPORT_SHA
    assert sha((stage / 'compatibility/pytorch-oracle-lock.json').read_bytes()) == LOCK_SHA
    old_steps = yaml.safe_load(before)['jobs']['reference']['steps']
    new_steps = yaml.safe_load(after)['jobs']['reference']['steps']
    added = yaml.safe_load('steps:\n' + '\n'.join(x[6:] for x in STEPS.rstrip().splitlines()) + '\n')['steps']
    assert len(added) == 4 and len(old_steps) == 41 and len(new_steps) == 45
    assert new_steps[:-6] + new_steps[-2:] == old_steps
    assert new_steps[-6:-2] == added
    envelope = lambda raw: sorted(set(re.findall(rb'tools/emit_ci_report\.py (build/[A-Za-z0-9_.-]+\.json)(?:\s|$)', raw)))
    old_reports, new_reports = envelope(before), envelope(after)
    assert len(old_reports) == 24 and len(new_reports) == 26
    assert set(new_reports) - set(old_reports) == {
        b'build/torch-retained-nonleaf-prepared.json', b'build/torch-retained-nonleaf.json'}
    executions = []

    def execute(label, argv, cwd, expected):
        result = subprocess.run(argv, cwd=cwd, capture_output=True)
        write(a.output / 'executions' / (label + '.stdout'), result.stdout)
        write(a.output / 'executions' / (label + '.stderr'), result.stderr)
        assert result.returncode == expected, (label, result.returncode, result.stderr[-2000:])
        row = {'name': label, 'argv': argv, 'cwd': str(cwd), 'exit_code': result.returncode,
               'stdout_sha256': sha(result.stdout), 'stderr_sha256': sha(result.stderr)}
        executions.append(row)
        return result

    for index in (0, 1):
        execute('workflow-' + str(index + 1), ['bash', '-e', '-o', 'pipefail', '-c', added[index]['run']], stage, 0)
    prepared_raw = (stage / 'build/torch-retained-nonleaf-prepared.json').read_bytes()
    assert prepared_raw == files[VALIDATION + '/prepared.json']
    prepared = json.loads(prepared_raw)
    assert (prepared['status'], prepared['expected_cases'], prepared['expected_variants'],
            prepared['cases'], prepared['torch_executed'], prepared['native_executed']) == (
                'PREPARED_NOT_RUN', 8, 12, 0, False, False)
    for altered in ('reference', 'lock'):
        source = 'tests/pytorch_oracle.py' if altered == 'reference' else 'compatibility/pytorch-oracle-lock.json'
        target = a.output / 'negative-inputs' / ('changed-' + Path(source).name)
        write(target, (stage / source).read_bytes() + b'\n')
        out = a.output / 'negative-inputs' / (altered + '-must-not-exist.json')
        command = [sys.executable, '-I', '-S', '-B', PROBE,
                   '--reference', str(target) if altered == 'reference' else 'tests/pytorch_oracle.py',
                   '--lock', str(target) if altered == 'lock' else 'compatibility/pytorch-oracle-lock.json',
                   '--repo', '.', '--output', str(out), '--prepare-only']
        refused = execute('reject-changed-' + altered, command, stage, 1)
        assert b'AssertionError' in refused.stderr and not out.exists()
    negative_stage = a.output / 'negative-probe-worktree'
    for path in (PROBE, 'tests/pytorch_oracle.py', 'compatibility/pytorch-oracle-lock.json',
                 VALIDATION + '/prepared.json', VALIDATION + '/independent-probe-review.json'):
        raw = (stage / path).read_bytes() + (b'\n' if path == PROBE else b'')
        write(negative_stage / path, raw)
    refused = execute('reject-changed-probe', ['bash', '-e', '-o', 'pipefail', '-c', added[0]['run']], negative_stage, 1)
    assert b'FAILED' in refused.stdout and not (negative_stage / 'build').exists()

    # A private index validates the entire proposed diff against the real base.
    # It never touches any worktree's normal index and never creates a commit.
    index = a.output / 'private-index'
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    subprocess.run(['git', 'read-tree', BASE], cwd=a.repository, env=env, check=True)
    candidate_patch = []
    for path, raw in sorted(files.items()):
        old = before if path == WORKFLOW else b''
        candidate_patch.append('diff --git a/' + path + ' b/' + path + '\n')
        if path != WORKFLOW:
            candidate_patch.append('new file mode 100644\n')
        candidate_patch.extend(difflib.unified_diff(old.decode().splitlines(True), raw.decode().splitlines(True),
                              fromfile='a/' + path if old else '/dev/null', tofile='b/' + path))
    initial_patch = a.output / 'proposal-input.patch'
    write(initial_patch, ''.join(candidate_patch))
    subprocess.run(['git', 'apply', '--cached', '--check', str(initial_patch)], cwd=a.repository, env=env, check=True)
    subprocess.run(['git', 'apply', '--cached', str(initial_patch)], cwd=a.repository, env=env, check=True)
    exact_patch = git('diff', '--cached', '--binary', '--full-index', '--no-ext-diff', BASE, env=env)
    write(a.output / 'retained-nonleaf-reference.patch', exact_patch)
    subprocess.run(['git', 'diff', '--cached', '--check', BASE], cwd=a.repository, env=env, check=True)
    changed = git('diff', '--cached', '--name-only', BASE, env=env).decode().splitlines()
    assert changed == sorted(files)
    for path, raw in files.items():
        assert git('show', ':' + path, env=env) == raw
        assert (stage / path).read_bytes() == raw
    native_before = git('show', BASE + ':.github/workflows/native.yml')
    assert git('show', ':.github/workflows/native.yml', env=env) == native_before
    for path in support_paths:
        assert git('show', ':' + path, env=env) == (stage / path).read_bytes()
    for item in json.loads(original_manifest)['files']:
        raw = (a.frozen_preparation / item['path']).read_bytes()
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    write(a.output / 'workflow-before.yml', before)
    write(a.output / 'added-reference-steps.yml', STEPS)
    report = {
        'schema': 'heptatorch.retained-nonleaf-reference-integration.v1',
        'status': 'PASS_PREPARATION_AND_PATCH_CHECKS_NOT_RUN', 'base_commit': BASE,
        'base_tree': git('rev-parse', BASE + '^{tree}').decode().strip(),
        'probe_sha256': PROBE_SHA, 'probe_bytes_changed': False,
        'original_preparation_manifest_sha256': ORIGINAL_MANIFEST,
        'original_independent_source_review_sha256': REVIEW_SHA,
        'source_custody': base_inputs, 'workflow_before_sha256': sha(before),
        'workflow_after_sha256': sha(after), 'exact_workflow_inverse': True,
        'old_steps_unchanged': True, 'reference_steps_before': 41, 'reference_steps_after': 45,
        'required_envelopes_before': 24, 'required_envelopes_after': 26,
        'added_required_envelopes': sorted((x.decode() for x in set(new_reports) - set(old_reports))),
        'collector_unchanged': True, 'native_workflow_unchanged': True,
        'native_workflow_sha256': sha(native_before), 'changed_repository_paths': changed,
        'patch_checked_and_applied_to_private_index': True, 'normal_index_modified': False,
        'commit_created': False, 'remote_actions': False,
        'prepare_cli_executed_from_final_yaml': True, 'verification_cli_executed_from_final_yaml': True,
        'prepare_exact_frozen_bytes': True, 'syntax_parsed': True,
        'wrong_reference_refused_before_execution': True, 'wrong_lock_refused_before_execution': True,
        'wrong_probe_refused_by_workflow': True, 'expected_cases': 8, 'expected_variants': 12,
        'new_Torch_executed': False, 'native_executed': False, 'observed_semantic_cases': 0,
        'executions': executions, 'patch_sha256': sha(exact_patch),
        'full_rewrite_complete': False, 'pytorch_schema_denominator': 2590,
    }
    write(a.output / 'integration-report.json', json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'base_commit', 'patch_sha256',
                        'workflow_after_sha256', 'reference_steps_after', 'required_envelopes_after')}))


if __name__ == '__main__':
    main()
