#!/usr/bin/env python3
"""Read-only, source-bound retain ownership and observer review receipt.

Mechanical checks below supplement the separately recorded manual control-flow
review. This does not execute producer Python, rerun Torch, or replace native
module admission and execution.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

NEW = '666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b'
OLD = '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
IMAGE = '7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5'
PEER = 'c925e33872dc87ead29ecc2125b1799763a87b13c8ae2eba85e966b4202f1d8e'
CHANGED = {'tgrad', 'zero_grad', 'state_header', 'init_state', 'tensor_row_release',
           'publish_tensor', 'so_state_init', 'so_state_header', 'so_audit_records',
           'backward_execute', 'partial_execute', 'zero_grad_none', 'optimizer_fill_group',
           'optimizer_audit_entry', 'gradient_object_command', 'gradient_gc_enqueue',
           'owned_adam_fill_group', 'owned_adam_audit_entry'}
FUNCTION = re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)', re.M | re.S)
REVIEW_INPUTS = {
    'source-map.json': 'a330ec49d36f7b54f77d19675df86174cebb77d9fb7e7e4843be7fa3a71150a9',
    'originals.py': '561d74b9cd5dabd2916ccb64ffcb7c9a7adab6b97bca132a0e5f5548169562d8',
    'overlay-authoring.py': '2487b959c406f4ee0687034e4121b2d17f5199951da5750b666ea36a5b6b7859',
    'authoring-custody.json': '93eba85ab97da186e1d6a04c18ff0ecf0e5cb987e8d7112b47bc2e8830b57d2d',
    'build.py': '9f5c25208518e9c2a7789e2db5bd44111a901bab5ba45a4c91b7daa7ff9ea119',
    'original-authoring-custody.json': '4f522958beb9d11df6e76acb16aa03f5fba459483378bb718896ebbb3672ebf7',
    'retained.py': 'f6f24385543890a89667be804f4a2fda5d8bc15ad257caca55935b447fb0dacd',
    'layout.py': '58b69bc880cc4cc2f559dcce30c6fd394734266110942b01fa5d683915d50170',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def functions(raw):
    rows = FUNCTION.findall(raw)
    result = {name.decode(): body for body, name in rows}
    assert len(rows) == len(result)
    return result


def python_functions(raw):
    return {node.name: node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}


def normalize(node):
    return ast.dump(node, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'previous-build', 'peer-runner', 'review-inputs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read(path, expected=None):
        path = path.resolve()
        data = path.read_bytes()
        if expected:
            assert sha(data) == expected, path
        inputs[path] = data
        return data

    read(Path(__file__))
    new_raw = read(a.build / 'runtime.hepta', NEW)
    old_raw = read(a.previous_build / 'runtime.hepta', OLD)
    read(a.build / 'tensor-retained-session', IMAGE)
    peer_raw = read(a.peer_runner, PEER)
    old, new = functions(old_raw), functions(new_raw)
    assert len(old) == 428 and len(new) == 446
    assert {name for name in old if new.get(name) != old[name]} == CHANGED
    assert sum(new.get(name) == body for name, body in old.items()) == 410
    assert not (old.keys() - new.keys())
    build = json.loads(read(a.build / 'build.json'))
    review_inputs = {name: read(a.review_inputs / name, expected) for name, expected in REVIEW_INPUTS.items()}
    mapping = json.loads(review_inputs['source-map.json'])
    assert build['source_sha256'] == NEW and build['native_sha256'] == IMAGE
    assert build['functions'] == len(new) and build['module_count'] == len(build['units']) == 9
    assert len(mapping) == len(new) and {r['name'] for r in mapping} == new.keys()
    for row in mapping:
        name = row['name']
        assert row['sha256'] == sha(new[name])
        assert row['unchanged_parent'] == (new[name] == old.get(name))
        assert row['parent_sha256'] == (sha(old[name]) if name in old else None)
    seen = {}
    units = []
    for row in build['units']:
        raw = read(a.build / (row['name'] + '.hepta'), row['source_sha256'])
        read(a.build / (row['name'] + '.o'), row['object_sha256'])
        bodies = functions(raw)
        assert len(bodies) == row['functions']
        assert raw.count(b' = ') == row['instructions']
        assert len(re.findall(rb'^block ', raw, re.M)) == row['blocks']
        if row['name'].startswith('retained_gradient_'):
            for name, body in bodies.items():
                assert name in new and body == new[name] and name not in seen
                seen[name] = row['name']
        units.append({'name': row['name'], 'source_sha256': sha(raw), 'object_sha256': row['object_sha256']})
    assert set(seen) == set(new)
    # The step/close owner ABI and RNG03 numerical module are unchanged bytes.
    for name in ('owners.hepta', 'rng03.hepta'):
        assert read(a.build / name) == read(a.previous_build / name)
    aliases = {'gradient_gc_enqueue': 'retained_gc_enqueue_original',
               'tensor_row_release': 'retained_row_release_original',
               'publish_tensor': 'retained_publish_original'}
    alias_rows = []
    for original, alias in aliases.items():
        restored = new[alias].replace(('fn ' + alias + '(').encode(), ('fn ' + original + '(').encode(), 1)
        assert restored == old[original]
        alias_rows.append({'old': original, 'alias': alias, 'old_body_sha256': sha(old[original]),
                           'new_alias_body_sha256': sha(new[alias]), 'name_only_inverse_exact': True})
    tag_rows = []
    for name, before, after in (
        ('init_state', b'HTGAD002', b'HTRET001'), ('state_header', b'HTGAD002', b'HTRET001'),
        ('so_state_init', b'HTOGAD02', b'HTORET01'), ('so_state_header', b'HTOGAD02', b'HTORET01'),
    ):
        literal = lambda b: b'{"value":' + str(int.from_bytes(b, 'little')).encode() + b'}'
        assert new[name].count(literal(after)) == 1
        assert new[name].replace(literal(after), literal(before)) == old[name]
        tag_rows.append({'function': name, 'old_tag': before.decode(), 'new_tag': after.decode(),
                         'single_literal_inverse_exact': True})

    original_raw = review_inputs['originals.py']
    overlay_raw = review_inputs['overlay-authoring.py']
    custody = json.loads(review_inputs['authoring-custody.json'])
    originals, overlay = python_functions(original_raw), python_functions(overlay_raw)
    assert set(custody['changes']) == set(originals) and len(originals) == 11
    assert custody['all_selected_originals_reemit_byte_identically'] is True and custody['parent_sha256'] == OLD
    author_rows = []
    for name, change in custody['changes'].items():
        assert normalize(python_functions(change['before'])[name]) == normalize(originals[name])
        assert normalize(python_functions(change['after'])[name]) == normalize(overlay[name])
        author_rows.append({'name': name, 'before_ast_sha256': sha(normalize(originals[name]).encode()),
                            'after_ast_sha256': sha(normalize(overlay[name]).encode()),
                            'before_matches_pinned_selected_original': True,
                            'after_matches_emitted_overlay_authoring': True,
                            'old_HIA_sha256': sha(old[name]), 'new_HIA_sha256': sha(new[name])})
    for filename in ('retained.py', 'layout.py'):
        selected = python_functions(review_inputs[filename])
        assert all(normalize(node) == normalize(overlay[name]) for name, node in selected.items())

    def where(name):
        node = overlay[name]
        return {'path': 'overlay-authoring.py', 'function': name, 'first_line': node.lineno,
                'last_line': node.end_lineno, 'normalized_ast_sha256': sha(normalize(node).encode())}

    findings = [
        {'topic': 'Weak self and committed Tensor flags',
         'finding': 'Enabling writes only T-row+56=1 after live public Tensor/requires/nonleaf checks. It does not increment node208/312/400. New descriptors start from zero row flags; same-Tensor republication restores the previous 0/1 flag.',
         'functions': [where(n) for n in ('retained_enable', 'retained_row', 'publish_tensor')]},
        {'topic': 'Real temporary reservation root',
         'finding': 'A fresh descriptor owns Storage/Version, node_retain precedes flag2, and public200 is cleared. GC/Tensor-row wrappers preserve that private reservation. Release clears flag2 before node_release can enqueue collection.',
         'functions': [where(n) for n in ('retained_reserve_one', 'retained_release_one', 'gradient_gc_enqueue', 'tensor_row_release')]},
        {'topic': 'Reservations are indexed by old nodes but contain fresh node indices',
         'finding': 'The table has limit+1 u64 entries inside a real private Hold. Existing descriptor admission limits node indices to below65536, so the new table size is at most524288 bytes. Cleanup iterates table entries, releases every contained fresh node even above output, then collects from node1 through the live graph range.',
         'functions': [where(n) for n in ('retained_reserve', 'retained_release_reservations')]},
        {'topic': '58 exception order and structured cleanup',
         'finding': 'Reserve occurs before execution. retained_apply precedes partial_check_node. completed increments only after successful node application and Saved release. On structured error, status is saved and cleared, marks/incoming scratch and all reservations are released, then the queue Hold is released before the exception payload is returned.',
         'functions': [where('partial_execute')]},
        {'topic': 'Fresh gradient ownership',
         'finding': 'The hook fills a fresh descriptor, binds it through the unchanged gradient_assign ownership path, clears its table entry and releases only the reservation root. Existing gradient storage/version is read without in-place mutation. Undefined G uses contiguous fresh layout; replacement layout uses actual incoming metadata.',
         'functions': [where('retained_apply'), where('retained_sum_layout')]},
        {'topic': 'Atomic entries and allocation refusal',
         'finding': '7/29/57 retain the old owner transaction. All new admission/reservation failures return through ordinary rollback; successful atomic paths also release all reservations and private Holds. The unchanged owners module performs the actual clone/commit/drop lifecycle.',
         'functions': [where('backward_execute')]},
        {'topic': 'Committed audit and Optimizer roles',
         'finding': 'Committed T rows permit only flags0/1, rejecting transient2/3. Nonleaf admission is broadened only through retained_parameter_blocker; Optimizer private roots and gradient binding reference counters remain real existing roles, not replaced by the retain flag.',
         'functions': [where(n) for n in ('so_audit_records', 'retained_parameter_blocker', 'optimizer_audit_entry', 'owned_adam_audit_entry')]},
    ]
    # Verify the specific text/order reviewed, while keeping the review's
    # assurance narrower than a general semantic proof.
    partial = ast.unparse(overlay['partial_execute'])
    assert partial.index('retained_apply(') < partial.index('partial_check_node(')
    assert partial.index('release_backward_scratch(') < partial.index('retained_release_reservations(') < partial.index('hold_release(')
    release = ast.unparse(overlay['retained_release_one'])
    assert release.index('so_uput(w, row + 56, 0)') < release.index('node_release(')
    peer_tree = ast.parse(peer_raw)
    replay = next(n for n in peer_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Replay')
    methods = {n.name: n for n in replay.body if isinstance(n, ast.FunctionDef)}
    observer_reviews = [
        ('gradient', 'Reads current G public flag before91; temporary publication is closed only when it was not already public. Explicit held aliases are retained.'),
        ('optimizer_snapshot', 'Uses76 public-token inventory before77. Fresh state getter tokens are immediately closed; already-public state aliases remain open.'),
        ('case_private', 'The temporary75 parameter publication is closed before78 drops the actual private Optimizer root. Temporary xG/seed public tokens are closed before lifecycle observations.'),
        ('expired', 'A successful weak43 lock creates a real Hold that is immediately released by45; expired6 checks do not create a Hold.'),
        ('case_weak', 'The q public root is explicitly dropped before weak observation/backward. Snapshots do not leave a hidden G token, except the intentionally held old-gradient recipe.'),
        ('weak_snapshot', 'Only Storage expiry participates in equality assertions; Python wrapper weakrefs are retained as diagnostic fields.'),
    ]
    observer_rows = []
    for name, finding in observer_reviews:
        node = methods[name]
        observer_rows.append({'method': name, 'finding': finding, 'first_line': node.lineno,
                              'last_line': node.end_lineno, 'normalized_ast_sha256': sha(normalize(node).encode())})
    assert all(path.read_bytes() == data for path, data in inputs.items())
    report = {'schema': 'heptatorch.retained-gradient-independent-source-review.v1', 'status': 'PASS',
              'review_kind': 'Independent read-only source and observer-lifetime review; no new native or Torch run.',
              'source_sha256': NEW, 'native_sha256': IMAGE, 'parent_source_sha256': OLD,
              'peer_runner_sha256': PEER, 'source_functions': 446, 'parent_functions': 428,
              'unchanged_parent_bodies': 410, 'changed_parent_bodies': sorted(CHANGED),
              'added_functions': sorted(new.keys() - old.keys()), 'all446_core_bodies_in_exactly_one_actual_unit': True,
              'units': units, 'aliases': alias_rows, 'tags': tag_rows, 'selected_authoring_comparisons': author_rows,
              'producer_original_reemit_receipt_observed': True, 'original_authoring_not_reexecuted_by_this_review': True,
              'owner_and_rng_modules_byte_identical_to_parent': True,
              'findings': findings, 'peer_observer_lifetime_review': observer_rows,
              'existing_negative_contract_change': {'opcodes': [93, 94], 'old_empty_args_status': 4,
                  'new_empty_args_status': 1, 'old_suite_not_relabelled_unchanged_PASS': True},
              'limits': ['Only checked producer/wire-created owner states are covered, not arbitrary foreign fabricated owner metadata.',
                         'Per-function authoring checks are not whole-aggregate Core admission; actual modules still pass native checks and whole-link stack admission.',
                         'This receipt is not a general proof of arbitrary code or every retained-gradient input.',
                         'Actual12 replay and the separate8 resource plus2 typed-handoff executions retain separate evidence/counts.'],
              'inputs': [{'path': str(path), 'bytes': len(data), 'sha256': sha(data)} for path, data in inputs.items()],
              'all_inputs_unchanged': True, 'native_executed_by_this_review': False, 'Torch_executed_by_this_review': False,
              'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'executed-source-review.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': report['status'], 'source_sha256': NEW, 'findings': len(findings),
                      'peer_observer_methods_reviewed': len(observer_rows),
                      'report_sha256': sha((out / 'report.json').read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
