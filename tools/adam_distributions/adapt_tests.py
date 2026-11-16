#!/usr/bin/env python3
"""Bind frozen native tests to the combined profile without changing expectations.

Only internal state tags, executable basename, and build receipt field names
change. Every edit is inverted and compared to the original pinned file.
"""
import argparse
import hashlib
import json
from pathlib import Path

SHA = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('atomic-tools', 'partial-tools', 'adam-tools', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    atomic, partial, adam = a.atomic_tools, a.partial_tools, a.adam_tools
    image = [('tensor-distributions-session', 'tensor-adam-distributions-session')]
    fields = [("build['native_sha256']", "build['executable_sha256']"),
              ("build['source_sha256']", "build['runtime_source_sha256']")]
    plans = [
        (atomic / 'test_controls.py', 'atomic_controls.py',
         '908dff3b128a15d6dc4559bdde9483517f13696c981122e5cd46fbe1bd9990f3',
         [("b'HTDIST01'", "b'HTADIST1'"), ("b'HTODIST1'", "b'HTOADIS1'")]),
        (partial / 'test_controls.py', 'partial_controls.py',
         '8ecd64a172405b21faa614f054284ebc2c8bdd6659050a8bd4580fc3515d22fa', image),
        (partial / 'test_entry.py', 'owned_entry.py',
         '7438d562ffbcf7bd3e52ffbb13194ab84167cc57e5ff80c79b2cbf2c1124b19b', image),
        (partial / 'test_handoffs.py', 'typed_handoffs.py',
         'a7b063f75696f1049360d0e3ed3e09bbf731e988e89d17f2ad3de454fe137a3e',
         image + [("b'HTDIST02'", "b'HTADIST1'")]),
        (partial / 'reference.py', 'reference.py',
         'a942a26c3f0a3286ec32e361aeff201574cadc0c52a855726ba52fe65f17b8b8', []),
        (adam / 'test_native.py', 'adam_controls.py',
         'ae4c85edbbff520590b1a1174bcc673a200dcade121d183521261433c4cf9982', []),
        (adam / 'test_guards.py', 'adam_guards.py',
         '6a88abc2eb91962f2d829bf8cae41b21cfe0acd38cdaa8cbc8a14e4f95faa5be', fields),
        (adam / 'test_step_counter.py', 'adam_step_counter.py',
         '19f653fa769f91bb2a208f4bb242d45ac0dbaafb946d712594182c75893ed104', fields),
    ]
    rows = []
    for source, name, digest, edits in plans:
        original = source.read_bytes()
        assert SHA(original) == digest, source
        text = original.decode()
        changes = []
        for before, after in edits:
            count = text.count(before)
            assert count > 0 and after not in text, (source, before, after)
            text = text.replace(before, after)
            changes.append({'before': before, 'after': after, 'count': count})
        inverse = text
        for change in reversed(changes):
            assert inverse.count(change['after']) == change['count']
            inverse = inverse.replace(change['after'], change['before'])
        assert inverse.encode() == original
        (a.output / name).write_text(text)
        (a.output / ('original-' + name)).write_bytes(original)
        assert source.read_bytes() == original
        rows.append({'name': name, 'original_sha256': digest,
                     'adapted_sha256': SHA(text.encode()), 'inverse_byte_identical': True,
                     'exact_edits': changes})
    # Preserve the old script above: it really fails on the now-defined 86/87,
    # whose empty arguments are invalid Tensor handles (1), not unknown opcodes
    # (4). This distinct derivative changes exactly those new command expectations.
    original = (a.output / 'atomic_controls.py').read_bytes()
    edits = [
        ("for opcode in (86, 87, 88, 89):\n            rollback(session, [(opcode,)], 4)",
         "for opcode in (86, 87, 88, 89):\n            rollback(session, [(opcode,)], 1 if opcode in (86, 87) else 4)"),
        ("'unsupported_reserved_opcodes': 4,", "'unsupported_reserved_opcodes': 2, 'new_partial_empty_arguments_type_errors': 2,"),
    ]
    text = original.decode()
    for before, after in edits:
        assert text.count(before) == 1 and after not in text
        text = text.replace(before, after)
    inverse = text
    for before, after in reversed(edits):
        assert inverse.count(after) == 1
        inverse = inverse.replace(after, before)
    assert inverse.encode() == original
    (a.output / 'atomic_controls_extended.py').write_text(text)
    extension = {
        'name': 'atomic_controls_extended.py', 'predecessor': 'atomic_controls.py',
        'predecessor_sha256': SHA(original), 'adapted_sha256': SHA(text.encode()),
        'inverse_byte_identical': True,
        'changed_input_packets': False, 'changed_expected_values': False,
        'changed_error_expectations': [{'opcode': 86, 'arguments': [], 'old': 4, 'new': 1},
                                       {'opcode': 87, 'arguments': [], 'old': 4, 'new': 1}],
        'reason': 'these opcodes are newly defined by frozen partial01; empty arguments are invalid typed Tensor handles',
        'old_script_retained': True, 'old_script_current_result_expected': 'FAIL',
        'exact_edits': [{'before': before, 'after': after, 'count': 1} for before, after in edits],
    }
    report = {
        'schema': 'heptatorch.adam-distributions.test-binding.v1', 'status': 'PASS',
        'scope': 'eight primary bindings change only profile tags, executable basename, and receipt field names; a separate explicitly named derivative changes two newly-defined command errors',
        'changed_input_packets': False, 'changed_expected_values': False,
        'changed_tolerances': False, 'primary_bindings_changed_error_or_commit_expectations': False,
        'explicit_derivative_changes_only_opcode86_87_empty_argument_errors': True,
        'scripts': rows, 'explicit_new_command_admission_derivative': extension,
    }
    (a.output / 'adaptation.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'scripts': len(rows)}))


if __name__ == '__main__':
    main()
