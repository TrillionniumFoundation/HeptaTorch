"""Reconstruct the exact pre-Half-logical source from the current source.

The three admitted HIA edits extend Bool reductions to F16 and BF16.  This
byte inverse supports existing source-identity checks; it neither executes
an earlier image nor transfers an earlier runtime result to this image.
"""

import hashlib
from pathlib import Path
import re


SOURCE = 'bf3641b1a9061c6f92b8cbeef75adfac89e82ee16d32a9f650cb938d9e918386'
PARENT_SOURCE = '51820849f7becec549be15babace00f00687601ae7f53bcee0e636ca02f721f9'
FUNCTIONS = 736
CHANGED = ('si_dispatch_sum', 'logical_plan', 'logical_truth')
EDITS = (
    ('si_dispatch_sum',
     b'  v114:u64 = const() {"value":4}\n',
     b'  v114:u64 = const() {"value":6}\n'),
    ('logical_plan',
     b'  v6:u64 = const() {"value":4}\n',
     b'  v6:u64 = const() {"value":6}\n'),
    ('logical_truth',
     b'  v12:bool = compare.ne(v11,v2)\n',
     b'  v18:u64 = const() {"value":5}\n'
     b'  v19:bool = compare.eq(v1,v18)\n'
     b'  v20:u64 = const() {"value":6}\n'
     b'  v21:bool = compare.eq(v1,v20)\n'
     b'  v22:bool = bool.or(v19,v21)\n'
     b'  v23:u64 = const() {"value":32767}\n'
     b'  v24:u64 = bit.and(v11,v23)\n'
     b'  v25:u64 = select(v22,v24,v11)\n'
     b'  v12:bool = compare.ne(v25,v2)\n'),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(raw):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',
                              raw, re.M | re.S))
    bodies = {match[1].decode('ascii'): match[0] for match in matches}
    require(len(matches) == len(bodies) == FUNCTIONS,
            'exact complete Half-logical function census')
    require(raw == raw.split(b'\n', 1)[0] + b'\n' + b''.join(bodies.values()),
            'complete canonical source order and bytes')
    return bodies


def restore_previous(source):
    """Restore every byte of the exact736 parent, accepting only this source."""
    require(isinstance(source, bytes) and sha(source) == SOURCE,
            'only the exact admitted Half-logical source')
    actual = functions(source)
    restored = source
    for name, before, after in EDITS:
        body = actual[name]
        require(body.count(after) == 1 and restored.count(body) == 1,
                'unique complete admitted body and edit: ' + name)
        previous = body.replace(after, before, 1)
        require(previous != body and previous.count(before) == 1,
                'one actual source edit: ' + name)
        restored = restored.replace(body, previous, 1)
    require(sha(restored) == PARENT_SOURCE,
            'every restored byte must match the complete exact736 parent')
    parent = functions(restored)
    require(list(parent) == list(actual)
            and tuple(name for name in parent if parent[name] != actual[name]) == CHANGED,
            'only the three named function bodies change; order is retained')
    forward = restored
    for name, before, after in EDITS:
        require(forward.count(parent[name]) == 1, 'unique complete forward body: ' + name)
        forward = forward.replace(parent[name], parent[name].replace(before, after, 1), 1)
    require(forward == source, 'exact whole-source forward round trip')
    return restored


def proof(source):
    """Return the verified source change record without running any program."""
    restored = restore_previous(source)
    actual, parent = functions(source), functions(restored)
    return dict(
        schema='heptatorch.half-logical-to-storage736.complete-source-inverse.v1',
        source_sha256=SOURCE, parent_source_sha256=PARENT_SOURCE,
        functions=FUNCTIONS, parent_functions=FUNCTIONS,
        changed_functions=list(CHANGED), unchanged_parent_functions=733,
        changed_bodies={name: dict(before_sha256=sha(parent[name]),
                                  after_sha256=sha(actual[name])) for name in CHANGED},
        added_functions=[], removed_functions=[],
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False,
        original_cases_or_expected_values_modified=False,
        runtime_qualification_transferred=False,
    )


def restore_source(source, bind):
    """Compatible source-proof entry point for the current admission chain."""
    bind(Path(__file__).resolve())
    return restore_previous(source), proof(source)
