"""Observe signbit results and reuse independently defined owner predicates."""

from numeric_predicates.native import (
    Live, Replay, Memory, load_wire, dump, Observer, require,
    compare_case, allocation_control, transaction_control,
)
from .cases import OPERATIONS


def packet_controls(n, row):
    generator, rng = n.begin()
    source, inputs = n.setup(row['spec'])
    original = n.tensor(source)[0]
    valid = [126, source, 0, 0, 0, 0, 0, 0]
    controls = 0
    for position in range(2, 8):
        bad = valid.copy()
        bad[position] = 1
        status, _, _, _ = n.issue(n.wire.packet([bad]))
        n.check('signbit reserved word rejected', status, 7)
        controls += 1
    for handle in (0, 1, generator):
        bad = valid.copy()
        bad[1] = handle
        status, _, _, _ = n.issue(n.wire.packet([bad]))
        n.check('signbit invalid public handle rejected', status, 1)
        controls += 1
    bad = valid.copy()
    bad[7] = 1
    status, _, _, _ = n.issue(n.wire.packet([valid, bad]))
    n.check('signbit allocation prefix rolls back on malformed suffix', status, 7)
    controls += 1
    for opcode in (125, 127):
        status, _, _, _ = n.issue(n.wire.packet([(opcode, source)]))
        n.check('unassigned opcode remains rejected', status, 4)
        controls += 1
    n.check('all signbit packet failures preserve input', n.tensor(source)[0], original)
    for token in inputs:
        n.one(41, token)
    status, _, _, _ = n.issue(n.wire.packet([(126, source)]))
    n.check('signbit stale public handle rejected', status, 1)
    controls += 1
    n.end(generator, rng)
    n.check('all signbit packet control recipes executed', controls, 13)
    return dict(controls=controls, complete_owner_rollback=True,
                original_unknown125_rejected=True, unknown127_rejected=True)


def recipes(reference_data):
    rows = reference_data['report']['rows']
    result = [('case-%03d' % row['index'], compare_case, (row,)) for row in rows]
    dense = [row for row in rows if row['spec']['name'].endswith('/special-bits')]
    require(len(dense) == 6, 'all six public dtype transaction fixtures')
    result.append(('packet-controls', packet_controls, (dense[0],)))
    result.extend(('transaction-' + row['spec']['dtype'], transaction_control, (row,)) for row in dense)
    allocation = [row for row in rows if row['spec']['name'].endswith('/all-65536-encodings')]
    require(len(allocation) == 2, 'both reduced precision exhaustive allocation fixtures')
    result.extend(('allocation-' + row['spec']['dtype'], allocation_control, (row,)) for row in allocation)
    return result
