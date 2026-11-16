"""Actual checked owner handoff: format rejection, exact preservation, and close."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import struct

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
Q = lambda words: struct.pack('<' + 'Q' * len(words), *words)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'build-receipt', 'protocol', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    snapshots = {p.resolve(): p.read_bytes() for p in (Path(__file__), a.executable, a.build_receipt, a.protocol)}
    receipt = json.loads(snapshots[a.build_receipt.resolve()])
    assert receipt['status'] == 'PASS' and receipt['qualification_only'] is True
    assert SHA(snapshots[a.executable.resolve()]) == receipt['native_sha256']
    assert SHA(snapshots[a.protocol.resolve()]) == '5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec = importlib.util.spec_from_file_location('optimizer_state_wire', a.protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    report = {'schema': 'heptatorch.optimizer-owner-state-handoff-execution.v1', 'status': 'RUNNING',
              'qualification_only': True, 'native_executed': True, 'Torch_executed_here': False,
              'native_sha256': receipt['native_sha256'], 'old_source_sha256': receipt['old_core_sha256'],
              'new_source_sha256': receipt['new_core_sha256'], 'rows': [],
              'inputs_sha256': {str(p): SHA(raw) for p, raw in snapshots.items()}}
    def save():
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    for own, other, close in [(0, 1, 3), (1, 0, 2)]:
        name = 'old-owner-to-new-step' if own == 0 else 'new-owner-to-old-step'
        s = wire.Session(a.executable, directory=a.output / name, address_limit=64 << 20)
        def run(route, commands, status=0):
            actual, records = wire.parse(s.exchange(Q([route]) + wire.packet(commands)))
            assert actual == status and len(records) == (len(commands) if status == 0 else 0)
            assert s.poll() is None
            return records
        empty = s.owner()
        status, records = wire.parse(s.exchange(Q([own]) + b'bad'))
        assert status == 100 and records == [] and s.owner() == empty
        x = run(own, [wire.tensor((2,), [2, 3], 1)])[0]['value']
        one = run(own, [wire.full((), 1)])[0]['value']
        added = run(own, [(2, 1, x, one)])[0]['value']
        loss = run(own, [(4, 7, added)])[0]['value']
        run(own, [(58, loss, 0, 1)])
        assert run(own, [(11, x)])[0]['payload'] == struct.pack('<2f', 1, 1)
        has_optimizer = own == 1 or receipt['old_has_optimizer']
        optimizer = 0
        if has_optimizer:
            defaults = Q([wire.bits(.125), wire.bits(.5), 0, 0, 0, 0])
            groups = Q([1, 1, 0, 0, 0, 0, 0, 0, 0, x])
            optimizer = run(own, [(71, 0, wire.Payload(defaults), wire.Payload(groups))])[0]['value']
            run(own, [(72, optimizer)])
            momentum = run(own, [(77, optimizer, 0, 0, 0)])[0]['value']
            assert wire.meta(run(own, [(13, momentum)])[0])['version'] == 0
            run(own, [(90, x, 0)])
            assert run(own, [(59, x)])[0]['value'] == 1
        run(own, [(70, x)])
        assert run(own, [(59, x)])[0]['value'] == 0
        before = s.owner()
        tag = receipt['old_tag'] if own == 0 else receipt['new_tag']
        assert before['length'] > 0 and before['bytes'][64:72] == tag.encode()
        (a.output / name / 'before.owner.bin').write_bytes(before['bytes'])
        run(other, [(8, x)], 7)
        after = s.owner()
        assert after == before
        (a.output / name / 'after.owner.bin').write_bytes(after['bytes'])
        if has_optimizer:
            run(own, [(90, x, 1)])
            assert run(own, [(77, optimizer, 0, 0, 0)])[0]['value'] == momentum
            assert run(own, [(10, momentum)])[0]['payload'] == struct.pack('<2f', 1, 1)
        run(own, [(58, loss, 0, 1), (53,)])
        assert run(own, [(11, x)])[0]['payload'] == struct.pack('<2f', 1, 1)
        pointer = s.owner()['pointer']
        assert s.mapped(pointer)
        closed_status, closed_records = wire.parse(s.exchange(Q([close])))
        assert closed_status == 0 and closed_records == []
        assert s.owner() == {'pointer': 0, 'length': 0, 'bytes': b'', 'live': 1}
        assert not s.mapped(pointer) and s.poll() is None
        z = run(other, [wire.full((), 7, 1)])[0]['value']
        run(other, [(7, z)])
        assert run(other, [(11, z)])[0]['payload'] == struct.pack('<f', 1)
        wanted = receipt['new_tag'] if own == 0 else receipt['old_tag']
        assert s.owner()['bytes'][64:72] == wanted.encode()
        s.finish()
        report['rows'].append({'case': name, 'status': 'PASS', 'frames': s.frames,
            'old_route': own, 'wrong_route': other, 'cross_format_close_route': close,
            'rejection_status': 7, 'entire_owner_bytes_unchanged': before['length'],
            'owner_sha256_before': SHA(before['bytes']), 'owner_sha256_after': SHA(after['bytes']),
            'pointer_and_length_unchanged': True, 'optimizer_with_real_momentum': has_optimizer,
            'resumed_gradient': [1, 1], 'cross_format_close_unmapped_while_process_alive': True,
            'reinitialized_opposite_from_live_empty': True, 'all_responses_before_stdin_eof': True})
        save()
    for path, raw in snapshots.items():
        assert path.read_bytes() == raw
    report.update(status='PASS', cases=len(report['rows']), frames=sum(row['frames'] for row in report['rows']),
                  entire_owner_bytes_unchanged=sum(row['entire_owner_bytes_unchanged'] for row in report['rows']))
    save()
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'frames', 'entire_owner_bytes_unchanged')}))


if __name__ == '__main__':
    main()
