"""One independent native ownership control: grow, reset, republish, rollback.

No Torch result is claimed here. Fixed dyadic values check that native private
Tensor/state roots survive real owner and descriptor-table migration. This uses
only the separately pinned framing/observer and request serializer.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import traceback

PINS = {
    'replay': 'ea748d4100b539e0144f032ca301b4886d0f9520c36676c980279e0bbc9bc034',
    'controls': '43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8',
    'observer': 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'replay', 'controls', 'observer', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    a = parser.parse_args()
    for name in ('binary', 'source', 'replay', 'controls', 'observer', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    paths = {name: getattr(a, name) for name in ('binary', 'source', 'replay', 'controls', 'observer')}
    before = {name: sha(path) for name, path in paths.items()}
    assert before == dict(PINS, binary=a.expected_sha256, source=a.expected_source_sha256)
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    b, transport, replay = (load('private_roots_'+name, getattr(a, name)) for name in ('controls', 'observer', 'replay'))
    a.output.mkdir(parents=True, exist_ok=False)
    archive = a.output/'executed-sources'; archive.mkdir()
    for path in (Path(__file__), a.replay, a.controls, a.observer):
        (archive/path.name).write_bytes(path.read_bytes())
    session = transport.OwnerSession(a.binary, directory=a.output/'trajectory', address_limit=64 << 20)
    c = b.Check(session); r = replay.Replay(c, b, 'optimizer')
    report = {'status': 'FAIL', 'native_executable_sha256': before['binary'], 'native_source_sha256': before['source']}
    try:
        p = r.p('p', [1., -2.])
        initial = list(b.words(c.one(54)))
        fillers = [rec['value'] for rec in c.run([b.tensor([0], [], False) for _ in range(300)])]
        opt = r.opt_create([(['p'], {})], {'momentum': .5})
        r.backward(p, [2., -1.]); r.step(opt)
        state = r.opt_state_handle(opt)
        c.values(p, [.75, -1.875]); c.values(state, [2., -1.]); c.values(p, [2., -1.], True)
        pmeta, smeta = list(b.words(c.one(13, p))), list(b.words(c.one(13, state)))
        pinv = list(b.words(c.one(74, opt, 0))); sinv = list(b.words(c.one(76, opt, 0, 0)))
        pw, sw = c.make(42, p), c.make(42, state)
        grown_table = list(b.words(c.one(54)))
        assert grown_table[0] > initial[0] and grown_table[11] > 256
        c.run([(41, p), (41, state), *[(41, tid) for tid in fillers], (33, 0)])
        for stale in (p, state):
            c.one(13, stale, status=1)
        c.expiry(pw, False); c.expiry(sw, False)
        owner_before_pool_grow = session.owner()
        big = c.make(38, 2, 0, 512, 1024, 0, 0, 0)
        owner_after_pool_grow = session.owner()
        assert owner_after_pool_grow['length'] > owner_before_pool_grow['length']
        assert owner_after_pool_grow['pointer'] != owner_before_pool_grow['pointer']
        got = c.one(39, big, 524287, 1)
        assert got['value'] == 1 and got['payload'] == bytes(4)
        c.one(41, big)
        current_p = c.make(75, opt, 0, 0); current_state = c.make(77, opt, 0, 0, 0)
        assert current_p != p and current_state != state
        new_pmeta, new_smeta = list(b.words(c.one(13, current_p))), list(b.words(c.one(13, current_state)))
        assert new_pmeta == pmeta and new_smeta == smeta
        current_pinv = list(b.words(c.one(74, opt, 0))); current_sinv = list(b.words(c.one(76, opt, 0, 0)))
        assert current_pinv[:9] == pinv[:9] and current_pinv[9] == current_p
        assert current_sinv[:3] == sinv[:3] and current_sinv[3] == current_state
        r.params['p'] = current_p
        c.values(current_p, [.75, -1.875]); c.values(current_state, [2., -1.]); c.values(current_p, [2., -1.], True)
        for _ in range(8):
            r.opt_add(opt, [])
        c.audit()
        invalid_group = r.words(r.group_words(['p'], {}))
        c.run([(72, opt), (79, opt, invalid_group)], status=7, label='late rejected add rolls back real SGD/state mutation')
        c.values(current_p, [.75, -1.875]); c.values(current_state, [2., -1.])
        assert list(b.words(c.one(13, current_p))) == pmeta
        assert list(b.words(c.one(13, current_state))) == smeta
        r.step(opt)
        c.values(current_p, [.375, -1.6875]); c.values(current_state, [3., -1.5])
        assert b.words(c.one(13, current_p))[13] == 2 and b.words(c.one(13, current_state))[13] == 2
        c.one(72, current_p, status=1, label='Tensor cannot be an Optimizer')
        c.one(13, opt, status=1, label='Optimizer cannot be a Tensor')
        c.one(78, opt)
        c.one(72, opt, status=1, label='closed Optimizer handle stays invalid')
        c.expiry(pw, False); c.expiry(sw, False)
        c.run([(41, current_p), (41, current_state)])
        c.expiry(pw, True); c.expiry(sw, True)
        c.run([(44, pw), (44, sw)])
        c.audit(); session.finish()
        report.update(status='PASS', cases=1, initial_stats=initial, table_growth_stats=grown_table,
            owner_bytes_before_pool_grow=owner_before_pool_grow['length'], owner_bytes_after_pool_grow=owner_after_pool_grow['length'],
            original_parameter_token=p, republished_parameter_token=current_p,
            original_state_token=state, republished_state_token=current_state,
            preserved_parameter_incarnation=pinv[8], preserved_state_incarnation=sinv[2],
            parameter_and_state_metadata_preserved_across_reset_and_growth=True,
            final_public_aliases_outlive_optimizer=True, final_weak_expiry_after_last_public_drop=True)
    except Exception as error:
        report.update(exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if session.poll() is None:
            os.kill(session.pid, signal.SIGKILL)
        try:
            session.finish(expected=-signal.SIGKILL)
        except Exception as cleanup:
            report['cleanup_exception'] = repr(cleanup)
    assert before == {name: sha(path) for name, path in paths.items()}
    report.update(schema='heptatorch.native-private-optimizer-root-control.v1',
        inputs=before, runner_sha256=sha(Path(__file__)), frames=c.frames,
        exact_value_and_helper_comparisons=c.comparisons, whole_owner_rollback_bytes=c.rollback_bytes,
        actual_Torch_reference_cases=0, host_numeric_implementation=False,
        scope='One ownership/migration/admission control with fixed dyadic expected words, separate from the actual hosted 24/6 replay.',
        full_rewrite_complete=False, pytorch_api_denominator=2590)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'frames': len(c.frames), 'exact_value_and_helper_comparisons': c.comparisons,
                      'whole_owner_rollback_bytes': c.rollback_bytes}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
