"""Independently replay three actual captured-accumulator Storage lifetimes.

Public parameter Storage expiration uses native Weak/Hold commands. The current
wire exposes no public .grad Tensor object: its private Hold/Storage lifetime is
read from our own child's committed owner, without creating a reference or
modifying bytes. Python-wrapper weakrefs and grad-object Versions remain open.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import traceback

OBSERVATIONS_SHA = 'b6cffbb632c9f71752353c1152e8023d454f9ba3f5837bd7fe7846670a913832'
PROBE_SHA = '1597219302665e79a840305f2ead58bdfd5472ddca97b0a4092c8d0dd8de96e2'
PINS = {
    'replay': 'ea748d4100b539e0144f032ca301b4886d0f9520c36676c980279e0bbc9bc034',
    'controls': '43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8',
    'observer': 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrivateView:
    def __init__(self, session):
        self.owner = session.owner()
        self.data = self.owner['bytes']
        assert self.data[64:72] == b'HTOPT002' and self.data[139264:139272] == b'HTOOPT02'
        self.slots = self.word(139264+16)
        self.pool = 139264+144+self.slots*552
        assert self.word(112) == 0

    def word(self, at):
        assert 0 <= at <= len(self.data)-8
        return struct.unpack_from('<Q', self.data, at)[0]

    def slot(self, kind, handle):
        assert (handle >> 24) & 255 == kind
        slot, generation = handle & 0xffffff, handle >> 32
        if not 1 <= slot <= self.slots or generation == 0:
            return 0
        at = 139264+128+(kind-1)*self.slots*64+(slot-1)*64
        return at if self.word(at) == generation and self.word(at+8) == 1 else 0

    def descriptor(self, node):
        assert 0 < node < self.word(8)
        page = self.word(4096+(node >> 8)*8)
        assert page >= 1 << 63
        at = self.pool+(page & ((1 << 63)-1))+(node & 255)*384
        assert at+384 <= len(self.data)
        assert self.word(at+336) == self.word(at+368) == self.word(at+376) == 0
        return at

    def storage_expired(self, storage):
        row = self.slot(2, storage)
        return row == 0 or self.word(row+40) == 0 or self.word(row+56) != 0

    def gradient(self, node):
        p = self.descriptor(node)
        if self.word(p+304) == 0:
            return {'node_live': False}
        presence = self.word(p+360)
        result = {'node_live': True, 'presence': presence, 'hold': self.word(p+248), 'address': self.word(p+24)}
        if not presence:
            assert result['hold'] == result['address'] == 0
            result['binary32_words'] = None
            return result
        hp = self.slot(6, result['hold'])
        assert hp and self.word(hp+24) == 1
        sid = self.word(hp+16); sp = self.slot(2, sid)
        assert sp and not self.storage_expired(sid)
        assert result['address'] == (1 << 63)+self.word(sp+16)
        n = self.word(p+8); assert self.word(sp+24) == n*4
        at = self.pool+self.word(sp+16)
        result.update(storage=sid, binary32_words=list(struct.unpack_from('<'+'I'*n, self.data, at)))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'replay', 'controls', 'observer', 'observations', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    a = parser.parse_args()
    for name in ('binary', 'source', 'replay', 'controls', 'observer', 'observations', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    paths = {name: getattr(a, name) for name in ('binary', 'source', 'replay', 'controls', 'observer', 'observations')}
    before = {name: sha(path) for name, path in paths.items()}
    assert before == dict(PINS, binary=a.expected_sha256, source=a.expected_source_sha256, observations=OBSERVATIONS_SHA)
    reference = json.loads(a.observations.read_bytes())
    assert reference['status'] == 'OBSERVED' and reference['cases'] == 3 and reference['probe_errors'] == 0
    assert reference['probe_sha256'] == PROBE_SHA
    assert len(reference['oracle_provenance']['checks']) == 11 and all(reference['oracle_provenance']['checks'].values())
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
        return result
    b, transport, replay = (load('accumulator_'+name, getattr(a, name)) for name in ('controls', 'observer', 'replay'))
    a.output.mkdir(parents=True, exist_ok=False)
    archive = a.output/'executed-sources'; archive.mkdir()
    for path in (Path(__file__), a.replay, a.controls, a.observer, a.observations):
        (archive/path.name).write_bytes(path.read_bytes())
    rows = []
    for number, actual in enumerate(reference['rows'], 1):
        directory = a.output/('case-'+str(number).zfill(2))
        session = transport.OwnerSession(a.binary, directory=directory, address_limit=64 << 20)
        c = b.Check(session); r = replay.Replay(c, b, 'requires')
        row = {'case': actual['case'], 'status': 'FAIL'}
        events = []
        try:
            captured = number != 3
            p = r.p('p', [1., -2.], requires=captured)
            if number == 2:
                r.backward(p, [3., 4.])
            other = c.new([], [2.]) if captured else c.new([2], [3., 4.], True)
            product = c.make(2, 3, p, other); root = c.make(4, 7, product); c.one(41, product)
            r.flag(p, not captured)
            weak = c.make(42, p)
            view = PrivateView(session); tr = view.slot(1, p); assert tr
            node = view.word(tr+48)
            initial_gradient = view.gradient(node)
            original_grad_storage = initial_gradient.get('storage')
            original_meta = list(b.words(c.one(13, p)))
            def observe(event):
                expected = event['observation']
                c.expiry(weak, expected['parameter_Storage_expired'])
                private = PrivateView(session); dp = private.descriptor(node)
                state = {'node_live': private.word(dp+304) != 0, 'native_Tensor_row_live': private.word(dp+240),
                         'captured_accumulator_references': private.word(dp+88), 'gradient': private.gradient(node)}
                if not expected['parameter_Storage_expired']:
                    hold = c.make(43, weak); rec = c.one(50, hold, 0, 2); c.one(45, hold)
                    words = list(struct.unpack('<2I', rec['payload']))
                    r.eq(words, expected['parameter']['binary32_words'], 'retained parameter Storage words')
                    r.eq(private.word(dp+32), int(expected['parameter']['requires_grad']), 'retained leaf current flag')
                    vp = private.slot(3, original_meta[12]); assert vp
                    r.eq(private.word(vp+16), expected['parameter']['version'], 'retained shared Version counter')
                    desired = expected['gradient']
                    r.eq(state['gradient']['presence'], int(desired is not None), 'private None/present gradient')
                    if desired is not None:
                        r.eq(state['gradient']['storage'], original_grad_storage, 'original private gradient Storage identity')
                        r.eq(state['gradient']['binary32_words'], desired['binary32_words'], 'retained private gradient words')
                    if captured:
                        assert state['captured_accumulator_references'] > 0 and state['native_Tensor_row_live'] == 1
                    else:
                        r.eq(state['captured_accumulator_references'], 0, 'no invented AccumulateGrad edge')
                if 'original_gradient_Storage_expired' in expected:
                    assert original_grad_storage is not None
                    r.eq(private.storage_expired(original_grad_storage), expected['original_gradient_Storage_expired'], 'original private gradient allocation expiration')
                    state['original_private_gradient_Storage_expired'] = private.storage_expired(original_grad_storage)
                events.append({'phase': event['phase'], 'reference_python_Tensor_alive_diagnostic_only': expected['python_Tensor_alive'],
                               'native_observation': state, 'reference_parameter_Storage_expired': expected['parameter_Storage_expired']})
            expected_events = actual['observation']['events']
            observe(expected_events[0]); c.one(41, p); c.one(13, p, status=1); observe(expected_events[1])
            if captured:
                for index, retain in ((2, 1), (3, 0)):
                    r.eq(expected_events[index]['backward']['outcome'], 'SUCCEEDED', 'actual disabled-leaf backward')
                    r.backward(root, retain=retain); observe(expected_events[index])
                c.one(41, root); observe(expected_events[4])
            else:
                r.eq(expected_events[2]['backward']['outcome'], 'SUCCEEDED', 'actual saved-only backward')
                r.backward(root); observe(expected_events[2])
                q = expected_events[2]['q']; qg = expected_events[2]['q_gradient']
                qmeta = b.words(c.one(13, other)); r.eq(qmeta[13], q['version'], 'q Version')
                r.eq(list(struct.unpack('<2I', c.one(10, other)['payload'])), q['binary32_words'], 'q values')
                r.eq(c.one(59, other)['value'], 1, 'q gradient defined')
                r.eq(list(struct.unpack('<2I', c.one(11, other)['payload'])), qg['binary32_words'], 'q gradient exact words')
                c.one(41, root); observe(expected_events[3])
            c.one(44, weak); c.audit(); session.finish()
            row.update(status='QUALIFIED_STORAGE_AND_PRIVATE_GRAD_LIFETIME', events=events)
        except Exception as error:
            row.update(exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), events=events)
            if session.poll() is None:
                os.kill(session.pid, signal.SIGKILL)
            try:
                session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup:
                row['cleanup_exception'] = repr(cleanup)
        row.update(frames=c.frames, exact_observation_comparisons=c.comparisons, rollback_bytes=c.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(row, indent=2)+'\n'); rows.append(row)
    assert before == {name: sha(path) for name, path in paths.items()}
    failures = sum(row['status'] == 'FAIL' for row in rows)
    report = {'schema': 'heptatorch.native-replay-captured-accumulator-lifetime.v1',
        'status': 'FAIL' if failures else 'QUALIFIED_STORAGE_AND_PRIVATE_GRAD_LIFETIME',
        'native_executable_sha256': before['binary'], 'native_source_sha256': before['source'],
        'reference_sha256': OBSERVATIONS_SHA, 'probe_sha256': PROBE_SHA, 'runner_sha256': sha(Path(__file__)),
        'inputs': before, 'oracle_provenance': reference['oracle_provenance'], 'cases': 3, 'qualified': 3-failures,
        'unexpected_failures': failures, 'frames': sum(len(row['frames']) for row in rows),
        'exact_observation_comparisons': sum(row['exact_observation_comparisons'] for row in rows),
        'rollback_bytes': sum(row['rollback_bytes'] for row in rows), 'rows': rows,
        'observer_contract': 'Read-only inspection of this test child committed owner; no new private/public Storage reference is created by reading grad Hold identity, bits or allocation lifetime.',
        'open_scope': ['Python wrapper weakref is diagnostic only and is not equated to native public-token liveness.',
            'Public grad Tensor identity, Version, layout and assignable aliases are not exposed. The existing private gradient Storage lifetime and exact words are compared.'],
        'pinned_Torch_executed_during_replay': False, 'host_numeric_implementation': False,
        'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'cases', 'qualified', 'unexpected_failures', 'frames', 'exact_observation_comparisons', 'rollback_bytes')}))
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
