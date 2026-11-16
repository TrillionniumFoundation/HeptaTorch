"""Replay ten actual pinned Torch gradient-clear observations on native HIA.

Parameter counters, presence and gradient bits are strict. Existing Adam value
tolerances are unchanged. Legacy explicitly supplied optimizer state is recorded
as an open lifecycle difference; no missing native step Tensor is fabricated.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import struct
import traceback

OBSERVATIONS_SHA = 'ddd585f50034005d8333b24e68c4e3aef675f45bb3c273d973e3525195ac77cf'
PROBE_SHA = 'f34e98011e62bdac2eb94beb5a07c043bde09d2fc8d9c416e52e22d278c96b50'
CONTROLS_SHA = '43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
RTOL, ATOL = 4e-5, 5e-6


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'controls', 'transport-observer', 'observations', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--expected-source-sha256', required=True)
    a = p.parse_args()
    for name in ('binary', 'source', 'controls', 'transport_observer', 'observations', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    paths = {name: getattr(a, name) for name in ('binary', 'source', 'controls', 'transport_observer', 'observations')}
    hashes = {name: sha(path.read_bytes()) for name, path in paths.items()}
    assert hashes == {'binary': a.expected_sha256, 'source': a.expected_source_sha256,
                      'controls': CONTROLS_SHA, 'transport_observer': OBSERVER_SHA, 'observations': OBSERVATIONS_SHA}
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    b = load('grad_clear_codec', a.controls)
    transport = load('grad_clear_transport', a.transport_observer)
    reference = json.loads(a.observations.read_bytes())
    assert reference['status'] == 'OBSERVED' and reference['cases'] == 10 and reference['probe_errors'] == 0
    assert reference['probe_sha256'] == PROBE_SHA
    assert len(reference['oracle_provenance']['checks']) == 11 and all(reference['oracle_provenance']['checks'].values())
    a.output.mkdir(parents=True, exist_ok=False)
    archive = a.output / 'executed-sources'
    archive.mkdir()
    files = (Path(__file__), a.controls, a.transport_observer, a.observations)
    source_hashes = {path.name: sha(path.read_bytes()) for path in files}
    for path in files:
        (archive / path.name).write_bytes(path.read_bytes())

    def create(c, info, requires=None):
        shape, words = info['shape'], info['binary32_words']
        return c.make(1, len(shape), int(info['requires_grad'] if requires is None else requires),
                      *shape, *([0] * (4-len(shape))), b.Blob(struct.pack('<'+'I'*len(words), *words)))

    def read_tensor(c, tensor, info=None, *, gradient=False, approximate=False, version=True):
        result = {}
        if not gradient:
            meta = b.words(c.one(13, tensor))
            assert len(meta) == 16
            result['metadata'] = list(meta)
            if info is not None:
                assert meta[0] == len(info['shape']) and tuple(meta[3:3+meta[0]]) == tuple(info['shape'])
                assert tuple(meta[7:7+meta[0]]) == tuple(info['strides'])
                assert meta[1] == len(info['binary32_words']) and meta[2] == int(info['requires_grad'])
                c.comparisons += 4 + 2*meta[0]
                if version:
                    assert meta[13] == info['version'], ('Version', tensor, meta[13], info['version'])
                    c.comparisons += 1
        rec = c.one(11 if gradient else 10, tensor)
        assert len(rec['payload']) == 4*rec['value']
        actual = list(struct.unpack('<'+'I'*rec['value'], rec['payload']))
        result['binary32_words'] = actual
        if info is not None:
            expected = info['binary32_words']
            assert len(actual) == len(expected)
            if approximate:
                left = struct.unpack('<'+'f'*len(actual), rec['payload'])
                right = struct.unpack('<'+'f'*len(expected), struct.pack('<'+'I'*len(expected), *expected))
                for x, y in zip(left, right):
                    assert math.isfinite(x) and math.isfinite(y) and abs(x-y) <= ATOL + RTOL*abs(y), (x, y)
                c.approximate += len(expected)
                c.approximate_bit_differences += sum(x != y for x, y in zip(actual, expected))
            else:
                assert actual == expected, ('gradient' if gradient else 'tensor', tensor, actual, expected)
                c.comparisons += len(expected)
        return result

    def private_gradient(session, tensor):
        owner = session.owner()
        data = owner['bytes']
        assert data[64:72] == b'HTOPT002'
        word = lambda at: struct.unpack_from('<Q', data, at)[0]
        slots = word(139264+16)
        pool_base = 139264+144+552*slots
        def lookup(kind, handle):
            assert handle >> 24 & 255 == kind
            slot, generation = handle & 0xffffff, handle >> 32
            assert 1 <= slot <= slots and generation != 0
            at = 139264+128+(kind-1)*slots*64+(slot-1)*64
            assert word(at) == generation and word(at+8) == 1
            return at
        trow = lookup(1, tensor)
        node = word(trow+48)
        assert 0 < node < word(8)
        page = word(4096+(node >> 8)*8)
        assert page >= 1 << 63
        desc = pool_base+(page & ((1 << 63)-1))+(node & 255)*384
        presence, hold, address = word(desc+360), word(desc+248), word(desc+24)
        assert word(desc+336) == word(desc+368) == word(desc+376) == 0
        if presence == 0:
            assert hold == address == 0, ('None owns a gradient allocation', node, hold, address)
            return {'presence': 0, 'hold': 0, 'storage': 0, 'address': 0, 'bytes': 0,
                    'committed_owner_bytes': len(data)}
        assert presence == 1 and hold != 0 and address >= 1 << 63
        hrow = lookup(6, hold)
        assert word(hrow+24) == 1
        storage = word(hrow+16)
        srow = lookup(2, storage)
        assert address == (1 << 63)+word(srow+16)
        assert word(srow+24) == word(desc+8)*4
        return {'presence': 1, 'hold': hold, 'storage': storage, 'address': address,
                'bytes': word(srow+24), 'committed_owner_bytes': len(data)}

    def run_case(c, session, case):
        observation = case['observation']
        actions = observation['actions']
        first = actions[0]['before']
        params = {name: create(c, info) for name, info in first['parameters'].items()}
        kind = observation.get('optimizer')
        states, steps = {}, {name: 0 for name in params}
        state_version_gaps, state_creation_gaps, step_gaps = [], [], []
        if kind:
            for name, info in first['parameters'].items():
                zero = dict(info, requires_grad=False, binary32_words=[0]*len(info['binary32_words']))
                fields = ('momentum_buffer',) if kind == 'SGD-momentum' else ('exp_avg', 'exp_avg_sq')
                states[name] = {field: create(c, zero) for field in fields}
        name = case['case']
        roots = {}
        if name == 'retained-graph-after-gradient-clear':
            roots['square'] = c.make(2, 3, params['p'], params['p'])
            roots['loss'] = c.make(4, 7, roots['square'])
        if name.startswith('partial-'):
            coefficient = c.new([], [2.])
            x = params.get('bad', params.get('p'))
            bad = c.make(2, 3, x, coefficient)
            good_parent = params.get('good', x)
            roots['good'] = c.make(2, 3, good_parent, good_parent)
            roots['loss'] = c.make(4, 7, c.make(2, 1, bad, roots['good']))
            c.one(12, coefficient, 0, b.bits(3.))

        def snapshot(expected, stage):
            result = {'parameters': {}, 'gradients': {}, 'states': {}, 'private_gradients': {}}
            for param_name, handle in params.items():
                result['parameters'][param_name] = read_tensor(c, handle, expected['parameters'][param_name], approximate=kind in ('Adam', 'AdamW'))
                gradient = expected['gradients'][param_name]
                presence = c.one(59, handle)
                assert presence['payload'] == b'' and presence['value'] == int(gradient is not None)
                c.comparisons += 1
                private = private_gradient(session, handle)
                assert private['presence'] == presence['value']
                c.comparisons += 4
                result['private_gradients'][param_name] = private
                if gradient is None:
                    zeros = c.one(11, handle)
                    assert zeros['value'] == len(expected['parameters'][param_name]['binary32_words'])
                    assert zeros['payload'] == bytes(4*zeros['value'])
                    c.comparisons += zeros['value']
                    result['gradients'][param_name] = None
                else:
                    result['gradients'][param_name] = read_tensor(c, handle, gradient, gradient=True)
                if kind:
                    ref_state = expected['optimizer_state'][param_name]
                    if not ref_state['entry_exists']:
                        state_creation_gaps.append({'stage': stage, 'parameter': param_name,
                            'native_state': 'explicit client-created Tensors', 'reference_state': 'no entry'})
                    state_rows = {}
                    for field, state in states[param_name].items():
                        info = ref_state['fields'].get(field, {}).get('tensor')
                        got = read_tensor(c, state, info, approximate=kind in ('Adam', 'AdamW'), version=kind != 'SGD-momentum')
                        state_rows[field] = got
                        if info is not None and got['metadata'][13] != info['version']:
                            assert kind == 'SGD-momentum'
                            state_version_gaps.append({'stage': stage, 'parameter': param_name, 'field': field,
                                'native_version': got['metadata'][13], 'reference_version': info['version']})
                    if 'step' in ref_state['fields']:
                        step_gaps.append({'stage': stage, 'parameter': param_name,
                            'reference_step': ref_state['fields']['step'],
                            'native_step_Tensor_exists': False, 'native_contract': 'explicit scalar step input'})
                    result['states'][param_name] = state_rows
            return result

        def backward(handle, seed_values=None, retain=0):
            seed_id = 0 if seed_values is None else c.new(list(first['parameters'][next(k for k, v in params.items() if v == handle)]['shape']), seed_values)
            return c.one(58, handle, seed_id, retain)

        def optimizer_step(expected_before):
            cfg = observation['config']
            commands = []
            for param_name, handle in params.items():
                step = steps[param_name]+1
                if kind == 'SGD-momentum':
                    fields = [b.bits(cfg[key]) for key in ('lr', 'momentum', 'dampening', 'weight_decay')]+[int(cfg['nesterov'])]
                    commands.append((25, handle, states[param_name]['momentum_buffer'], 0, step, b.Blob(struct.pack('<5Q', *fields))))
                else:
                    fields = [b.bits(cfg['lr']), b.bits(cfg['betas'][0]), b.bits(cfg['betas'][1]), b.bits(cfg['eps']), b.bits(cfg['weight_decay']), int(kind == 'AdamW')]
                    commands.append((24, handle, states[param_name]['exp_avg'], states[param_name]['exp_avg_sq'], step, b.Blob(struct.pack('<6Q', *fields))))
                if expected_before['gradients'][param_name] is not None:
                    steps[param_name] = step
            c.run(commands)

        traces = []
        for index, action in enumerate(actions):
            before = snapshot(action['before'], str(index)+' before')
            label = action['action']
            result = None
            clear_mode = None
            if name == 'already-None-true-and-false-are-noops':
                clear_mode = index != 2
            elif name == 'clear-true-then-first-negative-zero-gradient':
                if index in (1, 2): clear_mode = True
                else: result = backward(params['p'], [2., -3.] if index == 0 else [-0., -0.])
            elif name == 'clear-false-retains-defined-zero-gradient':
                if index in (1, 3): clear_mode = index == 3
                else: result = backward(params['p'], [-0., -0.])
            elif name == 'empty-gradient-clear-true-and-rebuild':
                if index in (0, 2, 3): clear_mode = index != 2
                else: result = backward(params['p'], [])
            elif name == 'retained-graph-after-gradient-clear':
                if index in (1, 3): clear_mode = True
                else: result = c.one(58, roots['loss'], 0, int(index == 0))
            elif name == 'partial-independent-ready-leaf-then-clear':
                if index in (1, 3): clear_mode = index == 1
                elif index == 0: result = c.one(58, roots['loss'], 0, 0)
                else: result = backward(params['good'], [-0., -0., -0.])
            elif name == 'partial-shared-pending-leaf-clear-is-noop':
                if index in (1, 3): clear_mode = True
                elif index == 0: result = c.one(58, roots['loss'], 0, 1)
                else: result = c.one(58, c.make(4, 7, roots['good']), 0, 0)
            else:
                assert kind in ('SGD-momentum', 'Adam', 'AdamW')
                if index in (2, 4, 7): clear_mode = index == 2
                elif index in (0, 5): result = backward(params['p'], [1., -2., .5] if index == 0 else [-0., -0., -0.])
                else: optimizer_step(action['before'])
            if clear_mode is not None:
                records = c.run([(70 if clear_mode else 8, handle) for handle in params.values()])
                assert all(r['payload'] == b'' and r['value'] == 1 for r in records)
                c.comparisons += len(records)
            if result is not None:
                assert len(result['payload']) == 32
                expected_error = 0
                if action['outcome'] == 'RAISED':
                    assert action['exception_type'] == 'RuntimeError'
                    if 'backward through the graph a second time' in action['message']: expected_error = 10
                    else:
                        assert 'modified by an inplace operation' in action['message']
                        expected_error = 6
                assert result['value'] == b.words(result)[0] == expected_error
                c.comparisons += 2
            else:
                assert action['outcome'] == 'SUCCEEDED'
            after = snapshot(action['after'], str(index)+' after')
            if clear_mode is not None:
                assert before['parameters'] == after['parameters'] and before['states'] == after['states']
                for param_name in params:
                    old_grad = before['private_gradients'][param_name]
                    new_grad = after['private_gradients'][param_name]
                    if clear_mode:
                        assert new_grad['presence'] == new_grad['hold'] == new_grad['storage'] == new_grad['address'] == 0
                    else:
                        assert old_grad == new_grad
                    c.comparisons += 2
            if kind and index == 3:
                assert before == after, 'None optimizer step mutated a native parameter, state or gradient allocation'
                c.comparisons += 1
            traces.append({'action': label, 'native_backward_result': None if result is None else list(b.words(result)),
                           'before': before, 'after': after, 'clear_mode': clear_mode})
        c.audit()
        return {'actions': traces, 'state_version_difference_observations': state_version_gaps,
                'state_creation_difference_observations': state_creation_gaps,
                'native_step_Tensor_absence_observations': step_gaps,
                'grad_Tensor_identity_and_Version_api': 'OPEN; only native private gradient Hold allocation and actual words/presence are compared.'}

    rows = []
    for index, case in enumerate(reference['rows'], 1):
        directory = a.output / ('case-' + str(index).zfill(2))
        session = transport.OwnerSession(a.binary, directory=directory, address_limit=64 << 20)
        c = b.Check(session)
        c.approximate = c.approximate_bit_differences = 0
        row = {'case': case['case'], 'status': 'FAIL'}
        try:
            details = run_case(c, session, case)
            session.finish()
            row.update(status='QUALIFIED_GRAD_CLEAR_WITH_OPEN_OBJECT_SCOPE', details=details)
        except Exception as error:
            row.update(exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            if session.poll() is None:
                os.kill(session.pid, signal.SIGKILL)
            try:
                session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup:
                row['cleanup_exception'] = repr(cleanup)
        row.update(frames=c.frames, exact_comparisons=c.comparisons,
                   Adam_tolerance_comparisons=c.approximate, Adam_different_bits=c.approximate_bit_differences)
        (directory / 'report.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
    assert hashes == {name: sha(path.read_bytes()) for name, path in paths.items()}
    assert source_hashes == {path.name: sha(path.read_bytes()) for path in files}
    failures = sum(row['status'] == 'FAIL' for row in rows)
    report = {'schema': 'heptatorch.native-replay-pinned-grad-clear.v1',
        'status': 'FAIL' if failures else 'QUALIFIED_GRAD_CLEAR_WITH_OPEN_OBJECT_SCOPE',
        'native_executable_sha256': hashes['binary'], 'native_source_sha256': hashes['source'],
        'observations_sha256': OBSERVATIONS_SHA, 'probe_sha256': PROBE_SHA,
        'test_source_hashes': source_hashes, 'oracle_provenance': reference['oracle_provenance'],
        'cases': len(rows), 'qualified': len(rows)-failures, 'unexpected_failures': failures,
        'frames': sum(len(row['frames']) for row in rows),
        'exact_comparisons': sum(row['exact_comparisons'] for row in rows),
        'Adam_tolerance_comparisons': sum(row['Adam_tolerance_comparisons'] for row in rows),
        'Adam_different_bits': sum(row['Adam_different_bits'] for row in rows),
        'state_version_difference_observations': sum(len(row.get('details', {}).get('state_version_difference_observations', [])) for row in rows),
        'state_creation_difference_observations': sum(len(row.get('details', {}).get('state_creation_difference_observations', [])) for row in rows),
        'native_step_Tensor_absence_observations': sum(len(row.get('details', {}).get('native_step_Tensor_absence_observations', [])) for row in rows),
        'numeric_policy': {'gradient_binary32_words': 'exact including signed zero', 'parameter_Version_and_presence': 'integer exact',
                           'SGD_and_nonoptimizer_values': 'binary32 exact', 'Adam_updated_values': {'rtol': RTOL, 'atol': ATOL}},
        'rows': rows, 'native_executed': True, 'pinned_Torch_executed_during_replay': False,
        'scope': 'True/False nullable-gradient semantics, strict None/present state, signed-zero reconstruction, partial/retained backward and skip invariance. Explicit legacy optimizer state, missing native step Tensor and public grad-object identity remain open.',
        'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'qualified', 'unexpected_failures', 'frames', 'exact_comparisons', 'Adam_tolerance_comparisons', 'Adam_different_bits', 'state_version_difference_observations', 'state_creation_difference_observations', 'native_step_Tensor_absence_observations')}))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
