"""Replay six fixed Torch existing-Optimizer / empty-native_dropout observations.

All numerical, identity, layout, Version, history and RNG expectations are read
from the immutable actual report. The reused observer only reads the native
owned object model; it does not import producer authoring or compute gradients.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback

REFERENCE_SHA = '6dea331ca744401bcf1f541462beaa50ab81cde662c2558c0b157ed30c69c013'
OBSERVER_SHA = '637710f944661adbbeca600177e96bf265805af06064f338170ced7ce203484f'
WIRE_SHA = '097a27e0155b22639759636583db8994cf5d610e07d15a85d8001aff36fef3bc'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'source', 'protocol', 'observer', 'reference', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {path.resolve(): path.read_bytes() for path in
              (a.binary, a.source, a.protocol, a.observer, a.reference, Path(__file__))}
    assert sha(inputs[a.reference.resolve()]) == REFERENCE_SHA
    assert sha(inputs[a.observer.resolve()]) == OBSERVER_SHA
    assert sha(inputs[a.protocol.resolve()]) == WIRE_SHA
    actual = json.loads(inputs[a.reference.resolve()])
    assert actual['status'] == 'OBSERVED' and actual['cases'] == 6
    assert actual['probe_error_count'] == 0 and actual['source_files_unchanged'] is True
    assert len(actual['oracle_provenance']['checks']) == 11
    assert all(v is True for v in actual['oracle_provenance']['checks'].values())
    wire = load(a.protocol, 'existing_optimizer_wire')
    observer = load(a.observer, 'fixed_identity_observer')
    rows = []
    for index, record in enumerate(actual['rows']):
        directory = output / record['name']
        r = observer.Replay(wire, a, directory, actual, index)
        result = {'name': record['name'], 'algorithm': record['algorithm'],
                  'initial_gradient': record['initial_gradient'], 'status': 'RUNNING'}
        algorithm = {'SGD': 0, 'Adam': 1, 'AdamW': 2}[record['algorithm']]
        held = 0
        mask = None
        x = 0
        opt = 0

        def get_tensor(command, expected, label):
            token = r.make(*command)
            already_public = token in r.public
            if token:
                r.public.add(token)
            value = r.tensor(token, expected, label)
            if token and not already_public:
                r.drop(token)
            return value

        def inventory(expected, label):
            group = r.one(74, opt, 0)
            words = list(struct.unpack('<' + 'Q' * (len(group['payload']) // 8), group['payload']))
            groups = expected['groups']
            r.eq(len(groups), 1, label + ' one actual group')
            r.eq(group['value'], len(groups[0]['parameters']), label + ' actual group occurrences')
            r.eq(words[:2], [0, 1], label + ' native group index/count')
            metadata = groups[0]['metadata']
            if algorithm == 0:
                config = [wire.bits(metadata[key]) for key in
                          ('lr', 'momentum', 'dampening', 'weight_decay')]
                config += [int(metadata['nesterov']), int(metadata['maximize'])]
            else:
                config = [wire.bits(metadata['lr']), wire.bits(metadata['betas'][0]),
                          wire.bits(metadata['betas'][1]), wire.bits(metadata['eps']),
                          wire.bits(metadata['weight_decay']), int(metadata['amsgrad']),
                          int(metadata['maximize']), 0]
            r.eq(words[2:2 + len(config)], config, label + ' admitted binary32 configuration')
            node, fields = r.node(tid=x)
            r.eq(words[2 + len(config):], [fields[320], x], label + ' actual parameter incarnation/token')
            for parameter in groups[0]['parameters']:
                r.identity(r.tensor_ids, parameter['tensor_identity'], fields[320], label + ' parameter identity')
                r.eq(parameter['is_input_object'], True, label + ' actual input is parameter')
            r.eq(r.make(75, opt, 0, 0), x, label + ' member getter returns same Tensor')
            inv = r.one(76, opt, 0, 0)
            data = list(struct.unpack('<' + 'Q' * (len(inv['payload']) // 8), inv['payload']))
            r.eq(data[0], fields[320], label + ' state key incarnation')
            present = expected['parameter_has_state_entry']
            r.eq(bool(inv['value']), present, label + ' lazy state presence')
            r.eq(expected['state_entry_count'], int(present), label + ' exact actual dictionary entry count')
            entries = expected['state_entries']
            r.eq(len(entries), int(present), label + ' actual entries')
            if not present:
                r.eq(data[1:], [0] * (len(data) - 1), label + ' absent inventory has no hidden state')
                for field in ([0] if algorithm == 0 else [1, 2, 3]):
                    r.eq(r.make(77, opt, 0, 0, field), 0, label + ' absent state getter')
                return
            entry = entries[0]
            r.eq(entry['parameter_is_input'], True, label + ' actual state keyed by input')
            r.identity(r.tensor_ids, entry['parameter_identity'], fields[320], label + ' actual state key')
            r.eq(data[1], 1 if algorithm == 0 else 14, label + ' exact native state field mask')
            names = [(0, 'momentum_buffer')] if algorithm == 0 else [(1, 'step'), (2, 'exp_avg'), (3, 'exp_avg_sq')]
            r.eq(set(entry['values']), {name for _, name in names}, label + ' exact state names')
            for ordinal, (field, name) in enumerate(names):
                expected_tensor = entry['values'][name]
                r.eq(expected_tensor['kind'], 'Tensor', label + '.' + name + ' is a true Tensor')
                value = get_tensor((77, opt, 0, 0, field), expected_tensor['tensor'], label + '.' + name)
                r.eq(value['incarnation'], data[2 + 2 * ordinal], label + '.' + name + ' private state incarnation')

        def snapshot(expected, label):
            r.phase = label
            r.tensor(x, expected['input'], label + '.input')
            r.eq(expected['gradient_observation']['accepted'], True, label + ' actual .grad query accepted')
            current = get_tensor((91, x), expected['gradient'], label + '.gradient')
            r.eq(bool(r.make(59, x)), expected['gradient'] is not None, label + ' real gradient presence')
            r.tensor(held, expected['held_initial_gradient'], label + '.held_initial_gradient')
            if held:
                held_inc = r.node(tid=held)[1][320]
                r.eq(current is not None and current['incarnation'] == held_inc,
                     expected['gradient_is_held'], label + ' binding still aliases explicitly held G')
            else:
                r.eq(expected['gradient_is_held'], None, label + ' no claimed held identity')
            r.tensor(mask or 0, expected['mask'], label + '.mask')
            if expected['current_graph'] is None:
                r.eq(r.node(tid=x)[1][40], 0, label + ' no current history')
            else:
                r.graph(expected['current_graph'], label + '.graph', tid=x)
                r.eq(expected['current_history_is_saved_handle'], True, label + ' actual history handle remains current')
                r.eq(expected['saved_mask_observation']['accepted'], True, label + ' actual Saved mask remains accessible')
                value = get_tensor((96, x, 1), expected['saved_mask'], label + '.Saved_mask')
                r.eq(value['incarnation'], r.node(tid=mask)[1][320], label + ' Saved owns same true mask Tensor')
            inventory(expected['optimizer'], label + '.optimizer')
            r.rng(expected['state'], label + '.rng')
            r.eq(r.make(53), 1, label + ' complete native ownership audit')

        try:
            x = r.empty()
            config = actual['configuration']
            if algorithm == 0:
                defaults = [wire.bits(config[key]) for key in ('lr', 'momentum', 'dampening', 'weight_decay')]
                defaults += [int(config['nesterov']), int(config['maximize'])]
            else:
                defaults = [wire.bits(config['lr']), wire.bits(config['betas'][0]), wire.bits(config['betas'][1]),
                            wire.bits(config['eps']), wire.bits(config['weight_decay']), 0, int(config['maximize']), 0]
            groups = [1, 1, 0] + [0] * len(defaults) + [x]
            opt = r.make(71, algorithm, wire.Payload(struct.pack('<'+'Q'*len(defaults), *defaults)),
                         wire.Payload(struct.pack('<'+'Q'*len(groups), *groups)))
            r.optimizers.add(opt)
            observation = record['observation']
            snapshot(observation['before_gradient_materialization'], 'before_gradient_materialization')
            if record['initial_gradient'] == 'defined-empty':
                r.eq(observation['materialize_gradient']['accepted'], True, 'actual materialization succeeded')
                r.eq(r.make(58, x, r.seed, 0), 0, 'native real direct backward materializes G')
                r.tensor(r.seed, observation['initial_seed'], 'initial_seed')
                held = r.hold_gradient(x)
            else:
                r.eq(observation['materialize_gradient']['outcome'], 'NOT_REQUESTED', 'actual None setup')
                r.eq(observation['initial_seed'], None, 'no actual seed for None setup')
            snapshot(observation['before_forward'], 'before_forward')
            r.eq(observation['native_dropout_call']['accepted'], True, 'actual forward accepted')
            out_tensor, mask = r.dropout(x)
            r.eq(out_tensor == x, observation['output_is_input_Tensor'], 'actual same Tensor result')
            r.eq(observation['output_is_input_object'], True, 'actual same Python Tensor object')
            snapshot(observation['after_forward'], 'after_forward')
            operations = [(73, opt, 0), (72, opt), (73, opt, 1), (72, opt)]
            for event, command in zip(observation['events'], operations, strict=True):
                label = event['name']
                snapshot(event['before'], label + '.before')
                r.eq(event['operation']['accepted'], True, label + ' actual operation accepted')
                r.eq(r.make(*command), 1, label + ' native operation succeeded')
                snapshot(event['after'], label + '.after')
            r.eq(record['state_unchanged'], True, 'actual complete RNG unchanged')
            r.rng(record['state_after'], 'final_rng')
            # The reference deliberately holds x, its current history, mask and
            # held G. It makes no weak lifetime claim. Explicit native cleanup
            # breaks that history cycle before closing remaining public roots.
            r.one(101, x)
            r.one(78, opt)
            r.optimizers.remove(opt)
            for tensor in list(r.public):
                r.drop(tensor)
            r.one(66, r.generator)
            r.eq(r.make(53), 1, 'final native object audit')
            result['status'] = 'PASS'
        except BaseException as error:
            result.update(status='FAIL', phase=r.phase, error=repr(error), traceback=traceback.format_exc())
        finally:
            result.update(frames=r.session.frames, assertions=r.assertions, tensor_snapshots=r.snapshots,
                          rng_bytes=r.rng_bytes, rollback_bytes=r.rollback_bytes)
            try:
                result['transport'] = r.session.finish()
            except BaseException as error:
                result.update(status='FAIL', transport_error=repr(error))
            (directory/'trace.json').write_text(json.dumps(r.trace, indent=2)+'\n')
            (directory/'owners.json').write_text(json.dumps(r.owners, indent=2)+'\n')
            (directory/'graphs.json').write_text(json.dumps(r.graph_observations, indent=2)+'\n')
            (directory/'report.json').write_text(json.dumps(result, indent=2)+'\n')
            rows.append(result)
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    result = {'schema':'heptatorch.native-dropout-existing-optimizer-replay.v1',
              'status':'PASS' if all(row['status']=='PASS' for row in rows) else 'FAIL',
              'cases':len(rows), 'passed':sum(row['status']=='PASS' for row in rows),
              'frames':sum(row['frames'] for row in rows), 'assertions':sum(row['assertions'] for row in rows),
              'tensor_snapshots':sum(row['tensor_snapshots'] for row in rows),
              'rng_bytes':sum(row['rng_bytes'] for row in rows),
              'source_sha256':sha(inputs[a.source.resolve()]), 'native_sha256':sha(inputs[a.binary.resolve()]),
              'reference_sha256':REFERENCE_SHA, 'readonly_observer_sha256':OBSERVER_SHA,
              'inputs_sha256':{str(path):sha(raw) for path,raw in inputs.items()}, 'rows':rows,
              'Torch_executed_here':False, 'native_execution':True,
              'strict_scope':'Exact exposed Tensor/Storage identity relations, logical words, dtype, shape/strides/byte extents/offset, Version, G presence, current graph and captured accumulator target, optimizer inventory/state tensors, Saved same-mask identity, and complete 5056-byte RNG. CPU numerical references come solely from the immutable actual6 report.',
              'unrepresented':'Python warning objects and private Python wrapper identity are not native packet fields. Configuration Python doubles are explicitly admitted as the existing binary32 optimizer interface; no general mixed-dtype or Python Optimizer API closure is claimed.',
              'full_pytorch_rewrite_complete':False, 'pytorch_api_denominator':2590}
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','cases','passed','frames','assertions','rng_bytes')}))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
