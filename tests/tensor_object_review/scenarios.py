"""Predeclared independent semantic traces; no target outputs form expectations."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import struct
from reference_model import Model, Rejected


def action(op, **fields):
    return dict(op=op, **fields)


def tensor(name, shape, values, requires=False):
    return action('create', name=name, shape=shape, values=values, requires=requires)


def check(**fields):
    return action('check', **fields)


def error(status, *actions):
    return action('frame', actions=list(actions), expected_status=status)


CASES = [
    {'case': 'independent-Tensor-Storage-Version-alias-clone-data-identities', 'actions': [
        tensor('x', [2, 3], [1, 2, 3, 4, 5, 6], True),
        action('transpose', name='t', source='x', a=0, b=1),
        action('detach', name='d', source='x'),
        action('clone', name='c', source='t'),
        action('detach', name='raw', source='x', separate_version=True),
        check(tensors=['x', 't', 'd', 'c', 'raw'], storage=['x', 'c'], version=['x', 'raw', 'c']),
        action('set', name='d', index=1, value=20),
        check(tensors=['x', 't', 'd', 'c', 'raw'], version=['x', 'raw', 'c']),
        action('set', name='raw', index=2, value=30),
        check(tensors=['x', 't', 'd', 'c', 'raw'], version=['x', 'raw', 'c']),
        action('drop', name='x'),
        error(1, action('values', name='x')),
        check(tensors=['t', 'd', 'c', 'raw'], storage=['x'], version=['x', 'raw']),
    ]},
    {'case': 'A-B-C-alias-weak-Saved-Hold-last-strong-and-internal-hole', 'actions': [
        tensor('A', [8], list(range(1, 9))), tensor('B', [8], list(range(11, 19))),
        tensor('C', [8], list(range(21, 29))),
        action('detach', name='aliasB', source='B'), action('make_weak', name='wB', tensor='B'),
        action('save', name='sB', tensor='B'), action('lock', name='hB', weak='wB'),
        check(tensors=['A', 'B', 'C', 'aliasB'], storage=['B'], version=['B']),
        action('drop', name='B'), error(1, action('values', name='B')),
        check(storage=['B'], version=['B'], saved=['sB'], holds=['hB']),
        action('drop', name='aliasB'), check(storage=['B'], version=['B']),
        action('drop_saved', name='sB'), check(storage=['B'], version=['B'], holds=['hB']),
        action('drop_hold', name='hB'), check(storage=['B'], version=['B']),
        error(6, action('lock', name='expired', weak='wB')),
        tensor('D', [8], list(range(31, 39))),
        check(tensors=['A', 'C', 'D'], storage=['A', 'B', 'C', 'D'], version=['D']),
        action('drop_weak', name='wB'), check(storage=['B']),
    ]},
    {'case': 'public-Saved-logical-transpose-survives-node-reset', 'actions': [
        tensor('x', [2, 3], [1, 2, 3, 4, 5, 6]),
        action('transpose', name='t', source='x', a=0, b=1),
        action('save', name='savedT', tensor='t'), action('make_weak', name='w', tensor='x'),
        action('reset_graph', keep=0), error(1, action('values', name='x')),
        error(1, action('values', name='t')), check(saved=['savedT'], storage=['x'], version=['x']),
        action('unpack', name='u', saved_name='savedT'),
        check(tensors=['u'], saved=['savedT'], storage=['x'], version=['u']),
        action('drop_saved', name='savedT'), check(tensors=['u'], storage=['u'], version=['u']),
        action('set', name='u', index=1, value=40), check(tensors=['u'], version=['u']),
        action('lock', name='h', weak='w'), check(holds=['h']),
        action('drop', name='u'), check(storage=['u'], version=['u']),
        action('drop_hold', name='h'), error(6, action('lock', name='expired', weak='w')),
        action('drop_weak', name='w'), check(storage=['u']),
    ]},
    {'case': 'gradient-topology-outlives-unneeded-nonleaf-payload', 'actions': [
        tensor('x', [3], [1, 2, 3], True), tensor('factor', [], [2]), tensor('one', [], [1]),
        action('binary', name='u', op_name='mul', a='x', b='factor'),
        action('binary', name='z', op_name='add', a='u', b='one'), action('total', name='loss', a='z'),
        action('make_weak', name='wu', tensor='u'), action('make_weak', name='wx', tensor='x'),
        action('make_weak', name='wf', tensor='factor'), action('drop', name='u'),
        action('drop', name='z'), action('drop', name='factor'), action('drop', name='one'),
        check(storage=['u', 'x', 'factor'], version=['u', 'x', 'factor']),
        error(6, action('lock', name='deadU', weak='wu')), action('backward', root='loss'),
        check(gradient=['x']), action('drop', name='x'), error(1, action('grad', name='x')),
        check(storage=['x', 'factor'], version=['x', 'factor']),
        action('drop', name='loss'), check(storage=['x', 'factor'], version=['x', 'factor']),
        error(6, action('lock', name='deadX', weak='wx')),
        error(6, action('lock', name='deadF', weak='wf')),
    ]},
    {'case': 'duplicated-inputs-have-two-Saved-obligations-and-nonuniform-repeat-VJP', 'actions': [
        tensor('x', [2, 2], [1, -2, 3, -4], True),
        action('binary', name='square', op_name='mul', a='x', b='x'),
        action('make_weak', name='w', tensor='x'),
        check(tensors=['square'], storage=['x'], version=['x']),
        action('backward', root='square', seed=[1, -2, '1/2', 4]),
        check(gradient=['x', 'square']),
        action('backward', root='square', seed=[1, -2, '1/2', 4]),
        check(gradient=['x', 'square']),
        action('drop', name='x'), check(storage=['x'], version=['x']),
        error(1, action('values', name='x')), action('drop', name='square'),
        check(storage=['x'], version=['x']), error(6, action('lock', name='dead', weak='w')),
    ]},
    {'case': 'mutation-version-policy-input-unsaved-versus-coefficient-saved', 'actions': [
        tensor('x', [3], [1, 2, 3], True), tensor('factor', [3], [2, -1, 4]),
        action('binary', name='y', op_name='mul', a='x', b='factor'), action('total', name='loss', a='y'),
        action('set', name='x', index=0, value=10), action('backward', root='loss'),
        check(gradient=['x'], storage=['x', 'factor'], version=['x', 'factor']),
        action('set', name='factor', index=0, value=8), error(6, action('backward', root='loss')),
        check(gradient=['x'], tensors=['x', 'factor'], version=['factor']),
    ]},
    {'case': 'ReLU-saves-output-and-accepts-input-mutation', 'actions': [
        tensor('x', [3], [-2, 1, 3], True), action('unary', name='y', op_name='relu', a='x'),
        action('total', name='loss', a='y'), action('set', name='x', index=0, value=2),
        action('backward', root='loss'), check(gradient=['x'], storage=['x', 'y'], version=['x', 'y']),
        action('set', name='y', index=0, value=1), error(6, action('backward', root='loss')),
        check(gradient=['x'], version=['y']),
    ]},
    {'case': 'clone-transpose-retain-layout-and-gradient-route-without-input-data', 'actions': [
        tensor('x', [2, 3], [1, 2, 3, 4, 5, 6], True),
        action('transpose', name='t', source='x', a=0, b=1), action('clone', name='c', source='t'),
        action('make_weak', name='wx', tensor='x'), action('drop', name='t'),
        action('set', name='x', index=0, value=9),
        action('backward', root='c', seed=[1, 2, 3, 4, 5, 6]),
        check(tensors=['x', 'c'], gradient=['x', 'c'], storage=['x', 'c'], version=['x', 'c']),
        action('drop', name='x'), check(storage=['x'], version=['x']),
        action('drop', name='c'), check(storage=['x'], version=['x']),
    ]},
    {'case': 'data-alias-independent-Version-does-not-pretend-to-track-storage-writes', 'actions': [
        tensor('x', [2], [2, 3], True), action('detach', name='raw', source='x', separate_version=True),
        action('binary', name='y', op_name='mul', a='x', b='raw'), action('total', name='loss', a='y'),
        check(storage=['x'], version=['x', 'raw']),
        action('set', name='x', index=0, value=10), action('backward', root='loss'),
        check(tensors=['x', 'raw', 'y'], gradient=['x'], version=['x', 'raw']),
        action('set', name='raw', index=0, value=7), error(6, action('backward', root='loss')),
        check(gradient=['x'], version=['x', 'raw']),
    ]},
    {'case': 'two-required-input-division-and-broadcast-gradients', 'actions': [
        tensor('a', [2, 2], [2, 4, 6, 8], True), tensor('b', [2], [2, 4], True),
        action('binary', name='q', op_name='div', a='a', b='b'),
        action('backward', root='q', seed=[1, 2, 4, 8]),
        check(tensors=['q'], gradient=['a', 'b'], storage=['a', 'b'], version=['a', 'b']),
        action('set', name='a', index=0, value=3), error(6, action('backward', root='q', seed=[1, 2, 4, 8])),
        check(gradient=['a', 'b']),
    ]},
    {'case': 'whole-frame-mutation-allocation-reset-and-Saved-failure-rollback', 'actions': [
        tensor('x', [2], [1, 2], True), tensor('factor', [2], [2, 4]),
        action('binary', name='y', op_name='mul', a='x', b='factor'), action('total', name='loss', a='y'),
        action('save', name='savedX', tensor='x'), action('backward', root='loss'),
        error(6, action('set', name='factor', index=0, value=8), action('backward', root='loss')),
        error(4, action('set', name='x', index=0, value=9), tensor('discarded', [8], [0]*8), action('unknown')),
        error(1, action('reset_graph', keep=0), action('values', name='x')),
        check(tensors=['x', 'factor', 'y'], gradient=['x'], saved=['savedX'],
              storage=['x', 'factor'], version=['x', 'factor']),
        action('reset_graph', keep=0), check(saved=['savedX'], storage=['x'], version=['x']),
        action('drop_saved', name='savedX'), check(storage=['x'], version=['x']),
    ]},
    {'case': 'zero-sized-Storage-and-Version-identities-are-still-independent', 'actions': [
        tensor('a', [2, 0, 3], [], True), tensor('b', [2, 0, 3], [], True),
        action('detach', name='d', source='a'), action('make_weak', name='w', tensor='a'),
        action('save', name='s', tensor='a'), action('drop', name='a'),
        check(tensors=['b', 'd'], saved=['s'], storage=['a', 'b'], version=['a', 'b']),
        action('drop', name='d'), action('drop_saved', name='s'),
        error(6, action('lock', name='dead', weak='w')), check(storage=['a'], version=['a']),
        action('drop_weak', name='w'), check(storage=['a']),
    ]},
    {'case': 'observed-PyTorch-empty-transpose-clone-preserves-strides', 'actions': [
        tensor('x', [0, 2, 3], [], True),
        action('transpose', name='t', source='x', a=1, b=2), action('clone', name='c', source='t'),
        check(tensors=['x', 't', 'c'], storage=['x', 'c'], version=['x', 'c']),
        action('backward', root='c', seed=[]), check(gradient=['x']),
    ]},
]


def apply(model, command):
    command = dict(command)
    operation = command.pop('op')
    if 'op_name' in command:
        command['op'] = command.pop('op_name')
    if operation == 'unknown':
        raise Rejected(4, 'unknown opcode')
    return getattr(model, operation)(**command)


def exact_bits(values):
    result = []
    for value in values:
        raw = struct.pack('<f', float(value))
        rounded, = struct.unpack('<f', raw)
        assert Fraction(rounded) == value, ('fixture not exactly binary32', value)
        result.append(struct.unpack('<I', raw)[0])
    return result


def observation(model, command):
    result = {}
    for category, names in command.items():
        if category == 'op':
            continue
        result[category] = {}
        for name in names:
            try:
                if category == 'tensors':
                    node = model._public(name)
                    values = model.values(name)
                    value = {'shape': node.layout.shape, 'strides': node.layout.strides,
                             'numel': node.layout.numel, 'requires': node.requires,
                             'storage_symbol': node.storage, 'version_symbol': node.version,
                             'counter': model.versions[node.version].counter,
                             'storage_offset_bytes': 4*node.layout.offset,
                             'storage_bytes': model.storages[node.storage].nbytes,
                             'bits': exact_bits(values)}
                elif category == 'storage':
                    value = model.storage_metadata(model.nodes[name].storage)
                elif category == 'version':
                    value = model.version_metadata(model.nodes[name].version)
                elif category == 'gradient':
                    value = exact_bits(model.grad(name))
                elif category == 'saved':
                    value = exact_bits(model.read_saved(name))
                elif category == 'holds':
                    value = exact_bits(model.read_hold(name))
                else:
                    raise AssertionError(category)
                result[category][name] = {'status': 0, 'value': value}
            except Rejected as exc:
                result[category][name] = {'status': exc.status}
    return result


def expected_trace(case):
    model, rows = Model(), []
    for index, command in enumerate(case['actions']):
        row = {'action_index': index, 'action': command}
        if command['op'] == 'check':
            row['expected'] = observation(model, command)
        elif command['op'] == 'frame':
            trial = model.fork()
            status = 0
            try:
                for action in command['actions']:
                    apply(trial, action)
            except Rejected as exc:
                status = exc.status
            assert status == command['expected_status'], (case['case'], index, status, command)
            row['expected_status'] = status
            if status == 0:
                model = trial
            else:
                row['committed_model_unchanged'] = True
        else:
            apply(model, command)
            row['expected_status'] = 0
            if command['op'] in ('create', 'transpose', 'detach', 'clone', 'binary',
                                  'unary', 'total', 'unpack', 'view'):
                row['published_tensor'] = observation(model, check(tensors=[command['name']]))['tensors'][command['name']]
        rows.append(row)
    return {'case': case['case'], 'actions': rows, 'final_model': model.snapshot()}


def main():
    root = Path(__file__).resolve().parent
    rows = [expected_trace(case) for case in CASES]
    report = {'schema': 'heptatorch.numeric-storage-independent-reference-traces.v1',
              'status': 'REFERENCE_GENERATED_TARGET_NOT_RUN',
              'reference_model_sha256': hashlib.sha256((root/'reference_model.py').read_bytes()).hexdigest(),
              'scenario_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'expected_cases': len(rows), 'cases': rows,
              'target_executed': False, 'pinned_pytorch_executed': False,
              'numerical_scope': 'Exact rational small fixtures; every emitted expected float is checked exactly binary32.',
              'lifetime_scope': 'v3 graph retained until explicit Tensor drop/reset, unlike PyTorch default backward disposal.'}
    path = root/'reference-traces.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'cases': len(rows),
                      'actions': sum(len(row['actions']) for row in rows),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
