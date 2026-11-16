"""Replay frozen independent object traces against a separately built HIA process.

The only imported service-adjacent helper is an explicitly supplied transport /
own-child memory observer. No target compiler, authoring, codec or numerical
implementation is imported. Failed native frames retain their original bytes.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import traceback
from reference_model import HandleCustody


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Blob:
    def __init__(self, data):
        self.data = data


def fbits(value):
    fraction = Fraction(value)
    raw = struct.pack('<f', float(fraction))
    assert Fraction(struct.unpack('<f', raw)[0]) == fraction
    return struct.unpack('<I', raw)[0]


def encode(commands):
    end = 32 + len(commands)*64
    records, data = bytearray(), bytearray()
    for command in commands:
        words = []
        for word in command:
            if isinstance(word, Blob):
                while (end+len(data)) % 8:
                    data.append(0)
                words.append(end+len(data))
                data.extend(word.data)
            else:
                words.append(int(word) & ((1 << 64)-1))
        assert len(words) <= 8
        records.extend(struct.pack('<8Q', *words, *([0]*(8-len(words)))))
    return b'HTORCH03' + struct.pack('<3Q', len(commands), end+len(data), 0) + records + data


def decode(body):
    magic, version, count, status = struct.unpack_from('<8s3Q', body)
    assert magic == b'HTORSP03' and version == 3
    if status:
        assert count == 0 and len(body) == 32
        return status, []
    at, result = 32, []
    for _ in range(count):
        op, error, value, size = struct.unpack_from('<4Q', body, at)
        at += 32
        assert error == 0 and at+size <= len(body)
        result.append({'op': op, 'value': value, 'payload': body[at:at+size]})
        at += size
    assert at == len(body)
    return 0, result


def tensor_command(shape, values, requires=False):
    payload = b''.join(struct.pack('<I', fbits(value)) for value in values)
    return (1, len(shape), int(requires), *shape, *([0]*(4-len(shape))), Blob(payload))


class Review:
    def __init__(self, session):
        self.session, self.handles = session, HandleCustody()
        self.tensors, self.objects, self.meta = {}, {}, {}
        self.frames, self.compared_fields, self.rollback_bytes = [], 0, 0
        self.storage_observations = {}
        self.action_index = -1

    def exchange(self, commands, expected=0, label=''):
        before = self.session.owner() if expected else None
        request = encode(commands)
        reply = self.session.exchange(request)
        status, result = decode(reply)
        row = {'frame': self.session.frames, 'action_index': self.action_index, 'label': label,
               'expected_status': expected, 'observed_status': status,
               'request_sha256': sha(request), 'response_sha256': sha(reply),
               'commands': len(commands), 'response_bytes': len(reply)}
        self.frames.append(row)
        assert status == expected, (label, status, expected)
        if expected:
            after = self.session.owner()
            assert before == after, ('whole returned owner changed on failure', label)
            self.rollback_bytes += before['length']
            row.update(whole_owner_unchanged=True, owner_bytes=before['length'],
                       owner_sha256=sha(before['bytes']), same_owner_pointer=True)
        else:
            assert len(result) == len(commands)
            assert [item['op'] for item in result] == [command[0] for command in commands]
        return result

    def eq(self, actual, expected, label):
        assert actual == expected, (label, actual, expected)
        self.compared_fields += len(expected) if isinstance(expected, (list, tuple)) else 1

    def bits(self, record, expected, label):
        self.eq(record['value'], len(expected), label+' result count')
        self.eq(len(record['payload']), len(expected)*4, label+' byte count')
        actual = list(struct.unpack('<'+'I'*len(expected), record['payload']))
        self.eq(actual, expected, label+' exact binary32')

    def tensor_meta(self, name, expected, *, values=True):
        tid = self.tensors[name]
        commands = [(13, tid)] + ([(10, tid)] if values else [])
        records = self.exchange(commands, label='Tensor '+name)
        record = records[0]
        self.eq(record['value'], tid, 'Tensor metadata result')
        self.eq(len(record['payload']), 128, 'Tensor metadata bytes')
        words = struct.unpack('<16Q', record['payload'])
        rank = len(expected['shape'])
        prefix = (rank, expected['numel'], int(expected['requires']),
                  *expected['shape'], *([0]*(4-rank)),
                  *expected['strides'], *([0]*(4-rank)))
        self.eq(words[:11], prefix, name+' shape/stride metadata')
        self.handles.bind(2, expected['storage_symbol'], words[11])
        self.handles.bind(3, expected['version_symbol'], words[12])
        self.eq(words[13:], (expected['counter'], expected['storage_offset_bytes'], expected['storage_bytes']),
                name+' Version/offset/bytes')
        self.meta[name] = expected
        if values:
            self.bits(records[1], expected['bits'], name)

    def wire(self, action, prepared):
        op = action['op']
        t = lambda name: self.tensors[name]
        o = lambda name: self.objects[name]
        if op == 'create': return tensor_command(action['shape'], action['values'], action.get('requires', False))
        if op == 'binary': return (2, {'add':1, 'sub':2, 'mul':3, 'div':4}[action['op_name']], t(action['a']), t(action['b']))
        if op == 'unary': return (3, {'relu':5, 'neg':6}[action['op_name']], t(action['a']))
        if op == 'total': return (4, 7, t(action['a']))
        if op == 'transpose': return (6, t(action['source']), action['a'], action['b'])
        if op == 'view': return (17, t(action['source']), len(action['shape']), *action['shape'])
        if op == 'clone': return (18, t(action['source']))
        if op == 'detach': return (55 if action.get('separate_version') else 19, t(action['source']))
        if op == 'set': return (12, t(action['name']), action['index'], fbits(action['value']))
        if op == 'zero': return (8, t(action['name']))
        if op == 'values': return (10, t(action['name']))
        if op == 'grad': return (11, t(action['name']))
        if op == 'drop': return (41, t(action['name']))
        if op == 'make_weak': return (42, t(action['tensor']))
        if op == 'lock': return (43, o(action['weak']))
        if op == 'drop_weak': return (44, o(action['name']))
        if op == 'drop_hold': return (45, o(action['name']))
        if op == 'save': return (46, t(action['tensor']))
        if op == 'drop_saved': return (48, o(action['name']))
        if op == 'unpack': return (56, o(action['saved_name']))
        if op == 'reset_graph': return (33, action['keep'])
        if op == 'unknown': return (999,)
        if op == 'backward':
            root = t(action['root'])
            if 'seed' not in action:
                return (7, root)
            # A real native Tensor carries each explicit seed. It is an auxiliary
            # fixture object, never a host numerical runtime or predicted handle.
            shape = self.meta[action['root']]['shape']
            record, = self.exchange([tensor_command(shape, action['seed'])], label='seed Tensor')
            seed = record['value']
            prepared.append(seed)
            return (29, root, seed)
        raise AssertionError(op)

    def check(self, expectations):
        for category, entries in expectations.items():
            for name, expectation in entries.items():
                status = expectation['status']
                expected = expectation.get('value')
                if category == 'tensors':
                    if status:
                        self.exchange([(13, self.tensors[name])], status, 'expired Tensor '+name)
                    else:
                        self.tensor_meta(name, expected)
                    continue
                if category == 'gradient':
                    records = self.exchange([(11, self.tensors[name])], status, 'gradient '+name)
                    if not status:
                        self.bits(records[0], expected, 'gradient '+name)
                elif category in ('saved', 'holds'):
                    code = 49 if category == 'saved' else 50
                    count = len(expected) if not status else 0
                    records = self.exchange([(code, self.objects[name], 0, count)], status, category+' '+name)
                    if not status:
                        self.bits(records[0], expected, category+' '+name)
                elif category in ('storage', 'version'):
                    kind = 2 if category == 'storage' else 3
                    symbol = self.meta[name]['storage_symbol' if kind == 2 else 'version_symbol']
                    handle = self.handles.bound[kind, symbol]
                    records = self.exchange([(51 if kind == 2 else 52, handle)], status, category+' '+name)
                    if status:
                        continue
                    record = records[0]
                    self.eq(record['value'], handle, category+' result')
                    self.eq(len(record['payload']), 64, category+' metadata bytes')
                    words = struct.unpack('<8Q', record['payload'])
                    self.eq(words[:2], (handle >> 32, 1), category+' generation/live')
                    if kind == 3:
                        self.eq(words[2:], (expected['counter'], expected['references'], 0, 0, 0, 0),
                                'Version counter/references')
                    else:
                        self.eq(words[3], expected['bytes'], 'Storage byte size')
                        self.eq(words[5:], (expected['strong'], expected['weak'], int(expected['dead'])),
                                'Storage strong/weak/dead')
                        aligned = 0 if expected['dead'] else (expected['bytes']+15)//16*16
                        self.eq(words[4], aligned, 'Storage aligned extent size')
                        assert words[2] % 16 == 0
                        if expected['dead']:
                            self.eq(words[2], 0, 'expired Storage extent cleared')
                        self.storage_observations.setdefault(name, []).append(list(words))
                else:
                    raise AssertionError(category)

    def run(self, case):
        for row in case['actions']:
            self.action_index = row['action_index']
            action = row['action']
            if action['op'] == 'check':
                self.check(row['expected'])
                record, = self.exchange([(53,)], label='independent object audit')
                self.eq(record['value'], 1, 'audit result')
                self.eq(record['payload'], b'', 'audit empty payload')
                continue
            actions = action['actions'] if action['op'] == 'frame' else [action]
            prepared = []
            commands = [self.wire(item, prepared) for item in actions]
            records = self.exchange(commands, row['expected_status'], action['op'])
            for seed in prepared:
                self.exchange([(41, seed)], label='release seed Tensor')
            if row['expected_status']:
                continue
            if 'published_tensor' in row:
                name, handle = action['name'], records[0]['value']
                self.tensors[name] = self.handles.bind(1, name, handle)
                self.tensor_meta(name, row['published_tensor']['value'])
            elif action['op'] in ('make_weak', 'lock', 'save'):
                kind = {'make_weak':4, 'lock':6, 'save':5}[action['op']]
                self.objects[action['name']] = self.handles.bind(kind, action['name'], records[0]['value'])
        record, stats = self.exchange([(53,), (54,)], label='final audit/stats')
        self.eq(record['value'], 1, 'final audit')
        self.eq(len(stats['payload']), 96, 'final stats bytes')
        if case['case'] == 'A-B-C-alias-weak-Saved-Hold-last-strong-and-internal-hole':
            old_extent = self.storage_observations['B'][0][2]
            new_extent = self.storage_observations['D'][0][2]
            self.eq(new_extent, old_extent, 'D reuses released interior B extent')
            assert self.tensors['D'] != self.tensors['B']
            assert self.handles.bound[2, self.meta['D']['storage_symbol']] != self.handles.bound[2, self.meta['B']['storage_symbol']]
        return list(struct.unpack('<12Q', stats['payload']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--transport-observer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-filter', default='')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary_before = sha(args.binary.read_bytes())
    assert binary_before == args.expected_sha256
    root = Path(__file__).resolve().parent
    source_paths = [Path(__file__), root/'reference_model.py', root/'scenarios.py', root/'reference-traces.json', args.transport_observer]
    source_hashes = {path.name: sha(path.read_bytes()) for path in source_paths}
    executed = args.output/'executed-sources'
    executed.mkdir()
    for path in source_paths:
        (executed/path.name).write_bytes(path.read_bytes())
    traces = json.loads((root/'reference-traces.json').read_text())
    assert traces['reference_model_sha256'] == source_hashes['reference_model.py']
    assert traces['scenario_source_sha256'] == source_hashes['scenarios.py']
    spec = importlib.util.spec_from_file_location('object_review_transport', args.transport_observer)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    results = []
    for index, case in enumerate(traces['cases']):
        if args.case_filter not in case['case']:
            continue
        directory = args.output/f'case-{index+1:02d}'
        session = observer.OwnerSession(args.binary, directory=directory, address_limit=64 << 20)
        review = Review(session)
        result = {'case': case['case'], 'status': 'FAIL'}
        try:
            result['final_stats'] = review.run(case)
            session.finish()
            result['status'] = 'PASS'
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            if session.poll() is None:
                os.kill(session.pid, signal.SIGKILL)
            try:
                session.finish(expected=-signal.SIGKILL)
            except Exception:
                pass
        result.update(frames=review.frames, exact_scalar_or_binary32_fields=review.compared_fields,
                      entire_owner_rollback_bytes=review.rollback_bytes,
                      observed_public_identities=len(review.handles.seen),
                      storage_observations=review.storage_observations)
        (directory/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        results.append(result)
    assert source_hashes == {path.name: sha(path.read_bytes()) for path in source_paths}, 'test sources changed during run'
    assert sha(args.binary.read_bytes()) == binary_before
    report = {'schema': 'heptatorch.numeric-storage-independent-native-review.v1',
              'status': 'PASS' if results and all(row['status']=='PASS' for row in results) else 'FAIL',
              'native_executable_sha256': binary_before, 'native_unchanged_during_run': True,
              'source_hashes': source_hashes, 'cases': len(results), 'rows': results,
              'passed': sum(row['status']=='PASS' for row in results),
              'frames': sum(len(row['frames']) for row in results),
              'exact_scalar_or_binary32_fields': sum(row['exact_scalar_or_binary32_fields'] for row in results),
              'entire_owner_rollback_bytes': sum(row['entire_owner_rollback_bytes'] for row in results),
              'native_target_execution': True, 'pinned_pytorch_executed': False,
              'reference': 'Predeclared exact rational and declarative ownership-obligation model, not target outputs.',
              'scope': 'Current binary32 rank<=4 objects and retain-until-release graph contract; not complete PyTorch.',
              'pytorch_api_denominator': 2590, 'full_rewrite_complete': False}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['status','cases','passed','frames','exact_scalar_or_binary32_fields','entire_owner_rollback_bytes']}))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
