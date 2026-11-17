#!/usr/bin/env python3
"""Strict direct differentiable-view replay of the immutable actual14 observations.

Only real wire calls and read-only native-owned-byte decoding execute here.
The numerical/geometry/Version/identity/lifetime oracle is the fixed Torch JSON.
Python wrapper multiplicity and diagnostic text are outside this wire surface;
warning conditions and normalized exception causes are checked explicitly.
"""
import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import lzma
import math
from pathlib import Path
import re
import shutil
import struct
import sys
import traceback

REFERENCE = '2270d5ba925bcf91c56db94fac663695e69079c74edbf80aff8fa086e7c81665'
PROBE = '1ca74384675f4f5bbe1b477be827e1d405bf3ca04bc3eeaa22e4628dfe0cc1ac'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
SUPPORT = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
SEED_VALUES = (1., -2., 3., -4., .5, -.25, 2., -1., .25, -.5, 4., -3.)
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested'}
KINDS = {3: 'MulBackward0', 43: 'ToCopyBackward0', 44: 'AsStridedBackward0'}
FUNCTIONS = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)', re.M | re.S)
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def safe(x):
    if (type(x).__name__ == 'Payload' and isinstance(getattr(x, 'data', None), bytes)
            and isinstance(getattr(x, 'alignment', None), int)):
        return {'wire_payload': {'bytes': len(x.data), 'hex': x.data.hex(),
                                 'alignment': x.alignment}}
    if isinstance(x, bytes):
        return {'bytes': len(x), 'hex': x.hex()}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (tuple, list)):
        return [safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    return x


def write_json(path, value):
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def expected_status(event):
    if event['accepted']:
        return 0
    error = event['error']
    assert error['type'] == 'RuntimeError', error
    message = error['message']
    if ('backward through the graph a second time' in message
            or 'saved tensors have already been freed' in message):
        return 10
    if ('modified by an inplace operation' in message
            and 'version' in message):
        return 6
    raise AssertionError(('UNMAPPED_REAL_EXCEPTION', error))


def event_normal(accepted, value_marker=False, value=None):
    x = {'accepted': accepted, 'outcome': 'SUCCEEDED' if accepted else 'RAISED'}
    if value_marker:
        x['value'] = value
    return x


class RequiredOperationFailed(Exception):
    pass


class Case:
    def __init__(self, args, observer, reference, row, directory):
        self.args, self.observer, self.reference, self.row = args, observer, reference, row
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.n = observer.Native(args.binary, args.source, args.protocol, directory)
        self.live, self.weak, self.tmap, self.treverse = {}, {}, {}, {}
        self.smap, self.sreverse = {}, {}
        self.failures, self.operations, self.snapshots, self.owner_records = [], [], [], []
        self.checks = self.words = self.rng_bytes = self.rollback_bytes = 0
        self.owner_blobs = {}
        self.current_path = 'initialization'
        self.gen = self.n.one(60, 1729)['value']
        start = bytes.fromhex(reference['states'][row['state_before_setup']]['hex'])
        self.n.one(64, self.gen, self.n.w.Payload(start), len(start))
        self.check('initial_rng', self.n.one(63, self.gen)['payload'], start)
        self.rng_bytes += len(start)

    def check(self, label, actual, expected, required=False):
        self.checks += 1
        if actual != expected:
            item = {'path': self.current_path + '/' + label,
                    'actual': safe(actual), 'expected': safe(expected)}
            self.failures.append(item)
            if required:
                raise RequiredOperationFailed(str(item))
            return False
        return True

    def owner(self, label):
        owner = self.n.s.owner()
        digest = sha(owner['bytes'])
        target = self.directory / 'owners' / (digest + '.bin.xz')
        if digest not in self.owner_blobs:
            target.parent.mkdir(exist_ok=True)
            packed = lzma.compress(owner['bytes'], format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64)
            target.write_bytes(packed)
            assert lzma.decompress(packed) == owner['bytes']
            self.owner_blobs[digest] = {'path': str(target.relative_to(self.directory)),
                                      'bytes': len(owner['bytes']), 'sha256': digest,
                                      'compressed_bytes': len(packed), 'compressed_sha256': sha(packed)}
        entry = {'label': label, 'frame': self.n.s.frames, 'pointer': owner['pointer'],
                 'length': owner['length'], 'sha256': digest}
        self.owner_records.append(entry)
        return owner

    def issue(self, command, label, *, partial=False):
        before = self.n.s.owner()
        status, records = self.n.w.parse(self.n.s.exchange(self.n.w.packet([command])))
        if status:
            self.check(label + '/no_records_on_frame_error', records, [])
            after = self.n.s.owner()
            self.check(label + '/rollback_pointer', after['pointer'], before['pointer'])
            self.check(label + '/rollback_length', after['length'], before['length'])
            self.check(label + '/rollback_all_bytes', after['bytes'], before['bytes'])
            self.rollback_bytes += len(before['bytes'])
            self.owner(label + '/frame_refusal')
            return status, None
        self.check(label + '/record_count', len(records), 1, True)
        self.check(label + '/record_opcode', records[0]['opcode'], command[0], True)
        record = records[0]
        if partial:
            self.check(label + '/partial_payload_bytes', len(record['payload']), 32, True)
            fields = struct.unpack('<4Q', record['payload'])
            self.check(label + '/partial_status_consistent', fields[0], record['value'])
            record['partial_fields_recorded_not_Torch_oracle_counters'] = {
                'status': fields[0], 'completed_nodes': fields[1],
                'completed_leaves': fields[2], 'failed_opcode': fields[3]}
            return record['value'], record
        return 0, record

    def mem(self):
        return self.n.memory()

    def raw_tensor(self, token=None, *, index=None):
        memory = self.mem()
        if index is None:
            index = memory.public_node(token)
        return memory.tensor(index)

    def bind_storage(self, expected, actual):
        if expected in self.smap:
            self.check('Storage_identity_stable', actual, self.smap[expected])
        else:
            self.check('Storage_identity_injective', self.sreverse.get(actual), None)
            self.smap[expected], self.sreverse[actual] = actual, expected

    def bind_tensor(self, expected, actual):
        key = (expected['tensor_identity'], expected['storage_identity'])
        value = actual['native_incarnation']
        if key in self.tmap:
            self.check('Tensor_identity_stable', value, self.tmap[key])
        else:
            self.check('Tensor_identity_injective', self.treverse.get(value), None)
            self.tmap[key], self.treverse[value] = value, key
        self.bind_storage(expected['storage_identity'], actual['native_storage_id'])

    def tensor(self, expected, actual, label):
        old = self.current_path
        self.current_path += '/' + label
        if expected is None or actual is None:
            self.check('None', actual is None, expected is None)
            self.current_path = old
            return
        self.bind_tensor(expected, actual)
        for name in ('dtype', 'element_size', 'shape', 'strides', 'numel', 'storage_bytes',
                     'storage_offset_elements', 'version', 'requires_grad', 'is_view'):
            self.check(name, actual[name], expected[name])
        self.check('storage_offset_bytes', actual['storage_offset_elements'] * actual['element_size'],
                   expected['storage_offset_bytes'])
        self.check('physical_storage_bytes', actual['storage_bytes_hex'], expected['storage_raw_hex'])
        self.check('logical_bytes', actual['logical_bytes_hex'], expected['logical_bytes_hex'])
        width = actual['element_size']
        logical = bytes.fromhex(actual['logical_bytes_hex'])
        words = [int.from_bytes(logical[i:i+width], 'little') for i in range(0, len(logical), width)]
        self.check('exact_words', words, expected['binary_words'])
        self.words += len(expected['binary_words'])
        if expected['base_tensor_identity'] is None:
            self.check('base_identity_absent', actual['native_base_incarnation'], None)
        else:
            base_key = (expected['base_tensor_identity'], expected['storage_identity'])
            self.check('base_identity_previously_observed', base_key in self.tmap, True)
            if base_key in self.tmap:
                self.check('ultimate_base_identity', actual['native_base_incarnation'], self.tmap[base_key])
        self.current_path = old

    def node_info(self, index, depth=4):
        if not index:
            return None
        memory = self.mem()
        node = memory.node(index)
        if node[5] == 0:
            target = node[76] & 65535
            if not target:
                target = index
            result = {'kind': 'AccumulateGrad', 'variable': memory.tensor(target)}
            if depth:
                result['next_edges'] = []
            return result
        assert node[5] in KINDS, ('unmapped_actual_native_node', index, node[5])
        result = {'kind': KINDS[node[5]]}
        if depth:
            count = 2 if node[5] == 3 else 1
            result['next_edges'] = [
                {'output_nr': 0, 'node': self.node_info(node[6+i], depth-1) if node[9+i] else None}
                for i in range(count)]
        return result

    def compare_node(self, expected, actual, label):
        old = self.current_path
        self.current_path += '/' + label
        if expected is None or actual is None:
            self.check('None', actual is None, expected is None)
        else:
            self.check('kind', actual['kind'], expected['kind'])
            self.check('variable_presence', 'variable' in actual, 'variable' in expected)
            if 'variable' in expected and 'variable' in actual:
                self.tensor(expected['variable'], actual['variable'], 'Accum_variable')
            self.check('next_edges_presence', 'next_edges' in actual, 'next_edges' in expected)
            if 'next_edges' in expected and 'next_edges' in actual:
                self.check('edge_count', len(actual['next_edges']), len(expected['next_edges']))
                for i, (e, a) in enumerate(zip(expected['next_edges'], actual['next_edges'])):
                    self.check('output_nr_' + str(i), a['output_nr'], e['output_nr'])
                    self.compare_node(e['node'], a['node'], 'edge_' + str(i))
        self.current_path = old

    def add_weak(self, name, token):
        self.check('new_Weak_name', name in self.weak, False, True)
        self.weak[name] = self.n.one(42, token)['value']

    def close_temporary(self, token, was_public):
        if token and not was_public:
            self.n.one(41, token)

    def gradient(self, token, expected):
        memory = self.mem()
        index = memory.public_node(token)
        node = memory.node(index)
        gindex = node[48] if node[45] else 0
        was_public = bool(memory.node(gindex)[25]) if gindex else False
        presence = self.n.one(59, token)['value']
        got = self.n.one(91, token)['value']
        self.check('gradient_presence_public', bool(presence), expected['tensor'] is not None)
        self.check('gradient_getter_public', bool(got), expected['tensor'] is not None)
        self.check('gradient_owner_binding', bool(got), bool(gindex))
        observed = self.raw_tensor(got) if got else None
        self.tensor(expected['tensor'], observed, 'gradient_tensor')
        self.check('gradient_getter_accepted', expected['accepted'], True)
        expected_warning = bool(expected['warnings'])
        retained = bool(self.n.one(94, token)['value'])
        current = self.mem().node(index)
        self.check('nonleaf_warning_condition', bool(current[5]) and not retained, expected_warning)
        if expected_warning:
            self.check('warning_count', len(expected['warnings']), 1)
            self.check('warning_type', expected['warnings'][0]['type'], 'UserWarning')
        if got and expected['tensor'] is not None:
            key = 'observed_grad_storage_' + expected['tensor']['storage_identity']
            if key not in self.weak:
                self.add_weak(key, got)
        self.close_temporary(got, was_public)
        return observed

    def check_expired(self, expected):
        self.check('Weak_names', list(self.weak), list(expected))
        result = {}
        for name, expiry in expected.items():
            if name not in self.weak:
                self.check('missing_Weak/' + name, False, True)
                continue
            status, record = self.issue((43, self.weak[name]), 'Weak/' + name)
            self.check('Weak_expired/' + name, status, 6 if expiry else 0)
            result[name] = status == 6
            if status == 0:
                self.n.one(45, record['value'])
        return result

    def saved_attributes(self, token, expected):
        index = self.mem().public_node(token)
        node = self.mem().node(index)
        op = node[5]
        keys = ('_saved_size', '_saved_storage_offset', '_saved_stride') if op == 44 else (
               ('_saved_other', '_saved_self') if op == 3 else ())
        self.check('Saved_attribute_names', list(keys), list(expected))
        observed = {}
        for name in keys:
            if name not in expected:
                continue
            event = expected[name]
            if op == 44:
                value = (list(node[12:12+node[0]]) if name == '_saved_size'
                         else list(node[16:16+node[0]]) if name == '_saved_stride'
                         else node[29] // {0:4,1:1,2:8}[node[72]])
                self.check(name + '/accepted', event['accepted'], True)
                self.check(name + '/value', value, event['value'])
                self.check(name + '/warnings', event['warnings'], [])
                observed[name] = event_normal(True, True, value)
            else:
                self.check(name + '/warnings', event['warnings'], [])
                slot = 1 if name == '_saved_other' else 0
                before = self.mem()
                publics = {i: bool(before.node(i)[25]) for i in range(1, before.q(8))
                           if before.node(i)[38] and before.node(i)[30]}
                status, record = self.issue((107, token, slot), name)
                self.check(name + '/normalized_status', status, expected_status(event))
                if status == 0:
                    got = record['value']
                    actual = self.raw_tensor(got) if got else None
                    value = event['value']
                    expected_tensor = value.get('tensor') if isinstance(value, dict) else None
                    self.tensor(expected_tensor, actual, name + '/original_saved_tensor')
                    observed[name] = event_normal(True, True, {'tensor': actual} if got else None)
                    if got:
                        saved_node = self.mem().public_node(got)
                        self.close_temporary(got, publics.get(saved_node, False))
                else:
                    observed[name] = {'accepted': False, 'normalized_wire_status': status}
        return observed

    def snapshot(self, expected, label):
        self.current_path = label
        self.check('getter_order', list(self.live), expected['tensor_getter_order'], True)
        output = {'label': label, 'tensor_getter_order': list(self.live), 'tensors': {}}
        for name in expected['tensor_getter_order']:
            old = self.current_path
            self.current_path += '/' + name
            token = self.live[name]
            e = expected['tensors'][name]
            before = self.raw_tensor(token)
            self.tensor(e['before_grad_fn_getter'], before, 'before_grad_fn_getter')
            meta = self.n.w.meta(self.n.one(13, token))
            memory = self.mem()
            index = memory.public_node(token)
            node = memory.node(index)
            fn = self.node_info(index) if node[5] else None
            self.check('grad_fn_getter_accepted', e['grad_fn_getter']['accepted'], True)
            self.check('grad_fn_getter_warnings', e['grad_fn_getter']['warnings'], [])
            self.compare_node(e['grad_fn_getter']['node'], fn, 'grad_fn_getter_node')
            self.check('requires_after_getter', bool(node[4]), e['requires_grad_after_grad_fn_getter'])
            self.check('public_metadata_requires', bool(meta['requires_grad']), bool(node[4]))
            self.check('public_metadata_shape', meta['shape'][:meta['rank']], list(node[12:12+node[0]]))
            self.check('public_metadata_strides', meta['strides'][:meta['rank']], list(node[16:16+node[0]]))
            self.check('leaf_getter_accepted', e['is_leaf_after_grad_fn_getter']['accepted'], True)
            self.check('leaf_after_getter', node[5] == 0, e['is_leaf_after_grad_fn_getter']['value'])
            retained = bool(self.n.one(94, token)['value'])
            self.check('retains_grad_accepted', e['retains_grad']['accepted'], True)
            self.check('retains_grad', retained, e['retains_grad']['value'])
            logical = self.n.one(98, token, 0, node[1])['payload']
            self.check('logical_public_vs_real_Storage', logical.hex(), self.raw_tensor(token)['logical_bytes_hex'])
            gradient = self.gradient(token, e['gradient'])
            output['tensors'][name] = {'before_grad_fn_getter': before, 'grad_fn': fn,
                                      'requires_after_getter': bool(node[4]),
                                      'is_leaf_after_getter': node[5] == 0,
                                      'retains_grad': retained, 'gradient': gradient}
            self.current_path = old
        relations = {}
        for left, right in itertools.combinations(self.live, 2):
            a, b = self.raw_tensor(self.live[left]), self.raw_tensor(self.live[right])
            rel = {'same_Python_object': a['native_incarnation'] == b['native_incarnation'],
                   'same_Tensor': a['native_incarnation'] == b['native_incarnation'],
                   'same_Storage': a['native_storage_id'] == b['native_storage_id'],
                   'same_current_Version_value': a['version'] == b['version']}
            relations[left+'|'+right] = rel
        self.check('all_live_identity_relations', relations, expected['relations'])
        current = {}
        for name, token in self.live.items():
            self.n.one(13, token)
            m = self.mem()
            index = m.public_node(token)
            current[name] = index if m.node(index)[5] else None
            self.check('edge_getter_accepted/' + name, expected['edge_getters'][name]['accepted'], True)
            self.check('edge_getter_warnings/' + name, expected['edge_getters'][name]['warnings'], [])
        saved, edges = {}, {}
        for name, index in current.items():
            self.current_path = label + '/Saved_and_edges/' + name
            saved[name] = self.saved_attributes(self.live[name],
                         expected['saved_getters_after_tensor_getters'][name])
            children = []
            if index:
                node = self.mem().node(index)
                count = 2 if node[5] == 3 else 1
                for i in range(count):
                    child = node[6+i] if node[9+i] else None
                    children.append({'output_nr': 0, 'node': self.node_info(child),
                                     'matches_live_current_grad_fn':
                                     [other for other, value in current.items()
                                      if child is not None and child == value]})
            wanted = expected['edges_after_all_tensor_getters'][name]
            self.check('edge_count', len(children), len(wanted))
            for i, (a, e) in enumerate(zip(children, wanted)):
                self.check('output_nr/' + str(i), a['output_nr'], e['output_nr'])
                self.check('live_grad_fn_identity/' + str(i), a['matches_live_current_grad_fn'],
                           e['matches_live_current_grad_fn'])
                self.compare_node(e['node'], a['node'], 'edge_node/' + str(i))
            edges[name] = children
        self.current_path = label
        output.update(relations=relations, saved=saved, edges=edges,
                      expired=self.check_expired(expected['expired']))
        rng = self.n.one(63, self.gen)['payload']
        expected_rng = bytes.fromhex(self.reference['states'][expected['state']]['hex'])
        self.check('complete_5056_MT', rng, expected_rng)
        self.rng_bytes += len(expected_rng)
        output['rng_sha256'] = sha(rng)
        self.check('native_audit', self.n.one(53)['value'], 1)
        owner = self.owner(label)
        output['owner_sha256'], output['frame'] = sha(owner['bytes']), self.n.s.frames
        self.snapshots.append(output)

    def view(self, source, shape, strides, offset):
        cfg = struct.pack('<10Q', len(shape), *shape, *([0]*(4-len(shape))),
                          *strides, *([0]*(4-len(shape))), offset)
        return (111, source, self.n.w.Payload(cfg), 80)

    def make_base(self):
        kind = self.row['base_recipe']
        if kind == 'matrix' or kind == 'transpose_clone':
            token = self.n.create(0, [2,3], struct.pack('<6f', 1.,2.,3.,4.,5.,6.))
            if kind == 'transpose_clone':
                transposed = self.n.one(6, token, 0, 1)['value']
                result = self.n.one(18, transposed)['value']
                self.n.one(41, transposed)
                self.n.one(41, token)
                return result
            return token
        if kind.startswith('flat'):
            count = int(kind[4:])
            return self.n.create(0, [count], struct.pack('<'+'f'*count, *range(1,count+1)))
        if kind == 'empty_offset':
            base = self.n.create(0, [0], b'')
            view = self.n.one(*self.view(base, [2,0,3], [17,101,2], 7))['value']
            result = self.n.one(19, view)['value']
            self.n.one(41, view)
            self.n.one(41, base)
            return result
        raise AssertionError(('unmapped_base_recipe', kind))

    def published(self, name, token):
        self.live[name] = token
        self.add_weak(name + '_storage', token)

    def execute(self, event, event_index):
        label = event['label']
        self.snapshot(event['before'], 'event%02d/%s/before' % (event_index, label))
        self.current_path = 'event%02d/%s/operation' % (event_index, label)
        commands, token, publish, status, record = [], None, None, 0, None
        wanted = expected_status(event)
        if label == 'construct_base':
            token = self.make_base()
            publish = 'base'
        elif label == 'construct_factor':
            token = self.n.create(0, [], struct.pack('<f',2.))
            publish = 'factor'
        elif label == 'construct_seed':
            shape = event['returned']['tensor']['shape']
            count = math.prod(shape)
            values = [SEED_VALUES[i % len(SEED_VALUES)] for i in range(count)]
            token = self.n.create(0, shape, struct.pack('<'+'f'*count, *values))
            publish = 'seed'
        elif label.startswith('as_strided_'):
            publish = label[len('as_strided_'):]
            if publish == 'view':
                source = self.live['middle' if self.row['family'] == 'nonleaf_direct' else 'base']
                shape = self.row.get('view_size',[3,2])
                strides = self.row.get('view_stride',[1,3])
                offset = self.row.get('view_storage_offset',0)
            else:
                source = self.live['base' if publish == 'view1' else 'view1']
                shape, strides = self.row[publish+'_size'], self.row[publish+'_stride']
                offset = self.row[publish+'_storage_offset']
            command = self.view(source,shape,strides,offset)
            status, record = self.issue(command,label)
            token = record['value'] if status == 0 else None
            commands = [command]
        elif label in ('base_requires_grad_true','base_requires_grad_true_at_version_zero',
                       'view1_own_requires_grad_true'):
            name = 'view1' if label.startswith('view1_') else 'base'
            command = (90,self.live[name],1)
            status,record=self.issue(command,label)
            token=self.live[name] if status==0 else None
            commands=[command]
        elif label.startswith('retain_grad_'):
            command=(93,self.live[label[len('retain_grad_'):]])
            status,record=self.issue(command,label);commands=[command]
        elif label in ('mul_base_factor','mul_view_by_same_view'):
            left,right,publish=('base','factor','middle') if label=='mul_base_factor' else ('view','view','root')
            command=(2,3,self.live[left],self.live[right])
            status,record=self.issue(command,label)
            token=record['value'] if status==0 else None;commands=[command]
        elif label.startswith('to_copy_'):
            publish='old_copy' if label=='to_copy_before_mutation_or_drop' else 'new_copy'
            command=(109,self.live['view2'],0,0)
            status,record=self.issue(command,label)
            token=record['value'] if status==0 else None;commands=[command]
        elif label.startswith('hold_'):
            name=('middle' if label=='hold_first_middle_gradient' else
                  'view1' if label=='hold_first_view1_gradient' else
                  'view2' if label=='hold_first_view2_gradient' or label=='hold_final_view_gradient' else 'view')
            publish='held_middle_grad' if name=='middle' else 'held_view1_grad' if name=='view1' else (
                    'held_view2_grad' if label=='hold_first_view2_gradient' else 'held_view_grad')
            command=(91,self.live[name])
            status,record=self.issue(command,label)
            token=record['value'] if status==0 else None;commands=[command]
        elif label.startswith('clear_before_'):
            name=label.rsplit('_',1)[-1]
            command=(92,self.live[name],0)
            status,record=self.issue(command,label);commands=[command]
        elif label=='base_add_scalar_one_no_grad':
            command=(95,self.live['base'],self.n.w.bits(1.))
            status,record=self.issue(command,label);commands=[command]
        elif label.startswith('backward_'):
            if self.row['family']=='saved_version':
                root='root'
            elif label=='backward_new_current_chain_after_refresh':
                root='new_copy'
            elif 'old_copy' in self.live and self.row['family'] in ('chain_refresh','weak_lifetime'):
                root='old_copy'
            else:
                root='view2' if 'view2' in self.live else 'view'
            retain=label in ('backward_first_retained','backward_old_copy_first_retained')
            command=(58,self.live[root],self.live['seed'],int(retain))
            status,record=self.issue(command,label,partial=True);commands=[command]
        elif label.startswith('final_drop_') or label in (
                'drop_public_base_before_backward','drop_public_intermediate_before_backward',
                'drop_copy_after_default_backward','drop_last_public_view'):
            name=(label[len('final_drop_'):] if label.startswith('final_drop_') else {
                 'drop_public_base_before_backward':'base','drop_public_intermediate_before_backward':'view1',
                 'drop_copy_after_default_backward':'old_copy','drop_last_public_view':'view2'}[label])
            drop_token=self.live[name]
            if sum(t==drop_token for t in self.live.values())==1:
                command=(41,drop_token)
                status,record=self.issue(command,label);commands=[command]
            del self.live[name]
        else:
            raise AssertionError(('UNMAPPED_EVENT',label))
        self.check('normalized_operation_status',status,wanted)
        operation={'label':label,'normalized_actual_status':status,'normalized_reference_status':wanted,
                   'reference_event':event_index,'commands':commands,'record':record}
        if wanted==0 and status!=0:
            self.operations.append(operation)
            raise RequiredOperationFailed(('native rejected accepted operation',label,status))
        if wanted!=0 and status==0:
            self.operations.append(operation)
            raise RequiredOperationFailed(('native accepted rejected operation',label,wanted))
        if status==0:
            returned=event.get('returned')
            if isinstance(returned,dict) and 'tensor' in returned:
                self.check('returned_Tensor_present',bool(token),True,True)
                actual=self.raw_tensor(token)
                self.tensor(returned['tensor'],actual,'returned_Tensor')
                self.check('returned_is_existing',{name:t==token for name,t in self.live.items()},
                           event['returned_is_existing'])
                operation['returned_tensor']=actual
            else:
                self.check('returned_primitive',returned,None)
            if publish is not None:
                self.check('publish_new_name',publish in self.live,False,True)
                self.published(publish,token)
        self.operations.append(operation)
        self.snapshot(event['after'],'event%02d/%s/after'%(event_index,label))

    def run(self):
        error=None
        try:
            for i,event in enumerate(self.row['events']):
                # before_final_drops is an additional real snapshot before the final_drop loop.
                if event['label'].startswith('final_drop_') and not getattr(self,'final_snap_done',False):
                    self.snapshot(self.row['before_final_drops'],'before_final_drops')
                    self.final_snap_done=True
                self.execute(event,i)
            if not getattr(self,'final_snap_done',False):
                self.snapshot(self.row['before_final_drops'],'before_final_drops')
            self.current_path='final'
            self.check('all_public_names_dropped',self.live,{})
            expired=self.check_expired(self.row['final_expired'])
            state=self.n.one(63,self.gen)['payload']
            expected=bytes.fromhex(self.reference['states'][self.row['state_after']]['hex'])
            self.check('final_complete_MT',state,expected);self.rng_bytes+=len(expected)
            self.check('final_native_audit',self.n.one(53)['value'],1)
            self.owner('final_before_observer_release')
            for token in self.weak.values():self.n.one(44,token)
            self.n.one(66,self.gen)
        except BaseException as failure:
            error={'type':type(failure).__name__,'message':str(failure),
                   'traceback':traceback.format_exc(),'at':self.current_path}
            try:self.owner('first_failure')
            except BaseException as nested:
                error['owner_capture_error']=repr(nested)
        finally:
            try:
                self.n.finish()
            except BaseException as failure:
                error=error or {'type':'EOF_FAILURE','message':repr(failure),'at':self.current_path}
        report={'index':self.row['index'],'name':self.row['name'],
                'status':'PASS' if error is None and not self.failures else 'FAIL',
                'assertions':self.checks,'exact_words':self.words,
                'full_rng_bytes':self.rng_bytes,'rollback_bytes':self.rollback_bytes,
                'frames':self.n.s.frames,'exit_code':self.n.s.exit_code,'error':error,
                'mismatches':self.failures,'operations':self.operations,
                'snapshots':self.snapshots,'owner_records':self.owner_records,
                'owner_blobs':self.owner_blobs,'reference_event_count':len(self.row['events'])}
        write_json(self.directory/'report.json',report)
        return report


def admit_reference(raw, probe):
    assert sha(raw)==REFERENCE and sha(probe)==PROBE
    data=json.loads(raw)
    assert data['status']=='OBSERVED' and data['cases']==14 and len(data['rows'])==14
    assert data['probe_error_count']==0 and not data['probe_errors']
    assert data['source_files_unchanged'] is True and data['probe_sha256']==PROBE
    checks=data['oracle_provenance']['checks']
    assert data['oracle_provenance']['status']=='PASS' and set(checks)==CHECKS and all(v is True for v in checks.values())
    assert data['num_threads']==1 and data['cpu_capability']=='AVX512'
    assert data['environment_before']==data['environment_after']
    assert data['environment_before']['deterministic_algorithms'] is True
    assert data['environment_before']['deterministic_fill'] is True
    assert data['environment_restored'] is True and data['original_outer_rng_restored'] is True
    assert data['operation_totals']=={'attempted':209,'succeeded':207,'raised':2}
    assert SUPPORT in data['inputs_sha256'].values() and LOCK in data['inputs_sha256'].values()
    for key,state in data['states'].items():
        rawstate=bytes.fromhex(state['hex'])
        assert len(rawstate)==5056 and state['bytes']==5056 and sha(rawstate)==key
    failures=[]
    for i,row in enumerate(data['rows']):
        assert row['index']==i and row['status']=='OBSERVED'
        snapshots = [row['before_final_drops']]
        for event in row['events']:
            snapshots.extend((event['before'], event['after']))
            if event['label'] == 'hold_final_view_gradient':
                assert row['family'] == 'weak_lifetime' and i == 13
                assert 'view2' in event['before']['tensor_getter_order']
            if not event['accepted']:failures.append((i,event['label'],expected_status(event)))
        for snapshot in snapshots:
            assert all(relation['same_Python_object'] == relation['same_Tensor']
                       for relation in snapshot['relations'].values())
    assert failures==[(7,'backward_retry_after_upstream_default',10),
                      (12,'backward_after_saved_version_change',6)]
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','build-report','protocol','observer','reference','probe','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    for name in ('source','native','observer'):
        ap.add_argument('--expected-'+name+'-sha256',required=True)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--expected-functions',type=int,required=True)
    ap.add_argument('--expected-native-units',type=int,required=True)
    a=ap.parse_args()
    paths=[Path(__file__).resolve(),a.binary,a.source,a.build_report,a.protocol,
           a.protocol.with_name('owner_client.py'),a.observer,a.reference,a.probe]
    inputs={p.resolve():p.read_bytes() for p in paths}
    assert sha(inputs[a.binary.resolve()])==a.expected_native_sha256
    assert sha(inputs[a.source.resolve()])==a.expected_source_sha256
    assert sha(inputs[a.observer.resolve()])==a.expected_observer_sha256
    assert sha(inputs[a.protocol.resolve()])==PROTOCOL
    assert sha(inputs[a.protocol.with_name('owner_client.py').resolve()])==CLIENT
    data=admit_reference(inputs[a.reference.resolve()],inputs[a.probe.resolve()])
    source=inputs[a.source.resolve()]
    functions=FUNCTIONS.findall(source)
    assert len(functions)==a.expected_functions
    build=json.loads(inputs[a.build_report.resolve()])
    assert build['status']=='PASS' and build['source_sha256']==a.expected_source_sha256
    assert build['native_sha256']==a.expected_native_sha256
    assert build['descriptor_bytes']==672 and build['functions']==a.expected_functions and len(build['units'])==a.expected_native_units
    a.output.mkdir(parents=True,exist_ok=False)
    common={'schema':'heptatorch.differentiable-view-actual14-native-replay.v1',
            'reference_sha256':REFERENCE,'probe_sha256':PROBE,
            'source_sha256':a.expected_source_sha256,'native_sha256':a.expected_native_sha256,
            'observer_sha256':a.expected_observer_sha256,
            'descriptor_bytes':672,'functions':a.expected_functions,'native_units':a.expected_native_units,
            'torch_executed_here':False,'full_pytorch_rewrite_complete':False,
            'closed_schema_count':0,'pytorch_api_denominator':2590,
            'expected_cases':14,'reference_events':209,
            'fixed_configuration':{'cpu_capability':'AVX512','threads':1,
                                   'deterministic_algorithms':True,'deterministic_fill':True},
            'comparison_scope':'All logical and physical bytes, geometry, Versions, Tensor/Storage relations, immediate/captured edges, Saved fields/status, grad None/retain/retry effects, final Weak, full5056 state; no Python multi-wrapper reference-count claim.',
            'diagnostics_scope':'RuntimeError causes normalized to exact wire statuses6/10; raw reference error/warning text remains in immutable reference. Nonleaf grad-warning predicate is checked; no Python warning-text frontend claim.',
            'reader_revision':'R5_EXPLICIT_HIA_PROFILE_BINDING',
            'same_Python_object_scope':'Every fixed actual14 live relation is admitted to have same_Python_object == same_Tensor. The capability comparison covers precisely these recorded relations, not general multiple Python wrappers.',
            'partial_prefix_scope':'Native58 status and the actual before/after Tensor gradients, retained hooks, Saved release and Versions are strictly compared. Its three additional execution counters are preserved verbatim/decoded, not represented as PyTorch-observed counters.'}
    if a.prepare_only:
        report={**common,'status':'PREPARED_NOT_RUN','native_executed':False,
                'all_inputs_unchanged':all(p.read_bytes()==b for p,b in inputs.items()),
                'inputs_sha256':{str(p):sha(b) for p,b in inputs.items()}}
        write_json(a.output/'report.json',report)
        print(json.dumps({k:report[k] for k in ('status','native_executed','expected_cases','reference_events')}))
        return 0
    for unit in build['units']:
        for suffix,key in (('.hepta','source_sha256'),('.o','object_sha256')):
            path=(a.build_report.parent/(unit['name']+suffix)).resolve()
            raw=path.read_bytes()
            assert sha(raw)==unit[key],(path,sha(raw),unit[key])
            inputs[path]=raw
    assert all(p.read_bytes()==b for p,b in inputs.items())
    copied=a.output/'inputs';copied.mkdir()
    identity={}
    for index,(path,raw) in enumerate(inputs.items()):
        dst=copied/('%03d-'%index+path.name)
        dst.write_bytes(raw)
        identity[str(path)]={'copy':str(dst.relative_to(a.output)),'bytes':len(raw),'sha256':sha(raw)}
    write_json(a.output/'input-custody.json',identity)
    observer=load_module(a.observer,'direct_view14_native_observer')
    rows=[]
    for row in data['rows']:
        result=Case(a,observer,data,row,a.output/('case-%02d'%row['index'])).run()
        rows.append({k:result[k] for k in ('index','name','status','assertions','exact_words',
                    'full_rng_bytes','rollback_bytes','frames','exit_code','error','mismatches')})
        print(json.dumps({'case':result['index'],'status':result['status'],'frames':result['frames'],
                          'mismatches':len(result['mismatches']),'error':result['error']['type'] if result['error'] else None}),flush=True)
    unchanged=all(p.read_bytes()==raw for p,raw in inputs.items())
    report={**common,'status':'PASS' if all(r['status']=='PASS' for r in rows) and unchanged else 'FAIL',
            'native_executed':True,'cases':len(rows),
            'passed_cases':sum(r['status']=='PASS' for r in rows),
            'failed_cases':sum(r['status']!='PASS' for r in rows),
            'all_inputs_unchanged':unchanged,'rows':rows,
            'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()}}
    for key in ('assertions','exact_words','full_rng_bytes','rollback_bytes','frames'):
        report[key]=sum(r[key] for r in rows)
    report['all_sessions_eof_exit_zero']=all(r['exit_code']==0 for r in rows)
    write_json(a.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','cases','passed_cases','failed_cases','frames','assertions','all_sessions_eof_exit_zero')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
