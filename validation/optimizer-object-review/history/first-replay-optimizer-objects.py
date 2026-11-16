"""Replay frozen, actual Torch Optimizer/leaf-flag observations on native HIA.

This program only serializes requests, reads native replies and compares them
with a separately observed reference. It imports no HIA authoring or numeric
implementation. SGD words and available object Versions are exact. Missing
public grad objects, retained nonleaf parameters, and Adam objects stay OPEN.
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

CONTROLS_SHA = '43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
REFERENCES = {
    'optimizer': ('7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160',
                  'f69340170dc2e55886bca73a8d37265ba2c95a8c793c7633c7e8a22234cfa386', 24),
    'requires': ('1bb9045b680cef2943eada06dd3288d2404ce9aada24b5f14a396a11c17f1844',
                 'aeccf1ed21501682df772bbc9e05ccdaf821e93bb243a6f2a045c05035e56ad3', 6),
}
CONFIG_KEYS = ('lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize')
DEFAULTS = dict(lr=.125, momentum=0., dampening=0., weight_decay=0., nesterov=False, maximize=False)
OPEN_CASES = {
    20: 'Public .grad Tensor/Storage/Version aliases are not exposed by this native profile.',
    21: 'Actual lazy Adam step/moment objects are not implemented by algorithm 0 (SGD).',
    22: 'Actual lazy AdamW step/moment objects are not implemented by algorithm 0 (SGD).',
    23: 'Adam duplicate occurrence planning and its real state objects remain unimplemented.',
    24: 'Actual Adam state Tensor ownership and external mutation remain unimplemented.',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Replay:
    def __init__(self, c, codec):
        self.c, self.b = c, codec
        self.params, self.opts, self.members = {}, {}, {}
        self.handle_incarnations, self.ref_tensor, self.ref_storage = {}, {}, {}
        self.steps, self.extra = [], {}

    def eq(self, actual, expected, label):
        assert actual == expected, (label, actual, expected)
        self.c.comparisons += 1

    def words(self, values):
        return self.b.Blob(struct.pack('<'+'Q'*len(values), *values))

    def config_words(self, config):
        return [self.b.bits(config[k]) for k in CONFIG_KEYS[:4]] + [int(config[k]) for k in CONFIG_KEYS[4:]]

    def group_words(self, names, overrides):
        mask = sum(1 << CONFIG_KEYS.index(k) for k in overrides)
        vals = [self.b.bits(overrides[k]) if k in overrides and i < 4
                else int(overrides[k]) if k in overrides else 0 for i, k in enumerate(CONFIG_KEYS)]
        return [len(names), mask, *vals, *(self.params[name] for name in names)]

    def opt_create(self, groups, overrides=None, status=0):
        cfg = dict(DEFAULTS, **(overrides or {}))
        body = [len(groups)]
        for names, local in groups:
            body.extend(self.group_words(names, local))
        rec = self.c.one(71, 0, self.words(self.config_words(cfg)), self.words(body), status=status)
        if status:
            return {'header_status': status}
        opt = rec['value']
        self.eq((opt >> 24) & 255, 8, 'Optimizer typed kind')
        occurrences = [self.params[name] for names, _ in groups for name in names]
        self.eq(list(self.b.words(rec)), [len(groups), len(occurrences), len(set(occurrences)),
                                        len(occurrences)-len(set(occurrences))], 'creation inventory')
        self.members[opt] = [list(names) for names, _ in groups]
        return opt

    def opt_add(self, opt, names, overrides=None, status=0):
        rec = self.c.one(79, opt, self.words(self.group_words(names, overrides or {})), status=status)
        if status:
            return {'header_status': status}
        self.members[opt].append(list(names))
        flattened = [self.params[name] for group in self.members[opt] for name in group]
        self.eq(rec['value'], 1, 'add group result')
        self.eq(list(self.b.words(rec)), [len(self.members[opt]), len(flattened), len(set(flattened)),
                                        len(flattened)-len(set(flattened))], 'add inventory')
        return {'header_status': 0}

    def bind_incarnation(self, tid, incarnation):
        assert incarnation > 0
        if tid in self.handle_incarnations:
            self.eq(self.handle_incarnations[tid], incarnation, 'same live token incarnation')
        self.handle_incarnations[tid] = incarnation
        for ref, value in list(self.ref_tensor.items()):
            if value == ('public', tid):
                self.ref_tensor[ref] = ('incarnation', incarnation)

    def identity(self, table, ref, value, label):
        if ref in table:
            self.eq(table[ref], value, label+' stable identity')
        else:
            assert value not in table.values(), (label+' distinct reference identities collided', ref, value)
            table[ref] = value
            self.c.comparisons += 1

    def tensor(self, tid, info, *, gradient=False):
        result = {}
        if not gradient:
            meta = list(self.b.words(self.c.one(13, tid)))
            self.eq(len(meta), 16, 'metadata width')
            rank = len(info['shape'])
            self.eq(meta[0], rank, 'rank')
            self.eq(meta[1], len(info['binary32_words']), 'numel')
            self.eq(meta[2], int(info['requires_grad']), 'requires_grad')
            self.eq(meta[3:3+rank], info['shape'], 'shape')
            self.eq(meta[7:7+rank], info['strides'], 'strides')
            self.eq(meta[13], info['version'], 'Version counter')
            self.eq(meta[14], info['storage_offset_bytes'], 'storage offset bytes')
            self.eq(meta[15], info['storage_bytes'], 'Storage byte size')
            key = ('incarnation', self.handle_incarnations[tid]) if tid in self.handle_incarnations else ('public', tid)
            self.identity(self.ref_tensor, info['tensor_identity'], key, 'Tensor')
            self.identity(self.ref_storage, info['storage_identity'], meta[11], 'Storage')
            result['metadata'] = meta
        rec = self.c.one(11 if gradient else 10, tid)
        expected = info['binary32_words']
        self.eq(rec['value'], len(expected), 'value count')
        assert len(rec['payload']) == 4*len(expected)
        got = list(struct.unpack('<'+'I'*len(expected), rec['payload']))
        assert got == expected, ('gradient bits' if gradient else 'tensor bits', tid, got, expected)
        self.c.comparisons += len(expected)
        result['binary32_words'] = got
        return result

    def groups(self, opt, expected, params):
        rows, locations = [], {}
        self.eq(len(self.members[opt]), len(expected), 'group count')
        ref_to_name = {info['tensor_identity']: name for name, info in params.items()}
        for gi, wanted in enumerate(expected):
            rec = self.c.one(74, opt, gi)
            vals = list(self.b.words(rec))
            self.eq(rec['value'], len(wanted['parameters']), 'group occurrences')
            self.eq(len(vals), 8+2*rec['value'], 'group payload width')
            self.eq(vals[:8], [gi, len(wanted['parameters']), *self.config_words(wanted['config'])], 'group configuration')
            self.eq([ref_to_name[identity] for identity in wanted['parameters']], self.members[opt][gi], 'ordered membership')
            for pi, ref in enumerate(wanted['parameters']):
                inc, public = vals[8+2*pi:10+2*pi]
                name = ref_to_name[ref]
                self.eq(public, self.params[name], 'current public parameter token')
                self.bind_incarnation(public, inc)
                if name in locations:
                    self.eq(locations[name][2], inc, 'duplicate occurrence identity')
                else:
                    locations[name] = (gi, pi, inc)
            rows.append(vals)
        return rows, locations

    def state(self, opt, location, expected):
        gi, pi, parameter_inc = location
        rec = self.c.one(76, opt, gi, pi)
        vals = list(self.b.words(rec))
        self.eq(len(vals), 4, 'state inventory width')
        self.eq(vals[0], parameter_inc, 'state key is Tensor incarnation')
        self.eq(rec['value'], int(expected['entry_exists']), 'lazy state entry presence')
        self.eq(vals[1], int(expected['entry_exists']), 'state fields mask')
        if not expected['entry_exists']:
            self.eq(expected['fields'], {}, 'reference absent fields')
            self.eq(vals[2:], [0, 0], 'no state Tensor exists')
            self.eq(self.c.one(77, opt, gi, pi, 0)['value'], 0, 'read absent state does not create it')
            self.eq(list(self.b.words(self.c.one(76, opt, gi, pi))), vals, 'absent state getter is read-only')
            return {'inventory': vals, 'momentum': None}
        self.eq(set(expected['fields']), {'momentum_buffer'}, 'SGD state field set')
        state = self.c.make(77, opt, gi, pi, 0)
        if vals[3]:
            self.eq(state, vals[3], 'state getter preserves current public token')
        self.bind_incarnation(state, vals[2])
        tensor = self.tensor(state, expected['fields']['momentum_buffer']['tensor'])
        # A transient observer must not retain a state Tensor after its owner is dropped.
        # Only close a public token created by this read; pre-existing user tokens stay live.
        if not vals[3]:
            self.c.one(41, state)
        return {'inventory': vals, 'momentum': tensor}

    def snapshot(self, expected, opt=None, params=None):
        names = self.params if params is None else params
        result = {'parameters': {}, 'gradients': {}}
        locations = {}
        if opt is not None:
            result['groups'], locations = self.groups(opt, expected['groups'], expected['parameters'])
            result['states'] = {}
        for name, tid in names.items():
            result['parameters'][name] = self.tensor(tid, expected['parameters'][name])
            gradient = expected['gradients'][name]
            rec = self.c.one(59, tid)
            self.eq(rec['payload'], b'', 'gradient presence payload')
            self.eq(rec['value'], int(gradient is not None), 'gradient None/present')
            if gradient is None:
                rec = self.c.one(11, tid)
                self.eq(rec['value'], len(expected['parameters'][name]['binary32_words']), 'None compatibility read count')
                self.eq(rec['payload'], bytes(4*rec['value']), 'None compatibility read zeros')
                result['gradients'][name] = None
            else:
                result['gradients'][name] = self.tensor(tid, gradient, gradient=True)
            if opt is not None:
                wanted = expected['optimizer_state'][name]
                if name in locations:
                    result['states'][name] = self.state(opt, locations[name], wanted)
                else:
                    self.eq(wanted, {'entry_exists': False, 'fields': {}}, 'nonmember has no state entry')
                    result['states'][name] = None
        if opt is not None:
            self.eq(sum(v is not None and v['momentum'] is not None for v in result['states'].values()),
                    expected['state_entries'], 'unique state entry count')
        return result

    def backward(self, target, values=None, retain=0, error=0, shape=None):
        if values is None:
            seed = 0
        else:
            if shape is None:
                meta = self.b.words(self.c.one(13, target))
                shape = list(meta[3:3+meta[0]])
            seed = self.c.new(shape, values)
        rec = self.c.one(58, target, seed, retain)
        vals = list(self.b.words(rec))
        self.eq(len(vals), 4, 'backward result width')
        self.eq(rec['value'], error, 'backward runtime error')
        self.eq(vals[0], error, 'backward payload error')
        if seed:
            self.c.one(41, seed)
        return {'header_status': 0, 'runtime_error': error, 'backward_result': vals}

    def step(self, opt):
        rec = self.c.one(72, opt)
        self.eq((rec['value'], rec['payload']), (1, b''), 'Optimizer step result')
        return {'header_status': 0}

    def clear(self, opt, none=True):
        rec = self.c.one(73, opt, int(none))
        self.eq((rec['value'], rec['payload']), (1, b''), 'Optimizer zero_grad result')
        return {'header_status': 0}

    def flag(self, tid, enabled):
        rec = self.c.one(90, tid, int(enabled))
        self.eq((rec['value'], rec['payload']), (tid, b''), 'requires_grad_ returns same Tensor token')
        return {'header_status': 0}

    def action(self, actual, function, opt=None, params=None):
        before = self.snapshot(actual['before'], opt, params)
        result = function()
        error = result.get('header_status', 0) or result.get('runtime_error', 0)
        self.eq(actual['outcome'], 'RAISED' if error else 'SUCCEEDED', 'reference action outcome')
        after = self.snapshot(actual['after'], opt, params)
        self.steps.append({'action': actual['action'], 'before': before, 'native_result': result, 'after': after})

    def actions(self, actual, functions, opt=None, params=None):
        self.eq(len(actual), len(functions), 'fixed recipe action count')
        for row, fn in zip(actual, functions):
            self.action(row, fn, opt, params)

    def p(self, name, values, shape=None, requires=True):
        self.params[name] = self.c.new([len(values)] if shape is None else shape, values, requires)
        return self.params[name]

    def opt_state_handle(self, opt, gi=0, pi=0):
        inv = self.b.words(self.c.one(76, opt, gi, pi))
        assert inv[1] == 1
        tid = self.c.make(77, opt, gi, pi, 0)
        self.bind_incarnation(tid, inv[2])
        return tid

    def storage_expiry(self, weak, expected):
        self.c.expiry(weak, expected)

    def one_version_literal_mutation(self, tensor, actual_after):
        before_stats = list(self.b.words(self.c.one(54)))
        before_meta = list(self.b.words(self.c.one(13, tensor)))
        bits = actual_after['binary32_words']
        assert bits
        temporary = 0xfe000001
        commands = [(55, tensor), (13, temporary)]
        commands += [(12, temporary, i, value) for i, value in enumerate(bits)]
        commands += [(12, tensor, 0, bits[0]), (41, temporary)]
        records = self.c.run(commands, label='explicit one-Version literal-write fixture')
        temporary_meta = list(self.b.words(records[1]))
        after_stats = list(self.b.words(self.c.one(54)))
        self.eq(temporary_meta[11], before_meta[11], 'fixture aliases actual Storage')
        assert temporary_meta[12] != before_meta[12]
        self.eq(temporary_meta[13], 0, 'fixture uses independent fresh Version')
        self.eq(after_stats[5:11], before_stats[5:11], 'fixture leaves no extra live T/S/V/W/Saved/Hold')
        self.eq(after_stats[4], before_stats[4], 'fixture frees all temporary extents')
        self.c.one(13, records[0]['value'], status=1)
        self.extra['literal_mutation_fixture'] = {
            'purpose': 'Reproduce an independently observed, fixed input mutation and exactly one shared Version increment.',
            'not_implemented_by_this_fixture': 'Tensor.add_ API or general in-place arithmetic',
            'literal_words_source': 'actual reference alias_after_write.binary32_words',
            'temporary_metadata': temporary_meta, 'before_stats': before_stats, 'after_stats': after_stats}

    def optimizer_case(self, number, obs):
        if number in (1, 2):
            p = self.p('p', [1., -2.]); opt = self.opt_create([(['p'], {})], obs['config'])
            f = [lambda: self.step(opt), lambda: self.backward(p, [0., 0.]), lambda: self.step(opt),
                 lambda: self.clear(opt), lambda: self.step(opt), lambda: self.backward(p, [2., -1.]), lambda: self.step(opt)]
            self.actions(obs['actions'], f, opt)
        elif number in (3, 4, 5, 6):
            p = self.p('p', [1., -2., .5]); opt = self.opt_create([(['p'], {})], obs['config'])
            f = []
            for i, values in enumerate(([2., -1., .5], [-1., 2., -.5], [0., 0., 0.])):
                if i:
                    f.append(lambda: self.clear(opt))
                f += [lambda v=values: self.backward(p, v), lambda: self.step(opt)]
            self.actions(obs['actions'], f, opt)
        elif number == 7:
            p = self.p('p', []); opt = self.opt_create([(['p'], {})], {'momentum': .5, 'weight_decay': .25})
            f = [lambda: self.step(opt), lambda: self.backward(p, []), lambda: self.step(opt),
                 lambda: self.step(opt), lambda: self.clear(opt), lambda: self.step(opt)]
            self.actions(obs['actions'], f, opt)
        elif number == 8:
            base = self.c.new([2, 3], [1., -2., 3., .5, 2., -1.])
            p = self.c.make(6, base, 0, 1); self.params['p'] = p
            self.flag(p, True); self.c.one(41, base)
            opt = self.opt_create([(['p'], {})], {'momentum': .5, 'dampening': .25})
            self.actions(obs['actions'], [lambda: self.backward(p, [1., 2., 3., -1., -2., -3.]),
                                         lambda: self.step(opt), lambda: self.step(opt)], opt)
        elif number in (9, 10, 11):
            p, q = self.p('p', [1., -2.]), self.p('q', [.5, 3.])
            if number == 9:
                opt = self.opt_create([(['p'], {'lr': .125, 'momentum': .5, 'weight_decay': .25}),
                                       (['q'], {'lr': .25, 'momentum': .75, 'dampening': .25})])
                f = [lambda: self.backward(p, [2., -1.]), lambda: self.step(opt), lambda: self.clear(opt),
                     lambda: self.backward(q, [-1., 2.]), lambda: self.step(opt), lambda: self.clear(opt),
                     lambda: self.backward(p, [1., 1.]), lambda: self.backward(q, [.5, -.5]), lambda: self.step(opt)]
            else:
                if number == 11:
                    self.eq(obs['construction']['outcome'], 'RAISED', 'reference across-group constructor rejection')
                    self.opt_create([(['p'], {}), (['p'], {})], {'momentum': .5}, status=7)
                opt = self.opt_create([(['p'], {})], {'momentum': .5})
                if number == 10:
                    f = [lambda: self.backward(p, [2., -1.]), lambda: self.step(opt),
                         lambda: self.opt_add(opt, ['q'], {'lr': .25, 'momentum': .75}), lambda: self.clear(opt),
                         lambda: self.backward(q, [-1., 2.]), lambda: self.step(opt)]
                else:
                    f = [lambda: self.opt_add(opt, ['p'], status=7), lambda: self.opt_add(opt, ['q', 'p'], status=7),
                         lambda: self.opt_add(opt, ['q'])]
            self.actions(obs['actions'], f, opt)
        elif number == 12:
            p = self.p('p', [1., -2.])
            opt = self.opt_create([(['p', 'p'], {})], {'momentum': .5, 'dampening': .25, 'weight_decay': .25})
            self.eq(obs['construction']['outcome'], 'SUCCEEDED', 'duplicate constructor succeeds')
            self.eq(len(obs['construction']['warnings']), 1, 'actual duplicate warning retained in reference')
            self.extra['warning_contract'] = 'The native inventory reports one duplicate; it does not create Python warning objects.'
            self.actions(obs['actions'], [lambda: self.backward(p, [2., -1.]), lambda: self.step(opt), lambda: self.step(opt)], opt)
        elif number == 13:
            self.eq(obs['empty_top_level']['outcome'], 'RAISED', 'empty top-level rejection')
            self.opt_create([], status=7)
            opt = self.opt_create([([], {})], {'momentum': .5})
            self.eq(obs['explicit_empty_group']['outcome'], 'SUCCEEDED', 'empty explicit group accepted')
            self.actions(obs['actions'], [lambda: self.step(opt), lambda: self.opt_add(opt, []), lambda: self.clear(opt)], opt)
        elif number == 14:
            p = self.p('p', [1., -2.])
            configs = ({'lr': -.125}, {'momentum': -.5}, {'weight_decay': -.25}, {'nesterov': True},
                       {'nesterov': True, 'momentum': .5, 'dampening': .25})
            for actual, cfg in zip(obs['default_constructor_boundaries'], configs):
                self.eq(actual['outcome'], 'RAISED', 'invalid defaults rejected by reference')
                self.opt_create([(['p'], {})], cfg, status=7)
            constant = self.c.new([], [2.]); self.params['nonleaf'] = self.c.make(2, 3, p, constant)
            self.eq(obs['nonleaf']['outcome'], 'RAISED', 'nonleaf rejection')
            self.opt_create([(['nonleaf'], {})], status=5)
            self.eq(obs['retained_nonleaf']['outcome'], 'SUCCEEDED', 'reference retained nonleaf accepted')
            self.extra['retained_nonleaf'] = 'OPEN: no native retain_grad/nonleaf optimizer membership interface; this sub-observation is not executed or counted as a match.'
            self.params['inert'] = self.c.new([1], [3.])
            inert_opt = self.opt_create([(['inert'], {})]); self.c.one(78, inert_opt)
            self.eq(obs['nonrequiring_leaf']['outcome'], 'SUCCEEDED', 'nonrequiring leaf accepted')
            del self.params['nonleaf'], self.params['inert']
            opt = self.opt_create([(['p'], {'lr': -.125})])
            self.eq(obs['group_override']['outcome'], 'SUCCEEDED', 'negative group override accepted')
            self.actions(obs['actions'], [lambda: self.backward(p, [2., -1.]), lambda: self.step(opt)], opt)
        elif number == 15:
            p = self.p('p', [1., -2.]); opt = self.opt_create([(['p'], {})], {'momentum': .5})
            square = self.c.make(2, 3, p, p); root = self.c.make(4, 7, square)
            self.backward(root); weak = self.c.make(42, p)
            before = self.snapshot(obs['before'], opt)
            old_inc = self.handle_incarnations[p]
            self.c.run([(41, p), (41, square), (41, root), (33, 0)])
            self.c.one(13, p, status=1)
            group = self.b.words(self.c.one(74, opt, 0))
            self.eq(list(group[8:]), [old_inc, 0], 'private parameter remains while public token is closed')
            self.storage_expiry(weak, obs['after_public_and_graph_drop']['storage_expired'])
            fresh = self.c.make(75, opt, 0, 0)
            assert fresh != p
            self.bind_incarnation(fresh, old_inc); self.params['p'] = fresh
            self.c.one(13, p, status=1)
            self.eq(obs['after_public_and_graph_drop']['tensor_alive'], True, 'reference privately retained Tensor')
            self.actions(obs['actions'], [lambda: self.step(opt)], opt)
            after = self.snapshot(obs['after'], opt)
            self.c.run([(41, fresh), (78, opt)])
            self.storage_expiry(weak, obs['after_optimizer_drop']['storage_expired'])
            self.c.one(13, fresh, status=1); self.c.one(44, weak)
            self.eq(obs['after_optimizer_drop']['tensor_alive'], False, 'reference final Tensor death')
            self.extra['private_lifetime'] = {'before': before, 'after': after, 'old_public': p, 'fresh_public': fresh,
                'stable_incarnation': old_inc, 'native_extra_control': 'graph_reset(0) and old-token refusal while private Optimizer retains the original Tensor binding'}
        elif number in (16, 17):
            p = self.p('p', [1., -2.]); opt = self.opt_create([(['p'], {})], {'momentum': .5})
            self.backward(p, [2., -1.]); self.step(opt)
            buffer = self.opt_state_handle(opt)
            q = self.p('q', [3., 4.]); product = self.c.make(2, 3, q, buffer); root = self.c.make(4, 7, product)
            if number == 16:
                alias = self.c.make(19, buffer); weak = self.c.make(42, buffer)
                self.tensor(buffer, obs['before']['state']); self.tensor(alias, obs['before']['alias']); self.tensor(q, obs['before']['q'])
                self.c.run([(41, buffer), (78, opt), (41, p)])
                self.storage_expiry(weak, obs['storage_expired_after_optimizer_drop'])
                self.action(obs['backward'], lambda: self.backward(root), params={'q': q})
                self.storage_expiry(weak, obs['storage_expired_with_alias_after_backward'])
                self.tensor(alias, obs['alias_before_write'])
                self.one_version_literal_mutation(alias, obs['alias_after_write'])
                self.tensor(alias, obs['alias_after_write'])
                self.c.one(41, alias)
                self.storage_expiry(weak, obs['storage_expired_after_last_alias_drop']); self.c.one(44, weak)
            else:
                self.tensor(buffer, obs['state_before_update'])
                self.action(obs['optimizer_update'], lambda: self.step(opt), opt, params={'p': p})
                self.tensor(buffer, obs['state_after_update'])
                self.action(obs['backward'], lambda: self.backward(root, error=6), params={'q': q})
        elif number == 18:
            p = self.p('p', [1., -2.])
            first = self.opt_create([(['p'], {})], {'momentum': .5})
            second = self.opt_create([(['p'], {})], {'lr': .25, 'momentum': .75, 'weight_decay': .25})
            self.backward(p, [2., -1.])
            for actual, opt in zip(obs['actions'], (first, second, first)):
                before = {k: self.snapshot(actual['before'][k], v) for k, v in [('first', first), ('second', second)]}
                self.eq(actual['outcome'], 'SUCCEEDED', 'two Optimizers action succeeds')
                self.step(opt)
                after = {k: self.snapshot(actual['after'][k], v) for k, v in [('first', first), ('second', second)]}
                self.steps.append({'action': actual['action'], 'before': before, 'after': after})
            x, y = self.opt_state_handle(first), self.opt_state_handle(second)
            self.eq(x == y, obs['same_state_Tensor'], 'two Optimizers state Tensor relationship')
            self.eq(self.b.words(self.c.one(13, x))[11] == self.b.words(self.c.one(13, y))[11],
                    obs['same_state_Storage'], 'two Optimizers state Storage relationship')
        elif number == 19:
            p = self.p('p', [1., -2.]); q = self.c.make(19, p); self.params['q'] = q; self.flag(q, True)
            opt = self.opt_create([(['p', 'q'], {})], {'momentum': .5, 'weight_decay': .25})
            self.eq(p == q, obs['same_parameter_Tensor'], 'distinct aliased Tensor identity')
            pm, qm = self.b.words(self.c.one(13, p)), self.b.words(self.c.one(13, q))
            self.eq(pm[11] == qm[11], obs['same_parameter_Storage'], 'shared parameter Storage')
            self.eq(pm[12], qm[12], 'detach aliases share Version identity')
            self.actions(obs['actions'], [lambda: self.backward(p, [2., -1.]), lambda: self.backward(q, [-1., 2.]), lambda: self.step(opt)], opt)
            x, y = self.opt_state_handle(opt, 0, 0), self.opt_state_handle(opt, 0, 1)
            self.eq(x == y, obs['same_state_Tensor'], 'distinct aliases have separate state Tensors')
            self.eq(self.b.words(self.c.one(13, x))[11] == self.b.words(self.c.one(13, y))[11],
                    obs['same_state_Storage'], 'distinct aliases have separate state Storage')
        else:
            raise AssertionError(('unmapped recipe', number))
        self.c.audit()
        return {'actions': self.steps, 'additional_observations': self.extra,
                'identity_relationships': {'Tensor': {k: list(v) for k, v in self.ref_tensor.items()}, 'Storage': self.ref_storage}}

    def requires_case(self, number, obs):
        if number == 1:
            p = self.p('p', [1., -2.], requires=False)
            f = [lambda value=value: self.flag(p, value) for value in (False, True, True, False, True)]
            self.actions(obs['actions'], f)
        elif number == 2:
            p = self.p('p', [1., -2.]); opt = self.opt_create([(['p'], {})], {'momentum': .5, 'weight_decay': .25})
            f = [lambda: self.backward(p, [2., -1.]), lambda: self.flag(p, False), lambda: self.step(opt),
                 lambda: self.clear(opt, False), lambda: self.step(opt), lambda: self.clear(opt, True),
                 lambda: self.step(opt), lambda: self.flag(p, True), lambda: self.backward(p, [-0., -0.])]
            self.actions(obs['actions'], f, opt)
        elif number in (3, 5):
            p = self.p('p', [1., -2.]); coefficient = self.p('coefficient', [2.], [], False)
            product = self.c.make(2, 3, p, coefficient); root = self.c.make(4, 7, product)
            if number == 3:
                f = [lambda: self.flag(p, False), lambda: self.backward(root, retain=1), lambda: self.flag(p, True),
                     lambda: self.backward(root, retain=1), lambda: self.flag(p, False), lambda: self.backward(root),
                     lambda: self.flag(p, True), lambda: self.backward(root, error=10)]
            else:
                def mutate():
                    self.c.one(12, coefficient, 0, self.b.bits(3.))
                    return {'header_status': 0}
                f = [lambda: self.flag(p, False), mutate, lambda: self.backward(root, error=6)]
            self.actions(obs['actions'], f)
        elif number == 4:
            p = self.p('p', [1., -2.], requires=False); q = self.p('q', [3., 4.])
            product = self.c.make(2, 3, p, q); root = self.c.make(4, 7, product)
            self.actions(obs['actions'], [lambda: self.flag(p, True), lambda: self.backward(root)])
        elif number == 6:
            p = self.p('p', [1., -2.]); q = self.c.make(19, p); self.params['q'] = q
            square = self.c.make(2, 3, p, p); root = self.c.make(4, 7, square)
            f = [lambda: self.flag(q, True), lambda: self.backward(q, [3., 4.]), lambda: self.flag(p, False),
                 lambda: self.flag(q, False), lambda: self.flag(p, True), lambda: self.flag(q, True), lambda: self.backward(root)]
            self.actions(obs['actions'], f)
            self.eq(p == q, obs['same_parameter_Tensor'], 'alias Tensor distinctness')
            pm, qm = self.b.words(self.c.one(13, p)), self.b.words(self.c.one(13, q))
            self.eq(pm[11] == qm[11], obs['same_parameter_Storage'], 'alias shared Storage')
            self.eq(pm[12], qm[12], 'alias shared Version identity')
        else:
            raise AssertionError(('requires recipe', number))
        self.c.audit()
        return {'actions': self.steps, 'additional_observations': self.extra,
                'identity_relationships': {'Tensor': {k: list(v) for k, v in self.ref_tensor.items()}, 'Storage': self.ref_storage}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'controls', 'transport-observer', 'observations', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--suite', choices=tuple(REFERENCES), required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    args = parser.parse_args()
    path_names = ('binary', 'source', 'controls', 'transport_observer', 'observations')
    for name in (*path_names, 'output'):
        setattr(args, name, getattr(args, name).resolve())
    paths = {name: getattr(args, name) for name in path_names}
    hashes = {name: sha(path.read_bytes()) for name, path in paths.items()}
    ref_sha, probe_sha, case_count = REFERENCES[args.suite]
    assert hashes == {'binary': args.expected_sha256, 'source': args.expected_source_sha256,
                      'controls': CONTROLS_SHA, 'transport_observer': OBSERVER_SHA, 'observations': ref_sha}
    reference = json.loads(args.observations.read_bytes())
    assert reference['status'] == 'OBSERVED' and reference['cases'] == case_count and reference['probe_errors'] == 0
    assert reference['probe_sha256'] == probe_sha
    assert len(reference['oracle_provenance']['checks']) == 11 and all(reference['oracle_provenance']['checks'].values())
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    codec = load('optimizer_replay_codec', args.controls)
    transport = load('optimizer_replay_transport', args.transport_observer)
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / 'executed-sources'; archive.mkdir()
    files = (Path(__file__), args.controls, args.transport_observer, args.observations)
    source_hashes = {path.name: sha(path.read_bytes()) for path in files}
    for path in files:
        (archive / path.name).write_bytes(path.read_bytes())
    rows = []
    for number, actual in enumerate(reference['rows'], 1):
        row = {'case': actual['case'], 'reference_status': actual['status'], 'status': 'FAIL'}
        if args.suite == 'optimizer' and number in OPEN_CASES:
            row.update(status='OPEN_NOT_EXECUTED', reason=OPEN_CASES[number], native_executed=False,
                       frames=[], exact_comparisons=0, rollback_bytes=0)
            rows.append(row)
            continue
        directory = args.output / ('case-'+str(number).zfill(2))
        session = transport.OwnerSession(args.binary, directory=directory, address_limit=64 << 20)
        c = codec.Check(session); replay = Replay(c, codec)
        try:
            details = (replay.optimizer_case if args.suite == 'optimizer' else replay.requires_case)(number, actual['observation'])
            session.finish()
            status = 'PARTIAL_WITH_RETAINED_NONLEAF_OPEN' if args.suite == 'optimizer' and number == 14 else 'QUALIFIED_EXPOSED_SCOPE'
            row.update(status=status, details=details)
        except Exception as error:
            row.update(exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                       completed_actions=replay.steps, additional_observations=replay.extra)
            if session.poll() is None:
                os.kill(session.pid, signal.SIGKILL)
            try:
                session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup:
                row['cleanup_exception'] = repr(cleanup)
        row.update(native_executed=True, frames=c.frames, exact_comparisons=c.comparisons, rollback_bytes=c.rollback_bytes)
        (directory / 'report.json').write_text(json.dumps(row, indent=2)+'\n')
        rows.append(row)
    assert hashes == {name: sha(path.read_bytes()) for name, path in paths.items()}
    assert source_hashes == {path.name: sha(path.read_bytes()) for path in files}
    failures = sum(row['status'] == 'FAIL' for row in rows)
    report = {'schema': 'heptatorch.native-replay-owned-optimizer-observations.v1',
        'suite': args.suite, 'status': 'FAIL' if failures else 'QUALIFIED_WITH_EXPLICIT_OPEN_SCOPE',
        'native_executable_sha256': hashes['binary'], 'native_source_sha256': hashes['source'],
        'observations_sha256': ref_sha, 'probe_sha256': probe_sha, 'test_source_hashes': source_hashes,
        'oracle_provenance': reference['oracle_provenance'], 'reference_cases': case_count,
        'executed_cases': sum(row['native_executed'] for row in rows),
        'qualified_exposed_scope': sum(row['status'] == 'QUALIFIED_EXPOSED_SCOPE' for row in rows),
        'partial_cases': sum(row['status'].startswith('PARTIAL_') for row in rows),
        'open_not_executed_cases': sum(row['status'] == 'OPEN_NOT_EXECUTED' for row in rows),
        'unexpected_failures': failures, 'frames': sum(len(row['frames']) for row in rows),
        'exact_comparisons': sum(row['exact_comparisons'] for row in rows),
        'rollback_bytes': sum(row['rollback_bytes'] for row in rows),
        'numeric_policy': 'Exact unsigned binary32 words, including signed zero; exact parameter/state Versions and gradient presence. No floating tolerance.',
        'identity_policy': 'Reference opaque identity relationships map to observed native stable Tensor incarnations or live public tokens and typed Storage IDs; host addresses are never compared.',
        'observer_policy': 'State inspection closes only a transiently published state token. It does not retain an extra public state root beyond a snapshot.',
        'common_open_scope': ['Public .grad Tensor identity, layout, Storage aliases and Version are not exposed; only exact words and None presence are compared.',
            'Python warning objects, mutable group dictionaries and intrusive_ptr reference counts are not implemented by this wire API.',
            'Arbitrary late-error partial Optimizer mutation behavior is not claimed; native Optimizer commands are frame atomic.',
            'The fixed literal alias mutation in case 16 is not a Tensor.add_ implementation.'],
        'rows': rows, 'native_executed': True, 'pinned_Torch_executed_during_replay': False,
        'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('suite', 'status', 'reference_cases', 'executed_cases', 'qualified_exposed_scope',
        'partial_cases', 'open_not_executed_cases', 'unexpected_failures', 'frames', 'exact_comparisons', 'rollback_bytes')}))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
