"""Independent expected-value/lifetime model for the numeric object protocol.

This is test code. It is never linked to, called by, or substituted for the HIA
runtime. Storage ownership is recomputed from a declarative graph of obligations;
it does not reproduce the target's increment/decrement or extent algorithms.
Numbers are exact Fractions. Differential fixtures must choose values whose
intermediate results are exactly representable as binary32.
"""
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from math import prod
import copy


class Rejected(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def contiguous(shape):
    result, step = [], 1
    for dim in reversed(shape):
        result.append(step)
        step *= max(dim, 1)
    return tuple(reversed(result))


def coordinates(shape):
    return list(product(*(range(dim) for dim in shape)))


def ravel(coord, shape):
    index = 0
    for axis, size in zip(coord, shape, strict=True):
        index = index * size + axis
    return index


def broadcast_shape(a, b):
    padded_a = (1,) * max(0, len(b)-len(a)) + tuple(a)
    padded_b = (1,) * max(0, len(a)-len(b)) + tuple(b)
    result = []
    for x, y in zip(padded_a, padded_b, strict=True):
        if x != y and x != 1 and y != 1:
            raise Rejected(2, 'incompatible broadcast shape')
        result.append(y if x == 1 else x)
    return tuple(result)


def projected_index(coord, shape):
    tail = coord[len(coord)-len(shape):] if shape else ()
    return ravel(tuple(0 if size == 1 else axis for axis, size in zip(tail, shape, strict=True)), shape)


@dataclass(frozen=True)
class Layout:
    shape: tuple
    strides: tuple
    offset: int = 0

    @property
    def numel(self):
        return prod(self.shape)

    def addresses(self):
        return [self.offset + sum(a*s for a, s in zip(coord, self.strides, strict=True))
                for coord in coordinates(self.shape)]


@dataclass
class Storage:
    values: list
    strong: int = 0
    explicit_weak: int = 0
    dead: bool = False
    exists: bool = True

    @property
    def nbytes(self):
        return len(self.values) * 4


@dataclass
class Version:
    counter: int = 0
    references: int = 0
    exists: bool = True


@dataclass(frozen=True)
class SavedValue:
    storage: str
    version: str
    counter: int
    layout: Layout
    role: str


@dataclass
class Node:
    ordinal: int
    storage: str
    version: str
    layout: Layout
    requires: bool
    op: str
    parents: tuple = ()
    saved: tuple = ()
    parameters: tuple = ()
    public: bool = True
    graph_live: bool = True
    row_live: bool = True
    gradient: list = field(default_factory=list)


class Model:
    """Declared v3 retain-until-release graph contract; not default torch.backward."""
    def __init__(self):
        self.nodes, self.storages, self.versions = {}, {}, {}
        self.weak, self.holds, self.saved = {}, {}, {}
        self.next_ordinal = 1
        self.reset_keep = None
        self._serial = 0

    def fork(self):
        return copy.deepcopy(self)

    def _fresh(self, prefix):
        self._serial += 1
        return prefix + str(self._serial)

    def _data(self, values):
        sid, vid = self._fresh('S'), self._fresh('V')
        self.storages[sid] = Storage([Fraction(value) for value in values])
        self.versions[vid] = Version()
        return sid, vid

    def _node(self, name, sid, vid, layout, requires, op, parents=(), saved=(), parameters=()):
        if name in self.nodes:
            raise ValueError('model symbolic names are never reused')
        if layout.numel < 0 or any(dim < 0 for dim in layout.shape):
            raise Rejected(2, 'negative shape')
        if any(address < 0 or address >= len(self.storages[sid].values) for address in layout.addresses()):
            raise Rejected(2, 'layout outside storage')
        self.nodes[name] = Node(self.next_ordinal, sid, vid, layout, bool(requires), op,
                                tuple(parents), tuple(saved), tuple(parameters),
                                gradient=[Fraction(0)] * layout.numel if requires else [])
        self.next_ordinal += 1
        self._refresh()
        return name

    def _public(self, name):
        if name not in self.nodes or not self.nodes[name].public or not self.nodes[name].graph_live:
            raise Rejected(1, 'invalid public Tensor')
        return self.nodes[name]

    def _snapshot(self, name, role):
        node = self.nodes[name]
        return SavedValue(node.storage, node.version, self.versions[node.version].counter,
                          node.layout, role)

    def _check_saved(self, value):
        if not self.versions[value.version].exists or self.versions[value.version].counter != value.counter:
            raise Rejected(6, 'saved Version changed')
        if self.storages[value.storage].dead:
            raise AssertionError('reference model released a live saved value')

    def _logical(self, sid, layout):
        if self.storages[sid].dead or not self.storages[sid].exists:
            raise Rejected(1, 'storage payload expired')
        return [self.storages[sid].values[address] for address in layout.addresses()]

    def values(self, name):
        node = self._public(name)
        return self._logical(node.storage, node.layout)

    def grad(self, name):
        node = self._public(name)
        if not node.requires or node.op != 'leaf':
            raise Rejected(5, 'wire gradient reads require a gradient leaf')
        return list(node.gradient)

    def create(self, name, shape, values, requires=False):
        shape, values = tuple(shape), list(values)
        if len(values) != prod(shape):
            raise ValueError('value/shape mismatch in reference fixture')
        sid, vid = self._data(values)
        return self._node(name, sid, vid, Layout(shape, contiguous(shape)), requires, 'leaf')

    def transpose(self, name, source, a, b):
        node = self._public(source)
        rank = len(node.layout.shape)
        a, b = a % rank, b % rank
        shape, strides = list(node.layout.shape), list(node.layout.strides)
        shape[a], shape[b] = shape[b], shape[a]
        strides[a], strides[b] = strides[b], strides[a]
        return self._node(name, node.storage, node.version,
                          Layout(tuple(shape), tuple(strides), node.layout.offset),
                          node.requires, 'transpose', (source,), parameters=(a, b))

    def view(self, name, source, shape):
        node = self._public(source)
        shape = tuple(shape)
        if prod(shape) != node.layout.numel or node.layout.strides != contiguous(node.layout.shape):
            raise Rejected(2, 'this fixture view requires matching contiguous storage')
        return self._node(name, node.storage, node.version,
                          Layout(shape, contiguous(shape), node.layout.offset),
                          node.requires, 'view', (source,))

    def detach(self, name, source, separate_version=False):
        node = self._public(source)
        vid = node.version
        if separate_version:
            vid = self._fresh('V')
            self.versions[vid] = Version()
        return self._node(name, node.storage, vid, node.layout, False,
                          'data' if separate_version else 'detach')

    def clone(self, name, source):
        node = self._public(source)
        addresses = node.layout.addresses()
        dense = not addresses or (len(set(addresses)) == len(addresses) and (
            max(addresses)-min(addresses)+1 == len(addresses)))
        strides = node.layout.strides if dense else contiguous(node.layout.shape)
        layout = Layout(node.layout.shape, strides)
        physical = [Fraction(0)] * layout.numel
        for address, value in zip(layout.addresses(), self.values(source), strict=True):
            physical[address] = value
        sid, vid = self._data(physical)
        return self._node(name, sid, vid, layout, node.requires, 'clone', (source,))

    def binary(self, name, op, a, b):
        if op not in ('add', 'sub', 'mul', 'div'):
            raise ValueError(op)
        na, nb = self._public(a), self._public(b)
        shape = broadcast_shape(na.layout.shape, nb.layout.shape)
        av, bv = self.values(a), self.values(b)
        values = []
        for coord in coordinates(shape):
            x, y = av[projected_index(coord, na.layout.shape)], bv[projected_index(coord, nb.layout.shape)]
            values.append({'add': lambda: x+y, 'sub': lambda: x-y,
                           'mul': lambda: x*y, 'div': lambda: x/y}[op]())
        saves = []
        if op in ('mul', 'div') and nb.requires:
            saves.append(self._snapshot(a, 'A'))
        if op == 'mul' and na.requires or op == 'div' and (na.requires or nb.requires):
            saves.append(self._snapshot(b, 'B'))
        sid, vid = self._data(values)
        return self._node(name, sid, vid, Layout(shape, contiguous(shape)),
                          na.requires or nb.requires, op, (a, b), saved=saves)

    def unary(self, name, op, a):
        node = self._public(a)
        if op == 'neg':
            values = [-value for value in self.values(a)]
        elif op == 'relu':
            values = [max(Fraction(0), value) for value in self.values(a)]
        else:
            raise ValueError('reference model only uses exactly representable unary fixtures')
        sid, vid = self._data(values)
        layout = Layout(node.layout.shape, contiguous(node.layout.shape))
        saved = (SavedValue(sid, vid, 0, layout, 'output'),) if op == 'relu' and node.requires else ()
        return self._node(name, sid, vid, layout, node.requires, op, (a,), saved=saved)

    def total(self, name, a):
        node = self._public(a)
        sid, vid = self._data([sum(self.values(a), Fraction(0))])
        return self._node(name, sid, vid, Layout((), ()), node.requires, 'sum', (a,))

    def set(self, name, index, value):
        node = self._public(name)
        if not 0 <= index < node.layout.numel:
            raise Rejected(7, 'Tensor logical index outside bounds')
        version = self.versions[node.version]
        if version.counter == (1 << 64)-1:
            raise Rejected(8, 'Version overflow')
        self.storages[node.storage].values[node.layout.addresses()[index]] = Fraction(value)
        version.counter += 1

    def zero(self, name):
        node = self._public(name)
        if not node.requires or node.op != 'leaf':
            raise Rejected(5, 'zero_grad requires a gradient leaf')
        node.gradient = [Fraction(0)] * node.layout.numel

    def drop(self, name):
        self._public(name).public = False
        self._refresh()

    def make_weak(self, name, tensor):
        if name in self.weak:
            raise ValueError('duplicate symbolic Weak')
        self.weak[name] = self._public(tensor).storage
        self._refresh()

    def drop_weak(self, name):
        if name not in self.weak:
            raise Rejected(1, 'invalid Weak')
        del self.weak[name]
        self._refresh()

    def lock(self, name, weak):
        if weak not in self.weak:
            raise Rejected(1, 'invalid Weak')
        sid = self.weak[weak]
        if self.storages[sid].dead:
            raise Rejected(6, 'expired weak storage')
        if name in self.holds:
            raise ValueError('duplicate symbolic Hold')
        self.holds[name] = sid
        self._refresh()

    def read_hold(self, name):
        if name not in self.holds:
            raise Rejected(1, 'invalid public Hold')
        return list(self.storages[self.holds[name]].values)

    def drop_hold(self, name):
        if name not in self.holds:
            raise Rejected(1, 'invalid public Hold')
        del self.holds[name]
        self._refresh()

    def save(self, name, tensor):
        self._public(tensor)
        if name in self.saved:
            raise ValueError('duplicate symbolic Saved')
        self.saved[name] = self._snapshot(tensor, 'public')
        self._refresh()

    def read_saved(self, name):
        if name not in self.saved:
            raise Rejected(1, 'invalid public Saved')
        saved = self.saved[name]
        self._check_saved(saved)
        return self._logical(saved.storage, saved.layout)

    def drop_saved(self, name):
        if name not in self.saved:
            raise Rejected(1, 'invalid public Saved')
        del self.saved[name]
        self._refresh()

    def unpack(self, name, saved_name):
        self.read_saved(saved_name)
        saved = self.saved[saved_name]
        return self._node(name, saved.storage, saved.version, saved.layout, False, 'unpack')

    def reset_graph(self, keep):
        if keep < 0 or keep >= self.next_ordinal:
            raise Rejected(7, 'invalid retained prefix')
        if self.reset_keep is not None and keep != self.reset_keep:
            raise Rejected(7, 'reset prefix cannot change after first reset')
        live_prefix = {n.ordinal: n for n in self.nodes.values()
                       if n.graph_live and n.ordinal <= keep}
        for ordinal in range(1, keep+1):
            n = live_prefix.get(ordinal)
            if n is None or not n.public or n.op != 'leaf':
                raise Rejected(6, 'retained prefix must contain public owned leaves')
        for node in self.nodes.values():
            if node.graph_live and node.ordinal > keep:
                node.public = False
        self._refresh()
        self.next_ordinal, self.reset_keep = keep+1, keep

    def backward(self, root, seed=None):
        target = self._public(root)
        if not target.requires:
            raise Rejected(5, 'backward target does not require gradients')
        if seed is None:
            if target.layout.numel != 1:
                raise Rejected(5, 'implicit backward seed requires scalar')
            seed = [Fraction(1)]
        seed = list(map(Fraction, seed))
        if len(seed) != target.layout.numel:
            raise Rejected(2, 'seed shape mismatch')
        active = set()
        pending = [root]
        while pending:
            name = pending.pop()
            if name in active:
                continue
            node = self.nodes[name]
            if not node.requires:
                continue
            active.add(name)
            pending.extend(node.parents)
        for name in active:
            for value in self.nodes[name].saved:
                self._check_saved(value)
        work = {name: [Fraction(0)] * self.nodes[name].layout.numel for name in active}
        work[root] = seed

        def add(parent, values):
            if parent in work:
                assert len(values) == len(work[parent])
                work[parent] = [a+b for a, b in zip(work[parent], values, strict=True)]

        ordered = sorted(active, key=lambda name: self.nodes[name].ordinal, reverse=True)
        for name in ordered:
            node, gradient = self.nodes[name], work[name]
            if node.op == 'leaf':
                continue
            if node.op in ('add', 'sub', 'mul', 'div'):
                a, b = node.parents
                na, nb = self.nodes[a], self.nodes[b]
                ga, gb = [Fraction(0)]*na.layout.numel, [Fraction(0)]*nb.layout.numel
                values = {s.role: self._logical(s.storage, s.layout) for s in node.saved}
                for coord, g in zip(coordinates(node.layout.shape), gradient, strict=True):
                    ai, bi = projected_index(coord, na.layout.shape), projected_index(coord, nb.layout.shape)
                    if na.requires:
                        da = g
                        if node.op == 'mul': da *= values['B'][bi]
                        if node.op == 'div': da /= values['B'][bi]
                        ga[ai] += da
                    if nb.requires:
                        db = -g if node.op == 'sub' else g
                        if node.op == 'mul': db *= values['A'][ai]
                        if node.op == 'div': db = -g*values['A'][ai]/values['B'][bi]**2
                        gb[bi] += db
                add(a, ga)
                add(b, gb)
            elif node.op in ('view', 'clone'):
                add(node.parents[0], gradient)
            elif node.op == 'transpose':
                parent = node.parents[0]
                result = [Fraction(0)] * self.nodes[parent].layout.numel
                a, b = node.parameters
                for coord, g in zip(coordinates(node.layout.shape), gradient, strict=True):
                    original = list(coord)
                    original[a], original[b] = original[b], original[a]
                    result[ravel(original, self.nodes[parent].layout.shape)] += g
                add(parent, result)
            elif node.op == 'sum':
                add(node.parents[0], [gradient[0]] * self.nodes[node.parents[0]].layout.numel)
            elif node.op == 'neg':
                add(node.parents[0], [-g for g in gradient])
            elif node.op == 'relu':
                saved, = node.saved
                values = self._logical(saved.storage, saved.layout)
                add(node.parents[0], [g if value > 0 else Fraction(0)
                                      for value, g in zip(values, gradient, strict=True)])
            else:
                raise AssertionError('unimplemented reference derivative: ' + node.op)
        for name, values in work.items():
            node = self.nodes[name]
            node.gradient = [a+b for a, b in zip(node.gradient, values, strict=True)]

    def _refresh(self):
        reachable, pending = set(), [name for name, node in self.nodes.items() if node.public]
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            node = self.nodes[name]
            if not node.graph_live:
                raise AssertionError('destroyed node became reachable again')
            reachable.add(name)
            if node.requires:
                pending.extend(node.parents)
        storage_owners, version_owners = Counter(), Counter()

        def own(sid, vid=None):
            storage_owners[sid] += 1
            if vid is not None:
                version_owners[vid] += 1

        for name, node in self.nodes.items():
            node.graph_live = name in reachable
            node.row_live = node.graph_live and (node.public or (node.op == 'leaf' and node.requires))
            if node.row_live:
                own(node.storage, node.version)
            if node.graph_live:
                for value in node.saved:
                    own(value.storage, value.version)
        for value in self.saved.values():
            own(value.storage, value.version)
        for sid in self.holds.values():
            own(sid)
        weak_owners = Counter(self.weak.values())
        for sid, storage in self.storages.items():
            if storage.dead and storage_owners[sid]:
                raise AssertionError('expired storage was resurrected')
            storage.strong = storage_owners[sid]
            storage.explicit_weak = weak_owners[sid]
            storage.dead = storage.strong == 0
            storage.exists = storage.strong + storage.explicit_weak > 0
        for vid, version in self.versions.items():
            if not version.exists and version_owners[vid]:
                raise AssertionError('released Version was resurrected')
            version.references = version_owners[vid]
            version.exists = version.references != 0

    def storage_metadata(self, sid):
        storage = self.storages[sid]
        if not storage.exists:
            raise Rejected(1, 'released Storage metadata')
        return {'bytes': storage.nbytes, 'strong': storage.strong,
                'weak': storage.explicit_weak + int(storage.strong > 0),
                'dead': storage.dead}

    def version_metadata(self, vid):
        version = self.versions[vid]
        if not version.exists:
            raise Rejected(1, 'released Version metadata')
        return {'counter': version.counter, 'references': version.references}

    def snapshot(self):
        def value(x):
            if isinstance(x, Fraction):
                return str(x)
            if isinstance(x, (tuple, list)):
                return [value(y) for y in x]
            if isinstance(x, dict):
                return {k: value(v) for k, v in x.items()}
            if hasattr(x, '__dataclass_fields__'):
                return {k: value(getattr(x, k)) for k in x.__dataclass_fields__}
            return x
        return value({'nodes': self.nodes, 'storages': self.storages,
                      'versions': self.versions, 'weak': self.weak,
                      'holds': self.holds, 'saved': self.saved,
                      'next_ordinal': self.next_ordinal, 'reset_keep': self.reset_keep})


class HandleCustody:
    """Check typed identity independently; never compute a target handle to use."""
    def __init__(self):
        self.bound, self.seen, self.high_generation = {}, {}, {}

    def bind(self, kind, symbolic, handle):
        if not isinstance(handle, int) or not 0 < handle < 1 << 64:
            raise AssertionError('invalid handle word')
        observed_kind, slot, generation = (handle >> 24) & 255, handle & 0xffffff, handle >> 32
        assert observed_kind == kind and 1 <= slot <= 65536 and 1 <= generation <= 0xffffffff
        key = (kind, symbolic)
        if key in self.bound:
            assert self.bound[key] == handle
            return handle
        assert handle not in self.seen, ('native identity reused', handle, self.seen.get(handle))
        previous = self.high_generation.get((kind, slot), 0)
        assert generation > previous, ('generation did not strictly increase', generation, previous)
        self.bound[key], self.seen[handle] = handle, key
        self.high_generation[kind, slot] = generation
        return handle
