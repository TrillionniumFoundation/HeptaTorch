"""Verified authoring only: this module never executes a Tensor operation."""
import ast
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

FUNCTION = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)', re.M | re.S)
CALL = re.compile(rb'\{"function":"(\w+)"\}')
PARENT = '6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
RNG = 'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
OWNERS = '9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'
EXTERNAL = {'rng_seed', 'rng_restore', 'rng_next', 'rng_next64', 'rng_unit32', 'rng_unit64'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def functions(raw):
    pairs = FUNCTION.findall(raw)
    result = {name.decode(): body for body, name in pairs}
    assert len(pairs) == len(result)
    return result


def once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new, 1)


class Emitter:
    def __init__(self, parent, repository, upstream, compactor):
        self.inputs = {}
        def read(path, pin=None):
            path = Path(path).resolve()
            raw = path.read_bytes()
            if pin is not None:
                assert sha(raw) == pin, (path, sha(raw), pin)
            self.inputs[path] = raw
            return raw
        self.parent = functions(read(parent / 'runtime.hepta', PARENT))
        self.rng_raw = read(parent / 'rng03.hepta', RNG)
        self.owner_raw = read(parent / 'owners.hepta', OWNERS)
        self.rng = functions(self.rng_raw)
        assert (len(self.parent), len(self.rng)) == (523, 17)
        emitter_path = repository / 'tools/owned_tensors/authoring.py'
        read(emitter_path, '2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6')
        profile = json.loads(read(repository / 'tools/owned_tensors/profile.json',
            '69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242'))
        read(compactor, '36730b55b58c29b210d921a426c62eab50d9f579ad5d7a1e4199b38799218f37')
        for name, pin in profile['upstream_authoring_sources'].items():
            read(upstream / name, pin)
        sys.path.insert(0, str(upstream))
        spec = importlib.util.spec_from_file_location('typed_native_emitter', emitter_path)
        self.emitter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.emitter)
        spec = importlib.util.spec_from_file_location('typed_bounded_compactor', compactor)
        self.compactor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.compactor)
        self.context = []
        self.signatures = {}
        self.checks = []
        for name, body in {**self.parent, **self.rng}.items():
            self.add_context(name, body)

    def add_context(self, name, body):
        match = re.match(rb'fn \w+\(([^)]*)\) -> \(([^)]*)\)', body)
        self.signatures[name] = ([p.split(b':')[1].decode() for p in match[1].split(b',') if p], match[2].decode())
        self.context.extend(self.emitter._parse_assembly_document(
            (b'hia-asm.v1 hepta.hia.program.v8 typed_context\n' + body).decode())['functions'])

    def emit(self, source):
        nodes = ast.parse(source).body
        assert all(isinstance(n, ast.FunctionDef) for n in nodes)
        signatures = dict(self.signatures)
        signatures.update({n.name: ([p.annotation.id for p in n.args.args], n.returns.id) for n in nodes})
        names = {n: n for n in signatures}
        total = self.emitter.total_scalar_functions(nodes)
        emitted = []
        for node in nodes:
            b = self.emitter.NativeBuilder(node, names, signatures, total)
            b.statements(node.body)
            emitted.append({'name': node.name, 'params': [b.param(p) for p in b.params],
                            'returns': [b.returns], 'effects': ['read', 'write', 'trap'],
                            'entry': 'b0', 'blocks': b.blocks})
        selected = {n.name for n in nodes}
        doc = [f for f in copy.deepcopy(self.context) if f['name'] not in selected] + emitted
        checked, records = self.compactor.compact(doc, selected)
        self.checks.append(records)
        return self.emitter._render_assembly_document(
            {'schema': 'hepta.hia.program.v8', 'module': 'typed_overlay', 'functions': checked}).encode()

    def unchanged(self):
        for path, raw in self.inputs.items():
            assert path.read_bytes() == raw, path
