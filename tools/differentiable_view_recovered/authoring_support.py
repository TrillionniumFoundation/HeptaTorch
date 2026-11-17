"""Recovery R1: same bounded emission algorithm; no Tensor evaluation."""
import ast
import copy
import hashlib
import re

FUNCTION = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)', re.M | re.S)
CALL = re.compile(rb'\{"function":"(\w+)"\}')
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
