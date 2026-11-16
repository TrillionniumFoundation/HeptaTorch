"""Temporary development authoring helper; emitted HIA is the repository source."""
import ast
import json
import re
import sys
from pathlib import Path

from tools.generate_hia_selfhost import Builder, compact_authoring_cfg, total_scalar_functions
from hepta.hia_agent_assembly import render_assembly, _parse_assembly_document, _render_assembly_document

ROOT = None

class NativeBuilder(Builder):
    def expr(self, expr):
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.FloorDiv, ast.Mod)):
            x, y = self.expr(expr.left), self.expr(expr.right)
            return self.op('int.div.checked' if isinstance(expr.op, ast.FloorDiv) else 'int.rem.checked', [x, y], x[1])
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in ('idiv', 'irem'):
            x, y = [self.expr(arg) for arg in expr.args]
            return self.op('int.div.checked' if expr.func.id == 'idiv' else 'int.rem.checked', [x, y], x[1])
        return super().expr(expr)

def emit(source, extra_signatures=None):
    nodes = ast.parse(source).body
    mapping = json.loads((ROOT / 'docs/functions.json').read_text())
    signatures = {name: ([p['type'] for p in f['params']], f['returns']) for name, f in mapping.items()}
    signatures.update(extra_signatures or {})
    signatures.update({n.name: ([a.annotation.id for a in n.args.args], n.returns.id) for n in nodes})
    names = {name: name for name in signatures}
    total = total_scalar_functions(nodes)
    functions = []
    for node in nodes:
        b = NativeBuilder(node, names, signatures, total)
        b.statements(node.body)
        functions.append({'name': node.name, 'params': [b.param(v) for v in b.params], 'returns': [b.returns], 'effects': ['read', 'write', 'trap'], 'entry': 'b0', 'blocks': b.blocks})
    doc = _parse_assembly_document((ROOT / 'src/heptatorch.hepta').read_text())
    new_names = {n.name for n in nodes}
    doc['functions'] = [f for f in doc['functions'] if f['name'] not in new_names] + functions
    doc = compact_authoring_cfg(doc)
    doc['functions'] = [f for f in doc['functions'] if f['name'] in new_names]
    text = _render_assembly_document(doc)
    new_mapping = {n.name: {'name': n.name, 'params': [{'name': a.arg, 'type': a.annotation.id} for a in n.args.args], 'returns': n.returns.id} for n in nodes}
    return text.split('\n', 1)[1], new_mapping
