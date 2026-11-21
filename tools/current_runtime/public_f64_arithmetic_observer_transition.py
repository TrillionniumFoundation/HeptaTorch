"""Exact refactor inverse: expose original negation recipe preparation only.

The complete original numeric reject closure and original default terminal
assertions are retained verbatim. The new arithmetic successor can reuse the
same prepared87 code object and perform its explicitly enlarged final census.
"""
import ast
import hashlib
import json
from pathlib import Path

sha=lambda raw:hashlib.sha256(raw).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)

def restore_negation_helper(source,bind):
    bind(Path(__file__).resolve())
    delta=json.loads(bind(Path(__file__).with_name('public_f64_arithmetic_observer_delta.json')))
    require(sha(source)==delta['source_sha256'],'complete current negation interface helper')
    previous=source
    for change in reversed(delta['replacements']):
        old,new=change['original'].encode(),change['current'].encode()
        require(previous.count(new)==1,'unique exact interface-only replacement')
        previous=previous.replace(new,old,1)
    require(sha(previous)==delta['parent_sha256'],'complete frozen771 negation observer restored')
    forward=previous
    for change in delta['replacements']:
        old,new=change['original'].encode(),change['current'].encode()
        require(forward.count(old)==1,'unique original interface location')
        forward=forward.replace(old,new,1)
    require(forward==source,'complete observer forward round trip')
    def nested(raw,method,name):
        cls=next(node for node in ast.parse(raw).body if isinstance(node,ast.ClassDef) and node.name=='Successor')
        fn=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==method)
        child=next(node for node in fn.body if isinstance(node,ast.FunctionDef) and node.name==name)
        return b''.join(raw.splitlines(keepends=True)[child.lineno-1:child.end_lineno])
    require(nested(source,'prepare_entry','reject')==nested(previous,'wrap_entry','reject'),
        'every original negation numerical and ownership predicate byte retained')
    require(nested(source,'wrap_entry','execute')==nested(previous,'wrap_entry','execute'),
        'default original86 refusal and one negation terminal contract retained')
    return previous,dict(schema='heptatorch.f6448.negation-interface-inverse.v1',
        source_sha256=sha(source),parent_sha256=sha(previous),
        original_numeric_reject_closure_bytes_unchanged=True,
        original_default_terminal_assertions_bytes_unchanged=True,
        original87_recipe_code_preserved=True,whole_observer_inverse_exact=True,
        whole_observer_forward_round_trip_exact=True)
