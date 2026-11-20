"""Exact source and closed-dispatch proof for native numeric predicates."""

import hashlib
from pathlib import Path
import re

SOURCE = '24011e3c861e3b0e77c6a25c07116375cded2225c1f05f42cd84aa1e4a139215'
PARENT_SOURCE = 'bf3641b1a9061c6f92b8cbeef75adfac89e82ee16d32a9f650cb938d9e918386'
FUNCTIONS = 740
PARENT_FUNCTIONS = 736
ADDED = ('predicate_bits','predicate_layout','predicate_tensor','predicate_command')
GUARD = b'block b0(v0:readbuf,v1:mutbuf,v2:mutbuf,v3:u64,v4:u64,v5:u64)\n  v6:u64 = call(v0,v3) {"function":"getr"}\n  v7:u64 = const() {"value":120}\n  v8:bool = compare.ge(v6,v7)\n  v9:u64 = const() {"value":124}\n  v10:bool = compare.le(v6,v9)\n  v11:bool = bool.and(v8,v10)\n  branch v11 b1(v0,v1,v2,v3,v4,v5,v6) b2(v0,v1,v2,v3,v4,v5)\n'
DELEGATE = b'block b2(v12:readbuf,v13:mutbuf,v14:mutbuf,v15:u64,v16:u64,v17:u64)\n  v18:u64 = call(v12,v13,v14,v15,v16,v17) {"function":"to_command"}\n  ret(v18)\n'
ENTRY_EDITS = (
    (b'  v142:u64 = const() {"value":119}\n', b'  v142:u64 = const() {"value":124}\n'),
    (b'  v171:u64 = call(v145,v146,v147,v170,v153,v151) {"function":"to_command"}\n',
     b'  v171:u64 = call(v145,v146,v147,v170,v153,v151) {"function":"predicate_command"}\n'),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def functions(source, count):
    matches=list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',source,re.M|re.S))
    result={m[1].decode('ascii'):m[0] for m in matches}
    require(len(result)==len(matches)==count, 'complete unique predicate function inventory')
    require(source==source.split(b'\n',1)[0]+b'\n'+b''.join(result.values()), 'complete canonical source envelope')
    return result


def restore_previous(source):
    require(isinstance(source,bytes) and sha(source)==SOURCE, 'exact admitted numeric predicate source')
    current=functions(source,FUNCTIONS)
    require(tuple(current)[-len(ADDED):]==ADDED, 'exact four appended native functions')
    restored=source[:-sum(len(current[name]) for name in ADDED)]
    entry=current['entry']
    for before,after in ENTRY_EDITS:
        require(entry.count(after)==1 and entry.count(before)==0, 'one exact admitted entry edit')
        entry=entry.replace(after,before,1)
    restored=restored.replace(current['entry'],entry,1)
    require(sha(restored)==PARENT_SOURCE, 'every reconstructed parent byte matches exact Half1391 source')
    parent=functions(restored,PARENT_FUNCTIONS)
    require(list(current)[:PARENT_FUNCTIONS]==list(parent), 'all parent functions remain in original order')
    require([name for name in parent if current[name]!=parent[name]]==['entry'], 'only the two entry instructions change')
    forward=restored
    patched=parent['entry']
    for before,after in ENTRY_EDITS:
        require(patched.count(before)==1, 'exact forward entry edit')
        patched=patched.replace(before,after,1)
    forward=forward.replace(parent['entry'],patched,1)+b''.join(current[name] for name in ADDED)
    require(forward==source, 'complete source round trip')
    return restored


def proof(source):
    restored=restore_previous(source)
    current,parent=functions(source,FUNCTIONS),functions(restored,PARENT_FUNCTIONS)
    wrapper=current['predicate_command']
    blocks={m[1]:m[0] for m in re.finditer(rb'^block (b\d+)\([^\n]*\)\n(?:(?!^block |^end$).)*',wrapper,re.M|re.S)}
    require(blocks[b'b0']==GUARD and blocks[b'b2']==DELEGATE, 'exact closed range and untouched argument delegation')
    require(GUARD.count(b' = call(')==1 and b'{"function":"getr"}' in GUARD
            and b'{"value":120}' in GUARD and b'{"value":124}' in GUARD
            and b' = compare.ge(' in GUARD and b' = compare.le(' in GUARD
            and b' = bool.and(' in GUARD, 'guard only reads the opcode then admits the closed120..124 range')
    require(DELEGATE.count(b' = call(')==1 and b'{"function":"to_command"}' in DELEGATE
            and wrapper.count(b'{"function":"to_command"}')==1,
            'outside the range the wrapper directly returns the original six-argument call')
    require(current['to_command']==parent['to_command'], 'the complete old dispatcher is unchanged')
    return dict(schema='heptatorch.predicates740-to-half736.complete-source-inverse.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,functions=FUNCTIONS,parent_functions=PARENT_FUNCTIONS,
        changed_functions=['entry'],added_functions=list(ADDED),removed_functions=[],unchanged_parent_functions=735,
        whole_source_inverse_exact=True,whole_round_trip_exact=True,every_other_byte_constrained_by_complete_parent_sha256=True,
        entry_only_upper_bound_and_dispatch_target_changed=True,
        dispatcher_guard=dict(opcode_min=120,opcode_max=124,complete_guard_block_sha256=sha(GUARD),
            reads_before_guard=['getr(request,at)'],exact_closed_range_verified=True),
        old_path_delegation=dict(target='to_command',parameter_count=6,argument_order_preserved=True,
            complete_delegate_block_sha256=sha(DELEGATE),complete_parent_body_unchanged=True,
            parent_body_sha256=sha(parent['to_command']),actual_body_sha256=sha(current['to_command'])),
        changed_bodies={'entry':dict(before_sha256=sha(parent['entry']),after_sha256=sha(current['entry']))},
        added_bodies={name:dict(bytes=len(current[name]),sha256=sha(current[name])) for name in ADDED},
        actual_executable_replaced=False,original_cases_or_expected_values_modified=False,runtime_qualification_transferred=False)


def restore_source(source, bind):
    bind(Path(__file__).resolve())
    return restore_previous(source),proof(source)
