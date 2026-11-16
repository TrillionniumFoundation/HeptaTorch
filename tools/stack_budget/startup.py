"""Author separate guarded-stack prefixes from pinned existing templates."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import struct


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def session_prefix(original, stack_bytes, observation=None):
    raw = Path(original).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'ab66f4ca7308799df362a281f255709c2d3bb9ace61d97dd0edb7da0ee73438e'
    nodes = ast.parse(raw).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == 'owned_session_template')
    source = ast.get_source_segment(raw.decode(), node)
    setup = '''    emit('49 89 a4 24');word(232)
    emit('31 ff be');word(STACK_BYTES+8192)
    emit('31 d2 41 ba 22 00 00 00 49 c7 c0 ff ff ff ff 45 31 c9')
    emit('b8 09 00 00 00 0f 05 48 3d 01 f0 ff ff');branch('0f 83','unmapped_io')
    emit('49 89 84 24');word(200)
    emit('49 c7 84 24');word(208);word(STACK_BYTES+8192)
    emit('48 8d b8 00 10 00 00 be');word(STACK_BYTES)
    emit('ba 03 00 00 00 b8 0a 00 00 00 0f 05 48 85 c0');branch('0f 85','unmapped_io')
    emit('49 8b 84 24');word(200)
    emit('48 8d a0');word(STACK_BYTES+4096)
    emit('49 89 a4 24');word(216)
    emit('49 89 64 24 38')
    emit('48 b8');budget_hole=len(code);quad(0)
    emit('49 89 84 24');word(224)
'''
    if observation in ('guard-low', 'guard-high'):
        setup += "    emit('49 8b 84 24');word(200)\n"
        if observation == 'guard-high':
            setup += "    emit('48 8d 80');word(STACK_BYTES+4096)\n"
        setup += "    emit('48 8b 00')\n"
    source = replace_once(source,
        "    emit('49 89 64 24 38')             # control+56 = initial aligned rsp\n", setup)
    release = '''    emit('45 8b 54 24 30')
    emit('49 8b a4 24');word(232)
    emit('49 8b bc 24');word(200)
    emit('49 8b b4 24');word(208)
    emit('b8 0b 00 00 00 0f 05 48 85 c0');branch('0f 85','unmapped_io')
    emit('4c 89 e7 be');word(MAPPING)
    emit('b8 0b 00 00 00 0f 05 48 85 c0');branch('0f 85','unmapped_io')
    emit('44 89 d7');branch('e9','exit')
'''
    if observation == 'cleanup':
        hook = '''    emit('48 83 ec 20 48 b8');quad(int.from_bytes(b'HTSTKCL1','little'))
    emit('48 89 04 24 4c 89 54 24 08')
    emit('b8 01 00 00 00 bf 03 00 00 00 48 89 e6 ba 10 00 00 00 0f 05')
    emit('48 83 f8 10');branch('0f 85','unmapped_io')
    emit('31 c0 bf 04 00 00 00 48 8d 74 24 10 ba 01 00 00 00 0f 05')
    emit('48 83 f8 01');branch('0f 85','unmapped_io')
    emit('48 83 c4 20')
'''
        release = replace_once(release, "    emit('44 89 d7');branch('e9','exit')\n",
                               hook + "    emit('44 89 d7');branch('e9','exit')\n")
    source = replace_once(source, "    emit('41 8b 7c 24 30');branch('e9','exit')\n", release)
    source = replace_once(source, '    return prefix,176+step_hole,176+close_hole,metadata',
        "    metadata['stack']={'bytes':STACK_BYTES,'guard_bytes_each':4096,'mapping_bytes':STACK_BYTES+8192,'budget_reserve':4096,'budget_hole':176+budget_hole,'base_control':200,'top_control':216,'required_control':224,'original_rsp_control':232,'normal_cleanup_unmaps_stack_and_io':True}\n"
        '    return prefix,176+step_hole,176+close_hole,metadata')
    namespace = dict(hashlib=hashlib, struct=struct, MIB=1<<20, CONTROL=4096,
                     INPUT=4096, OUTPUT=4096+(1<<20), MAPPING=4096+(2<<20),
                     REQUEST_MAGIC=b'HTOWNR01', RESPONSE_MAGIC=b'HTOWNS01', STACK_BYTES=stack_bytes)
    exec(compile(source, '<separate-guarded-stack-session-authoring>', 'exec'), namespace)
    result = namespace['owned_session_template']()
    result[3]['stack']['qualification_observation'] = observation
    assert Path(original).read_bytes() == raw
    return (*result, source)


def filter_prefix(module, stack_bytes, workspace_bytes, io_bytes=1048576):
    source = inspect.getsource(module.template)
    old = """    emit('4c 89 e7 4c 89 ee 49 8d 94 24');word(OUTPUT_OFFSET)
    emit('b9');word(output_bytes)
    emit('4d 8d 84 24');word(workspace_offset)
    emit('41 b9');word(workspace_bytes)
    emit('e8');hole=len(code);word(0)
"""
    new = """    emit('49 89 a4 24');word(WIRE_BYTES+2048)
    emit('31 ff be');word(STACK_BYTES+8192)
    emit('31 d2 41 ba 22 00 00 00 49 c7 c0 ff ff ff ff 45 31 c9')
    emit('b8 09 00 00 00 0f 05 48 3d 01 f0 ff ff');branch('0f 83','io')
    emit('49 89 84 24');word(WIRE_BYTES+2056)
    emit('48 8d b8 00 10 00 00 be');word(STACK_BYTES)
    emit('ba 03 00 00 00 b8 0a 00 00 00 0f 05 48 85 c0');branch('0f 85','io')
    emit('49 8b 84 24');word(WIRE_BYTES+2056)
    emit('48 8d a0');word(STACK_BYTES+4096)
    emit('48 83 ec 40 4c 89 24 24 4c 89 6c 24 08')
    emit('49 8d 84 24');word(OUTPUT_OFFSET)
    emit('48 89 44 24 10 48 c7 44 24 18');word(output_bytes)
    emit('49 8d 84 24');word(workspace_offset)
    emit('48 89 44 24 20 48 c7 44 24 28');word(workspace_bytes)
    emit('48 89 e7 48 8d 74 24 30')
    emit('e8');hole=len(code);word(0)
    emit('44 8b 74 24 30')
    emit('49 8b a4 24');word(WIRE_BYTES+2048)
    emit('49 8b bc 24');word(WIRE_BYTES+2056)
    emit('be');word(STACK_BYTES+8192)
    emit('b8 0b 00 00 00 0f 05 48 85 c0');branch('0f 85','io')
    emit('44 89 f0')
"""
    source = replace_once(source, old, new)
    namespace = dict(module.template.__globals__, WIRE_BYTES=io_bytes,
                     OUTPUT_OFFSET=io_bytes+4096, STACK_BYTES=stack_bytes)
    exec(compile(source, '<separate-guarded-stack-filter-authoring>', 'exec'), namespace)
    prefix, hole = namespace['template'](output_bytes=io_bytes, workspace_bytes=workspace_bytes)
    return prefix, hole, {'stack_bytes':stack_bytes,'guard_bytes_each':4096,'mapping_bytes':stack_bytes+8192,
                         'budget_reserve':4096,'workspace_bytes':workspace_bytes,'input_bytes':io_bytes,'output_bytes':io_bytes,
                         'normal_cleanup_unmaps_stack':True}, source
