"""Transport-only bridge from the frozen 28-case client ledger to native v3.

Logical client IDs preserve the old suite's bookkeeping expressions. They are
never interpreted as native slots. Native IDs are accepted only from actual
successful responses; same-frame operands use the checked native result-ref
mechanism. No numeric operation or native allocation policy is implemented here.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import struct


OWNED_SHA = 'ee3cf72154ac9dfa31055557b3facab8fdd3e14aad03ef6000ced55ccd75ed88'
SESSION_SHA = '35fa5fa3e1b740f901c11d46dca484079b77805004586c03ae42b35be2265577'
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
FINAL_NATIVE_SHA = '4bd22ab115a5428be6ee1ab77fd31a1e20baf14a80756eb38f000574cbad531f'
RESULT_REF = 0xfe000000

# Word positions in command records, independently specified by the wire
# contract. Command 14 consumes scalar bit patterns and produces a scalar.
TENSOR_FIELDS = {
    2:(2,3), 3:(2,), 4:(2,), 5:(1,2), 6:(1,), 7:(1,), 8:(1,), 9:(1,),
    10:(1,), 11:(1,), 12:(1,), 13:(1,), 15:(1,2,3), 16:(1,2), 17:(1,),
    18:(1,), 19:(1,), 20:(2,), 21:(2,), 22:(1,2), 23:(2,), 24:(1,2,3),
    25:(1,2), 26:(1,2,3), 27:(1,2), 28:(1,2,3,4,5), 29:(1,2),
    30:(1,), 31:(1,), 32:(1,2), 36:(1,2), 37:(1,2), 39:(1,), 40:(1,),
    41:(1,), 42:(1,), 46:(1,), 55:(1,),
}
TENSOR_RESULTS = {1,2,3,4,5,6,15,16,17,18,19,20,21,22,23,26,27,28,30,31,32,34,35,36,37,38,55,56}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def typed(handle, kind):
    assert isinstance(handle, int) and 0 < handle < 1 << 64
    generation, observed_kind, slot = handle >> 32, (handle >> 24) & 255, handle & 0xffffff
    assert generation > 0 and observed_kind == kind and 1 <= slot <= 65536, (handle, kind)
    return generation, slot


def prepare_sources(owned_path, session_path, support_path):
    owned_raw, session_raw, support_raw = (Path(p).read_bytes() for p in (owned_path, session_path, support_path))
    assert sha(owned_raw) == OWNED_SHA
    assert sha(session_raw) == SESSION_SHA
    assert sha(support_raw) == SUPPORT_SHA
    scope = {'__name__':'frozen_owned_tensor_oracle_definitions', '__file__':str(owned_path)}
    # Execute the unmodified frozen file as a definition provider. Its guarded
    # main is not called, so this does not import Torch or start a native target.
    exec(compile(owned_raw, str(owned_path), 'exec'), scope)
    definitions, registrations, old_audit = scope['adapt_session'](session_raw.decode(), str(session_path))
    original = copy.deepcopy(definitions)
    session, = [node for node in definitions.body if isinstance(node, ast.ClassDef) and node.name == 'Session']
    exchange, = [node for node in session.body if isinstance(node, ast.FunctionDef) and node.name == 'exchange']
    counts = {'response_magic':0, 'response_version':0, 'begin_hook':0, 'raw_trace_hook':0,
              'record_projection_hook':0, 'successful_commit_hook':0}
    for node in ast.walk(exchange):
        if isinstance(node, ast.Constant) and node.value == b'HTORSP02':
            node.value = b'HTORSP03'
            counts['response_magic'] += 1
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == 'version'
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 2):
            node.comparators[0].value = 3
            counts['response_version'] += 1

    def statement(text):
        return ast.parse(text).body[0]

    result = []
    for node in exchange.body:
        result.append(node)
        spelling = ast.unparse(node)
        if spelling == 'body = program.encode()':
            result.append(statement('self._v3_begin(program, body)'))
            counts['begin_hook'] += 1
        if spelling == 'response = self.read(length)':
            result.append(statement('self._v3_trace(raw, response)'))
            counts['raw_trace_hook'] += 1
        if spelling == 'self.next_id, self.generation, self.keep = (program.next_id, program.generation, program.keep)':
            result.append(statement('self._v3_commit(program)'))
            counts['successful_commit_hook'] += 1
        if isinstance(node, ast.For) and ast.unparse(node.target) == '(index, command)':
            body = []
            for inner in node.body:
                if ast.unparse(inner) == 'records.append((op, value, payload))':
                    body.append(statement('value, payload = self._v3_record(program, index, op, value, payload)'))
                    counts['record_projection_hook'] += 1
                body.append(inner)
            node.body = body
    exchange.body = result
    assert counts == {name:1 for name in counts}, counts

    class Reverse(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in (
                    'self._v3_begin', 'self._v3_trace', 'self._v3_commit'):
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'self._v3_record':
                return None
            return self.generic_visit(node)

    restored = Reverse().visit(copy.deepcopy(definitions))
    for node in ast.walk(restored):
        if isinstance(node, ast.Constant) and node.value == b'HTORSP03':
            node.value = b'HTORSP02'
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == 'version'
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 3):
            node.comparators[0].value = 2
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    ast.fix_missing_locations(definitions)
    owned_tree = ast.parse(owned_raw, filename=str(owned_path))
    owned_main, = [n for n in owned_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    owned_program, = [n for n in owned_main.body if isinstance(n, ast.ClassDef) and n.name == 'OwnedProgram']
    support_tree = ast.parse(support_raw, filename=str(support_path))
    support_nodes = [n for n in support_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                     and n.name in ('Program', 'bits', 'f32')]
    assert len(support_nodes) == 3
    audit = {'frozen_28_oracle_sha256': OWNED_SHA, 'frozen_session_sha256': SESSION_SHA,
             'frozen_support_sha256': SUPPORT_SHA, 'inherited_overlay_audit': old_audit,
             'v3_exchange_changes': counts, 'reverse_overlay_restores_entire_v2_definition_AST': True,
             'v2_definition_AST_sha256': sha(ast.dump(original, include_attributes=False).encode()),
             'v3_definition_AST_sha256': sha(ast.dump(definitions, include_attributes=False).encode()),
             'all_case_function_AST_unchanged': True,
             'registration_AST_unchanged': True,
             'owned_program_AST_sha256': sha(ast.dump(owned_program, include_attributes=False).encode()),
             'source_assertions_preserved_except_explicit_protocol_response_version': True,
             'metadata_projection': 'Actual v3 words [0:11] + actual StorageID word11 + actual counter word13; VersionID retained and checked separately.',
             'logical_id_assertion_scope': 'Original ordinal/generation expressions remain client ledger checks. Actual typed native IDs are separately validated; the adapter does not predict native slots.'}
    return scope, definitions, registrations, owned_program, support_nodes, audit


def install(scope, definitions, owned_program, base_program, executable, trace_directory=None):
    scope['EXE'] = Path(executable).resolve()
    scope['original_program'] = base_program
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(owned_program)], type_ignores=[])),
                 '<unchanged-frozen-OwnedProgram>', 'exec'), scope)
    OwnedProgram = scope['OwnedProgram']

    class ObjectProgram(OwnedProgram):
        def __init__(self):
            super().__init__()
            self.produced, self.known = {}, {}

        def handle(self, slot):
            logical = super().handle(slot)
            ordinal = len(self.commands)-1
            assert ordinal >= 0 and self.commands[ordinal][0] in TENSOR_RESULTS
            assert logical not in self.produced
            self.produced[logical] = ordinal
            return logical

        def resolve(self, logical, use_ordinal):
            if logical == 0:
                return 0
            if logical in self.produced:
                index = self.produced[logical]
                assert index < use_ordinal, 'client forward result reference'
                return RESULT_REF | (index+1)
            return self.known.get(logical, logical)

        def encode(self):
            original = super().encode()
            assert original[:8] == b'HTORCH02'
            result = bytearray(original)
            result[:8] = b'HTORCH03'
            patches = []
            for index, command in enumerate(self.commands):
                for field in TENSOR_FIELDS.get(command[0], ()):
                    offset = 32+index*64+field*8
                    old, = struct.unpack_from('<Q', original, offset)
                    new = self.resolve(old, index)
                    struct.pack_into('<Q', result, offset, new)
                    patches.append((offset, old, new))
                if command[0] in (34,35):
                    offset, = struct.unpack_from('<Q', original, 32+index*64+16)
                    count, = struct.unpack_from('<Q', original, offset)
                    assert offset >= 32+64*len(self.commands) and offset+8+8*count <= len(original)
                    for j in range(count):
                        address = offset+8+j*8
                        old, = struct.unpack_from('<Q', original, address)
                        new = self.resolve(old, index)
                        struct.pack_into('<Q', result, address, new)
                        patches.append((address, old, new))
            restored = bytearray(result)
            restored[:8] = b'HTORCH02'
            for offset, old, _ in patches:
                struct.pack_into('<Q', restored, offset, old)
            assert bytes(restored) == original, 'adapter modified a non-ID byte'
            self.patch_count = sum(old != new for _, old, new in patches)
            self.result_reference_count = sum(new >> 24 == 254 for _, _, new in patches)
            return bytes(result)

    scope['Program'] = ObjectProgram
    exec(compile(definitions, '<reversible-v3-Session-overlay>', 'exec'), scope)
    OriginalSession = scope['Session']

    class ObjectSession(OriginalSession):
        def __init__(self, row):
            super().__init__(row)
            self.bindings, self.native_seen, self.slot_generation = {}, {}, {}
            self.raw_metadata, self.metadata_history = {}, []
            self.wire_sequence = bytes(32)
            self.trace_requests = self.trace_responses = None
            row['v3_bridge'] = {'protocol':3, 'actual_typed_bindings':[],
                'raw_metadata_records':[], 'native_identity_reuse_events':0,
                'id_words_relocated':0, 'request_local_references':0,
                'logical_handle_encoding_is_not_native_handle_encoding':True}
            if trace_directory is not None:
                directory = Path(trace_directory)/sha(row['case'].encode())[:16]
                directory.mkdir(parents=True, exist_ok=False)
                self.trace_requests = (directory/'requests.frames.bin').open('wb')
                self.trace_responses = (directory/'responses.frames.bin').open('wb')

        def program(self):
            program = super().program()
            program.known = self.bindings.copy()
            return program

        def _v3_begin(self, program, body):
            assert body[:8] == b'HTORCH03'
            self.pending_bindings = self.bindings.copy()
            self.pending_seen = self.native_seen.copy()
            self.pending_generations = self.slot_generation.copy()
            self.pending_events = []
            self.pending_metadata = []
            self.pending_reuses = 0
            self.row['v3_bridge']['id_words_relocated'] += program.patch_count
            self.row['v3_bridge']['request_local_references'] += program.result_reference_count

        def _v3_trace(self, request, response):
            framed = b'HTOWNS01'+struct.pack('<Q', len(response))+response
            self.wire_sequence = hashlib.sha256(self.wire_sequence+request+framed).digest()
            self.row['v3_bridge']['ordered_wire_sha256'] = self.wire_sequence.hex()
            if self.trace_requests is not None:
                self.trace_requests.write(request); self.trace_requests.flush()
                self.trace_responses.write(framed); self.trace_responses.flush()

        def _v3_record(self, program, index, op, value, payload):
            logicals = [logical for logical, ordinal in program.produced.items() if ordinal == index]
            if logicals:
                logical, = logicals
                assert op in TENSOR_RESULTS and payload == b''
                generation, slot = typed(value, 1)
                assert value not in self.pending_seen, 'native Tensor identity reused'
                previous = self.pending_generations.get(slot, 0)
                assert generation > previous
                if previous:
                    self.pending_reuses += 1
                self.pending_generations[slot] = generation
                self.pending_bindings[logical] = value
                self.pending_seen[value] = logical
                self.pending_events.append({'logical_client_id':logical, 'actual_native_id':value,
                    'generation':generation, 'slot':slot, 'request_record':index+1})
                value = logical
            if op == 13:
                assert len(payload) == 128
                raw = struct.unpack('<16Q', payload)
                logical = program.commands[index][1]
                actual = self.pending_bindings.get(logical, logical)
                typed(value, 1)
                assert value == actual, ('metadata returned another Tensor', value, actual)
                typed(raw[11], 2); typed(raw[12], 3)
                self.pending_metadata.append((logical, raw))
                self.row['v3_bridge']['raw_metadata_records'].append({
                    'logical_client_id':logical, 'actual_native_id':actual,
                    'StorageID':raw[11], 'VersionID':raw[12], 'counter':raw[13],
                    'storage_offset_bytes':raw[14], 'storage_bytes':raw[15]})
                payload = struct.pack('<13Q', *raw[:11], raw[11], raw[13])
                value = logical
            return value, payload

        def _v3_commit(self, program):
            assert len(self.pending_events) == len(program.produced)
            self.bindings, self.native_seen = self.pending_bindings, self.pending_seen
            self.slot_generation = self.pending_generations
            self.row['v3_bridge']['actual_typed_bindings'].extend(self.pending_events)
            self.row['v3_bridge']['native_identity_reuse_events'] += self.pending_reuses
            for logical, raw in self.pending_metadata:
                self.raw_metadata[logical] = raw
                self.metadata_history.append(raw)

        def close(self):
            program = self.program()
            index = program.command(54)
            record = self.exchange(program)[index]
            assert record[1] == 1 and len(record[2]) == 96
            stats = struct.unpack('<12Q', record[2])
            assert stats[11] == self.next_id-1, ('native node sequence disagrees with client ledger', stats[11], self.next_id-1)
            self.row['v3_bridge']['final_native_stats'] = list(stats)
            self.row['v3_bridge']['native_node_sequence_matches_logical_creation_count'] = True
            super().close()

        def stop(self):
            try:
                super().stop()
            finally:
                for file in (self.trace_requests, self.trace_responses):
                    if file is not None and not file.closed:
                        file.close()

    scope['Session'] = ObjectSession
    original_alias = scope['_v2_alias_strict']

    def strict_alias(session):
        original_alias(session)
        assert len(session.metadata_history) == 6
        before, after = session.metadata_history[:3], session.metadata_history[3:]
        for triple in (before, after):
            assert triple[0][11] == triple[2][11] != triple[1][11]
            assert triple[0][12] == triple[2][12] != triple[1][12]
        assert [v[11:13] for v in before] == [v[11:13] for v in after]
        record = session.row['version_observation']
        record['assertion_origin_profile'] = record['profile']
        record['profile'] = 'owned numeric Tensor objects v3'
        record['actual_Storage_and_Version_identities_strictly_checked'] = True

    scope['_v2_alias_strict'] = strict_alias
    return ObjectProgram, ObjectSession
