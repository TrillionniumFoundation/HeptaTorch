"""Reversible wire and identity conversion; no Tensor numerical operations."""
import hashlib
import struct

MIB = 1 << 20
RESULT_REF = 0xfe000000
TENSOR_FIELDS = {
    2:(2,3), 3:(2,), 4:(2,), 5:(1,2), 6:(1,), 7:(1,), 8:(1,), 9:(1,),
    10:(1,), 11:(1,), 12:(1,), 13:(1,), 15:(1,2,3), 16:(1,2), 17:(1,),
    18:(1,), 19:(1,), 20:(2,), 21:(2,), 22:(1,2), 23:(2,), 24:(1,2,3),
    25:(1,2), 26:(1,2,3), 27:(1,2), 28:(1,2,3,4,5), 29:(1,2),
    30:(1,), 31:(1,), 32:(1,2), 36:(1,2), 37:(1,2),
}
TENSOR_RESULTS = {1,2,3,4,5,6,15,16,17,18,19,20,21,22,23,26,27,28,30,31,32,34,35,36,37}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def typed(value, kind):
    generation, actual_kind, slot = value >> 32, (value >> 24) & 255, value & 0xffffff
    require(generation > 0 and actual_kind == kind and 1 <= slot <= 65536,
            'actual typed identity kind/generation/slot')
    return generation, slot


def parse_request(raw):
    require(32 <= len(raw) <= MIB, 'original request extent')
    magic, count, extent, reserved = struct.unpack_from('<8sQQQ', raw)
    require(magic == b'HTORCH01' and extent == len(raw) and reserved == 0,
            'original request envelope')
    require(count <= (len(raw)-32)//64, 'original command extent')
    commands = [list(struct.unpack_from('<8Q', raw, 32+64*i)) for i in range(count)]
    return commands


def allocations(command):
    """Only the old published descriptor-bookkeeping contract in docs/protocol.md."""
    op = command[0]
    if op not in TENSOR_RESULTS:
        return 0
    if op == 15:
        return 2 + bool(command[3])
    if op == 16:
        return 3
    if op == 26:
        return 8 + bool(command[2]) + bool(command[3])
    if op == 27:
        return 5
    if op == 28:
        return 7 + 2*bool(command[4]) + 2*bool(command[5])
    return 1


def history(commands):
    produced, internal, returned, owners, capacity = {}, {}, {}, {}, []
    generation, keep, next_slot = 0, 0, 1
    for index, command in enumerate(commands):
        count = allocations(command)
        if count:
            last = next_slot + count - 1
            for slot in range(next_slot, last):
                if slot <= 255:
                    logical = slot if slot <= keep else (generation << 8) | slot
                    internal[logical] = index
            if last <= 255:
                logical = last if last <= keep else (generation << 8) | last
                require(logical not in produced, 'ambiguous original returned logical identity')
                produced[logical], returned[index] = index, logical
                owners[logical] = owners.get(command[1]) if command[0] in (6,17,19) else logical
            else:
                capacity.append(dict(command=index, old_descriptor_limit=255,
                                     would_allocate_through=last))
            next_slot = last + 1
        elif command[0] == 33:
            keep, next_slot = command[1], command[1]+1
            generation += 1
    return dict(produced=produced, internal=internal, returned=returned, owners=owners,
                capacity_crossings=capacity, final_next=next_slot,
                final_generation=generation, final_keep=keep)


def convert_request(raw, prefix=None):
    commands = parse_request(raw)
    count = len(commands) if prefix is None else prefix
    require(0 <= count <= len(commands), 'original command prefix range')
    ledger = history(commands)
    actual, changes, unresolved, private = bytearray(raw), {}, [], []
    actual[:8] = b'HTORCH03'
    struct.pack_into('<Q', actual, 8, count)

    def relocate(address, ordinal, role):
        logical, = struct.unpack_from('<Q', raw, address)
        origin = ledger['produced'].get(logical)
        if origin is None:
            if logical in ledger['internal'] and ledger['internal'][logical] < ordinal:
                private.append(dict(command=ordinal, address=address, logical=logical,
                                    origin=ledger['internal'][logical], role=role))
            elif logical:
                unresolved.append(dict(command=ordinal, address=address, literal=logical, role=role))
            return
        replacement = RESULT_REF | (origin + 1)
        record = dict(address=address, original=logical, actual=replacement,
                      origin_command=origin, uses=[ordinal], role=role)
        if address in changes:
            require(changes[address]['actual'] == replacement, 'shared payload identity disagreement')
            changes[address]['uses'].append(ordinal)
        else:
            changes[address] = record
            struct.pack_into('<Q', actual, address, replacement)

    for index, command in enumerate(commands[:count]):
        for field in TENSOR_FIELDS.get(command[0], ()):
            relocate(32 + 64*index + 8*field, index, 'tensor-command-field')
        if command[0] in (34,35):
            offset = command[2]
            # Malformed offsets/counts remain exact original bits. They are
            # refused, or accepted, only by the actual native command.
            if offset % 8 == 0 and offset >= 32+64*len(commands) and offset <= len(raw)-8:
                length, = struct.unpack_from('<Q', raw, offset)
                if length <= (len(raw)-offset-8)//8:
                    for at in range(length):
                        relocate(offset+8+8*at, index, 'tensor-list-member')
    inverse = bytearray(actual)
    inverse[:8] = b'HTORCH01'
    struct.pack_into('<Q', inverse, 8, len(commands))
    for address, row in changes.items():
        require(struct.unpack_from('<Q', inverse, address)[0] == row['actual'], 'inverse identity word')
        struct.pack_into('<Q', inverse, address, row['original'])
    require(bytes(inverse) == raw, 'whole original packet inverse differs')
    proof = dict(original_sha256=sha(raw), actual_sha256=sha(actual), original_bytes=len(raw),
        actual_bytes=len(actual), original_commands=len(commands), executed_commands=count,
        original_payload_offsets_unchanged=True, numeric_and_index_payload_bytes_changed=False,
        command_codes_and_scalar_bits_changed=False, whole_original_packet_inverse_exact=True,
        retained_unused_suffix_bytes=64*(len(commands)-count),
        identity_changes=list(changes.values()), unresolved_literals=unresolved,
        unpublished_composite_inputs=private, capacity_crossings=ledger['capacity_crossings'])
    return bytes(actual), proof, commands[:count], ledger


def response(raw, version):
    require(len(raw) >= 32, 'complete response header')
    magic, observed_version, count, status = struct.unpack_from('<8sQQQ', raw)
    require(magic == (b'HTORSP01' if version == 1 else b'HTORSP03')
            and observed_version == version, 'response magic/version')
    rows, at = [], 32
    for index in range(count):
        require(at+32 <= len(raw), 'complete response record header')
        op, error, value, length = struct.unpack_from('<4Q', raw, at)
        at += 32
        require(length <= len(raw)-at, 'complete response payload')
        rows.append(dict(op=op, error=error, value=value, payload=raw[at:at+length], index=index))
        at += length
    require(at == len(raw), 'response has extra bytes')
    if version == 3 and status:
        require(count == 0 and len(raw) == 32, 'actual v3 atomic failure framing')
    return status, rows


def project_success(raw, commands, ledger, owner_snapshot):
    status, rows = response(raw, 3)
    require(status == 0 and len(rows) == len(commands), 'all current records must actually succeed')
    bindings, seen, generations, proof, pieces = {}, set(), {}, [], []

    def live_tensor(token):
        generation, slot = typed(token, 1)
        table = owner_snapshot['tables']['1']
        require(slot <= len(table), 'metadata root Tensor slot not observed')
        row = table[slot-1]
        require(row[0] == generation and row[1] == 1, 'metadata root no longer live at final owner observation')
        return row

    for index, (command, row) in enumerate(zip(commands, rows)):
        op, value, payload = row['op'], row['value'], row['payload']
        require(op == command[0] and row['error'] == 0, 'actual opcode/error differs')
        if op in TENSOR_RESULTS:
            require(index in ledger['returned'] and payload == b'', 'old constructor identity is outside representable contract')
            logical = ledger['returned'][index]
            generation, slot = typed(value, 1)
            require(value not in seen and generation > generations.get(slot, 0), 'actual native identity reuse')
            seen.add(value); generations[slot] = generation; bindings[logical] = value
            proof.append(dict(record=index, kind='TensorID', actual=value, logical=logical))
            value = logical
        elif op == 13:
            require(len(payload) == 128 and command[1] in bindings
                    and value == bindings[command[1]], 'actual metadata Tensor identity')
            words = struct.unpack('<16Q', payload)
            typed(words[11], 2); typed(words[12], 3)
            root = ledger['owners'].get(command[1])
            require(root in bindings, 'original metadata owner identity is not declared')
            actual_root = live_tensor(bindings[root])
            require(actual_root[2] == words[11] and actual_root[3] == words[12],
                    'actual Storage/Version identities do not match original owner root')
            require(words[14] <= words[15], 'actual metadata storage byte extent')
            proof.append(dict(record=index, kind='metadata', logical_tensor=command[1],
                logical_storage_owner=root, actual_TensorID=value, actual_StorageID=words[11],
                actual_VersionID=words[12], actual_counter=words[13],
                actual_offset_bytes=words[14], actual_storage_bytes=words[15],
                original_version_counter_offset_applied=0))
            value = command[1]
            payload = struct.pack('<13Q', *words[:11], root, words[13])
        pieces.append(struct.pack('<4Q', op, 0, value, len(payload)) + payload)
    projected = struct.pack('<8sQQQ', b'HTORSP01', 1, len(rows), 0) + b''.join(pieces)
    # Keep the actual typed identities and added metadata as an explicit
    # inverse sidecar. No comparison value is taken from the old fixture here.
    _, projected_rows = response(projected, 1)
    changes = {row['record']: row for row in proof}
    inverse_records = []
    for index, row in enumerate(projected_rows):
        value, payload = row['value'], row['payload']
        patch = changes.get(index)
        if patch and patch['kind'] == 'TensorID':
            require(value == patch['logical'], 'projected constructor identity inverse')
            value = patch['actual']
        elif patch:
            words = struct.unpack('<13Q', payload)
            require(value == patch['logical_tensor'] and words[11] == patch['logical_storage_owner']
                    and words[12] == patch['actual_counter'], 'metadata identity inverse')
            value = patch['actual_TensorID']
            payload = struct.pack('<16Q', *words[:11], patch['actual_StorageID'],
                patch['actual_VersionID'], words[12], patch['actual_offset_bytes'], patch['actual_storage_bytes'])
        inverse_records.append(struct.pack('<4Q', row['op'], row['error'], value, len(payload)) + payload)
    inverse = struct.pack('<8sQQQ', b'HTORSP03', 3, len(rows), 0) + b''.join(inverse_records)
    require(inverse == raw, 'complete actual response inverse differs')
    return projected, dict(actual_response_sha256=sha(raw), projected_response_sha256=sha(projected),
        mappings=proof, numerical_payload_bytes_modified=False, version_counter_offsets_applied=False,
        every_constructor_identity_from_actual_native_response=True, complete_actual_response_inverse_exact=True)
