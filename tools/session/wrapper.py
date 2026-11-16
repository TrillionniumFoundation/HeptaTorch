"""Native Linux x86-64 session adapter emitted by a real HIA packer.

The production prefix owns only I/O framing, separate mapped buffers, complete
workspace copying and commit/swap. Tensor/AD/optimizer execution stays in HIA.
"""
import struct
MIB = 1 << 20
CONTROL = 4096
INPUT = CONTROL
OUTPUT = CONTROL + MIB
DURABLE = CONTROL + 2*MIB
SHADOW = CONTROL + 3*MIB
MAPPING = CONTROL + 4*MIB
REQUEST_MAGIC = b'HTOSES01'
RESPONSE_MAGIC = b'HTOSRP01'


def session_template(*, output_bytes=MIB, workspace_bytes=MIB, receipt=None):
    """Linux x86-64 syscall adapter, following the real upstream HIA packer.

    The HIA packer writes these instructions, patches its flat-ABI call and
    copies the compiler-produced image. There is no runtime host assembler.
    Each tensor workspace is exactly one MiB; the second is a transaction copy.
    """
    assert output_bytes == workspace_bytes == MIB
    code = bytearray()
    labels = {}
    branches = []

    def emit(value):
        code.extend(bytes.fromhex(value))

    def word(value):
        code.extend(struct.pack('<I', value))

    def quad(value):
        code.extend(struct.pack('<Q', value))

    def label(name):
        assert name not in labels
        labels[name] = len(code)

    def branch(op, name):
        emit(op)
        branches.append((len(code), name))
        word(0)

    # argc=1; one private anonymous mapping, initially all zeros.
    emit('48 83 3c 24 01'); branch('0f 85', 'usage')
    emit('31 ff be'); word(MAPPING)
    emit('ba 03 00 00 00 41 ba 22 00 00 00 49 c7 c0 ff ff ff ff')
    emit('45 31 c9 b8 09 00 00 00 0f 05 48 3d 01 f0 ff ff')
    branch('0f 83', 'io')
    emit('49 89 c4')                              # r12 = mapping
    emit('49 8d 84 24'); word(DURABLE)
    emit('49 89 44 24 20')                       # control+32 durable ptr
    emit('49 8d 84 24'); word(SHADOW)
    emit('49 89 44 24 28')                       # control+40 shadow ptr

    # Real Linux rt_sigaction(SIGPIPE, SIG_IGN, NULL, 8).  The anonymous
    # control page supplies a zeroed x86-64 kernel sigaction at +64:
    # handler=1, flags=0, restorer=0, mask=0.  Closed stdout thus produces
    # -EPIPE and the same deterministic I/O exit as any failed write.
    emit('49 c7 44 24 40 01 00 00 00')
    emit('bf 0d 00 00 00 49 8d 74 24 40 31 d2 41 ba 08 00 00 00')
    emit('b8 0d 00 00 00 0f 05 48 85 c0'); branch('0f 88', 'io')

    label('frame')
    emit('4d 89 e5 41 be 10 00 00 00')           # header ptr, 16 bytes
    branch('e8', 'read_exact')
    emit('85 c0'); branch('0f 84', 'header')
    emit('49 83 fe 10'); branch('0f 84', 'success')  # EOF between frames
    branch('e9', 'invalid')                       # short frame header

    label('header')
    emit('48 b8'); quad(int.from_bytes(REQUEST_MAGIC, 'little'))
    emit('49 39 04 24'); branch('0f 85', 'invalid')
    emit('4d 8b 74 24 08 49 81 fe'); word(MIB)
    branch('0f 87', 'space')                     # before body read/copy
    emit('4d 8d ac 24'); word(INPUT)
    branch('e8', 'read_exact')
    emit('85 c0'); branch('0f 85', 'invalid')     # short body

    # Transaction starts only after the complete bounded frame arrives.
    emit('49 8b 74 24 20 49 8b 7c 24 28 b9'); word(MIB // 8)
    emit('fc f3 48 a5')                          # cld; rep movsq
    emit('49 8d bc 24'); word(INPUT)             # s pointer
    emit('49 8b 74 24 08')                       # s length
    emit('49 8d 94 24'); word(OUTPUT)            # o pointer
    emit('b9'); word(MIB)                        # o length
    emit('4d 8b 44 24 28')                       # w shadow pointer
    emit('41 b9'); word(MIB)                     # w length, never 2 MiB
    emit('41 54 48 83 ec 08 e8')                # preserve r12; aligned call
    hole = len(code); word(0)
    emit('48 83 c4 08 41 5c')
    emit('83 f8 20'); branch('0f 82', 'invalid')
    emit('3d'); word(MIB); branch('0f 87', 'invalid')
    emit('89 c0 49 89 44 24 18')                # response byte count
    emit('48 b8'); quad(int.from_bytes(RESPONSE_MAGIC, 'little'))
    emit('49 89 44 24 10')

    emit('49 83 bc 24'); word(OUTPUT + 24); emit('00')
    branch('0f 85', 'publish')                  # error: durable unchanged
    emit('49 8b 44 24 20 49 8b 54 24 28')
    emit('49 89 54 24 20 49 89 44 24 28')      # successful commit = swap

    label('publish')
    emit('4d 8d 6c 24 10 41 be 10 00 00 00')
    branch('e8', 'write_all')
    emit('4d 8d ac 24'); word(OUTPUT)
    emit('4d 8b 74 24 18')
    branch('e8', 'write_all')
    branch('e9', 'frame')                       # respond before next read

    # r13 pointer, r14 remaining; 0=complete, 1=EOF. No over-read across frames.
    label('read_exact')
    emit('4d 85 f6'); branch('0f 84', 'read_done')
    emit('31 c0 31 ff 4c 89 ee 4c 89 f2 0f 05 48 83 f8 fc')
    branch('0f 84', 'read_exact')               # EINTR without progress
    emit('48 85 c0'); branch('0f 88', 'io'); branch('0f 84', 'read_eof')
    emit('49 01 c5 49 29 c6'); branch('e9', 'read_exact')
    label('read_done'); emit('31 c0 c3')
    label('read_eof'); emit('b8 01 00 00 00 c3')

    label('write_all')
    emit('4d 85 f6'); branch('0f 84', 'write_done')
    emit('b8 01 00 00 00 bf 01 00 00 00 4c 89 ee 4c 89 f2 0f 05')
    emit('48 83 f8 fc'); branch('0f 84', 'write_all')
    emit('48 85 c0'); branch('0f 8e', 'io')
    emit('49 01 c5 49 29 c6'); branch('e9', 'write_all')
    label('write_done'); emit('c3')

    for name, status in [('success', 0), ('invalid', 2), ('space', 3),
                         ('io', 4), ('usage', 64)]:
        label(name); emit('bf'); word(status); branch('e9', 'exit')
    label('exit'); emit('b8 3c 00 00 00 0f 05 0f 0b')
    for at, target in branches:
        struct.pack_into('<i', code, at, labels[target] - at - 4)
    header = (b'\x7fELF\x02\x01\x01' + bytes(9)
              + struct.pack('<HHIQQQIHHHHHH', 2, 62, 1, 0x4000b0, 64, 0, 0,
                            64, 56, 2, 0, 0, 0)
              + struct.pack('<IIQQQQQQ', 1, 5, 0, 0x400000, 0x400000, 0, 0, 4096)
              + struct.pack('<IIQQQQQQ', 0x6474e551, 6, 0, 0, 0, 0, 0, 16))
    assert len(header) == 176
    metadata = {
        'elf_header_bytes': 176, 'call_hole': 176 + hole,
        'labels': {name: 176 + value for name, value in labels.items()},
        'mapping_bytes': MAPPING, 'input_offset': INPUT,
        'output_offset': OUTPUT, 'workspace_offsets': [DURABLE, SHADOW],
        'each_workspace_bytes': MIB, 'durable_pointer_offset': 32,
        'shadow_pointer_offset': 40,
        'sigpipe_action_offset': 64, 'sigpipe_disposition': 'SIG_IGN',
    }
    if receipt is not None:
        receipt.update(metadata)
    return header + code, 176 + hole
