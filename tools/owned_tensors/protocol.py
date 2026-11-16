"""Independent v2 tensor wire helpers used only by development qualification."""
from pathlib import Path
import importlib.util
import struct

CLIENT = Path(__file__).resolve().parent.parent / 'owned_sessions/client.py'
spec = importlib.util.spec_from_file_location('owned_tensor_transport', CLIENT)
transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
MIB = 1 << 20


class Payload:
    def __init__(self, data, alignment=8):
        self.data = bytes(data); self.alignment = alignment


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def packet(commands):
    end = 32 + 64 * len(commands)
    data = bytearray(); records = bytearray()
    for command in commands:
        assert 1 <= len(command) <= 8
        words = []
        for word in command:
            if isinstance(word, Payload):
                while (end + len(data)) % word.alignment: data.append(0)
                words.append(end + len(data)); data.extend(word.data)
            else:
                words.append(int(word) & ((1 << 64) - 1))
        records.extend(struct.pack('<8Q', *words, *([0] * (8 - len(words)))))
    body = b'HTORCH02' + struct.pack('<3Q', len(commands), end + len(data), 0) + records + data
    assert len(body) <= MIB
    return bytes(body)


def parse(raw):
    assert len(raw) >= 32
    magic, version, count, status = struct.unpack_from('<8s3Q', raw)
    assert magic == b'HTORSP02' and version == 2
    if status:
        assert count == 0 and len(raw) == 32
        return status, []
    result = []; at = 32
    for _ in range(count):
        assert at + 32 <= len(raw)
        opcode, error, value, size = struct.unpack_from('<4Q', raw, at); at += 32
        assert error == 0 and at + size <= len(raw)
        result.append({'opcode': opcode, 'value': value, 'payload': raw[at:at + size]})
        at += size
    assert at == len(raw)
    return 0, result


def tensor(shape, values, requires=0):
    raw = struct.pack('<' + 'f' * len(values), *values)
    return (1, len(shape), requires, *shape, *([0] * (4 - len(shape))), Payload(raw, 4))


def full(shape, value, requires=0):
    return (38, len(shape), requires, *shape, *([0] * (4 - len(shape))), bits(value))


def meta(record):
    assert record['opcode'] == 13 and len(record['payload']) == 104
    rank, count, requires, *rest = struct.unpack('<13Q', record['payload'])
    return {'rank': rank, 'numel': count, 'requires_grad': requires, 'shape': rest[:4],
        'strides': rest[4:8], 'owner_id': rest[8], 'version': rest[9]}


class Session(transport.OwnerSession):
    def read_exact(self, fd, count, timeout=120):
        return super().read_exact(fd, count, timeout=timeout)

    def run(self, commands):
        status, records = parse(self.exchange(packet(commands)))
        assert status == 0 and len(records) == len(commands), (status, records)
        assert [record['opcode'] for record in records] == [command[0] for command in commands]
        return records

    def error(self, commands, expected):
        before = self.owner()
        status, records = parse(self.exchange(packet(commands)))
        assert status == expected and not records, (status, expected)
        after = self.owner()
        assert before == after
        return before['length']

    def read_constant(self, tid, count, value, gradient=False):
        step = 131072
        expected = struct.pack('<I', bits(value))
        for start in range(0, count, step):
            amount = min(step, count - start)
            record, = self.run([(40 if gradient else 39, tid, start, amount)])
            assert record['value'] == amount and record['payload'] == expected * amount
        return count
