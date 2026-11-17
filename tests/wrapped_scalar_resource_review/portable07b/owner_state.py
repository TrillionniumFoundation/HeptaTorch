"""Exact parent raw State class with two explicit format-tag bindings."""
import struct
BASE=139264

class State:
    """Decode table/descriptor/extent fields, without broad graph certification."""
    def __init__(self, raw):
        self.raw = raw
        assert raw[64:72] == b'HTVSHP02' and raw[BASE:BASE+8] == b'HTOVSHP2'
        self.slots = self.q(BASE + 16)
        self.pool = self.q(BASE + 24)
        self.data = BASE + 144 + 552 * self.slots
        assert len(raw) == self.data + self.pool
        self.next = self.q(8)

    def q(self, offset):
        assert 0 <= offset <= len(self.raw) - 8
        return struct.unpack_from('<Q', self.raw, offset)[0]

    def row(self, handle, kind=None, live=True):
        k = (handle >> 24) & 255
        slot = handle & 0xffffff
        assert 1 <= k <= 8 and (kind is None or k == kind) and 1 <= slot <= self.slots
        at = BASE + 128 + ((k - 1) * self.slots + slot - 1) * 64
        row = struct.unpack_from('<8Q', self.raw, at)
        assert row[0] == handle >> 32
        if live:
            assert row[1] == 1
        return row

    def table(self, kind):
        result = {}
        for slot in range(1, self.slots + 1):
            at = BASE + 128 + ((kind - 1) * self.slots + slot - 1) * 64
            row = struct.unpack_from('<8Q', self.raw, at)
            if row[1]:
                assert row[1] == 1 and row[0] != 0
                result[(row[0] << 32) | (kind << 24) | slot] = row
        return result

    def counts(self):
        return [len(self.table(kind)) for kind in range(1, 9)]

    def node(self, node):
        assert 0 < node < self.next
        page = self.q(4096 + (node >> 8) * 8)
        assert page >> 63 == 1
        at = self.data + (page & ((1 << 63) - 1)) + (node & 255) * 672
        return struct.unpack_from('<84Q', self.raw, at)

    def tensor_node(self, tid):
        node = self.row(tid, 1)[6]
        desc = self.node(node)
        assert desc[192 // 8] == tid and desc[304 // 8] == 1
        return node, desc

    def storage_bytes(self, sid):
        row = self.row(sid, 2)
        assert row[5] != 0 and row[7] == 0 and row[2] + row[3] <= self.pool
        return self.raw[self.data + row[2]:self.data + row[2] + row[3]]

    def rng(self, generator):
        hold = self.row(generator, 7)[2]
        sid = self.row(hold, 6)[2]
        raw = self.storage_bytes(sid)
        assert len(raw) == 5056
        return raw

    def free(self):
        start = BASE + 128 + 512 * self.slots
        count = self.q(BASE + 32)
        assert count <= self.slots + 1
        spans = [struct.unpack_from('<2Q', self.raw, start + 16 * i) for i in range(count)]
        assert all(size != 0 and offset + size <= self.pool for offset, size in spans)
        assert all(spans[i][0] + spans[i][1] < spans[i+1][0] for i in range(len(spans)-1))
        return spans

    def partition(self):
        spans = [(offset, size, 'free') for offset, size in self.free()]
        for sid, row in self.table(2).items():
            if row[5]:
                assert row[7] == 0 and row[4] == (row[3] + 15) & ~15
                if row[4]:
                    spans.append((row[2], row[4], str(sid)))
            else:
                # The real producer retains logical byte_size in a tombstone.
                assert row[2] == row[4] == 0 and row[7] == 1 and row[6] > 0
        at = 0
        for offset, size, label in sorted(spans):
            assert offset == at, ('extent partition', offset, at, label)
            at += size
        assert at == self.pool
        return len(spans)

    def final_pages_only(self):
        counts = self.counts()
        assert all(counts[k-1] == 0 for k in (1, 3, 4, 5, 7, 8)), counts
        pages = []
        for i in range(256):
            page, hold = self.q(4096 + 8*i), self.q(6144 + 8*i)
            if page:
                assert hold and self.row(hold, 6)[3] == 1
                sid = self.row(hold, 6)[2]
                storage = self.row(sid, 2)
                assert storage[3] == storage[4] == 256*672 and storage[5] == storage[6] == 1
                assert page == (1 << 63) | storage[2]
                pages.append(hold)
            else:
                assert hold == 0
        assert counts[1] == counts[5] == len(pages)
        self.partition()
        return {'live_counts_8_kinds': counts, 'descriptor_pages': len(pages),
                'all_remaining_storage_and_holds_are_actual_descriptor_pages': True}
