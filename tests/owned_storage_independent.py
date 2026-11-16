"""Independent logical reference for real interactive native storage objects.

No implementation authoring code, agent codec or native allocator is imported.
The model derives references from live Python records; native owns all target
allocation, lifetime transitions, copies, checks and responses.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import select
import struct
import subprocess
import time
import traceback

LIMIT = 1 << 20
sha = lambda b: hashlib.sha256(b).hexdigest()


def cmd(op, *values):
    assert len(values) <= 7
    return (op, *values, *([0] * (7-len(values))))


def packet(commands, slots=0, pool=0):
    return struct.pack('<8s3Q', b'HTOSOBJ1', len(commands), slots, pool) + b''.join(
        struct.pack('<8Q', *c) for c in commands)


class Client:
    def __init__(self, executable, root):
        self.p = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  bufsize=0, env={'PATH': ''})
        self.records = []
        self.requests = (root/'requests.bin').open('wb')
        self.responses = (root/'responses.bin').open('wb')

    def read(self, size):
        data = bytearray()
        end = time.monotonic()+20
        while len(data) < size:
            left = end-time.monotonic()
            assert left > 0 and select.select([self.p.stdout], [], [], left)[0], 'response timeout'
            block = os.read(self.p.stdout.fileno(), size-len(data))
            assert block, ('premature EOF', self.p.poll())
            data.extend(block)
        return bytes(data)

    def exchange(self, commands, status=0, slots=0, pool=0):
        body = packet(commands, slots, pool)
        request = b'HTOWNR01'+struct.pack('<Q', len(body))+body
        data = memoryview(request)
        while data:
            written = os.write(self.p.stdin.fileno(), data)
            assert written > 0
            data = data[written:]
        magic, length = struct.unpack('<8sQ', self.read(16))
        assert magic == b'HTOWNS01' and 32 <= length <= LIMIT
        response = self.read(length)
        assert self.p.poll() is None and not self.p.stdin.closed
        self.requests.write(struct.pack('<Q', len(request))+request)
        self.responses.write(struct.pack('<Q', len(response))+response)
        header, version, count, observed = struct.unpack_from('<8s3Q', response)
        self.records.append({'frame': len(self.records)+1, 'request_sha256': sha(request),
                             'response_sha256': sha(response), 'status': observed,
                             'response_before_stdin_eof': True})
        assert header == b'HTOSRES1' and version == 1 and observed == status, (observed, status, commands)
        if status:
            assert count == 0 and len(response) == 32
            return []
        assert count == len(commands)
        result, at = [], 32
        for command in commands:
            row = struct.unpack_from('<8Q', response, at)
            at += 64
            assert row[0] == command[0] and row[1] == 0
            payload = response[at:at+row[7]]
            assert len(payload) == row[7]
            at += row[7]
            result.append((tuple(row[2:7]), payload))
        assert at == len(response)
        return result

    def one(self, op, *values):
        return self.exchange([cmd(op, *values)])[0][0]

    def dump(self):
        return self.exchange([cmd(18)])[0][1]

    def finish(self):
        self.p.stdin.close()
        assert self.p.wait(timeout=10) == 0
        assert self.p.stdout.read() == self.p.stderr.read() == b''

    def stop(self):
        self.requests.close()
        self.responses.close()
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait(timeout=10)


class Reference:
    def __init__(self, client):
        self.c = client
        self.t = {}
        self.s = {}
        self.v = {}
        self.w = {}
        self.saved = {}
        self.hold = {}
        self.generations = {}
        self.issued = set()
        self.retired = []
        self.comparisons = 0
        self.rollback_checks = 0

    def compare(self, actual, expected):
        assert actual == expected, (actual, expected)
        self.comparisons += len(expected) if isinstance(expected, tuple) else 1

    def new_id(self, handle, kind):
        assert handle not in self.issued
        generation, actual_kind, slot = handle >> 32, (handle >> 24) & 255, handle & 0xffffff
        assert actual_kind == kind and slot >= 1
        assert generation == self.generations.get((kind, slot), 0)+1
        self.generations[kind, slot] = generation
        self.issued.add(handle)
        return handle

    def strong(self, sid):
        return (sum(t[0] == sid for t in self.t.values())
                + sum(s[0] == sid for s in self.saved.values())
                + sum(value == sid for value in self.hold.values()))

    def vrefs(self, vid):
        return sum(t[1] == vid for t in self.t.values())+sum(s[1] == vid for s in self.saved.values())

    def weak(self, sid):
        return sum(value == sid for value in self.w.values())

    def collect(self):
        for vid in list(self.v):
            if not self.vrefs(vid):
                del self.v[vid]
                self.retired.append(vid)
        for sid in list(self.s):
            if not self.strong(sid):
                self.s[sid]['data'] = None
                if not self.weak(sid):
                    del self.s[sid]
                    self.retired.append(sid)

    def create(self, size, fill):
        tid, sid, vid, extent, length = self.c.one(1, size, fill)
        self.compare(length, size)
        assert extent % 16 == 0
        for handle, kind in [(tid, 1), (sid, 2), (vid, 3)]:
            self.new_id(handle, kind)
        self.t[tid] = (sid, vid, 0, size)
        self.s[sid] = {'data': bytearray([fill])*size, 'extent': extent, 'size': size}
        self.v[vid] = 0
        self.metadata(vid)
        return tid

    def alias(self, parent, offset, length, fresh=False):
        ps, pv, po, pl = self.t[parent]
        tid, sid, vid, origin, size = self.c.one(19 if fresh else 2, parent, offset, length)
        self.new_id(tid, 1)
        if fresh:
            self.new_id(vid, 3)
            self.v[vid] = 0
        else:
            self.compare(vid, pv)
        self.compare((sid, origin, size), (ps, po+offset, length))
        self.t[tid] = (sid, vid, origin, size)
        return tid

    def write(self, tid, offset, value):
        sid, vid, origin, length = self.t[tid]
        result = self.c.one(11, tid, offset, value)
        self.s[sid]['data'][origin+offset] = value
        self.v[vid] += 1
        self.compare(result, (self.v[vid], 0, 0, 0, 0))

    def read_tensor(self, tid, offset):
        sid, _, origin, _ = self.t[tid]
        self.compare(self.c.one(12, tid, offset), (self.s[sid]['data'][origin+offset], 0, 0, 0, 0))

    def make_weak(self, tid):
        wid, sid, a, b, c = self.c.one(4, tid)
        self.new_id(wid, 4)
        self.compare((sid, a, b, c), (self.t[tid][0], 0, 0, 0))
        self.w[wid] = sid

    def lock(self, wid):
        sid = self.w[wid]
        if not self.strong(sid):
            self.c.exchange([cmd(5, wid)], status=6)
            return
        hold, observed, a, b, c = self.c.one(5, wid)
        self.new_id(hold, 6)
        self.compare((observed, a, b, c), (sid, 0, 0, 0))
        self.hold[hold] = sid

    def save(self, tid):
        sid, vid, origin, length = self.t[tid]
        saved, observed_s, observed_v, version, reserved = self.c.one(8, tid)
        self.new_id(saved, 5)
        self.compare((observed_s, observed_v, version, reserved), (sid, vid, self.v[vid], 0))
        self.saved[saved] = (sid, vid, version, origin, length)

    def check_saved(self, saved):
        sid, vid, snapshot, origin, length = self.saved[saved]
        if snapshot != self.v[vid]:
            self.c.exchange([cmd(9, saved)], status=5)
            if length:
                self.c.exchange([cmd(14, saved, length-1)], status=5)
        else:
            self.compare(self.c.one(9, saved), (snapshot, snapshot, 0, 0, 0))
            if length:
                self.compare(self.c.one(14, saved, length-1), (self.s[sid]['data'][origin+length-1], 0, 0, 0, 0))

    def drop(self, kind, handle):
        mapping, opcode = {1: (self.t, 3), 4: (self.w, 6), 5: (self.saved, 10), 6: (self.hold, 7)}[kind]
        self.compare(self.c.one(opcode, handle), (0, 0, 0, 0, 0))
        del mapping[handle]
        self.retired.append(handle)
        self.collect()

    def metadata(self, handle):
        actual = self.c.one(15, handle)
        kind = (handle >> 24) & 255
        if kind == 1:
            sid, vid, origin, size = self.t[handle]
            self.compare(actual, (sid, vid, origin, size, self.v[vid]))
        elif kind == 2:
            item = self.s[handle]
            strong, weak = self.strong(handle), self.weak(handle)
            if strong:
                self.compare(actual, (item['extent'], item['size'], strong, weak+1, 0))
            else:
                self.compare(actual[2:], (0, weak, 1))
        elif kind == 3:
            self.compare(actual, (self.v[handle], self.vrefs(handle), 0, 0, 0))
        elif kind == 5:
            self.compare(actual, self.saved[handle])
        else:
            sid = (self.w if kind == 4 else self.hold)[handle]
            self.compare(actual, (sid, 0, 0, 0, 0))

    def check(self, rng):
        live = list(self.t)+list(self.s)+list(self.v)+list(self.w)+list(self.saved)+list(self.hold)
        for handle in rng.sample(live, min(6, len(live))):
            self.metadata(handle)
        for tid in rng.sample(list(self.t), min(3, len(self.t))):
            if self.t[tid][3]:
                self.read_tensor(tid, rng.randrange(self.t[tid][3]))
        for hold in rng.sample(list(self.hold), min(2, len(self.hold))):
            data = self.s[self.hold[hold]]['data']
            if data:
                index = rng.randrange(len(data))
                self.compare(self.c.one(13, hold, index), (data[index], 0, 0, 0, 0))
        slots, pool, extents, free, epoch = self.c.one(16)
        used = sum(((x['size']+15)//16)*16 for sid, x in self.s.items() if self.strong(sid))
        self.compare(free, pool-used)
        assert slots >= 4 and slots & (slots-1) == 0 and pool >= 16 and pool % 16 == 0

    def rollback(self, rng):
        original = self.c.dump()
        commands = [cmd(1, 17, 0xE3)]
        nonempty = [tid for tid, value in self.t.items() if value[3]]
        if nonempty:
            commands.append(cmd(11, rng.choice(nonempty), 0, 0xD7))
        commands.append(cmd(99))
        self.c.exchange(commands, status=7)
        assert self.c.dump() == original, 'failed trial published data, counters or allocation metadata'
        self.rollback_checks += 1
        if self.retired:
            self.c.exchange([cmd(15, rng.choice(self.retired))], status=1)
            assert self.c.dump() == original
            self.rollback_checks += 1


def campaign(executable, output, seed, iterations):
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    client = Client(executable, output)
    model = Reference(client)
    result = {'seed': seed, 'iterations': iterations, 'status': 'FAIL'}
    try:
        client.exchange([], slots=4, pool=16)
        for step in range(iterations):
            choice = rng.randrange(100)
            if len(model.t) < 3 or (choice < 20 and len(model.t) < 24):
                model.create(rng.choice([0, 1, 3, 15, 16, 17, 31, 33, 47]), rng.randrange(256))
            elif choice < 36 and len(model.t) < 36:
                tid = rng.choice(list(model.t)); length = model.t[tid][3]
                offset = rng.randrange(length+1)
                model.alias(tid, offset, rng.randrange(length-offset+1), fresh=bool(choice & 1))
            elif choice < 48:
                nonempty = [tid for tid, value in model.t.items() if value[3]]
                if nonempty:
                    tid = rng.choice(nonempty)
                    model.write(tid, rng.randrange(model.t[tid][3]), rng.randrange(256))
            elif choice < 56 and len(model.w) < 20:
                model.make_weak(rng.choice(list(model.t)))
            elif choice < 64 and model.w and len(model.hold) < 20:
                model.lock(rng.choice(list(model.w)))
            elif choice < 73 and len(model.saved) < 20:
                model.save(rng.choice(list(model.t)))
            elif choice < 80 and model.saved:
                model.check_saved(rng.choice(list(model.saved)))
            elif choice < 96:
                available = [(kind, handle) for kind, mapping in [(1, model.t), (4, model.w), (5, model.saved), (6, model.hold)] for handle in mapping]
                if available:
                    model.drop(*rng.choice(available))
            else:
                wanted_slots = rng.choice([4, 8, 16, 32, 64])
                wanted_pool = rng.choice([16, 33, 65, 129, 257, 513, 1025, 2049])
                observed = client.one(17, wanted_slots, wanted_pool)
                assert observed[0] >= wanted_slots and observed[1] >= wanted_pool
            if step % 7 == 0:
                model.check(rng)
            if step % 19 == 0:
                model.rollback(rng)
            if step % 31 == 0:
                client.one(20)
        for kind, mapping in [(1, model.t), (5, model.saved), (6, model.hold)]:
            for handle in list(mapping):
                model.drop(kind, handle)
        for wid in list(model.w):
            model.lock(wid)
            model.drop(4, wid)
        assert not model.t and not model.s and not model.v and not model.w and not model.saved and not model.hold
        slots, pool, extents, free, epoch = client.one(16)
        model.compare((extents, free), (1, pool))
        client.one(20)
        for old in rng.sample(list(model.issued), min(30, len(model.issued))):
            client.exchange([cmd(15, old)], status=1)
        client.finish()
        result.update(status='PASS', exact_scalar_comparisons=model.comparisons,
                      full_state_rollback_comparisons=model.rollback_checks,
                      issued_unique_handles=len(model.issued), all_references_released=True,
                      final_capacity={'slots': slots, 'pool': pool, 'resize_epoch': epoch})
    except Exception as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        client.stop()
    result.update(interactive_frames=len(client.records), all_observed_before_stdin_eof=True,
                  request_transcript_sha256=sha((output/'requests.bin').read_bytes()),
                  response_transcript_sha256=sha((output/'responses.bin').read_bytes()), rows=client.records)
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1201, 5527, 9031])
    args = parser.parse_args()
    executable = args.executable.resolve()
    original = executable.read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [campaign(executable, args.output/str(seed), seed, args.iterations) for seed in args.seeds]
    passed = all(r['status'] == 'PASS' for r in rows) and executable.read_bytes() == original
    report = {'schema': 'heptatorch.independent-owned-storage-model-review.v1',
              'status': 'PASS' if passed else 'FAIL', 'executable_sha256': sha(original),
              'reviewer_sha256': sha(Path(__file__).read_bytes()), 'native_unchanged': executable.read_bytes() == original,
              'campaigns': len(rows), 'interactive_frames': sum(r['interactive_frames'] for r in rows),
              'host_role': 'Independent logical reference and wire transport only; no production allocation or state execution.',
              'rows': [{k: v for k, v in r.items() if k != 'rows'} for r in rows]}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)
