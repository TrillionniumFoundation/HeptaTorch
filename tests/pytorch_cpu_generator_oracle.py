"""Independent actual-PyTorch comparison for owned native CPU Generator objects.

Only the wire transport and reference live in Python. The native process owns
every target Generator, mutates its MT state and computes every target value.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import runpy
import select
import struct
import subprocess
import sys
import time
import traceback

MASK = (1 << 64)-1
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
SEEDS = [0, 1, 17, 5489, (1 << 31)-1, (1 << 32)-1, 1 << 32,
         (1 << 32)+17, (1 << 63)-1, 1 << 63, MASK-1, MASK]
EXTRA = ['default-construction', 'six-interleaved-generators', 'clone-across-twist',
         'state-export-restore', 'float-normal-cache-clones', 'double-normal-cache-clones',
         'reseed-clears-both-caches', 'native-system-seed-roundtrip',
         'release-compaction-preserves-survivors', 'zero-count-does-not-advance',
         'sixty-four-independent-generators', 'long-cloned-trajectory']
PLAN = ['mixed-seed-'+str(seed) for seed in SEEDS]+EXTRA
assert len(PLAN) == len(set(PLAN)) == 24
sha = lambda raw: hashlib.sha256(raw).hexdigest()


class Native:
    def __init__(self, executable):
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': ''})
        self.frames = 0
        self.request_hash, self.response_hash = hashlib.sha256(), hashlib.sha256()
        self.epoch = self.last = self.live = 0

    def read_exact(self, count):
        deadline, result = time.monotonic()+60, bytearray()
        while len(result) < count:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError(('native response', count, len(result), self.process.poll()))
            block = os.read(self.process.stdout.fileno(), count-len(result))
            assert block, ('native EOF', self.process.poll(), count, len(result))
            result.extend(block)
        return bytes(result)

    def call(self, op, handle=0, count=0, a=0, b=0, payload=b'', status=0):
        body = b'HTCGEN01'+struct.pack('<7Q', op, handle, count, a & MASK, b & MASK, len(payload), 0)+payload
        request = b'HTOWNR01'+struct.pack('<Q', len(body))+body
        self.process.stdin.write(request); self.process.stdin.flush()
        prefix = self.read_exact(16)
        assert prefix[:8] == b'HTOWNS01'
        length = struct.unpack_from('<Q', prefix, 8)[0]
        assert 64 <= length <= 1 << 20
        response = self.read_exact(length)
        assert response[:8] == b'HTCGRS01'
        code, size, live, allocation, last, epoch, reserved = struct.unpack_from('<7Q', response, 8)
        assert code == status and reserved == 0 and size == length-64, (op, code, status, length, size)
        assert self.process.poll() is None, 'response must precede process EOF'
        self.frames += 1; self.request_hash.update(request); self.response_hash.update(prefix+response)
        mutation = op in (1, 2, 3, 6, 7, 16, 19) or (8 <= op <= 15 and count != 0)
        assert epoch == self.epoch + (1 if status == 0 and mutation else 0), (op, epoch, self.epoch)
        if status:
            assert (live, last) == (self.live, self.last) and not size
        elif op in (1, 7, 19):
            assert (live, last) == (self.live+count, self.last+count)
            assert list(struct.unpack('<'+'Q'*count, response[64:])) == list(range(self.last+1, self.last+count+1))
        elif op == 16:
            assert (live, last) == (self.live-1, self.last)
        else:
            assert (live, last) == (self.live, self.last)
        assert allocation == (0 if last == 0 else 64+5088*live)
        self.epoch, self.last, self.live = epoch, last, live
        return response[64:]

    def words(self, *args, **kwargs):
        raw = self.call(*args, **kwargs)
        assert len(raw) % 8 == 0
        return list(struct.unpack('<'+'Q'*(len(raw)//8), raw))

    def close(self, failed=False):
        if failed:
            self.process.kill(); self.process.communicate(); return
        self.process.stdin.close(); self.process.stdin = None
        stdout, stderr = self.process.communicate(timeout=30)
        assert self.process.returncode == 0 and not stdout and not stderr, (self.process.returncode, stdout, stderr)


class Comparison:
    def __init__(self, native, torch):
        self.native, self.torch, self.generators = native, torch, {}
        self.exact_values = self.state_bytes = self.raw32_projection_values = 0
        self.reference_values_hash, self.reference_states_hash = hashlib.sha256(), hashlib.sha256()
        self.nondeterministic_seed_observations = []

    def state(self, handle):
        expected = bytes(self.generators[handle].get_state().tolist())
        actual = self.native.call(5, handle)
        assert len(expected) == len(actual) == 5056 and actual == expected, (handle, 'full serialized state')
        self.state_bytes += 5056; self.reference_states_hash.update(expected)
        assert self.native.words(4, handle) == [self.generators[handle].initial_seed()]
        return expected

    def create(self, seed=0, count=1, default=False):
        handles = self.native.words(19 if default else 1, count=count, a=0 if default else seed)
        for handle in handles:
            generator = self.torch.Generator(device='cpu')
            if not default:
                generator.manual_seed(seed)
            self.generators[handle] = generator
            self.state(handle)
        return handles

    def clone(self, handle, count=1):
        handles = self.native.words(7, handle, count)
        for new in handles:
            self.generators[new] = self.generators[handle].clone_state()
            self.state(new)
        self.state(handle)
        return handles

    def restore(self, handle, state):
        self.native.call(6, handle, payload=state)
        self.generators[handle].set_state(self.torch.frombuffer(bytearray(state), dtype=self.torch.uint8).clone())
        self.state(handle)

    def reseed(self, handle, seed):
        self.native.call(2, handle, a=seed)
        self.generators[handle].manual_seed(seed)
        self.state(handle)

    def draw(self, handle, op, count, low=0, high=0):
        t, generator = self.torch, self.generators[handle]
        before = bytes(generator.get_state().tolist())
        actual = self.native.words(op, handle, count, low, high)
        if op in (8, 9):
            # Import the raw engine into CPython, independently of HIA twist code.
            index = struct.unpack_from('<Q', before, 16)[0]
            words = struct.unpack_from('<624Q', before, 24)
            engine = random.Random()
            engine.setstate((3, tuple(words)+(624 if index == 0 else index,), None))
            raw = [engine.getrandbits(32) for _ in range(count)]
            projection = t.empty(count, dtype=t.int32).random_(generator=generator).tolist()
            assert projection == [value & 0x7fffffff for value in raw]
            after = bytes(generator.get_state().tolist())
            assert engine.getstate()[1] == tuple(struct.unpack_from('<624Q', after, 24))+(struct.unpack_from('<Q', after, 16)[0] or 624,)
            self.raw32_projection_values += count
            expected = raw if op == 9 else []
        elif op == 10:
            values = t.empty(count, dtype=t.int64).random_(-(1 << 63), None, generator=generator)
            expected = [value & MASK for value in values.tolist()]
        elif op in (11, 12, 13, 14):
            dtype, view, mask = {11:(t.float32,t.int32,(1 << 32)-1),
                                 12:(t.float64,t.int64,MASK),
                                 13:(t.float16,t.int16,65535),
                                 14:(t.bfloat16,t.int16,65535)}[op]
            values = t.rand(count, dtype=dtype, generator=generator)
            expected = [value & mask for value in values.view(view).tolist()]
        else:
            assert op == 15
            expected = [value & MASK for value in t.randint(low, high, (count,), dtype=t.int64, generator=generator).tolist()]
        assert actual == expected, (op, handle, count, [(i, hex(a), hex(b)) for i,(a,b) in enumerate(zip(actual,expected)) if a != b][:16])
        self.exact_values += len(expected)
        self.reference_values_hash.update(b''.join(struct.pack('<Q', value) for value in expected))
        self.state(handle)
        return actual

    def release(self, handle):
        self.native.call(16, handle); del self.generators[handle]
        self.native.call(4, handle, status=4)
        for other in self.generators:
            self.state(other)


def mixed(c, seed):
    a,b = c.create(seed, 2)
    for handle,op,count,low,high in [(a,11,17,0,0),(b,12,19,0,0),(a,13,113,0,0),
                                    (b,14,127,0,0),(a,15,31,-9999,10001),
                                    (b,15,37,-17,(1 << 28)-17),(a,10,23,0,0),
                                    (b,8,623,0,0),(b,9,17,0,0)]:
        c.draw(handle,op,count,low,high)
    clone = c.clone(a)[0]
    assert c.draw(a,12,43) == c.draw(clone,12,43)
    c.reseed(a, seed ^ MASK)
    c.draw(a,11,29); c.draw(clone,15,31,-(1 << 63),(1 << 63)-1)


def extra(c, name):
    t = c.torch
    if name == 'default-construction':
        a,b = c.create(count=2, default=True)
        assert c.generators[a].initial_seed() == 67280421310721
        assert c.draw(a,9,625) == c.draw(b,9,625)
    elif name == 'six-interleaved-generators':
        handles = [c.create(seed)[0] for seed in [0,17,5489,1 << 32,1 << 63,MASK]]
        for turn in range(12):
            for index,handle in enumerate(reversed(handles)):
                c.draw(handle, 11+(index+turn)%4, 1+(index*13+turn*17)%67)
    elif name == 'clone-across-twist':
        a = c.create(5489)[0]; c.draw(a,8,623)
        clones = c.clone(a,3)
        for handle in [a]+clones: c.draw(handle,10,2); c.draw(handle,9,627)
        states = [c.state(handle) for handle in [a]+clones]; assert len(set(states)) == 1
        c.draw(clones[1],12,127)
        assert c.state(a) == c.state(clones[0]) != c.state(clones[1])
    elif name == 'state-export-restore':
        a,b = c.create(17,2); c.draw(a,12,397); saved = c.state(a)
        c.draw(a,13,999); c.restore(b,saved); first = c.draw(b,11,1000)
        c.restore(a,saved); assert c.draw(a,11,1000) == first
    elif name in ('float-normal-cache-clones','double-normal-cache-clones','reseed-clears-both-caches'):
        handle = c.create(20261116)[0]
        dtypes = [t.float32] if name.startswith('float-') else [t.float64] if name.startswith('double-') else [t.float32,t.float64]
        for dtype in dtypes: t.randn(1,dtype=dtype,generator=c.generators[handle])
        c.restore(handle, bytes(c.generators[handle].get_state().tolist()))
        clones = c.clone(handle,2)
        for value in [handle]+clones: c.draw(value,11,43); c.draw(value,12,29)
        assert len({c.state(value) for value in [handle]+clones}) == 1
        for index,value in enumerate([handle]+clones): c.reseed(value, MASK-index); c.draw(value,14,139)
    elif name == 'native-system-seed-roundtrip':
        handle = c.create(0)[0]
        for turn in range(4):
            seed = c.native.words(3,handle)[0]
            reference_seed = c.generators[handle].seed()
            assert c.generators[handle].initial_seed() == reference_seed
            c.generators[handle].manual_seed(seed)
            c.nondeterministic_seed_observations.append({'native_returned_seed':seed,'reference_independent_seed':reference_seed})
            c.state(handle); c.draw(handle,11,17+turn)
    elif name == 'release-compaction-preserves-survivors':
        handles = [c.create(seed)[0] for seed in [1,2,3,4,5]]
        for index,handle in enumerate(handles): c.draw(handle,12,225+index)
        c.release(handles[1]); c.release(handles[-1]); c.release(handles[0])
        new = c.create(17)[0]; assert new > max(handles)
        for handle in list(c.generators): c.draw(handle,15,103,-(1 << 31),1 << 31)
        for handle in list(c.generators): c.release(handle)
        c.draw(c.create(default=True)[0],11,31)
    elif name == 'zero-count-does-not-advance':
        handle = c.create(MASK)[0]; before = c.state(handle); epoch = c.native.epoch
        for op in range(8,16): c.draw(handle,op,0,-3 if op==15 else 0,7 if op==15 else 0)
        assert c.native.epoch == epoch and c.state(handle) == before
    elif name == 'sixty-four-independent-generators':
        handles = c.create(5489,64)
        for index,handle in enumerate(handles): c.draw(handle,8,index*17); c.draw(handle,10,2)
        for handle in handles: c.state(handle)
    elif name == 'long-cloned-trajectory':
        original = c.create(17)[0]; c.draw(original,9,627); handles = [original]+c.clone(original,3)
        for turn in range(128):
            handle = handles[(turn*3)%4]
            op = 11+turn%5
            c.draw(handle,op,(turn*31)%97, -(1 << 62) if op==15 else 0, (1 << 62)+1 if op==15 else 0)
    else:
        raise AssertionError(name)


def emit(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report, indent=2, allow_nan=False)+'\n'
    path.write_text(raw); print(raw, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--transport-smoke', action='store_true')
    args = parser.parse_args()
    args.native, args.repo, args.output = args.native.resolve(), args.repo.resolve(), args.output.resolve()
    paths = {'native':args.native,'oracle':Path(__file__).resolve(),
             'reference_support':args.repo/'tests/pytorch_oracle.py',
             'lock':args.repo/'compatibility/pytorch-oracle-lock.json'}
    before = {name:sha(path.read_bytes()) for name,path in paths.items()}
    assert before['native'] == args.expected_native_sha256
    assert before['reference_support'] == SUPPORT_SHA and before['lock'] == LOCK_SHA
    base = {'schema':'heptatorch.live-pytorch-owned-cpu-generator.v1',
            'native_executable_sha256':before['native'],'source_custody':before,'case_plan':PLAN,
            'expected_cases':24,'pytorch_api_denominator':2590,'full_rng_gate_closed':False,
            'whole_pytorch_rewrite_complete':False}
    if args.prepare_only:
        emit(args.output,dict(base,status='NOT_RUN',cases=0,reason='Source pins and independent 24-case plan prepared; no native or Torch case executed.')); return 0
    if args.transport_smoke:
        n = Native(args.native)
        try:
            handle = n.words(1,count=1,a=5489)[0]
            assert n.words(9,handle,10) == [3499211612,581869302,3890346734,3586334585,545404204,4161255391,3922919429,949333985,2715962298,1323567403]
            state = n.call(5,handle); assert len(state)==5056
            clone = n.words(7,handle,1)[0]; assert n.call(5,clone)==state
            n.call(16,handle); assert n.call(5,clone)==state
            n.close()
        except Exception:
            n.close(True); raise
        emit(args.output,dict(base,status='NATIVE_SMOKE_ONLY',cases=1,frames=n.frames,
                              exact_raw32_values=10,pytorch_executed=False)); return 0
    provenance_path = args.output.with_name(args.output.stem+'-provenance.json')
    previous_argv,previous_cwd = sys.argv,Path.cwd()
    try:
        os.chdir(args.repo); sys.argv=[str(paths['reference_support']),str(args.native),str(provenance_path),str(paths['lock'])]
        support = runpy.run_path(str(paths['reference_support']),run_name='owned_cpu_generator_reference_support')
    except (SystemExit, ModuleNotFoundError) as error:
        emit(args.output,dict(base,status='NOT_RUN',cases=0,reason=str(error),
                              reference_initialization=json.loads(provenance_path.read_text()) if provenance_path.exists() else None)); return 2
    finally:
        sys.argv=previous_argv; os.chdir(previous_cwd)
    provenance = support['PROVENANCE']; assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
    rows=[]
    for name in PLAN:
        n=Native(args.native); c=Comparison(n,support['torch'])
        row={'case':name,'status':'PASS'}
        try:
            if name.startswith('mixed-seed-'): mixed(c,int(name.removeprefix('mixed-seed-')))
            else: extra(c,name)
            n.close()
        except Exception:
            row.update(status='FAIL',error=traceback.format_exc());n.close(True)
        row.update(frames=n.frames,exact_values=c.exact_values,serialized_state_bytes_compared=c.state_bytes,
                   raw32_reference_projection_values=c.raw32_projection_values,
                   request_stream_sha256=n.request_hash.hexdigest(),response_stream_sha256=n.response_hash.hexdigest(),
                   reference_values_sha256=c.reference_values_hash.hexdigest(),reference_states_sha256=c.reference_states_hash.hexdigest(),
                   nondeterministic_seed_observations=c.nondeterministic_seed_observations)
        rows.append(row)
    unchanged={name:sha(path.read_bytes())==before[name] for name,path in paths.items()}
    passed=all(row['status']=='PASS' for row in rows) and all(unchanged.values())
    report=dict(base,status='PASS' if passed else 'FAIL',cases=len(rows),failed_cases=sum(row['status']=='FAIL' for row in rows),
                oracle_provenance=provenance,rows=rows,source_files_unchanged_during_run=unchanged,
                frames=sum(row['frames'] for row in rows),exact_values=sum(row['exact_values'] for row in rows),
                serialized_state_bytes_compared=sum(row['serialized_state_bytes_compared'] for row in rows),
                raw32_reference_boundary='Full CPython MT19937 words plus actual Torch int32 low31 projection and full state. No host RNG in target.',
                nondeterministic_seed_boundary='Native and Torch independently call seed(); target seed is then passed to reference manual_seed for complete state and value comparison.',
                scope='Canonical CPU Generator states and specified uniform/integer draws; no normal target generation, Tensor filling, default global generator, concurrent access or accelerator RNG.')
    emit(args.output,report); return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
