"""Two separately observed allocation failures for explicit integral-to-F32 sum.

This module only defines controls. The native reader decides when to execute
them. No storage contents or numeric sum result are calculated by the controls.
"""
from pathlib import Path
import resource
import struct

from common import dump, require, sha


ELEMENTS = 65536 * 4
CAST_BYTES = ELEMENTS * 4
OUTPUT_BYTES = 65536 * 4


def _tensor_bytes(c, token):
    # A 1 MiB cast does not fit in one transport response once its header is
    # included. Read the actual native-owned bytes through the frozen observer;
    # do not create a spurious output-buffer OOM while observing the witness.
    memory = c.n.memory()
    return memory.tensor(memory.public_node(token))


def _inputs(c):
    return {name: _tensor_bytes(c, c.live[name]) for name in ('backing', 'input')}


def _rng(c, label):
    raw = c.n.one(63, c.gen)['payload']
    c.check(label + '/RNG-width', len(raw), 5056)
    c.check(label + '/complete5056-RNG', raw == c.rng, True)
    return raw


def _resources(c, stage, label, *, pool=None):
    state, endpoint, raw = c.n.capture_owner(label)
    next_node = state['header']['next']
    needed_pages = sorted({next_node >> 8, (next_node + 1) >> 8})
    actual_pages = sorted(p['page'] for p in state['pages'])
    require(next_node + 2 <= 65536 and set(needed_pages) <= set(actual_pages),
            'both-cast-and-result-descriptor-pages-preexist', next_node=next_node,
            needed_pages=needed_pages, actual_pages=actual_pages)
    require(state['header']['incarnation'] + 2 < 1 << 64,
            'two-new-descriptor-incarnations-fit')
    # Kinds 1, 2, 3 and 6 are Tensor/public capability, Storage, Version,
    # and temporary Hold. No Saved/AD allocation is required by this contract.
    available = {str(kind): sum(row[1] == 0 and row[0] < 0xffffffff
                               for row in state['tables'][str(kind)])
                 for kind in (1, 2, 3, 6)}
    require(min(available.values()) >= 4, 'all-T-S-V-Hold-and-public-slots-preexist',
            available=available)
    free_extents = state['free_extents']
    total = sum(length for _, length in free_extents)
    largest = max((length for _, length in free_extents), default=0)
    if pool is not None:
        c.check(label + '/pool-has-not-grown', state['header']['pool'], pool)
    if stage == 'cast-storage':
        require(16 * 1024 <= total <= 32 * 1024 and largest < CAST_BYTES,
                'cast-cannot-fit-in-any-existing-extent', total=total, largest=largest)
    else:
        require(CAST_BYTES + 32 * 1024 <= total <= CAST_BYTES + 64 * 1024
                and largest >= CAST_BYTES and total - CAST_BYTES < OUTPUT_BYTES,
                'cast-fits-but-cast-plus-output-cannot-fit', total=total, largest=largest)
    result = dict(stage=stage, owner=endpoint, owner_bytes=len(raw), pool=state['header']['pool'],
                  free_extents=free_extents, total_free_bytes=total, largest_free_extent=largest,
                  cast_requested_bytes=CAST_BYTES, output_requested_bytes=OUTPUT_BYTES,
                  free_after_cast_upper_bound=total - CAST_BYTES, next_node=next_node,
                  needed_descriptor_pages=needed_pages, present_descriptor_pages=actual_pages,
                  available_slots=available, public_T_slots_available=available['1'],
                  complete_owner_sha256=sha(raw))
    dump(c.directory / (label + '.json'), result)
    return state, result


def _fill(c, stage):
    # Stage B first grows a genuine pool using an actual F32 allocation and
    # releases it. Both dimensions obey the existing 262144-per-axis limit.
    if stage == 'result-storage-after-cast':
        token = c.n.one(38, 2, 0, 262144, 2, 0, 0, 0)['value']
        c.n.one(41, token)
    initial, _, _ = c.n.capture_owner('before-filling')
    pool = initial['header']['pool']
    lower, upper = ((16 * 1024, 32 * 1024) if stage == 'cast-storage'
                    else (CAST_BYTES + 32 * 1024, CAST_BYTES + 64 * 1024))
    fillers = []
    for _ in range(64):
        state, _, _ = c.n.capture_owner('fill-inspection')
        total = sum(length for _, length in state['free_extents'])
        if lower <= total <= upper:
            break
        require(total > upper, 'bounded-real-allocation-setup-free-space', total=total, target=upper)
        amount = min(262144, (total - upper) & ~15)
        require(amount > 0, 'positive-aligned-filler')
        fillers.append(c.n.create(1, [amount], bytes(amount), requires=0))
        after, _, _ = c.n.capture_owner('filled')
        c.check('filler-stays-in-existing-pool', after['header']['pool'], pool)
    else:
        raise AssertionError('bounded allocation setup did not reach target')
    return pool, fillers


def _mapping_failure(c, command, label):
    before = Path('/proc/%d/maps' % c.n.s.pid).read_bytes()
    status, record = c.issue(command, label)
    after = Path('/proc/%d/maps' % c.n.s.pid).read_bytes()
    (c.directory / (label + '.maps-before')).write_bytes(before)
    (c.directory / (label + '.maps-after')).write_bytes(after)
    c.check(label + '/actual-allocation-cause', status, 3)
    c.check(label + '/no-published-result', record, None)
    c.check(label + '/all-mappings-restored', after == before, True)
    _rng(c, label)


def _execute(c, stage):
    c.live['backing'] = c.n.create(1, [4], b'\0\1\0\0', requires=0)
    c.live['input'] = c.view(c.live['backing'], [65536, 4], [0, 1], 0)
    original_inputs = _inputs(c)
    dump(c.directory / 'inputs-before.json', original_inputs)
    pool, fillers = _fill(c, stage)
    state, ready = _resources(c, stage, 'ready-before-limit', pool=pool)
    sum_command = (115, c.live['input'], 1, c.n.w.Payload(struct.pack('<q', 1)), 0, 0, 0, 0)
    cast_command = (108, c.live['input'], 0, 1, 0, 0, 0, 0)
    pid = c.n.s.pid
    old_limit = resource.prlimit(pid, resource.RLIMIT_AS)
    raw_status = Path('/proc/%d/status' % pid).read_bytes()
    (c.directory / 'before-limit.status').write_bytes(raw_status)
    (c.directory / 'before-limit.limits').write_bytes(Path('/proc/%d/limits' % pid).read_bytes())
    vm = int(next(line.split()[1] for line in raw_status.decode().splitlines()
                  if line.startswith('VmSize:'))) * 1024
    page = lambda value: (value + 4095) & ~4095
    # Same-width transaction, a complete RNG query and small wire buffers fit;
    # a larger owner/pool needed by the denied Storage allocation does not.
    soft = vm + page(state['length'] + 16) + page(5056 + 16) + 65536
    require(old_limit[1] == resource.RLIM_INFINITY or soft < old_limit[1],
            'child-limit-fits-existing-hard-limit', soft=soft, hard=old_limit[1])
    dump(c.directory / 'allocation-setup.json', dict(ready=ready, source_logical_elements=ELEMENTS,
         filler_count=len(fillers), current_vm=vm, old_limit=list(old_limit), applied_soft_limit=soft,
         cast_command=list(cast_command), sum_request_sha256=sha(c.n.w.packet([sum_command])),
         witness_contract='command108 F32 copy=True preserve_format under the same RLIMIT as command115'))
    resource.prlimit(pid, resource.RLIMIT_AS, (soft, old_limit[1]))
    try:
        c.check('same-capacity-query-under-limit', c.n.one(94, c.live['input'])['value'], 0)
        _rng(c, 'ordinary-query-under-limit')
        _resources(c, stage, 'ready-for-cast-witness', pool=pool)
        if stage == 'cast-storage':
            _mapping_failure(c, cast_command, 'cast-only108-allocation-failure')
        else:
            status, record = c.issue(cast_command, 'cast-only108-under-same-limit')
            c.check('same-limit-cast-actually-succeeds', status, 0)
            require(record is not None and record['opcode'] == 108, 'cast-witness-real-record')
            c.live['cast_witness'] = record['value']
            cast = _tensor_bytes(c, record['value'])
            c.check('cast-witness-dtype', cast['dtype'], 'torch.float32')
            c.check('cast-witness-width', cast['element_size'], 4)
            c.check('cast-witness-requested-storage', cast['storage_bytes'], CAST_BYTES)
            c.check('cast-witness-shape', cast['shape'], [65536, 4])
            c.check('cast-witness-preserve-dense-strides', cast['strides'], [4, 1])
            c.check('cast-witness-no-grad', cast['requires_grad'], False)
            dump(c.directory / 'cast-witness-observed.json', cast)
            after_cast, endpoint, _ = c.n.capture_owner('cast-witness-live')
            c.check('cast-witness-no-pool-growth', after_cast['header']['pool'], pool)
            require(sum(n for _, n in after_cast['free_extents']) < OUTPUT_BYTES,
                    'actual-post-cast-free-pool-cannot-fit-output')
            dump(c.directory / 'cast-witness-live-resources.json',
                 dict(owner=endpoint, free_extents=after_cast['free_extents'],
                      total_free_bytes=sum(n for _, n in after_cast['free_extents']), pool=pool))
            c.drop('cast_witness')
        c.check('cast-witness-does-not-change-input-bytes-or-identities', _inputs(c) == original_inputs, True)
        _rng(c, 'after-cast-witness')
        # In stage B the real successful cast/drop advances incarnation and
        # slot generations. Recheck everything after that actual release.
        _resources(c, stage, 'ready-for-sum-after-witness', pool=pool)
        _mapping_failure(c, sum_command, 'sum115-' + stage + '-allocation-failure')
        c.check('failed-sum-input-bytes-and-identities-unchanged', _inputs(c) == original_inputs, True)
    finally:
        resource.prlimit(pid, resource.RLIMIT_AS, old_limit)
        c.check('child-original-resource-limit-restored', list(resource.prlimit(pid, resource.RLIMIT_AS)), list(old_limit))
    status, record = c.issue(sum_command, 'same-sum115-after-resource-recovery')
    c.check('same-command-recovers', status, 0)
    require(record is not None and record['opcode'] == 115, 'recovered-real-sum-record')
    c.live['result'] = record['value']
    result = c.n.tensor(record['value'])
    c.check('recovered-output-dtype', result['dtype'], 'torch.float32')
    c.check('recovered-output-width', result['element_size'], 4)
    c.check('recovered-output-requested-bytes', result['storage_bytes'], OUTPUT_BYTES)
    c.check('recovered-output-shape', result['shape'], [65536])
    c.check('recovered-output-no-grad', result['requires_grad'], False)
    after, _, _ = c.n.capture_owner('after-resource-recovery')
    c.check('actual-pool-growth-after-limit-restored', after['header']['pool'] > pool, True)
    c.check('successful-sum-input-bytes-and-identities-unchanged', _inputs(c) == original_inputs, True)
    _rng(c, 'after-resource-recovery')
    dump(c.directory / 'inputs-after.json', _inputs(c))
    for token in reversed(fillers):
        c.n.one(41, token)
    for role in list(c.live)[::-1]:
        c.drop(role)
    c.finish_controls()


def run_allocation_controls(args, observer, owner, run_one):
    result = []
    for stage in ('cast-storage', 'result-storage-after-cast'):
        def execute(c, stage=stage):
            _execute(c, stage)
        result.append(run_one(args, observer, owner, args.output / ('allocation-' + stage),
            'integral-F32-' + stage + '-OOM-full-rollback-and-recovery', execute))
    require(len(result) == 2, 'both-distinct-allocation-stages-executed')
    return result
