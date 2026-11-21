"""Literal inputs and operations; expected bytes come only from real Torch."""
import hashlib
import itertools
import json
import math

WORDS = (0, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000,
         0x7ff800123456789a, 0xfff00123456789ab, 1, 0x7fefffffffffffff,
         0x7ff0000000000000, 0xfff0000000000000, 0x0010000000000000,
         0x000fffffffffffff, 0x8000000000000001, 0x7ff0000000000001,
         0xfff8000000000081, 0x4008000000000000)
WRITE_WORDS = (0x7ff8007654321098, 0x8000000000000000, 0xfff0000000000042)
MASK = (1 << 64) - 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def base(shape=(64,), words=WORDS):
    return dict(shape=list(shape), words=list(words))


def raw_input(spec):
    count = math.prod(spec['shape'])
    words = spec['words']
    assert words and all(type(value) is int and 0 <= value <= MASK for value in words)
    return b''.join(words[index % len(words)].to_bytes(8, 'little') for index in range(count))


def view(name, source, shape, strides, offset):
    return dict(operation='view', name=name, source=source, shape=list(shape),
                strides=list(strides), offset=offset)


def write(name, index, word):
    return dict(operation='write', name=name, index=index, word=word)


def release(name):
    return dict(operation='release', name=name)


def ordinary(name, shape, strides, offset, *, root=None, child=None, category='geometry'):
    root = root or base()
    operations = [view('view', 'root', shape, strides, offset)]
    target, target_shape = 'view', shape
    if child is not None:
        operations.append(view('child', 'view', child['shape'], child['strides'], child['offset']))
        target, target_shape = 'child', child['shape']
    if math.prod(target_shape):
        operations.extend((write(target, 0, WRITE_WORDS[0]),
                           write(target, math.prod(target_shape) - 1, WRITE_WORDS[1])))
    if math.prod(root['shape']):
        operations.append(write('root', 0, WRITE_WORDS[2]))
    operations.append(release('root'))
    if child is not None:
        operations.append(release('view'))
    operations.append(release(target))
    return dict(name=name, category=category, base=root, operations=operations)


def recipes():
    # Geometries from the earlier isolated probe are independently observed anew.
    for name, shape, strides, offset in (
        ('scalar-offset', [], [], 3), ('dense-offset', [2, 3], [3, 1], 2),
        ('transpose', [3, 2], [1, 3], 3), ('gapped', [3, 4], [8, 2], 3),
        ('overlap', [2, 3], [1, 1], 2), ('expanded', [3, 4], [0, 1], 2),
        ('rank3', [2, 3, 4], [12, 1, 3], 2),
        ('rank4', [2, 3, 2, 4], [24, 1, 12, 3], 2),
        ('singleton-rank4', [1, 1, 1, 1], [0, 7, 0, 3], 2),
        ('empty', [0, 3], [3, 1], 0), ('empty-beyond-storage', [0, 3], [3, 1], 68)):
        yield ordinary('probe-geometry/' + name, shape, strides, offset)
    for name, strides in (('nested', [4, 1]), ('nested-gapped', [8, 1])):
        yield ordinary('probe-geometry/' + name, [4, 4], strides, 3,
                       child=dict(shape=[2, 2], strides=[1, 8], offset=4))
    for source_rank, result_rank in itertools.product(range(5), repeat=2):
        strides = [0 if axis % 2 == 0 else axis + 3 for axis in range(result_rank)]
        yield ordinary('unit-ranks/%d-%d' % (source_rank, result_rank), [1] * result_rank,
                       strides, 0, root=base([1] * source_rank), category='all25-unit-rank-pairs')
    for rank in range(1, 5):
        shape = [2] * rank
        dense = [2 ** axis for axis in range(rank - 1, -1, -1)]
        candidates = [('dense', dense), ('permuted', list(reversed(dense))),
                      ('gapped', [stride * 2 for stride in dense]),
                      ('broadcast', [0] + dense[1:]), ('overlap', [1] * rank), ('zero', [0] * rank)]
        seen = set()
        for name, strides in candidates:
            if tuple(strides) in seen:
                continue
            seen.add(tuple(strides))
            for offset in (0, 3):
                yield ordinary('layout/r%d-%s-offset%d' % (rank, name, offset), shape,
                               strides, offset, category='layout-and-offset')
        for axis in range(rank):
            shape = [2] * rank
            shape[axis] = 0
            for offset in (0, 64, 68):
                yield ordinary('empty/r%d-axis%d-offset%d' % (rank, axis, offset), shape,
                               [3] * rank, offset, category='empty-view')
    for source_rank, result_rank, offset in itertools.product(range(1, 5), range(1, 5), (0, 7)):
        yield ordinary('empty-storage/r%d-r%d-offset%d' % (source_rank, result_rank, offset),
                       [2] * (result_rank - 1) + [0], [0] * result_rank, offset,
                       root=base([0] + [2] * (source_rank - 1)), category='empty-storage')
    for parent_rank, child_rank in itertools.product(range(5), repeat=2):
        operations = [view('parent', 'root', [1] * parent_rank, [3] * parent_rank, 8),
                      release('root'),
                      view('child', 'parent', [1] * child_rank, [0] * child_rank, 3),
                      write('child', 0, WRITE_WORDS[0]), release('parent'),
                      write('child', 0, WRITE_WORDS[1]), release('child')]
        yield dict(name='nested-ranks/%d-%d' % (parent_rank, child_rank),
                   category='all25-nested-unit-rank-pairs', base=base(), operations=operations)
    for name, parent, child in (
        ('outside-parent-logical-extent', ([1], [1], 10), ([3], [2], 0)),
        ('nonempty-from-empty-parent', ([0], [1], 68), ([2, 2], [3, 1], 1)),
        ('empty-from-scalar-parent', ([], [], 4), ([0, 2], [9, 1], 68))):
        yield ordinary('nested/' + name, *parent,
                       child=dict(shape=child[0], strides=child[1], offset=child[2]),
                       category='nested-storage-extent')
    for order in (('root', 'left', 'right'), ('left', 'root', 'right'), ('right', 'root', 'left')):
        yield dict(name='siblings/release-' + '-'.join(order), category='sibling-release-order', base=base(),
            operations=[view('left', 'root', [3, 2], [1, 3], 2),
                        view('right', 'root', [2, 3], [0, 2], 1),
                        write('left', 0, WRITE_WORDS[0]), write('right', 5, WRITE_WORDS[1]),
                        *[release(name) for name in order]])
    yield dict(name='nested/depth-three', category='nested-depth-three', base=base(), operations=[
        view('first', 'root', [4, 4], [8, 1], 3), release('root'),
        view('second', 'first', [2, 2], [1, 8], 4), release('first'),
        view('third', 'second', [], [], 17), write('third', 0, WRITE_WORDS[0]),
        release('second'), write('third', 0, WRITE_WORDS[1]), release('third')])
    # Exact raw values and geometry from the three immutable original requests.
    for label, words, packet in (
        ('unsupported-F64-view-111', [0x3ff0000000000000, 0x8000000000000000], False),
        ('same-packet-F64-public-alias-view', [0x3ff0000000000000, 0x4000000000000000], True),
        ('public-f64-as-strided', [0, 0x8000000000000000], False)):
        row = ordinary('legacy/' + label, [2], [1], 0, root=base([2], words), category='original-request-successor')
        row.update(legacy_label=label, legacy_same_frame=packet)
        yield row
    for name, root, shape, strides, offset, error, status in (
        ('past-storage', base(), [1], [1], 64, 'storage', 2),
        ('last-element-past-storage', base(), [2], [1], 63, 'storage', 2),
        ('strided-past-storage', base(), [2, 3], [100, 1], 0, 'storage', 2),
        ('nonempty-from-empty-storage', base([0]), [1], [1], 0, 'storage', 2),
        ('negative-stride', base(), [2], [-1], 1, 'negative-stride', 3),
        ('negative-stride-empty', base(), [0], [-1], 0, 'negative-stride', 3),
        ('negative-offset', base(), [1], [1], -1, 'negative-offset', 3),
        ('negative-offset-empty', base(), [0], [1], -1, 'negative-offset', 3),
        ('negative-size', base(), [-1], [1], 0, 'negative-size', 3),
        ('negative-size-rank4', base(), [1, -1, 1, 1], [1, 1, 1, 1], 0, 'negative-size', 3)):
        yield dict(name='error/' + name, category='actual-Torch-geometry-error', base=root,
                   operations=[view('view', 'root', shape, strides, offset)],
                   expected_error=error, native_error=status)
    # The locked wheel actually accepts this exact zero-element negative shape.
    # Keep its original input and position; record the native incompatibility.
    yield dict(name='error/negative-size-with-empty',
               category='known-incompatibility-empty-negative-size', base=base(),
               operations=[view('view', 'root', [0, -1], [1, 1], 0)],
               expected_Torch_acceptance=True, native_error=3,
               known_incompatibility='negative_dimension_on_empty_view')


CASES = list(recipes())
CASE_COUNT = len(CASES)
CASE_SHA256 = hashlib.sha256(canonical(CASES)).hexdigest()
CATEGORY_COUNTS = {name: sum(row['category'] == name for row in CASES)
                   for name in sorted({row['category'] for row in CASES})}
KNOWN_INCOMPATIBILITY_INDEX = 187
ORIGINAL_CASE_SHA256 = '57c00c0f5803d905975c14138c1868e0d9e8c5f62d63b71146db2338bc54bf90'
_original_cases = [dict(row) for row in CASES]
_original_cases[-1] = dict(name=CASES[-1]['name'], category='actual-Torch-geometry-error',
    base=CASES[-1]['base'], operations=CASES[-1]['operations'],
    expected_error='negative-size', native_error=3)
assert hashlib.sha256(canonical(_original_cases)).hexdigest() == ORIGINAL_CASE_SHA256
CASE_INPUT_SHA256 = hashlib.sha256(canonical([
    {key: row[key] for key in ('name', 'base', 'operations')} for row in CASES])).hexdigest()
assert [{key: row[key] for key in ('name', 'base', 'operations')} for row in CASES] == [
    {key: row[key] for key in ('name', 'base', 'operations')} for row in _original_cases]
assert len(CASES) == 188 and CASES[KNOWN_INCOMPATIBILITY_INDEX]['known_incompatibility'] == 'negative_dimension_on_empty_view'
assert sum(bool(row.get('expected_error')) for row in CASES) == 10
LEGACY_LABELS = ('unsupported-F64-view-111', 'same-packet-F64-public-alias-view', 'public-f64-as-strided')
ALLOCATION_VIEWS = 256
ALLOCATION_SPEC = dict(base=base([2], [0x3ff0000000000000, 0x8000000000000000]),
                       views=ALLOCATION_VIEWS, shape=[2], strides=[1], offset=0)
ALLOCATION_SHA256 = hashlib.sha256(canonical(ALLOCATION_SPEC)).hexdigest()
assert len({row['name'] for row in CASES}) == CASE_COUNT
assert tuple(row['legacy_label'] for row in CASES if 'legacy_label' in row) == LEGACY_LABELS
assert CATEGORY_COUNTS['all25-unit-rank-pairs'] == CATEGORY_COUNTS['all25-nested-unit-rank-pairs'] == 25
