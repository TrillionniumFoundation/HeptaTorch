"""Literal bounded signed-empty F64 recipes; expected values come from Torch."""
import hashlib
import itertools
import json

BASE = {'shape': [2], 'words': [0x3ff0000000000000, 0x8000000000000000]}
WORD = 0x4045000000000000
DIMENSIONS = (0, 1, 2, -1, -2, -(1 << 62))
MASK = (1 << 64) - 1

def recipe(name, shape, strides=None, offset=0, scope='bounded-comparison'):
    return dict(name=name, category='signed-empty-family', base=BASE,
        operations=[dict(operation='view', name='view', source='root', shape=list(shape),
                         strides=list(strides) if strides is not None else [1] * len(shape), offset=offset)],
        comparison_scope=scope, native_error=3)

EMPTY_CASES = []
for rank in range(1, 5):
    for shape in itertools.product(DIMENSIONS, repeat=rank):
        if 0 in shape:
            EMPTY_CASES.append(recipe('ordered-rank%d-%04d' % (rank, len(EMPTY_CASES)), shape))
ORDERED_CASE_COUNT = len(EMPTY_CASES)
assert ORDERED_CASE_COUNT == 774

for shape in ((0, -1), (-1, 0), (1, -1, 0), (0, -1, 1, 2)):
    for stride in (0, 2, (1 << 63) - 1, -1, -(1 << 63)):
        EMPTY_CASES.append(recipe('stride-boundary-%04d' % len(EMPTY_CASES), shape, [stride] * len(shape)))
    for offset in (1, 2, 134217728, -1, 134217729, (1 << 63) - 1):
        scope = 'known-native-offset-capacity-gap' if offset > 134217728 else 'bounded-comparison'
        EMPTY_CASES.append(recipe('offset-boundary-%04d' % len(EMPTY_CASES), shape, offset=offset, scope=scope))

for dimension in (-(1 << 63), -(1 << 62) - 1, -(1 << 62), -(1 << 62) + 1,
                  262144, 262145, (1 << 63) - 1):
    for shape in ((0, dimension), (dimension, 0)):
        scope = 'known-native-positive-dimension-capacity-gap' if dimension > 262144 else 'bounded-comparison'
        EMPTY_CASES.append(recipe('signed-domain-boundary-%04d' % len(EMPTY_CASES), shape, scope=scope))

for shape in ((-1,), (-2,), (1, -1, 1, 1), (2, -1, 0), (-1, 2, 0),
              (0, -1, 2), (-2, -2, 0), (0, -2, -2), (1, -1, 0, -1)):
    EMPTY_CASES.append(recipe('ordered-explicit-%04d' % len(EMPTY_CASES), shape))

EMPTY_ALLOCATION_SPEC = dict(base=BASE, views=256, shape=[0, -1], strides=[1, 1], offset=0)
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
EMPTY_CASE_SHA256 = hashlib.sha256(canonical(EMPTY_CASES)).hexdigest()
EMPTY_ALLOCATION_SHA256 = hashlib.sha256(canonical(EMPTY_ALLOCATION_SPEC)).hexdigest()


MAX=(1<<63)-1
WIDE_SHAPES=(
    [MAX,2,0], [2,MAX,0], [MAX,3,0], [3,MAX,0], [MAX,MAX,0],
    [0,MAX,MAX,MAX], [MAX,0,-1,MAX], [-1,MAX,0], [MAX,-1,0],
    [1,-1,0,MAX], [-(1<<62),1,0,MAX], [1<<62,3,0], [1<<62,4,0],
    [0,1<<62,4], [1<<32,(1<<32)-1,0], [1<<32,1<<32,0],
    [1<<62,2,0], [1<<62,2,1,0], [MAX,2,1,0], [MAX,0,-(1<<62),0],
)
WIDE_CASES=[recipe('wide-inline-empty-%02d'%i,shape) for i,shape in enumerate(WIDE_SHAPES)]

WIDE_CASE_SHA256 = hashlib.sha256(canonical(WIDE_CASES)).hexdigest()

HIGH = 1 << 63
CAP = 134217728
OFFSETS = (CAP-1, CAP, CAP+1, (1<<31)-1, 1<<31, (1<<60)-1, 1<<60,
           (1<<61)-1, 1<<61, HIGH-2, HIGH-1, -1, -(1<<62), -(1<<62)-1, -HIGH)
OFFSET_SHAPES = ((0,), (0,-1), (-1,0), (HIGH-1,2,0), (0,HIGH-1,HIGH-1,HIGH-1))
OFFSET_CASES = [recipe('element-offset-boundary-%03d' % (i*len(OFFSETS)+j), shape, offset=offset)
         for i,shape in enumerate(OFFSET_SHAPES) for j,offset in enumerate(OFFSETS)]
CHAINS = [dict(name='element-offset-chain-%02d'%i, base=BASE, shape=list(shape),
               strides=[1]*len(shape), offsets=[HIGH-1,CAP+1,CAP,0,HIGH-2])
          for i,shape in enumerate((OFFSET_SHAPES[1],OFFSET_SHAPES[2],OFFSET_SHAPES[4]))]
HIGH_ALLOCATION_SPEC = dict(base=BASE, views=256, shape=[0,-1], strides=[1,1], offset=HIGH-1)

# Each entry explicitly names every diagnostic byte word. Unlisted bytes must
# remain identical and are checked against the full physical owner readback.
OFFSET_FAULTS = (
    ('tag-on-nonempty-f64','root',(('descriptor','root',232,HIGH|CAP+1),('row','root',32,HIGH|CAP+1))),
    ('tag-on-f32-empty','f32',(('descriptor','view',232,HIGH|CAP+1),('row','view',32,HIGH|CAP+1))),
    ('tag-with-requires-grad','high',(('descriptor','view',32,1),)),
    ('tag-with-operation-history','high',(('descriptor','view',40,44),)),
    ('tag-with-dead-descriptor','high',(('descriptor','view',304,0),)),
    ('noncanonical-zero-tag','high',(('descriptor','view',232,HIGH),('row','view',32,HIGH))),
    ('noncanonical-cap-tag','high',(('descriptor','view',232,HIGH|CAP),('row','view',32,HIGH|CAP))),
    ('tag-descriptor-mismatch','high',(('descriptor','view',232,HIGH|CAP+2),)),
    ('tag-row-mismatch','high',(('row','view',32,HIGH|CAP+2),)),
    ('ordinary-descriptor-forged-tag','low',(('descriptor','view',232,HIGH|CAP+1),)),
    ('ordinary-row-forged-tag','low',(('row','view',32,HIGH|CAP+1),)),
    ('tag-with-nonzero-span','high',(('row','view',40,8),)),
    ('tag-with-retain-flag','high',(('row','view',56,1),)),
    ('tag-data-pointer-displaced','high',(('descriptor','view',16,{'add':8}),)),
    ('tag-false-root-index','high',(('descriptor','view',640,0),)),
    ('tag-false-root-incarnation','high',(('descriptor','view',648,2),)),
    ('tag-future-version','high',(('descriptor','view',656,1),)),
    ('tag-root-marked-view','high',(('descriptor','root',408,1),)),
    ('tag-root-offset-displaced','high',(('descriptor','root',232,8),)),
    ('tag-cleared-in-descriptor','high',(('descriptor','view',232,CAP*8),)),
    ('tag-cleared-in-row','high',(('row','view',32,CAP*8),)),
)
canonical = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
OFFSET_CASE_SHA256 = hashlib.sha256(canonical(OFFSET_CASES)).hexdigest()
CHAIN_SHA256 = hashlib.sha256(canonical(CHAINS)).hexdigest()
OFFSET_FAULT_SHA256 = hashlib.sha256(canonical(OFFSET_FAULTS)).hexdigest()
HIGH_ALLOCATION_SHA256 = hashlib.sha256(canonical(HIGH_ALLOCATION_SPEC)).hexdigest()
assert len(OFFSET_CASES) == 75 and len(CHAINS) == 3 and len(OFFSET_FAULTS) == 21

ORIGINAL_OFFSET_FAULTS = OFFSET_FAULTS
OFFSET_FAULTS = ORIGINAL_OFFSET_FAULTS + (
    ('tag-on-nonempty-shape-with-forged-zero-count','nonempty',
     (('descriptor','view',8,0),('descriptor','view',232,HIGH|CAP+1),
      ('row','view',32,HIGH|CAP+1),('row','view',40,0))),
)
OFFSET_FAULT_SHA256 = hashlib.sha256(canonical(OFFSET_FAULTS)).hexdigest()
assert len(OFFSET_FAULTS) == 22 and OFFSET_FAULTS[:-1] == ORIGINAL_OFFSET_FAULTS

ADDITIONAL_FAULTS = (
    ('nonview-ancestor-index', 'root', 640, 1, False),
    ('nonview-ancestor-incarnation', 'root', 648, 1, False),
    ('nonview-created-version', 'root', 656, 1, False),
    ('nonview-retain-count-without-child', 'root', 600, 1, False),
    ('view-ancestor-index-zero', 'view', 640, 0, True),
    ('view-ancestor-index-self', 'view', 640, 2, True),
)

assert len(EMPTY_CASES) == 841 and len(WIDE_CASES) == 20 and len(OFFSET_CASES) == 75
