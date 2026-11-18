"""Two literal input recipes for the explicitly added Bool destination types."""
import hashlib
import json
import struct

CASES=(
 dict(name='F32_explicit_Bool',dtype='float32',words=[0x3f800000,0xbf800000],
      shape=[2],strides=[1],axes=None,keepdim=False,offset=0,requires_grad=False,operation='sum_bool'),
 dict(name='wrapped_F64_explicit_Bool',dtype='float64',words=[0x3ff0000000000000],
      shape=[],strides=[],axes=None,keepdim=False,offset=0,requires_grad=False,operation='sum_bool'),
)
def raw_input(spec):
    return struct.pack('<'+{'float32':'I','float64':'Q'}[spec['dtype']]*len(spec['words']),*spec['words'])
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
CASE_SHA256=hashlib.sha256(canonical(CASES)).hexdigest()
assert len(CASES)==2
