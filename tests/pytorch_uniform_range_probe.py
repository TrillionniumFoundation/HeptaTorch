#!/usr/bin/env python3
"""Fixed real Torch observation of binary64 uniform-bound admission.

No native implementation is executed; no expected acceptance is imposed.
"""
import argparse,hashlib,json,runpy,struct,sys
from pathlib import Path

SHA=lambda b:hashlib.sha256(b).hexdigest()
SUPPORT_SHA='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
def value(word):return struct.unpack('<f',struct.pack('<I',word))[0]
def bits(number):return struct.unpack('<I',struct.pack('<f',number))[0]

def corpus():
    maximum=0x7f7fffff;rows=[]
    small=[0,1,0x007fffff,0x00800000,bits(1.),0x647fffff,0x64800000,0x64800001,0x73000000,0x73800000]
    for number in small:
        rows.append((number|0x80000000,maximum));rows.append((maximum|0x80000000,number))
    for larger in (maximum-1,maximum-2,0x7f400000,0x7f000000):
        gap=bits(value(maximum)-value(larger))
        for smaller in (gap-1,gap,gap+1):
            rows.append((smaller|0x80000000,larger));rows.append((larger|0x80000000,smaller))
    return list(dict.fromkeys(rows))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('support','lock','executable','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert SHA(a.support.read_bytes())==SUPPORT_SHA and SHA(a.lock.read_bytes())==LOCK_SHA
    old=sys.argv
    try:
        sys.argv=[str(a.support),str(a.executable),str(a.output.with_suffix('.provenance.json')),str(a.lock)]
        ctx=runpy.run_path(str(a.support),run_name='uniform_range_provenance_only')
    finally:sys.argv=old
    t=ctx['torch'];provenance=ctx['PROVENANCE']
    assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
    rows=[]
    for i,(lo,hi) in enumerate(corpus()):
        for count in (0,4):
            g=t.Generator().manual_seed(99);before=bytes(g.get_state().tolist());tensor=t.full((count,),-7.,dtype=t.float32)
            error=None
            try:tensor.uniform_(value(lo),value(hi),generator=g)
            except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
            raw=bytes(g.get_state().tolist());observed=[int(x)&0xffffffff for x in tensor.view(t.int32).tolist()]
            rows.append({'case':i,'from_bits':lo,'to_bits':hi,'elements':count,'error':error,
                'accepted':error is None,'version':tensor._version,'generator_state_unchanged':raw==before,
                'state_sha256':SHA(raw),'output_bits':observed,
                'host_binary64_predicate':value(hi)-value(lo)<=value(0x7f7fffff)})
    report={'schema':'heptatorch.pytorch-uniform-binary64-admission-probe.v1','status':'OBSERVED',
        'probe_sha256':SHA(Path(__file__).read_bytes()),'oracle_provenance':provenance,'rows':rows,
        'cases':len(rows),'native_executed':False,'native_qualified':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
