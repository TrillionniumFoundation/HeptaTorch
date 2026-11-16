#!/usr/bin/env python3
"""Pinned PyTorch-only diagnostics: exact float32 rounding, traversal and version.

This does not execute or qualify a native implementation. Mathematical reference
operations use signed integers and powers of two, not a host floating RNG.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import runpy
import struct
import sys
import traceback

SUPPORT_SHA='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
SHA=lambda raw:hashlib.sha256(raw).hexdigest()
bits=lambda value:struct.unpack('<I',struct.pack('<f',value))[0]
value=lambda word:struct.unpack('<f',struct.pack('<I',word))[0]

def parts(word):
    e=(word>>23)&255;m=word&0x7fffff
    assert e!=255
    if e:m|=1<<23
    return (-m if word>>31 else m),e-150 if e else -149

def add(a,b):
    exponent=min(a[1],b[1]);return (a[0]<<(a[1]-exponent))+(b[0]<<(b[1]-exponent)),exponent

def rne(pair):
    number,exponent=pair;sign=0
    if number<0:sign=1<<31;number=-number
    if not number:return sign
    top=number.bit_length()-1+exponent
    if top>127:return sign|0x7f800000
    quantum=max(top-23,-149);shift=quantum-exponent
    if shift>0:
        mantissa,remainder=divmod(number,1<<shift);half=1<<(shift-1)
        if remainder>half or remainder==half and mantissa&1:mantissa+=1
    else:mantissa=number<<(-shift)
    if mantissa>=1<<24:mantissa>>=1;quantum+=1
    if mantissa<1<<23:return sign|mantissa
    exp=quantum+150
    if exp>=255:return sign|0x7f800000
    return sign|(exp<<23)|(mantissa&0x7fffff)

def pipelines(raw,lo,hi):
    left=parts(lo);right=parts(hi)
    width=rne(add(right,(-left[0],left[1])))
    unit=rne((raw&0xffffff,-24));u=parts(unit);w=parts(width)
    product=(u[0]*w[0],u[1]+w[1])
    fused=rne(add(product,left));split=rne(add(parts(rne(product)),left))
    def clamp(word):return lo if word==hi or (word&0x7fffffff)==(hi&0x7fffffff)==0 else word
    return clamp(fused),clamp(split),unit,width

def raw_model(state,count):
    index=struct.unpack_from('<Q',state,16)[0]
    engine=random.Random();engine.setstate((3,tuple(struct.unpack_from('<624Q',state,24))+(index if index else 624,),None))
    return [engine.getrandbits(32) for _ in range(count)],engine.getstate()[1]

def state(g):return bytes(g.get_state().tolist())

def axis_order(shape,strides):
    order=list(reversed(range(len(shape))))
    for i in range(1,len(order)):
        anchor=i
        for j in range(i-1,-1,-1):
            left,right=order[j],order[anchor];a,b=strides[left],strides[right]
            if not a or not b:continue
            if a>b or a==b and shape[left]>shape[right]:order[j],order[anchor]=order[anchor],order[j];anchor=j
            elif a<b:break
    return order

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--support',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--executable',type=Path,required=True,help='Provenance loader argument only; this probe never invokes it')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert SHA(a.support.read_bytes())==SUPPORT_SHA and SHA(a.lock.read_bytes())==LOCK_SHA
    old=sys.argv
    try:
        sys.argv=[str(a.support),str(a.executable),str(a.output.with_suffix('.provenance.json')),str(a.lock)]
        ctx=runpy.run_path(str(a.support),run_name='uniform_probe_provenance_only')
    finally:sys.argv=old
    t=ctx['torch'];provenance=ctx['PROVENANCE'];assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
    chooser=random.Random(0x1703cd)
    for _ in range(4096):
        x=chooser.getrandbits(32)
        if (x>>23)&255!=255 and x&0x7fffffff:assert rne(parts(x))==x
    ranges=[(0,0x3f800000),(0xbf800000,0x3f800000),(0xc1980000,0x40400000),
        (0x3f800001,0x40200003),(0xbf000003,0x3f7fffff),(0x80800000,0x00800001),
        (0x7e7fffff,0x7f3fffff),(0x3f800001,0x3f800001),(0x80000000,0),(0,0x80000000)]
    for _ in range(20):
        l=chooser.randrange(1,0x7e800000);h=chooser.randrange(1,0x7e800000)
        if chooser.randrange(2):l|=1<<31
        if chooser.randrange(2):h|=1<<31
        if value(l)>value(h):l,h=h,l
        ranges.append((l,h))
    arithmetic=[];total_fma=total_split=distinguishing=0
    for number,(lo,hi) in enumerate(ranges):
        g=t.Generator(device='cpu').manual_seed(1729+number);before=state(g)
        count=1024;raw,final=raw_model(before,count)
        output=t.empty(count,dtype=t.float32).uniform_(value(lo),value(hi),generator=g)
        observed=[int(x)&0xffffffff for x in output.view(t.int32).tolist()]
        expected=[pipelines(x,lo,hi) for x in raw]
        fma_errors=[i for i,(actual,ex) in enumerate(zip(observed,expected)) if actual!=ex[0]]
        split_errors=[i for i,(actual,ex) in enumerate(zip(observed,expected)) if actual!=ex[1]]
        different=[i for i,ex in enumerate(expected) if ex[0]!=ex[1]]
        after=state(g)
        assert tuple(struct.unpack_from('<624Q',after,24))+(struct.unpack_from('<Q',after,16)[0],)==final
        total_fma+=len(fma_errors);total_split+=len(split_errors);distinguishing+=len(different)
        witnesses=[]
        for i in list(dict.fromkeys(different+fma_errors+split_errors))[:16]:
            fused,split,unit,width=expected[i]
            witnesses.append(dict(index=i,raw32=raw[i],unit_bits=unit,width_bits=width,
                                  fused_bits=fused,separate_bits=split,observed_bits=observed[i]))
        arithmetic.append(dict(case='arithmetic-'+str(number),from_bits=lo,to_bits=hi,values=count,
            fused_mismatches=len(fma_errors),separate_mismatches=len(split_errors),distinguishing_values=len(different),
            observed_bits_sha256=SHA(b''.join(struct.pack('<I',x) for x in observed)),state_sha256=SHA(after),witnesses=witnesses))
    layouts=[('contiguous',(2,3),(3,1),0,12),('transpose',(3,2),(1,3),1,12),
        ('holes',(2,3),(9,2),2,24),('interleaved-nonoverlap',(2,3),(3,2),1,12),
        ('nonzero-overlap',(2,2),(1,1),1,8),('equal-stride-shape-tie',(3,2),(1,1),0,8),
        ('zero-stride-size-one',(2,1,3),(4,0,1),0,12),('expanded',(2,3),(0,1),0,8),
        ('empty-expanded',(0,3),(0,1),0,8),('empty-large-offset',(0,3),(0,1),999,8),
        ('channels-last',(2,3,2,2),(12,1,6,3),0,24),('scalar',(),(),3,8)]
    traversal=[]
    for name,shape,strides,offset,storage in layouts:
        g=t.Generator(device='cpu').manual_seed(99);before=state(g);base=t.full((storage,),-7.,dtype=t.float32)
        version=base._version;error=None;actual=[];expected=[];order=axis_order(shape,strides)
        try:
            view=base.as_strided(shape,strides,offset);view.uniform_(0,1,generator=g)
            actual=[int(x)&0xffffffff for x in base.view(t.int32).tolist()]
            count=view.numel();raw,_=raw_model(before,count);expected=[bits(-7.)]*storage
            for i,x in enumerate(raw):
                q=i;at=offset
                for axis in order:at+=(q%shape[axis])*strides[axis];q//=shape[axis]
                expected[at]=rne((x&0xffffff,-24))
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
        traversal.append(dict(case=name,shape=shape,strides=strides,storage_offset=offset,storage_elements=storage,
            comparison_order=order,error=error,version_before=version,version_after=base._version,
            generator_state_unchanged=state(g)==before,expected_order_matches=(actual==expected) if error is None else None,
            actual_storage_bits=actual,expected_storage_bits=expected))
    boundary=[]
    for name,shape,lo,hi,requires,no_grad in [('empty', (0,),0.,1.,False,False),
        ('empty-invalid-order',(0,),2.,1.,False,False),('empty-overwide',(0,),-value(0x7f7fffff),value(0x7f7fffff),False,False),
        ('leaf-grad',(3,),0.,1.,True,False),('leaf-no-grad',(3,),0.,1.,True,True)]:
        g=t.Generator().manual_seed(17);before=state(g);tensor=t.empty(shape,requires_grad=requires);version=tensor._version;error=None
        try:
            if no_grad:
                with t.no_grad():tensor.uniform_(lo,hi,generator=g)
            else:tensor.uniform_(lo,hi,generator=g)
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
        boundary.append(dict(case=name,error=error,version_before=version,version_after=tensor._version,
                             generator_state_unchanged=state(g)==before))
    report={'schema':'heptatorch.pytorch-uniform-boundary-probe.v1','status':'OBSERVED','probe_sha256':SHA(Path(__file__).read_bytes()),
        'oracle_provenance':provenance,'torch_config':t.__config__.show(),'arithmetic':arithmetic,'traversal':traversal,'boundaries':boundary,
        'fused_mismatches':total_fma,'separate_mismatches':total_split,'distinguishing_values':distinguishing,
        'mathematical_reference':'exact signed dyadic arithmetic with integer RNE to binary32',
        'native_executed':False,'native_uniform_qualified':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
