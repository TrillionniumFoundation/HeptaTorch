#!/usr/bin/env python3
"""Independent nonuniform VJP/page/owner-migration review of native tensors.

The numerical reference is exact rational differentiation of a small graph.
It is also compared with the frozen canonical native kernel. Host code never
executes in place of either target and never constructs an owned capability.
"""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys
import traceback

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import Program,bits,parse,words,f32bits

TARGET_SHA='05c5544da2b7b34a8db9bd019a3ad0fa930f9ed5c18fd4b7582e7150f21eaeea'
sha=lambda b:hashlib.sha256(b).hexdigest()
flatten=lambda a:[v for row in a for v in row]
transpose=lambda a:[list(row) for row in zip(*a)]


def matmul(a,b):
    return [[sum((F(x)*F(y) for x,y in zip(row,col)),F(0)) for col in zip(*b)] for row in a]


def reference():
    x=[[F(1),F(-2),F(3)],[F(4),F(-5),F(6)]]
    w=[[F(1,2),F(-1)],[F(2),F(1,4)],[F(-1),F(3,2)]]
    seed=[[F(1,2),F(-5,4),F(2)],[F(3),F(-3,4),F(3,2)]]
    index=[[1,0,1],[0,1,0]];columns=[1,1,0]
    m=matmul(x,w);tr=transpose(m)
    selected=[[r[c] for c in columns] for r in tr]
    g=[[selected[index[i][j]][j] for j in range(3)] for i in range(2)]
    q=[[v*v for v in row] for row in g]
    ds=[[F(0) for _ in range(3)] for _ in range(2)]
    for i in range(2):
        for j in range(3):ds[index[i][j]][j]+=2*g[i][j]*seed[i][j]
    dt=[[F(0),F(0)] for _ in range(2)]
    for i in range(2):
        for j,c in enumerate(columns):dt[i][c]+=ds[i][j]
    dm=transpose(dt);dx=matmul(dm,transpose(w));dw=matmul(transpose(x),dm)
    updated=[[v-F(1,2)*grad for v,grad in zip(row,grow)] for row,grow in zip(x,dx)]
    return {'x':x,'w':w,'seed':seed,'gather':g,'square':q,'dx':dx,'dw':dw,'updated':updated}


def graph(p,x,w):
    m=p.matmul(x,w);tr=p.transpose(m,0,1)
    selected=p.select(tr,-1,[1,1,0])
    gathered=p.gather(selected,0,(2,3),[1,0,1,0,1,0])
    squared=p.binary(3,gathered,gathered)
    return {'matmul':m,'transpose':tr,'selected':selected,'gathered':gathered,'squared':squared}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--canonical-binary',type=Path,required=True)
    ap.add_argument('--transport-observer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert sha(a.binary.read_bytes())==TARGET_SHA
    observer_raw=a.transport_observer.read_bytes()
    spec=importlib.util.spec_from_file_location('native_owner_observer',a.transport_observer)
    observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    r=reference();comparisons=0;checks=[]
    def check_values(record,values,label):
        nonlocal comparisons
        values=list(values);expected=tuple(bits(v) for v in values);actual=f32bits(record)
        assert actual==expected,(label,actual,expected)
        comparisons+=len(values)
        checks.append({'check':label,'values':len(values),'expected_bits':expected,'actual_bits':actual,'pass':True})

    # Independent unchanged-native comparison uses the same small mathematical
    # graph, without capacity expansion or the new full/range wire operations.
    old=Program(version=1)
    x=old.tensor((2,3),flatten(r['x']),True);w=old.tensor((3,2),flatten(r['w']),True)
    seed=old.tensor((2,3),flatten(r['seed']))
    g=graph(old,x,w);q=old.read(g['squared'])
    old.backward(g['squared'],seed);dx=old.read(x,True);dw=old.read(w,True)
    old.backward(g['squared'],seed);dx2=old.read(x,True);dw2=old.read(w,True)
    old.sgd(x,F(1,4));updated=old.read(x)
    result=subprocess.run([str(a.canonical_binary.resolve())],input=old.encode(),capture_output=True,env={'PATH':''},timeout=30)
    (a.output/'canonical.request.bin').write_bytes(old.encode());(a.output/'canonical.actual.bin').write_bytes(result.stdout)
    (a.output/'canonical.stderr').write_bytes(result.stderr)
    assert result.returncode==0 and not result.stderr
    status,old_rows=parse(result.stdout,1);assert status==0
    for pos,values,label in [(q,flatten(r['square']),'canonical-square'),(dx,flatten(r['dx']),'canonical-dx'),
          (dw,flatten(r['dw']),'canonical-dw'),(dx2,[2*v for v in flatten(r['dx'])],'canonical-repeat-dx'),
          (dw2,[2*v for v in flatten(r['dw'])],'canonical-repeat-dw'),(updated,flatten(r['updated']),'canonical-SGD')]:
        check_values(old_rows[pos],values,label)

    session=observer.OwnerSession(a.binary,directory=a.output/'interactive',address_limit=64<<20)
    frame_rows=[];rollback_bytes=0;created=[]
    def exchange(p,status=0,label=''):
        raw=p.encode();reply=session.exchange(raw)
        code,rows=parse(reply);assert code==status,(label,code,status)
        assert len(rows)==(len(p.commands) if code==0 else 0)
        frame_rows.append({'label':label,'expected_status':status,'observed_status':code,
                           'request_sha256':sha(raw),'reply_sha256':sha(reply),'reply_bytes':len(reply)})
        return rows
    def failure(p,status,label):
        nonlocal rollback_bytes
        before=session.owner();rows=exchange(p,status,label);after=session.owner()
        assert before['pointer']==after['pointer'] and before['length']==after['length']
        assert before['bytes']==after['bytes'],label
        rollback_bytes+=before['length']
        frame_rows[-1].update(entire_owner_bytes_unchanged=True,owner_bytes=before['length'],
                              same_returned_owner_pointer=True,owner_sha256=sha(before['bytes']))
        return rows
    try:
        first=Program()
        x=first.tensor((2,3),flatten(r['x']),True);w=first.tensor((3,2),flatten(r['w']),True)
        seed=first.tensor((2,3),flatten(r['seed']))
        for _ in range(257):first.full((0,))
        mx=first.meta(x);mw=first.meta(w);mp=first.meta(260)
        rows=exchange(first,label='260-owned-leaf-prefix')
        assert [record['result'] for record in rows[:260]]==list(range(1,261))
        created.extend(record['result'] for record in rows[:260]);comparisons+=260
        for i,tid in [(mx,x),(mw,w),(mp,260)]:
            meta=words(rows[i]);assert meta[11]==tid and meta[12]==0
            comparisons+=2
        initial_owner=session.owner()
        raw=initial_owner['bytes'];page1=struct.unpack_from('<Q',raw,4104)[0]
        assert initial_owner['length']>=131072 and page1>=57344 and page1+49152<=struct.unpack_from('<Q',raw,16)[0]
        p=Program(next_slot=261);g=graph(p,x,w);big=p.full((512,1024),F(1,8))
        sq=p.read(g['squared']);tail=p.read(big,start=524285,count=3)
        p.backward(g['squared'],seed);dx=p.read(x,True);dw=p.read(w,True)
        mt=p.meta(g['transpose']);mb=p.meta(big)
        rows=exchange(p,label='nonuniform-VJP-across-descriptor-page-and-owner-growth')
        assert big==266 and g['squared']==265
        check_values(rows[sq],flatten(r['square']),'owned-square')
        check_values(rows[tail],[F(1,8)]*3,'two-MiB-data-tail')
        check_values(rows[dx],flatten(r['dx']),'owned-dx');check_values(rows[dw],flatten(r['dw']),'owned-dw')
        assert words(rows[mt])[11:]==(261,0) and words(rows[mb])[1]==524288
        expanded=session.owner();assert expanded['length']>initial_owner['length'] and expanded['length']>=4<<20
        assert struct.unpack_from('<Q',expanded['bytes'],4104)[0]==page1
        p=Program(next_slot=267);p.backward(g['squared'],seed);dx=p.read(x,True);dw=p.read(w,True)
        p.sgd(x,F(1,4));rx=p.read(x);xm=p.meta(x)
        rows=exchange(p,label='repeat-backward-and-SGD-after-owner-migration')
        check_values(rows[dx],[2*v for v in flatten(r['dx'])],'owned-repeat-dx')
        check_values(rows[dw],[2*v for v in flatten(r['dw'])],'owned-repeat-dw')
        check_values(rows[rx],flatten(r['updated']),'owned-SGD');assert words(rows[xm])[12]==1
        p=Program(next_slot=267);p.backward(g['squared'],seed)
        failure(p,6,'saved-version-failure-after-SGD')
        p=Program(next_slot=267);p.reset(261)
        failure(p,6,'reset-cannot-retain-nonleaf')
        p=Program(next_slot=267);p.command(12,x,0,bits(999));p.full((1024,2048),1);p.command(999)
        failure(p,4,'mutation-growth-and-later-unknown-op-whole-frame-rollback')
        p=Program(next_slot=267);p.full((4096,4096),0)
        failure(p,3,'real-mmap-failure-retains-original-four-MiB-owner')
        p=Program(next_slot=267);p.full((262145,),0)
        failure(p,3,'per-dimension-refusal-not-capacity-retry')
        p=Program(next_slot=267);source=p.full((1,1,128,128),1)
        p.make(23,23,source,(128<<32)|128,(1<<32)|1,(64<<32)|64,1)
        failure(p,3,'pool-work-budget-refusal-not-capacity-retry')
        p=Program(next_slot=267);p.reset(260);mp=p.meta(260);rx=p.read(x)
        rows=exchange(p,label='reset-retains-descriptor-page-end')
        assert words(rows[mp])[0:2]==(1,0);check_values(rows[rx],flatten(r['updated']),'retained-parameter-after-reset')
        reset_owner=session.owner();cursor=struct.unpack_from('<Q',reset_owner['bytes'],16)[0]
        assert cursor>=page1+49152 and struct.unpack_from('<Q',reset_owner['bytes'],4104)[0]==page1
        assert not any(reset_owner['bytes'][4112:6144])
        p=Program(next_slot=261,generation=1,keep=260);p.meta(261)
        failure(p,1,'old-generation-graph-handle-refused')
        p=Program(next_slot=261,generation=1,keep=260)
        fresh=[p.full((),F(3,4)) for _ in range(258)]
        mp=p.meta(256);read=p.read(fresh[0]);rx=p.read(x)
        rows=exchange(p,label='reuse-slots-and-allocate-next-descriptor-page')
        assert fresh[0]==(1<<16)|261 and fresh[-1]==(1<<16)|518
        assert [v['result'] for v in rows[:258]]==fresh
        created.extend(fresh);comparisons+=258
        assert words(rows[mp])[0:2]==(1,0)
        check_values(rows[read],[F(3,4)],'new-generation-scalar')
        check_values(rows[rx],flatten(r['updated']),'retained-parameter-after-page2-allocation')
        owner=session.owner();page2=struct.unpack_from('<Q',owner['bytes'],4112)[0]
        assert page2>=cursor and page2+49152<=struct.unpack_from('<Q',owner['bytes'],16)[0]
        p=Program(next_slot=519,generation=1,keep=260);p.reset(260);new=p.full((),F(1,2));m=p.meta(256)
        rows=exchange(p,label='second-reset-discards-page2-and-advances-generation')
        assert new==(2<<16)|261 and rows[1]['result']==new and words(rows[m])[0:2]==(1,0)
        owner=session.owner();assert struct.unpack_from('<Q',owner['bytes'],4112)[0]==0
        p=Program(next_slot=262,generation=2,keep=260);p.read(fresh[0])
        failure(p,1,'previous-reused-handle-refused')
        p=Program(next_slot=262,generation=2,keep=260);p.reset(259)
        failure(p,7,'retained-prefix-cannot-change')
        p=Program(next_slot=262,generation=2,keep=260);p.read(x,start=(1<<64)-1,count=1)
        failure(p,2,'range-index-overflow-refused')
        session.finish()
        report={'schema':'heptatorch.independent-owned-tensor-migration-review.v1','status':'PASS',
                'candidate_executable_sha256':TARGET_SHA,'canonical_executable_sha256':sha(a.canonical_binary.read_bytes()),
                'transport_observer_sha256':sha(observer_raw),'review_script_sha256':sha(Path(__file__).read_bytes()),
                'wire_encoder_sha256':sha((Path(__file__).parent/'protocol.py').read_bytes()),
                'reference':'Exact rational nonuniform VJP plus separate unchanged canonical native execution',
                'hosted_pytorch_execution_claimed':False,'native_target_execution':True,
                'interactive_frames':len(frame_rows),'responses_before_stdin_eof':True,
                'exact_scalar_or_f32_fields_compared':comparisons,'entire_owner_rollback_bytes_compared':rollback_bytes,
                'created_handle_count':len(created),'all_created_handles_unique':len(set(created))==len(created),
                'initial_owner_capacity':initial_owner['length'],'grown_owner_capacity':expanded['length'],
                'relative_page1_offset_preserved':page1,'frame_rows':frame_rows,'numerical_checks':checks,
                'pytorch_api_denominator':2590,'full_rewrite_complete':False}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('status','interactive_frames','exact_scalar_or_f32_fields_compared','entire_owner_rollback_bytes_compared','initial_owner_capacity','grown_owner_capacity')}))
    except BaseException:
        (a.output/'failure.txt').write_text(traceback.format_exc())
        if session.poll() is None:
            import os
            os.kill(session.pid,signal.SIGKILL)
        raise


if __name__=='__main__':main()
