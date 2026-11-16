#!/usr/bin/env python3
"""Execute the native storage service and inspect independent lifetime facts.

Python only constructs packets, sets process limits, and checks observations.
No Python storage implementation is reachable from the target executable.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import struct
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import body,trace,parse_trace,handle


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_dump(raw):
    """Observe coverage/refcounts independently, without reproducing allocation."""
    assert len(raw) >= 144 and raw[:8] == b'HTOSST01'
    words = struct.unpack_from('<16Q',raw)
    _,version,n,p,free_count,epoch,provider,*reserved = words
    assert version == provider == 1 and not any(reserved)
    assert 4 <= n <= 65536 and n & (n-1) == 0
    assert 16 <= p <= 1 << 30 and p % 16 == 0
    base = 424*n+144
    assert len(raw) == base+p and 0 <= free_count <= n+1
    assert raw[144+400*n:base] == bytes(24*n)
    tables = {k:{} for k in range(1,7)}
    for k in tables:
        for slot in range(1,n+1):
            at = 128+((k-1)*n+slot-1)*64
            row = struct.unpack_from('<8Q',raw,at)
            gen,live,*fields = row
            assert gen <= 0xffffffff and live in (0,1)
            if live:
                assert gen != 0
                tables[k][handle(k,slot,gen)] = fields
            else:
                assert not any(fields)
    strong,weak,version_refs = Counter(),Counter(),Counter()
    for kind in (1,5):
        for sid,vid,a,b,c,d in tables[kind].values():
            assert sid in tables[2] and vid in tables[3]
            strong[sid] += 1
            version_refs[vid] += 1
            offset,length = (a,b) if kind == 1 else (b,c)
            assert offset <= tables[2][sid][1] and length <= tables[2][sid][1]-offset
            assert d == 0
            if kind == 1:
                assert c == 0
    for kind in (4,6):
        for sid,*rest in tables[kind].values():
            assert sid in tables[2] and not any(rest)
            (weak if kind == 4 else strong)[sid] += 1
    regions = []
    for sid,(off,size,cap,sc,wc,dead) in tables[2].items():
        assert sc == strong[sid] and wc == weak[sid]+(sc>0) and wc > 0
        if sc:
            assert dead == 0 and cap == (size+15)//16*16
            assert off % 16 == 0 and off <= p and cap <= p-off
            if cap:
                regions.append((off,off+cap,'live'))
            else:
                assert off == 0
        else:
            assert dead == 1 and off == cap == 0
    for vid,(counter,refs,*rest) in tables[3].items():
        assert refs == version_refs[vid] and refs > 0 and not any(rest)
    free_at = 128+384*n
    free = [struct.unpack_from('<2Q',raw,free_at+i*16) for i in range(free_count)]
    for i,(off,size) in enumerate(free):
        assert size > 0 and size % 16 == off % 16 == 0 and off+size <= p
        if i:
            assert free[i-1][0]+free[i-1][1] < off
        regions.append((off,off+size,'free'))
    assert raw[free_at+free_count*16:free_at+(n+1)*16] == bytes((n+1-free_count)*16)
    cursor = 0
    for start,end,kind in sorted(regions):
        assert start == cursor and end > start, (cursor,start,end,kind)
        cursor = end
    assert cursor == p
    return {'slots':n,'pool':p,'resize_epoch':epoch,
            'live_counts':{str(k):len(v) for k,v in tables.items()},
            'free_extents':free,'state_sha256':sha(raw),'tables':tables}


class Suite:
    def __init__(self,binary,output):
        self.binary,self.output = binary,output
        self.rows,self.frames,self.values,self.dumps = [],0,0,0

    def values_at(self,frames,frame,record,expected):
        observed = frames[frame]['records'][record]['values']
        assert observed[:len(expected)] == tuple(expected), (frame,record,observed,expected)
        self.values += len(expected)

    def dump_at(self,frames,frame,record):
        raw = frames[frame]['records'][record]['payload']
        info = check_dump(raw)
        self.dumps += 1
        return raw,info

    def run(self,name,requests,statuses,verify=None,address_limit=None):
        out = self.output/name
        out.mkdir()
        raw = trace(requests)
        (out/'request.bin').write_bytes(raw)
        for i,request in enumerate(requests):
            (out/f'frame-{i:03d}.request.bin').write_bytes(request)
        limit = None
        if address_limit is not None:
            def limit():
                resource.setrlimit(resource.RLIMIT_AS,(address_limit,address_limit))
        result = subprocess.run([str(self.binary)],input=raw,capture_output=True,
                                env={'PATH':''},preexec_fn=limit,timeout=30)
        (out/'actual.bin').write_bytes(result.stdout)
        (out/'stderr.bin').write_bytes(result.stderr)
        row = {'case':name,'request_sha256':sha(raw),'response_sha256':sha(result.stdout),
               'exit_code':result.returncode,'expected_frame_statuses':statuses,
               'address_space_limit_bytes':address_limit,'pass':False}
        self.rows.append(row)
        try:
            assert result.returncode == 0 and not result.stderr, (result.returncode,result.stderr)
            trace_status,frames = parse_trace(result.stdout)
            assert trace_status == 0
            assert [f['status'] for f in frames] == statuses, [f['status'] for f in frames]
            for i,frame in enumerate(frames):
                (out/f'frame-{i:03d}.actual.bin').write_bytes(frame['raw'])
            before_values,before_dumps = self.values,self.dumps
            if verify:
                verify(frames)
            row.update(pass_=True)
            row['pass'] = row.pop('pass_')
            row['frames'] = len(frames)
            row['scalar_fields_compared'] = self.values-before_values
            row['state_dumps_independently_checked'] = self.dumps-before_dumps
            row['observed_records'] = [[{'op':r['op'],'values':r['values'],
                                       'payload_bytes':len(r['payload']),
                                       'payload_sha256':sha(r['payload']) if r['payload'] else None}
                                      for r in f['records']] for f in frames]
            self.frames += len(frames)
        except BaseException:
            row['failure'] = traceback.format_exc()
            (out/'report.json').write_text(json.dumps(row,indent=2)+'\n')
            self.save('FAIL')
            raise
        (out/'report.json').write_text(json.dumps(row,indent=2)+'\n')
        self.save('RUNNING')
        print(name+' PASS',flush=True)

    def save(self,status):
        report = {'schema':'heptatorch.native-storage-object-qualification.v1','status':status,
                  'binary_sha256':sha(self.binary.read_bytes()),'transport':'finite native HIA trace driver',
                  'interactive_claimed':False,'target_supplied_by_python':False,
                  'cases':len(self.rows),'frames':self.frames,'scalar_fields_compared':self.values,
                  'state_dumps_independently_checked':self.dumps,'rows':self.rows,
                  'pytorch_api_denominator':2590,'full_rewrite_complete':False}
        (self.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    suite = Suite(a.binary.resolve(),a.output.resolve())
    T,S,V,W,D,H = (lambda slot,g=1,k=k:handle(k,slot,g) for k in range(1,7))
    f = [body([(1,16,65),(1,32,66),(1,16,67),(2,T(2),4,8),(4,T(2)),(8,T(4)),
               (5,W(1)),(3,T(2)),(12,T(4),0),(13,H(1),31),(14,D(1),7),(15,S(2)),(20,),(18,)],4,128),
         body([(3,T(4)),(10,D(1)),(13,H(1),31),(15,S(2)),(20,)]),
         body([(7,H(1)),(15,S(2)),(1,32,68),(12,T(1),15),(12,T(3),15),
               (12,T(2,2),31),(20,),(18,)]),
         body([(5,W(1))]),body([(12,T(2),0)]),body([(12,T(4),0)]),body([(15,V(2))]),
         body([(15,W(1)),(15,S(2)),(18,),(6,W(1)),(1,0,69),(4,T(4,2)),(20,),(18,)]),
         body([(5,W(1))]),body([(15,S(2))]),body([(12,T(4),0)]),
         body([(15,W(1,2)),(15,S(2,2)),(12,T(2,2),31),(20,)])]
    def central(r):
        suite.values_at(r,0,0,[T(1),S(1),V(1),0,16])
        suite.values_at(r,0,1,[T(2),S(2),V(2),16,32])
        suite.values_at(r,0,2,[T(3),S(3),V(3),48,16])
        for i in (8,9,10): suite.values_at(r,0,i,[66])
        suite.values_at(r,0,11,[16,32,3,2,0])
        suite.values_at(r,1,2,[66]);suite.values_at(r,1,3,[16,32,1,2,0])
        suite.values_at(r,2,1,[0,32,0,1,1])
        suite.values_at(r,2,2,[T(2,2),S(4),V(2,2),16,32])
        for i,value in ((3,65),(4,67),(5,68)):suite.values_at(r,2,i,[value])
        suite.dump_at(r,0,13)
        before,_ = suite.dump_at(r,2,7)
        after,_ = suite.dump_at(r,7,2)
        assert before == after, 'failed weak/stale accesses mutated state'
        suite.values_at(r,7,4,[T(4,2),S(2,2),V(4),0,0])
        suite.values_at(r,7,5,[W(1,2),S(2,2)])
        suite.dump_at(r,7,7)
        suite.values_at(r,11,0,[S(2,2)])
        suite.values_at(r,11,1,[0,0,1,2,0])
        suite.values_at(r,11,2,[68])
    suite.run('abc-interior-reuse-and-distinct-generations',f,[0,0,0,6,1,1,1,0,1,1,1,0],central)

    f = [body([(1,4,10),(2,T(1),0,4),(8,T(1)),(19,T(1),0,4),(8,T(3)),
               (11,T(2),1,55),(12,T(1),1),(12,T(3),1),(15,V(1)),(15,V(2)),(20,),(18,)]),
         body([(9,D(1))]),body([(14,D(1),1)]),
         body([(9,D(2)),(14,D(2),1),(3,T(1)),(3,T(2)),(15,V(1)),(20,),(18,)]),
         body([(10,D(1)),(15,V(2)),(20,)])]
    def versions(r):
        suite.values_at(r,0,3,[T(3),S(1),V(2),0,4])
        suite.values_at(r,0,5,[1]);suite.values_at(r,0,6,[55]);suite.values_at(r,0,7,[55])
        suite.values_at(r,0,8,[1,3]);suite.values_at(r,0,9,[0,2])
        suite.dump_at(r,0,11)
        suite.values_at(r,3,0,[0,0]);suite.values_at(r,3,1,[55]);suite.values_at(r,3,4,[1,1])
        suite.dump_at(r,3,6);suite.values_at(r,4,1,[0,2])
    suite.run('independent-version-identity-and-saved-validation',f,[0,5,5,0,0],versions)

    f = [body([(1,16,i) for i in range(1,6)]+[(16,),(20,),(18,)],4,64),
         body([(17,32,1000)]+[(12,T(i),15) for i in range(1,6)]+[(16,),(20,),(18,)]),
         body([(17,1,1),(20,),(18,)])]
    def growth(r):
        suite.values_at(r,0,5,[8,128,1,48,1]);suite.dump_at(r,0,7)
        suite.values_at(r,1,0,[32,1008,1,928,2])
        for i in range(1,6):suite.values_at(r,1,i,[i])
        before,_ = suite.dump_at(r,1,8);after,_ = suite.dump_at(r,2,2)
        assert before == after, 'no-op reserve changed state'
    suite.run('actual-table-and-pool-migration',f,[0,0,0],growth)

    f = [body([(1,16,1),(18,)],4,64),
         body([(11,T(1),0,9),(1,32,2),(3,T(1)),(99,)]),
         body([(18,),(12,T(1),0),(1,32,2),(20,)])]
    def transaction(r):
        before,_ = suite.dump_at(r,0,1);after,_ = suite.dump_at(r,2,0)
        assert before == after
        suite.values_at(r,2,1,[1]);suite.values_at(r,2,2,[T(2),S(2),V(2),16,32])
    suite.run('whole-frame-rollback-including-ids-and-cursors',f,[0,7,0],transaction)

    bad_commands = [(12,S(1),0),(12,T(1),16),(11,T(1),0,256),
                    (2,T(1),0xffffffffffffffff,1),(2,T(1),16,1),
                    (1,(1<<30)+1,0),(17,65537,0),(17,0,(1<<30)+1),
                    (3,T(1),1),(0xffffffffffffffff,)]
    f = [body([(1,16,7),(18,)],4,64)]+[body([c]) for c in bad_commands]+[body([(18,),(12,T(1),15),(20,)])]
    def bounds(r):
        before,_ = suite.dump_at(r,0,1);after,_ = suite.dump_at(r,len(r)-1,0)
        assert before == after
        suite.values_at(r,len(r)-1,1,[7])
    suite.run('wrong-kind-bounds-reserved-and-capacity-refusals',f,[0,1,2,2,2,2,2,2,2,7,7,0],bounds)

    malformed = [b'', b'HTOSOBJ1'+bytes(8),body()+b'X',b'BADMAGIC'+body()[8:],
                 b'HTOSOBJ1'+struct.pack('<3Q',1025,0,0)]
    f = malformed+[body([(1,16,5),(18,)],4,64),body([],slots=4),body([(18,),(20,)])]
    def malformed_check(r):
        suite.values_at(r,5,0,[T(1),S(1),V(1),0,16])
        before,_ = suite.dump_at(r,5,1);after,_ = suite.dump_at(r,7,0)
        assert before == after
    suite.run('malformed-first-frames-and-reinitialization-refused',f,[100]*5+[0,2,0],malformed_check)

    f = [body([(1,0,4),(2,T(1),0,0),(4,T(1)),(8,T(2)),(3,T(1)),(3,T(2)),(5,W(1)),(20,),(18,)],4,16),
         body([(13,H(1),0)]),body([(14,D(1),0)]),
         body([(10,D(1)),(7,H(1)),(15,S(1)),(16,),(20,)]),
         body([(5,W(1))]),body([(6,W(1)),(1,0,9),(20,),(18,)])]
    def zero(r):
        suite.values_at(r,0,0,[T(1),S(1),V(1),0,0])
        suite.dump_at(r,0,8);suite.values_at(r,3,2,[0,0,0,1,1])
        suite.values_at(r,3,3,[4,16,1,16,0]);suite.dump_at(r,5,3)
    suite.run('zero-length-owned-storage-lifetimes',f,[0,2,2,0,6,0],zero)

    f = [body([(1,1048593,7),(12,T(1),0),(12,T(1),1048592),(11,T(1),1048592,42),(16,),(20,)],4,256),
         body([(17,0,3145729),(12,T(1),1048592),(12,T(1),0),(16,),(20,)]),
         body([(18,)]),body([(12,T(1),1048592),(20,)])]
    def large(r):
        suite.values_at(r,0,0,[T(1),S(1),V(1),0,1048593])
        suite.values_at(r,0,1,[7]);suite.values_at(r,0,2,[7]);suite.values_at(r,0,3,[1])
        suite.values_at(r,0,4,[4,2097152,1,1048544,1])
        suite.values_at(r,1,0,[4,3145744,1,2097136,2])
        suite.values_at(r,1,1,[42]);suite.values_at(r,1,2,[7]);suite.values_at(r,3,0,[42])
    suite.run('owned-payload-exceeds-one-mib-and-migrates',f,[0,0,3,0],large)

    f = [body([(1,16,7),(18,)],4,64),body([(17,0,134217728)]),body([(18,),(12,T(1),15),(1,16,8),(20,)])]
    def oom(r):
        before,_ = suite.dump_at(r,0,1);after,_ = suite.dump_at(r,2,0)
        assert before == after
        suite.values_at(r,2,1,[7]);suite.values_at(r,2,2,[T(2),S(2),V(2),16,16])
    suite.run('actual-mmap-failure-preserves-committed-owner',f,[0,3,0],oom,64<<20)
    f = [body([],4,134217728),body([(1,16,77),(20,)])]
    def initial_oom(r):
        suite.values_at(r,1,0,[T(1),S(1),V(1),0,16])
    suite.run('actual-initial-allocation-failure-returns-live-empty',f,[3,0],initial_oom,64<<20)

    f = [body([(1,16,7),(18,),(18,)],4,600000),body([(1,16,9),(12,T(1),15),(20,),(18,)])]
    def output_capacity(r):
        suite.values_at(r,1,0,[T(1),S(1),V(1),0,16]);suite.values_at(r,1,1,[9]);suite.dump_at(r,1,3)
    suite.run('reply-capacity-failure-rolls-back-provisional-state',f,[3,0],output_capacity)

    f = [body([(1,16,i) for i in range(1,5)]+[(3,T(1)),(3,T(3)),(20,),(18,)],4,64),
         body([(3,T(2)),(3,T(4)),(16,),(20,),(18,)]),
         body([(1,64,88),(12,T(1,2),63),(16,),(20,)])]
    def coalesce(r):
        _,d = suite.dump_at(r,0,7);assert d['free_extents'] == [(0,16),(32,16)]
        suite.values_at(r,1,2,[4,64,1,64,0]);_,d = suite.dump_at(r,1,4);assert d['free_extents'] == [(0,64)]
        suite.values_at(r,2,0,[T(1,2),S(1,2),V(1,2),0,64]);suite.values_at(r,2,1,[88]);suite.values_at(r,2,2,[4,64,0,0,0])
    suite.run('free-extents-merge-both-neighbors-and-full-pool-reuse',f,[0,0,0],coalesce)

    report = suite.save('PASS')
    print(json.dumps({k:report[k] for k in ('status','binary_sha256','cases','frames','scalar_fields_compared','state_dumps_independently_checked')}))


if __name__ == '__main__':
    main()
