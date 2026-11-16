"""Independent native gradient ownership, growth, rollback and lifetime controls.

The host serializes fixed requests and observes the native owner read-only.
It imports no implementation or authoring code. Exact small dyadic values and
documented object invariants are used, not a substitute numerical runtime.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, struct, traceback

CONTROLS='43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(name,path,pin):
    assert sha(path.read_bytes())==pin
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','observer','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True);parser.add_argument('--expected-source-sha256',required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={p.resolve():p.read_bytes() for p in (Path(__file__),a.binary,a.source,a.controls,a.observer)}
    assert sha(a.binary.read_bytes())==a.expected_sha256 and sha(a.source.read_bytes())==a.expected_source_sha256
    b=load('gradient_lifetime_codec',a.controls,CONTROLS);observer=load('gradient_lifetime_observer',a.observer,OBSERVER)
    def words(c,*command):return b.words(c.one(*command))
    def full(c,shape,value,requires=False):return c.make(38,len(shape),int(requires),*shape,*([0]*(4-len(shape))),b.bits(value))
    def blob(values):return b.Blob(struct.pack('<'+'Q'*len(values),*values))
    def optimizer(c,p,momentum=.5):
        return c.make(71,0,blob([b.bits(.125),b.bits(momentum),0,0,0,0]),blob([1,1,0,0,0,0,0,0,0,p]))
    def back(c,t,seed=0,retain=0):
        r=c.one(58,t,seed,retain);assert r['value']==0;rfields=words_record(r)
        assert rfields[0]==0;return r
    def words_record(r):return struct.unpack('<'+'Q'*(len(r['payload'])//8),r['payload'])
    def nodes(c):
        raw=c.session.owner()['bytes'];q=lambda at:struct.unpack_from('<Q',raw,at)[0]
        assert raw[64:72]==b'HTDTYP02' and raw[139264:139272]==b'HTODTY02'
        assert all(q(offset)==0 for offset in (112,3968,3976,3984,3992,4000,4008))
        pool=139264+144+552*q(139264+16);result={}
        for node in range(1,q(8)):
            page=q(4096+(node>>8)*8)
            if page==0:continue
            assert page>>63
            at=pool+(page&((1<<63)-1))+(node&255)*640
            assert all(q(at+field)==0 for field in range(448,576,8)),('temporary incoming field persisted',node)
            if q(at+304)==0:continue
            assert all(q(at+field)==0 for field in (24,248,336,368,376,416,424,432,440))
            # The inherited constructor may own a pre-backward work cache.
            # Incoming-value fields above must still be zero at every commit.
            work,hold=q(at+160),q(at+256)
            assert (work==0)==(hold==0),('work cache/Hold pair mismatch',node)
            if hold:
                assert hold>>32 and (hold>>24)&255==6 and work>>63
                hslot=hold&0xffffff;slots=q(139264+16)
                assert 0<hslot<=slots
                hrow=139264+128+(5*slots+hslot-1)*64
                assert q(hrow)==hold>>32 and q(hrow+8)==1 and q(hrow+24)==1
            record={field:q(at+field) for field in (8,32,40,192,200,208,216,224,312,320,360,384,392,400)}
            assert (record[360]==0)==(record[384]==0)
            if record[384]:
                gn=record[384];gp=q(4096+(gn>>8)*8);ga=pool+(gp&((1<<63)-1))+(gn&255)*640
                assert q(ga+304)==1 and q(ga+320)==record[392] and q(ga+400)>0
            result[node]=record
        c.comparisons+=len(result)
        return result
    def record(c,t):
        matches=[r for r in nodes(c).values() if r[192]==t and r[200]==1]
        assert len(matches)==1;return matches[0]
    def checks(c):c.audit();return len(nodes(c))
    report={'schema':'heptatorch.gradient-incoming-native-lifetimes.v1','status':'RUNNING',
        'native_sha256':a.expected_sha256,'source_sha256':a.expected_source_sha256,
        'runner_sha256':sha(Path(__file__).read_bytes()),'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()},
        'native_executed':True,'pinned_Torch_executed_here':False,'rows':[],
        'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def case(name,body,limit=128<<20):
        s=observer.OwnerSession(a.binary.resolve(),directory=out/name,address_limit=limit);c=b.Check(s)
        row={'case':name,'status':'RUNNING','address_limit':limit};report['rows'].append(row);save()
        try:
            row['details']=body(c);row['committed_live_nodes_checked']=checks(c);row['status']='PASS'
        except Exception as error:
            row.update(status='FAIL',exception=repr(error),traceback=traceback.format_exc())
            try:
                owner=s.owner();(out/name/'failure.owner.bin').write_bytes(owner['bytes'])
                row['failure_owner']={'bytes':owner['length'],'sha256':sha(owner['bytes'])}
            except Exception as observation_error:row['owner_observation_exception']=repr(observation_error)
        finally:
            try:row['transport']=s.finish()
            except Exception as error:row.update(status='FAIL',transport_exception=repr(error))
            row.update(frames=s.frames,exact_checks=c.comparisons,rollback_bytes=c.rollback_bytes,frame_records=c.frames);save()

    def gradient_chain(c):
        tensors=[c.new([],[float(i%8)]) for i in range(300)]
        for first,second in zip(tensors,tensors[1:]):c.one(92,first,second)
        c.one(92,tensors[-1],tensors[0],status=7,label='gradient ownership cycle is rejected atomically')
        weak=c.make(42,tensors[-1]);c.run([(41,t) for t in tensors[1:]])
        c.expiry(weak,False);middle=nodes(c)
        assert len(middle)==300 and sum(r[400] for r in middle.values())==299
        assert c.make(91,tensors[0])!=0
        public_second=c.make(91,tensors[0]);c.one(41,public_second)
        c.one(41,tensors[0]);c.expiry(weak,True);c.one(44,weak);c.one(33,0)
        assert not nodes(c)
        return {'chain_nodes':300,'private_gradient_refs':299,'cycle_rejected':True,
            'iterative_cascade_released_all_nodes':True,'more_than_one_descriptor_page':True}
    case('gradient-reference-chain-cycle-and-iterative-release',gradient_chain)

    def private_roots(c):
        p=c.new([2],[1.,-2.],True);g=c.new([2],[2.,-1.]);c.one(92,p,g)
        opt=optimizer(c,p);wp=c.make(42,p);wg=c.make(42,g)
        pm=words(c,13,p);gm=words(c,13,g);pi=record(c,p)[320];gi=record(c,g)[320]
        c.run([(41,p),(41,g)]);c.expiry(wp,False);c.expiry(wg,False)
        before=words(c,54);fillers=[full(c,[],1) for _ in range(300)]
        large=full(c,[512,1024],3)
        after=words(c,54);assert after[3]>before[3] and after[2]>before[2]
        c.one(33,0)
        c.one(13,p,status=1);c.one(13,g,status=1);c.one(13,large,status=1)
        renewed=c.make(75,opt,0,0);gradient=c.make(91,renewed)
        assert renewed!=p and gradient!=g
        assert record(c,renewed)[320]==pi and record(c,gradient)[320]==gi
        assert words(c,13,renewed)==pm and words(c,13,gradient)==gm
        c.one(72,opt);c.values(renewed,[.75,-1.875]);c.values(gradient,[2.,-1.])
        assert words(c,13,renewed)[13]==1 and words(c,13,gradient)[13]==0
        c.run([(78,opt),(41,renewed)]);c.expiry(wp,True);c.expiry(wg,False)
        c.one(41,gradient);c.expiry(wg,True);c.run([(44,wp),(44,wg)]);c.one(33,0)
        assert not nodes(c)
        return {'owner_bytes_before_growth':before[3],'owner_bytes_after_growth':after[3],
            'parameter_and_gradient_incarnations_preserved':True,'real_private_optimizer_root_survived_reset':True,
            'old_tokens_stayed_stale':True,'all_private_refs_released':True,'filler_descriptors':len(fillers),'large_data_bytes':2<<20}
    case('private-optimizer-and-gradient-roots-across-pages-growth-reset',private_roots)

    def saved_gradient(c):
        p=c.new([2],[1.,-2.],True);out=c.make(17,p,2,1,2,0,0);seed=c.new([1,2],[-0.,2.])
        back(c,out,seed);g=c.make(91,p);alias=c.make(19,g);saved=c.make(46,g);weak=c.make(42,g)
        gm=words(c,13,g);sm=words(c,13,seed);assert gm[11:14]==sm[11:14] and gm[13]==0
        c.run([(92,p,0),(41,g),(41,seed)]);c.expiry(weak,False);assert c.make(47,saved)==1
        c.one(12,alias,0,b.bits(4.));assert words(c,13,alias)[13]==1
        c.one(47,saved,status=6);c.one(49,saved,0,2,status=6);c.one(56,saved,status=6)
        c.values(alias,[4.,2.]);c.one(41,alias);c.expiry(weak,False)
        c.one(48,saved);c.expiry(weak,True);c.one(44,weak)
        c.run([(41,out),(41,p)]);c.one(33,0);assert not nodes(c)
        return {'saved_has_real_shared_gradient_Version':True,'mutation_rejected_saved_check_read_unpack':True,
            'saved_storage_outlives_all_public_gradient_aliases':True,'last_saved_release_expires_storage':True}
    case('saved-gradient-shared-Version-and-last-reference-release',saved_gradient)

    def transactions(c):
        p=c.new([2],[1.,-2.],True);out=c.make(17,p,2,1,2,0,0);seed=c.new([1,2],[1.,2.])
        before=c.session.owner();c.run([(58,out,seed,0),(999,)],status=4,label='later error rolls back genuine Storage transfer')
        assert c.session.owner()==before and c.make(91,p)==0 and c.make(59,p)==0
        back(c,out,seed);g=c.make(91,p);gm=words(c,13,g);sm=words(c,13,seed);assert gm[11:14]==sm[11:14]
        c.run([(92,p,0),(58,out,seed,0),(999,)],status=4,label='clear and rematerialization roll back together')
        assert c.make(91,p)==g and words(c,13,g)==gm
        for field in (3,4,5,6,7):
            cmd=[91,p,0,0,0,0,0,0];cmd[field]=1;c.run([cmd],status=7)
        for command,status in [((91,1),1),((92,p,p),2),((93,p,1),7),((94,p,1),7)]:c.run([command],status=status)
        for entry in (7,29,57,58):
            x=c.new([],[2.],True);coefficient=c.new([],[3.]);y=c.make(2,3,x,coefficient);s=c.new([],[1.])
            call=(7,y) if entry==7 else (29,y,s) if entry==29 else (entry,y,s,1)
            c.one(*call);grad=c.make(91,x);ident=record(c,grad)[320];c.values(grad,[3.]);assert words(c,13,grad)[13]==0
            c.one(*call);assert c.make(91,x)==grad and record(c,grad)[320]==ident;c.values(grad,[6.]);assert words(c,13,grad)[13]==1
            c.one(70,x);assert c.make(91,x)==0;c.one(*call)
            new=c.make(91,x);assert new!=grad and record(c,new)[320]!=ident
            owner=c.session.owner()['bytes'];u=lambda at:struct.unpack_from('<Q',owner,at)[0]
            for node,r in nodes(c).items():
                if r[192] in (x,y):
                    page=u(4096+(node>>8)*8);base=139264+144+552*u(139264+16)+(page&((1<<63)-1))+(node&255)*640
                    assert u(base+160)==u(base+256)==0
            c.values(new,[3.]);c.values(grad,[6.]);assert words(c,13,new)[13]==0;checks(c)
        return {'frame_failure_preserves_entire_owner':True,'four_backward_entries_materialize_real_gradient_objects':True,
            'retained_graph_has_no_committed_temporary_cache_or_Version':True,'retained_93_94_reserved_fields_rejected':True}
    case('atomic-admission-and-four-backward-entry-materialization',transactions)

    def actual_oom(c):
        x=full(c,[256,512],2,True);loss=c.make(4,7,x);back(c,loss);c.one(70,x)
        assert c.make(91,x)==0
        large=full(c,[1024,1024],1);fillers=[full(c,[256,1024],1) for _ in range(2)]
        before=c.session.owner();(out/'real-allocation-failure-and-recovery'/'before.owner.bin').write_bytes(before['bytes'])
        c.one(58,loss,0,0,status=3,label='real allocator failure while reserving incoming values and gradient')
        after=c.session.owner();assert after==before
        (out/'real-allocation-failure-and-recovery'/'after.owner.bin').write_bytes(after['bytes'])
        assert c.make(59,x)==0 and record(c,x)[384]==0;checks(c)
        c.one(41,fillers[-1]);back(c,loss);assert c.make(59,x)==1
        for start in range(0,131072,16384):
            r=c.one(40,x,start,16384);assert r['payload']==struct.pack('<I',b.bits(1.))*16384;c.comparisons+=16384
        g=c.make(91,x);assert words(c,13,g)[13]==1;c.one(41,g);c.one(70,x);assert c.make(91,x)==0
        return {'actual_address_space_limit_bytes':32<<20,'entire_owner_unchanged_bytes':before['length'],
            'owner_sha256_before':sha(before['bytes']),'owner_sha256_after':sha(after['bytes']),
            'unready_gradient_objects_absent_after_failure':True,'recovered_after_releasing_unrelated_extent':True,
            'recovered_exact_gradient_words':131072,'live_large_tensor':large}
    case('real-allocation-failure-and-recovery',actual_oom,32<<20)
    assert all(path.read_bytes()==raw for path,raw in inputs.items())
    report.update(status='FAIL' if any(row['status']=='FAIL' for row in report['rows']) else 'PASS',
        cases=len(report['rows']),passed=sum(row['status']=='PASS' for row in report['rows']),
        frames=sum(row['frames'] for row in report['rows']),exact_checks=sum(row['exact_checks'] for row in report['rows']),
        rollback_bytes=sum(row['rollback_bytes'] for row in report['rows']))
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('rows','inputs_sha256')},indent=2))
    return int(report['status']!='PASS')

if __name__=='__main__':raise SystemExit(main())
