#!/usr/bin/env python3
"""Actual Saved-original-T ownership, cleanup and copied adversarial controls."""
import argparse,copy,gzip,json,struct,traceback
from pathlib import Path
import owner640_mixed10 as observer
from native_case import Case
from resource_support import Custody,load,sha,write_json


def f32(c,label,shape,values,requires=0):
    return c.typed(label,0,shape,values,requires)


def backward(c,label,root,seed,retain=0):
    record=c.one(label,58,root,seed,retain,structural_only=True)
    c.check(label+':runtime error',struct.unpack('<4Q',record['payload'])[0],0)


def mixed_view_release(c):
    base=c.typed('integer backing',2,[5],[0,3,0,5,0])
    view=c.view('real integer holes view',base,[2],[2],1)
    x=f32(c,'F32 requires', [2],[2.,4.],1)
    gen=c.one('preserved Generator',60,1729)['value']
    weak=c.one('observe actual original Storage',42,view)['value']
    base_i,base_n=c.node(base);view_i,view_n=c.node(view)
    y=c.one('mixed multiply graph',103,3,x,view)['value']
    root_i,root_n=c.node(y)
    snap=c.mark('mixed_saved_view')
    c.check('Saved holds original I64 view',snap['binary_saved_roots'][str(root_i)],
            [0,0,view_i,view_n[40]])
    c.check('Saved private Hold distinct from input nodes',root_n[73]>65535,True)
    c.check('no saved self when only x requires',c.one('getter absent',107,y,0)['value'],0)
    c.one('getter bad slot',107,y,2,expected=7)
    c.one('getter reserved',107,y,1,1,expected=7)
    c.one('drop original public base',41,base);c.one('drop original public view',41,view)
    held=c.one('publish original Saved view',107,y,1)['value']
    hi,hn=c.node(held)
    c.check('getter is genuine original T/S/V/inc',(hi,hn[27],hn[28],hn[40]),
            (view_i,view_n[27],view_n[28],view_n[40]))
    seed=f32(c,'seed',[2],[1.,1.])
    before=c.mark('held_before_default_backward')
    backward(c,'default backward',y,seed)
    after=c.mark('held_after_default_backward')
    c.check('whole5056 RNG unchanged across backward',after['generator_states'][str(gen)],before['generator_states'][str(gen)])
    c.check('actual I64 coefficient derivative',c.one('gradient raw',40,x,0,2)['payload'],struct.pack('<2f',3.,5.))
    c.check('runtime releases Saved private Hold',(after['nodes'][str(root_i)][73],after['nodes'][str(root_i)][74]),(0,0))
    c.one('spent actual Saved accessor',107,y,1,expected=10)
    c.one('absent remains absent when released',107,y,0)
    c.one('release published Saved view',41,held)
    dead=c.mark('rowless_view_metadata_after_public_drop')
    node=dead['nodes'][str(view_i)]
    c.check('metadata node is still held by disabled graph edge',(node[38],node[26]),(1,1))
    c.check('metadata node no longer owns a Tensor row/base',(node[30],node[73],node[74]),(0,0,0))
    c.check('ultimate base Tensor is really reclaimed',dead['nodes'][str(base_i)][38],0)
    c.one('original Storage really expired',43,weak,expected=6)
    c.one('drop Weak',44,weak);c.one('clear grad',70,x)
    for label,t in [('root',y),('seed',seed),('leaf',x)]:c.one('drop '+label,41,t)
    c.one('drop Generator',66,gen)


def duplicate_slots(c):
    x=f32(c,'shared input',[2],[2.,3.],1)
    y=c.one('pure F32 x*x',2,3,x,x,structural_only=True)['value']
    xi,xn=c.node(x);yi,yn=c.node(y)
    snapshot=c.mark('duplicate_original_T_slots')
    c.check('two original pairs really share input',snapshot['binary_saved_roots'][str(yi)],
            [xi,xn[40],xi,xn[40]])
    c.check('two actual private references',snapshot['nodes'][str(xi)][75],2)
    c.check('two graph plus two actual owner references',snapshot['nodes'][str(xi)][26],4)
    c.check('first getter actual identity',c.one('get self',107,y,0)['value'],x)
    c.check('second getter actual identity',c.one('get other',107,y,1)['value'],x)
    c.one('drop graph releases both pairs',41,y)
    _,after=c.node(x)
    c.check('both private and graph references gone',(after[75],after[26],after[11]),(0,0,0))
    c.one('drop shared input',41,x)


def gradient_detach_cleanup(c):
    x=f32(c,'parameter',[2],[7.,9.],1)
    input=f32(c,'source requires',[2],[2.,3.],1)
    q=c.one('real nonleaf public gradient',2,3,input,input,structural_only=True)['value']
    qi,qn=c.node(q);ii,inode=c.node(input)
    c.one('assign original nonleaf as grad',92,x,q)
    before=c.mark('assigned_G_with_two_Saved_roots')
    c.check('grad points to exact public q',before['nodes'][str(c.node(x,before)[0])][48],qi)
    c.check('input owned twice by q',before['nodes'][str(ii)][75],2)
    c.one('zero_grad false detaches actual G',9,x,structural_only=True)
    after=c.mark('assigned_G_after_zero_false')
    node=after['nodes'][str(qi)]
    c.check('G retains its T/S/V and loses graph',(node[30],node[27],node[28],node[5],node[4]),
            (1,qn[27],qn[28],0,0))
    c.check('new Saved roles really released',(node[73],node[74],node[35],node[36]),(0,0,0,0))
    c.check('original source no hidden private refs',(after['nodes'][str(ii)][75],after['nodes'][str(ii)][26]),(0,0))
    c.check('original G zeroed natively',c.one('read zero G',98,q,0,2)['payload'],bytes(8))
    c.one('clear parameter grad',70,x)
    for label,t in [('q',q),('input',input),('x',x)]:c.one('drop '+label,41,t)


def saved_version_and_history(c):
    x=f32(c,'leaf',[0],[],1)
    q=c.one('empty pureF32 graph',2,3,x,x,structural_only=True)['value']
    qi,qn=c.node(q);xi,xn=c.node(x)
    before=c.mark('empty_binary_history_before')
    c.one('detach real nonleaf in place',101,q)
    after=c.mark('binary_history_transfer')
    oldhold=qn[73]
    prior=[i for i,n in after['nodes'].items() if n[73]==oldhold]
    # No graph consumer needs the old graph in this case: actual detach may
    # collect the transferred history immediately. The release is exact.
    c.check('unneeded history Hold released',prior,[])
    c.check('target stays exactT/S/V/inc',tuple(after['nodes'][str(qi)][j] for j in (24,27,28,40)),
            tuple(qn[j] for j in (24,27,28,40)))
    c.check('input private roots cleared',after['nodes'][str(xi)][75],0)
    c.one('drop detached target',41,q);c.one('drop leaf',41,x)
    # Independently exercise the actual Version guard, with full state rollback.
    i=c.typed('I64 saved mutation input',2,[2],[2,4])
    f=f32(c,'F32 graph input',[2],[1.,1.],1)
    y=c.one('version guarded mixed root',103,3,f,i)['value']
    c.one('mutate shared I64 Version once',106,i,1)
    c.one('getter Version failure is atomic',107,y,1,expected=6)
    c.one('drop version-failed graph',41,y)
    c.one('drop integer',41,i);c.one('drop float',41,f)


def negatives(cases,out):
    groups={c.row['name']:c for c in cases}
    sample=groups['duplicate-slots'].saved['duplicate_original_T_slots']
    owner=next(i for i,n in sample['nodes'].items() if n[73]>65535)
    sn=sample['nodes'][owner];hid=sn[73];pairs=sample['binary_saved_roots'][owner]
    input=str(pairs[0]);stored=sn[35]
    rows=[]
    def rehash(s):
        s['hold_regions'][str(hid)]['sha256']=sha(struct.pack('<4Q',*s['binary_saved_roots'][owner]))
    tests=[]
    def add(name,fn,rule,base=sample):tests.append((name,fn,rule,base))
    def ns(word,value):return lambda s:s['nodes'][owner].__setitem__(word,value)
    add('private Hold pool offset forged',ns(74,sn[74]+16),'binary-Saved-relative-address')
    add('Saved owner lost requires',ns(4,0),'binary-Saved-owner-role')
    add('spent graph retained Saved owner',ns(41,1),'binary-Saved-owner-role')
    add('nonbinary owner retains role',ns(5,1),'binary-Saved-owner-role')
    add('pair missing from decoded inventory',lambda s:s['binary_saved_roots'].pop(owner),'binary-Saved-pair-coverage')
    add('extra pair index outside actual owner',lambda s:s['binary_saved_roots'].__setitem__('65535',[0]*4),'binary-Saved-pair-coverage')
    add('pair bytewidth wrong',lambda s:s['binary_saved_roots'][owner].pop(),'binary-Saved-pair-width')
    add('raw private Hold not matching pairs',lambda s:s['binary_saved_roots'][owner].__setitem__(1,pairs[1]+1),'binary-Saved-Hold-content')
    def stale(s):s['binary_saved_roots'][owner][1]+=1;rehash(s)
    add('incarnation cannot be fixed by rehash',stale,'binary-Saved-original-Tensor-incarnation')
    def absent(s):s['binary_saved_roots'][owner][0]=0;s['binary_saved_roots'][owner][1]=0;rehash(s)
    add('real Saved slot cannot lose Tensor owner',absent,'binary-Saved-slot-presence')
    add('missing target600 multiplicity',lambda s:s['nodes'][input].__setitem__(75,1),'private-strong-Tensor-count')
    add('missing target208 multiplicity',lambda s:s['nodes'][input].__setitem__(26,3),'node-reference-count')
    add('missing private Hold role',lambda s:s['hold_regions'].pop(str(hid)),'private-hold-coverage')
    hr=(hid&0xffffff)-1
    add('private Hold forged public',lambda s:s['tables']['6'][hr].__setitem__(3,0),'private-hold-coverage')
    add('private Hold declares integer payload dtype',lambda s:s['tables']['6'][hr].__setitem__(4,2),'private-hold-role')
    sv=(stored&0xffffff)-1
    add('Saved offset no longer originalT',lambda s:s['tables']['5'][sv].__setitem__(5,4),'binary-Saved-original-S-V-offset')
    rowless=groups['mixed-view-release'].saved['rowless_view_metadata_after_public_drop']
    view=next(i for i,n in rowless['nodes'].items() if n[38] and n[72]==2 and n[51]==1 and n[30]==0)
    add('metadata view must not regain actual base edge',lambda s:s['nodes'][view].__setitem__(73,1),
        'I64-rowless-view-base-retained',rowless)
    for name,fn,rule,base in tests:
        original=observer.canonical(base);observer.verify_snapshot(base)
        altered=copy.deepcopy(base);fn(altered)
        observed=None
        try:observer.verify_snapshot(altered)
        except AssertionError as e:
            if e.args and isinstance(e.args[0],dict):observed=e.args[0].get('observer_rule')
        row={'name':name,'expected_rule':rule,'observed_rule':observed,'native_snapshot_unchanged':observer.canonical(base)==original}
        rows.append(row)
        with gzip.open(out/f'negative-{len(rows)}.json.gz','wb') as f:f.write(observer.canonical(altered))
        write_json(out/'negative-controls.json',{'status':'RUNNING','cases':rows})
        assert observed==rule,row
        assert row['native_snapshot_unchanged']
    result={'status':'PASS','cases':rows,'negative_count':len(rows),'new_native_executions':0,'target_memory_written':False}
    write_json(out/'negative-controls.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    custody=Custody(out);cases=[]
    report={'status':'RUNNING','source_sha256':observer.SOURCE,'native_sha256':observer.IMAGE,
            'profile':observer.PROFILE,'cases':[],'new_Torch_reference_cases':0,'whole_pytorch':'HOLD',
            'pytorch_api_denominator':2590,'closed_schema_count':0,'full_pytorch_rewrite_complete':False}
    try:
        for name in ('test_saved_owner10.py','native_case.py','owner640_mixed10.py','build_profile.py','adaptation.json'):
            custody.bind(Path(__file__).with_name(name),copy='observer/'+name)
        wire,image,info,_=load(custody,a.build,a.protocol,expected_source=observer.SOURCE,expected_image=observer.IMAGE)
        observer.pin_profile(image,a.build/'runtime.hepta')
        for name,fn in [('mixed-view-release',mixed_view_release),('duplicate-slots',duplicate_slots),
                        ('gradient-detach-cleanup',gradient_detach_cleanup),('Saved-Version-history',saved_version_and_history)]:
            c=Case(wire,image,out,name,report);cases.append(c)
            try:fn(c);c.close()
            except BaseException:
                c.row['status']='FAIL';c.row['traceback']=traceback.format_exc()
                try:
                    if c.s.poll() is None:c.s.finish()
                except BaseException:c.row['cleanup_traceback']=traceback.format_exc()
                c.snapshot_stream.close();raise
            write_json(out/'report.json',report)
        report['negative_controls']=negatives(cases,out)
        report['checked_inputs']=custody.finish();report['status']='PASS'
    except BaseException:report.update(status='FAIL',traceback=traceback.format_exc())
    for key in ('frames_before_EOF','snapshot_checks','isolated_transition_checks','ordinary_rollback_bytes','checks'):
        report[key]=sum((c.s.frames if key=='frames_before_EOF' else c.row[key]) for c in cases)
    write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','checked_inputs','negative_controls','traceback')}))
    if report['status']!='PASS':print(report.get('traceback'));raise SystemExit(1)


if __name__=='__main__':main()
