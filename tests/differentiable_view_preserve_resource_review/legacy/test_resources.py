"""Actual to-copy resource/ownership controls; not a numerical runtime."""
from pathlib import Path
import argparse,copy,gzip,json,struct,traceback
import owner_view05 as observer
from owner_state import State
from support import SOURCE,IMAGE,REFERENCE,Custody,load,create_packet,raw_file,typed_bytes,dump,sha


class Case:
    def __init__(self,name,wire,image,out,report,limit=None):
        self.out=out/name;self.out.mkdir();self.w=wire
        self.row={'name':name,'status':'RUNNING','checks':0,'snapshots':[], 'events':[],
                  'rollback_bytes':0,'RLIMIT_AS_bytes':limit,'negative_controls':[]}
        report['cases'].append(self.row)
        self.s=wire.Session(image,directory=self.out/'transport',address_limit=limit)
        self.gen=self.one('seeded Generator',60,1729)['value']
        self.rng0=self.state().rng(self.gen)
    def check(self,name,actual,expected):
        def fmt(x):return {'bytes':len(x),'sha256':sha(x)} if isinstance(x,bytes) else x
        assert actual==expected,(name,fmt(actual),fmt(expected))
        self.row['checks']+=1
    def mark(self,label):
        owner=self.s.owner();raw=owner['bytes'];snap=observer.decode_owner(raw,owner['pointer'])
        result=observer.verify_snapshot(snap)
        event={'label':label,'owner':raw_file(self.out,raw),'result':result}
        path=self.out/(str(len(self.row['snapshots'])).zfill(3)+'.snapshot.json.gz')
        with path.open('wb') as f:
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(observer.canonical(snap))
        event['snapshot_file']=path.name;event['snapshot_sha256']=sha(observer.canonical(snap))
        self.row['snapshots'].append(event)
        return snap
    def one(self,label,*command):
        record,=self.s.run([command]);self.mark(label);return record
    def state(self):return State(self.s.owner()['bytes'])
    def meta(self,t):return self.w.meta(self.one('metadata',13,t))
    def node(self,t):return self.state().tensor_node(t)
    def typed(self,dtype,shape,values=None,requires=0,raw=None,opcode=97):
        raw=typed_bytes(dtype,values) if raw is None else raw
        status,records=self.w.parse(self.s.exchange(create_packet(dtype,shape,raw,requires,opcode)))
        self.check('constructor status',status,0);self.check('constructor records',len(records),1)
        self.mark('typed construction');return records[0]['value']
    def view(self,t,shape,strides,offset=0,opcode=111):
        config=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(strides))),offset)
        return self.one('layout alias',opcode,t,self.w.Payload(config),80)['value']
    def read(self,t,count):return self.one('actual raw read',98,t,0,count)['payload']
    def refusal(self,label,commands,expected,packet=None,maps=False):
        before=self.s.owner();rng=self.state().rng(self.gen)
        map_path=Path(f'/proc/{self.s.pid}/maps');mb=map_path.read_bytes() if maps else None
        raw=self.s.exchange(self.w.packet(commands) if packet is None else packet)
        status,records=self.w.parse(raw);after=self.s.owner()
        ma=map_path.read_bytes() if maps else None
        self.row['events'].append({'name':label,'status':status,'expected':expected,
            'before':raw_file(self.out,before['bytes']),'after':raw_file(self.out,after['bytes'])})
        self.check(label+' status',status,expected);self.check(label+' records',records,[])
        self.check(label+' whole owner',after,before);self.check(label+' full RNG',self.state().rng(self.gen),rng)
        if maps:
            (self.out/(label+'.maps-before')).write_bytes(mb);(self.out/(label+'.maps-after')).write_bytes(ma)
            self.check(label+' exact mappings',ma,mb)
        self.row['rollback_bytes']+=len(before['bytes']);self.mark(label)
    def negative(self,label,snapshot,mutate,rule):
        changed=copy.deepcopy(snapshot);mutate(changed)
        raw=observer.canonical(changed);path=self.out/(label+'.negative.json.gz')
        with gzip.open(path,'wb') as f:f.write(raw)
        try:observer.verify_snapshot(changed)
        except AssertionError as e:
            self.check(label+' exact refusal',e.args[0]['observer_rule'],rule)
        else:raise AssertionError((label,'copied corruption admitted'))
        self.row['negative_controls'].append({'name':label,'rule':rule,'status':'PASS_EXPECTED_REJECTION',
                                              'file':path.name,'input_sha256':sha(raw)})
    def closed(self):
        self.check('entire5056 RNG unchanged',self.state().rng(self.gen),self.rng0)
        self.one('drop Generator',66,self.gen)
        self.check('final native committed audit',self.one('audit',53)['value'],1)
        self.row['final_pages_only']=self.state().final_pages_only();self.mark('final')
    def finish(self):
        self.row['frames_before_EOF']=self.s.frames
        self.s.finish();self.row['EOF_exit_code']=self.s.exit_code


def base_detach_copy_lifetime(c,ref):
    for dtype in (0,1,2):
        base=c.typed(dtype,[5],[0,1,0,1,0])
        view=c.view(base,[2],[2],1);b,bn=c.node(base);v,vn=c.node(view)
        copy_t=c.one('new independent Storage copy',109,view,dtype,0)['value']
        alias=c.one('real detached Storage alias',19,view)['value']
        alias_n=c.node(alias)[1]
        c.check('detach is not a view root', (alias_n[51],alias_n[80],alias_n[81]),(0,0,0))
        wb=c.one('base weak Storage',42,base)['value'];wc=c.one('copy weak Storage',42,copy_t)['value']
        initial=c.mark('typed_base_roots_'+str(dtype));n=initial['nodes'][str(b)]
        c.check('actual private base count',(n[75],n[26]),(1,1))
        c.check('real base node/inc',(vn[80],vn[81]),(b,bn[40]))
        c.check('copy has actual typed byte extent',c.meta(copy_t)['storage_bytes'],2*(4,1,8)[dtype])
        c.check('copy Storage differs',c.meta(copy_t)['storage_id']!=c.meta(base)['storage_id'],True)
        before=c.state();same=c.one('sameT to no-copy',108,view,dtype,0,1)['value'];after=c.state()
        c.check('suggest-format Contiguous aliases real view',same,view)
        c.check('no-copy allocates no Tensor/incarnation',after.next,before.next)
        c.check('no-copy live record counts unchanged',after.counts(),before.counts())
        c.negative('base-incarnation-'+str(dtype),initial,lambda s:s['nodes'][str(v)].__setitem__(81,bn[40]+1),'typed-base-incarnation')
        c.negative('base-reference-count-'+str(dtype),initial,lambda s:s['nodes'][str(b)].__setitem__(75,2),'private-strong-Tensor-count')
        c.one('drop public original base',41,base)
        c.check('private root keeps base T row',c.state().node(b)[30],1)
        c.one('drop final true view',41,view)
        dead=c.state().node(b)
        c.check('base private T is reclaimed',(dead[38],dead[30],dead[75]),(0,0,0))
        hold=c.one('detached alias independently retains Storage',43,wb)['value'];c.one('release lock Hold',45,hold)
        c.one('drop detached alias',41,alias);c.refusal('base expired dtype'+str(dtype),[(43,wb)],6)
        c.check('copy data remains independent',c.read(copy_t,2),typed_bytes(dtype,[1,1]))
        c.one('drop independent copy',41,copy_t);c.refusal('copy expired dtype'+str(dtype),[(43,wc)],6)
        c.one('drop base Weak',44,wb);c.one('drop copy Weak',44,wc)
    c.closed()


def raw_bool_boundary(c,ref):
    row=ref['rows'][43];assert row['name']=='special-bool-to-bool'
    raw=bytes.fromhex(row['input_before']['storage_raw_hex'])
    c.refusal('old97 rejects noncanonical bool',[],7,packet=create_packet(1,[10],raw))
    x=c.typed(1,[10],raw=raw,opcode=110)
    before=c.meta(x)
    same=c.one('raw identity no-copy',108,x,1,0,0)['value'];c.check('raw sameT',same,x)
    c.check('identity preserves original noncanonical bytes',c.read(x,10),raw)
    expected=next(r for r in row['conversions'] if r['flavor']=='aten_to_copy')['output']
    y=c.one('actual bool typed copy',109,x,1,0)['value']
    c.check('fixed raw bool normalization',c.read(y,10),bytes.fromhex(expected['storage_raw_hex']))
    c.check('copy exact extent',c.meta(y)['storage_bytes'],10)
    c.one('mutate source after copy',100,x,0,1)
    c.check('source V increment',c.meta(x)['version'],before['version']+1)
    c.check('copy V independent',c.meta(y)['version'],0)
    c.check('copy data independent',c.read(y,10),bytes.fromhex(expected['storage_raw_hex']))
    for t in (x,y):c.one('drop bool Tensor',41,t)
    c.closed()


def no_saved_and_prefix(c,ref):
    assert ref['rows'][45]['saved_before']=={'grad_fn':'ToCopyBackward0','attributes':{}}
    x=c.typed(0,[2],[2.,3.],requires=1);seed=c.typed(0,[2],[1.,1.])
    y=c.one('real ToCopyBackward',109,x,0,0)['value'];yi,yn=c.node(y)
    c.check('real op43 and no Saved',(yn[5],yn[35],yn[36],yn[37],yn[73],yn[74]),(43,0,0,0,0,0))
    snap=c.mark('op43_no_Saved')
    c.negative('op43-invented-Saved',snap,lambda s:s['nodes'][str(yi)].__setitem__(35,1),'ToCopy-no-Saved-Tensor')
    for n in range(1,4):
        r=c.one('repeat no-Saved backward',58,y,seed,0)
        c.check('repeat native success',struct.unpack('<4Q',r['payload'])[0],0)
        c.check('real accumulated derivative',c.one('leaf gradient',40,x,0,2)['payload'],struct.pack('<2f',n,n))
    c.one('clear leaf gradient',70,x)
    for t in (y,x,seed):c.one('drop repeat Tensor',41,t)
    x=c.typed(0,[2],[2.,4.],requires=1);factor=c.typed(0,[2],[3.,5.])
    q=c.one('upstream Saved multiply',2,3,x,factor)['value'];y=c.one('downstream ToCopy',109,q,0,0)['value']
    seed=c.typed(0,[2],[1.,1.]);sentinel=c.typed(0,[1],[7.])
    c.one('retain downstream gradient',93,y)
    c.one('invalidate upstream real Saved Version',95,factor,0x3f800000)
    raw=c.s.exchange(c.w.packet([(95,sentinel,0x3f800000),(58,y,seed,0),(95,sentinel,0x42c80000)]))
    status,records=c.w.parse(raw)
    c.check('58 normal header',status,0);c.check('58 stops suffix',[r['opcode'] for r in records],[95,58])
    c.check('upstream actual version error',struct.unpack('<4Q',records[1]['payload'])[0],6)
    c.check('prefix single write committed',c.read(sentinel,1),struct.pack('<f',8.))
    c.check('prefix Version once',c.meta(sentinel)['version'],1)
    after=c.mark('retained_prefix_and_incoming_cleanup');yi,yn=c.node(y)
    c.check('ToCopy no Saved after upstream failure',(yn[35],yn[36],yn[37],yn[73],yn[74]),(0,0,0,0,0))
    g=c.one('actual retained prefix G',91,y)['value'];c.check('retained gradient bytes',c.read(g,2),struct.pack('<2f',1.,1.))
    c.refusal('57 same failure is whole-frame atomic',[(95,sentinel,0x3f800000),(57,y,seed,0),(95,sentinel,0x42c80000)],6)
    c.one('drop public retained G',41,g)
    c.one('reset all graph roots',33,0)
    c.closed()


def ordinary_admission(c,ref):
    f=c.typed(0,[2],[1.,2.]);i=c.typed(2,[2],[1,2]);r=c.typed(0,[2],[1.,2.],requires=1)
    nonleaf=c.one('requires copy node',109,r,0,0)['value']
    cfg=struct.pack('<10Q',1,2,0,0,0,1,0,0,0,0)
    tests=[('bad target dtype',(108,f,3,0,0),7),('bad copy flag',(108,f,0,2,0),7),
        ('bad format',(109,f,0,2),7),('reserved108',(108,f,0,0,0,1),7),
        ('reserved109',(109,f,0,0,1),7),('invalid handle',(108,1,0,0,0),1),
        ('111 requires source',(111,r,c.w.Payload(cfg),80),5),('111 nonleaf source',(111,nonleaf,c.w.Payload(cfg),80),5),
        ('111 wrong layout length',(111,f,c.w.Payload(cfg),79),7)]
    for name,command,code in tests:
        c.refusal(name,[(95,f,0x3f800000),(67,c.gen,9),command],code)
    c.refusal('old97 nonfloat requires',[],5,packet=create_packet(2,[1],typed_bytes(2,[1]),1))
    c.refusal('new110 nonfloat requires',[],5,packet=create_packet(1,[1],b'\x02',1,110))
    c.refusal('new110 payload truncated',[],100,packet=create_packet(1,[2],b'\x02\x03',0,110)[:-1])
    detached=c.view(f,[2],[1],opcode=69)
    c.check('old69 remains detached alias',c.node(detached)[1][51],0)
    c.one('reset admission roots',33,0);c.closed()


def empty_view_metadata(c,ref):
    x=c.typed(0,[0],raw=b'');v=c.view(x,[2,0,3],[17,101,2],7)
    xi,xn=c.node(x);vi,vn=c.node(v);weak=c.one('empty source weak',42,x)['value']
    c.check('empty real offset',c.meta(v)['storage_offset'],28)
    c.one('view becomes real requires leaf',90,v,1)
    q=c.one('empty real ToCopy graph',109,v,0,0)['value']
    c.one('drop base public',41,x);c.one('drop view public',41,v)
    seed=c.typed(0,[2,0,3],raw=b'')
    result=c.one('empty ToCopy backward',58,q,seed,0)
    c.check('empty backward success',struct.unpack('<4Q',result['payload'])[0],0)
    c.one('drop copy graph',41,q)
    c.refusal('empty original Storage expired after last captured root',[(43,weak)],6)
    c.one('drop empty seed',41,seed);c.one('drop empty Weak',44,weak)
    c.closed()


def actual_oom_recovery(c,ref):
    count=65536;x=c.typed(1,[count],raw=b'\x01'*count)
    fillers=[]
    while True:
        state=c.state();free=sum(size for _,size in state.free())
        if state.pool==16<<20 and 262144<=free<524288:break
        assert state.pool<=16<<20 and len(fillers)<80,('bounded real OOM setup',state.pool,free,len(fillers))
        fillers.append(c.typed(2,[32768],raw=bytes(262144)))
    c.row['setup']={'pool':state.pool,'free_bytes':free,'fillers':len(fillers),'owner_bytes':len(state.raw)}
    (c.out/'child-limits.txt').write_bytes(Path(f'/proc/{c.s.pid}/limits').read_bytes())
    (c.out/'child-status.txt').write_bytes(Path(f'/proc/{c.s.pid}/status').read_bytes())
    c.refusal('copy_true_real_OOM',[(108,x,2,1,0)],3,maps=True)
    c.refusal('to_copy_real_OOM',[(109,x,2,0)],3,maps=True)
    for _ in range(3):c.one('release real interior filler',41,fillers.pop())
    y=c.one('same conversion recovers',109,x,2,0)['value']
    c.check('recovery no pool growth',c.state().pool,16<<20)
    c.check('actual full I64 converted payload',c.read(y,count),struct.pack('<Q',1)*count)
    c.check('actual eight-byte destination',c.meta(y)['storage_bytes'],count*8)
    c.one('drop recovered copy',41,y)
    for t in fillers+[x]:c.one('drop OOM source/filler',41,t)
    c.closed()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-build','protocol','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--only',choices=['base_detach_copy_lifetime','raw_bool_boundary','no_saved_and_prefix','ordinary_admission','empty_view_metadata','actual_oom_recovery'])
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    custody=Custody();report={'schema':'heptatorch.to-copy-independent-resources05.v1','status':'RUNNING',
        'source_sha256':SOURCE,'native_sha256':IMAGE,'reference_sha256':REFERENCE,'cases':[],
        'native_executed':False,'Torch_executed_here':False,'whole_pytorch_gate':'HOLD','full_pytorch_parity':False}
    try:
        custody.bind(Path(__file__));wire,image,build,ref=load(custody,a.native_build,a.protocol,a.reference)
        plan=[(base_detach_copy_lifetime,None),(raw_bool_boundary,None),(no_saved_and_prefix,None),
              (ordinary_admission,None),(empty_view_metadata,None),(actual_oom_recovery,56<<20)]
        report['declared_case_names']=[f.__name__ for f,_ in plan]
        for f,limit in plan:
            if a.only and f.__name__!=a.only:continue
            c=None
            try:
                c=Case.__new__(Case)
                c.__init__(f.__name__,wire,image,out,report,limit);report['native_executed']=True
                f(c,ref);c.row['status']='PASS'
            except BaseException as error:
                row=c.row if c else report['cases'][-1]
                row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
                if c and hasattr(c,'s'):
                    try:row['failed_owner']=raw_file(c.out,c.s.owner()['bytes'])
                    except BaseException as extra:row['failed_owner_read_error']=repr(extra)
            finally:
                if c and hasattr(c,'s'):
                    try:c.finish()
                    except BaseException as error:c.row.update(status='FAIL',transport_error=repr(error))
                    dump(c.out/'report.json',c.row)
                dump(out/'report.json',report)
        report['status']='PASS' if report['cases'] and all(r['status']=='PASS' for r in report['cases']) else 'FAIL'
    except BaseException as error:report.update(status='FAIL',preparation_error=repr(error),traceback=traceback.format_exc())
    try:report['inputs']=custody.finish();report['inputs_after_unchanged']=True
    except BaseException as error:report.update(status='FAIL',custody_error=repr(error))
    report.update(case_count=len(report['cases']),passed=sum(r['status']=='PASS' for r in report['cases']),
        frames_before_EOF=sum(r.get('frames_before_EOF',0) for r in report['cases']),
        checks=sum(r['checks'] for r in report['cases']),copied_negative_cases=sum(len(r['negative_controls']) for r in report['cases']),
        full_owner_rollback_bytes=sum(r['rollback_bytes'] for r in report['cases']))
    dump(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','case_count','passed','frames_before_EOF','checks','copied_negative_cases','preparation_error')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
