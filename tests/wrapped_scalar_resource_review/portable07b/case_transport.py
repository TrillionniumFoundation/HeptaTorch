"""Exact inherited Case class; original six recipes are not run."""
from pathlib import Path
import copy,gzip,json,struct
import owner_view05 as observer
from owner_state import State
from support import create_packet,raw_file,typed_bytes,dump,sha

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
