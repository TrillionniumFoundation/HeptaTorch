"""Own-child transport controls; host observer only, no target math."""
import copy,gzip,json,struct
from pathlib import Path
import owner640_mixed10 as observer
from resource_support import create_packet,integer_raw,dump_raw,sha

class Case:
    def __init__(self, wire, image, output, name, report):
        self.w = wire
        self.directory = output/name
        self.directory.mkdir()
        self.s = wire.Session(image, directory=self.directory/'wire')
        self.row = {'name':name, 'status':'RUNNING', 'checks':0, 'events':[],
                    'snapshot_checks':0, 'isolated_transition_checks':0,
                    'ordinary_rollback_bytes':0}
        report['cases'].append(self.row)
        self.snapshot_stream = gzip.open(self.directory/'snapshots.jsonl.gz','wb')
        self.saved = {}
        self.initial = self.observe('initial')

    def check(self, name, actual, expected):
        def display(v):
            return {'bytes':len(v), 'sha256':sha(v)} if isinstance(v,bytes) else v
        assert actual == expected, (name, display(actual), display(expected))
        self.row['checks'] += 1

    def observe(self, label, remember=False):
        raw = self.s.owner()
        # The actual raw bytes are saved before checker admission so that a
        # first checker/target failure remains inspectable.
        archive = dump_raw(self.directory, 'owner-'+str(self.row['snapshot_checks']), raw['bytes'])
        snapshot = observer.decode_owner(raw['bytes'],raw['pointer'])
        self.snapshot_stream.write(observer.canonical({'label':label,'snapshot':snapshot,
                                  'raw':archive})+b'\n')
        self.snapshot_stream.flush()
        result = observer.verify_snapshot(snapshot)
        self.row['snapshot_checks'] += 1
        if remember:
            self.saved[label] = copy.deepcopy(snapshot)
        return snapshot, result

    def packet(self, label, body, command=None, expected=0, structural_only=False):
        before, _ = self.observe(label+':before')
        raw = self.s.exchange(body)
        status, rows = self.w.parse(raw)
        after, result = self.observe(label+':after')
        event = {'label':label, 'response_sha256':sha(raw), 'status':status,
                 'records':[dict(r,payload=r['payload'].hex()) for r in rows],
                 'before_owner_sha256':before['full_sha256'],
                 'after_owner_sha256':after['full_sha256'], 'summary':result}
        self.row['events'].append(event)
        self.check(label+':frame-status',status,expected)
        if expected:
            self.check(label+':no partial public records',rows,[])
            assert command is not None
            event['transition'] = observer.verify_isolated_transition(before,after,command,rows,status)
            self.row['ordinary_rollback_bytes'] += before['length']
            self.row['isolated_transition_checks'] += 1
        elif command is not None and not structural_only:
            event['transition'] = observer.verify_isolated_transition(before,after,command,rows,0)
            self.row['isolated_transition_checks'] += 1
        else:
            event['transition'] = 'STRUCTURAL_ONLY_EXPLICIT_MULTICOMMAND_OR_RESET'
        return rows

    def one(self, label, *cmd, expected=0, structural_only=False):
        body = self.w.packet([cmd])
        words = list(struct.unpack_from('<8Q',body,32))
        rows = self.packet(label,body,words,expected,structural_only)
        return rows[0] if rows else None

    def typed(self,label,code,shape,values=(),requires=0):
        raw = integer_raw(values) if code==2 else bytes(values) if code==1 else struct.pack('<'+'f'*len(values),*values)
        body=create_packet(code,shape,raw,requires)
        return self.packet(label,body,list(struct.unpack_from('<8Q',body,32)))[0]['value']

    def view(self,label,t,shape,strides,offset=0):
        config=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),
                           *strides,*([0]*(4-len(strides))),offset)
        return self.one(label,69,t,self.w.Payload(config),80)['value']

    def node(self, handle, snapshot=None):
        if snapshot is None:
            snapshot,_=self.observe('node-read')
        row=snapshot['tables']['1'][(handle&0xffffff)-1]
        self.check('node handle generation',(row[0],row[1]),(handle>>32,1))
        return row[6],snapshot['nodes'][str(row[6])]

    def mark(self,label):
        return self.observe(label,True)[0]

    def close(self):
        self.one('committed-native-audit',53)
        last,result=self.observe('final')
        self.row['final_summary']=result
        self.row['frames_before_EOF']=self.s.frames
        self.s.finish()
        self.row['EOF_exit_code']=self.s.exit_code
        self.check('clean native EOF',self.s.exit_code,0)
        self.snapshot_stream.close()
        self.row['all_responses_before_stdin_EOF']=True
        self.row['status']='PASS'
