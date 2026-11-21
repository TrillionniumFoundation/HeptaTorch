"""Real native forward/backward predicates and complete physical offline replay."""
import math
import struct
from numeric_predicates.native import Observer as BaseObserver,Live,Replay,Memory,load_wire,dump
from forward_nograd.cases import raw_input,seed_raw

MASK=(1<<64)-1

class Observer(BaseObserver):
    def __init__(self,*args):
        super().__init__(*args)
        self.numerical=dict(values=0,exact_words=0,finite_words=0,nan_words=0,infinity_words=0,
            max_absolute_error=0.0,max_relative_error_scaled_by_atol=0.0)

    def setup(self,spec):
        raw=raw_input(spec)
        if spec['strides'] is None:
            source=self.create(0,spec['shape'],raw,spec['requires_grad']);return source,[source],source
        backing=self.create(0,[spec['storage_count']],raw,spec['requires_grad'])
        shape,strides=spec['shape'],spec['strides']
        cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
        source=self.one(111,backing,self.wire.Payload(cfg),80)['value'];return source,[source,backing],backing

    def inputs(self,spec):
        a,ah,ar=self.setup(spec['a']);b=br=None;bh=[]
        if spec['b'] is not None:
            b,bh,br=(a,[],ar) if spec['same_input'] else self.setup(spec['b'])
        return a,b,ah+bh,ar,br

    def command(self,spec,a,b=None):
        if spec['operation']=='matmul':return (32,a,b)
        return (21,spec['opcode'],a,spec['axis'])

    def values(self,label,actual,expected,spec):
        self.check(label+' complete float32 byte count',len(actual),len(expected))
        self.check(label+' float32 extent alignment',len(actual)%4,0)
        self.numerical['values']+=len(actual)//4
        if spec['comparison']=='exact':
            self.check(label+' all independently observed exact words',actual,expected)
            self.numerical['exact_words']+=len(actual)//4
            return
        self.check(label+' fixed comparison policy',[spec['comparison'],spec['rtol'],spec['atol']],['float32',1e-5,1e-6])
        bad=[]
        for index,((a,),(b,)) in enumerate(zip(struct.iter_unpack('<f',actual),struct.iter_unpack('<f',expected))):
            if math.isnan(b):
                self.numerical['nan_words']+=1
                if not math.isnan(a):bad.append((index,'NaN class'))
            elif math.isinf(b):
                self.numerical['infinity_words']+=1
                if a!=b:bad.append((index,'infinity sign'))
            else:
                self.numerical['finite_words']+=1
                if not math.isfinite(a):bad.append((index,'finite class'));continue
                error=abs(a-b)
                self.numerical['max_absolute_error']=max(self.numerical['max_absolute_error'],error)
                self.numerical['max_relative_error_scaled_by_atol']=max(
                    self.numerical['max_relative_error_scaled_by_atol'],error/max(abs(b),spec['atol']))
                if error>spec['atol']+spec['rtol']*abs(b):bad.append((index,'fixed finite tolerance',a,b))
        self.check(label+' all finite values and every NaN/infinity class',bad,[])

    def result(self,label,actual,value,spec):
        expected=self.expected(value)
        self.check(label+' all dtype shape strides storage/version and AD metadata',
            {key:item for key,item in actual.items() if key not in ('logical','storage')},
            {key:item for key,item in expected.items() if key not in ('logical','storage')})
        for field in ('logical','storage'):self.values(label+' '+field,actual[field],expected[field],spec)


def expected_status(row):
    if row['spec']['native_limit'] is not None:
        assert row['accepted'] and row['spec']['operation']=='matmul'
        return 3
    if row['accepted']:return 0
    error=row['error'];message=error['message']
    assert error['type'] in ('RuntimeError','IndexError')
    assert any(part in message for part in ('Dimension out of range','at least 1D','size mismatch','inconsistent tensor size',
        'cannot be multiplied','size of tensor','expected scalar type','same dtype')),(error,row['spec'])
    return 2


def transaction(n,row,mode='success'):
    spec=row['spec'];sources=[spec['a']]+([spec['b']] if spec['b'] is not None else [])
    count=len(sources)+1+(mode=='suffix');end=32+64*count
    commands=[];data=bytearray()
    for source in sources:
        at=end+len(data);raw=raw_input(source);shape=source['shape']
        assert source['strides'] is None and not source['requires_grad']
        commands.append((97,0,at,0,0,0,0,0))
        data+=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),at+64,len(raw))+raw
        while (end+len(data))%8:data.append(0)
    source=n.wire.reference(1)
    if mode in ('self','future'):source=n.wire.reference(len(sources)+(1 if mode=='self' else 2))
    if spec['operation']=='matmul':
        commands.append((32,source,n.wire.reference(1 if mode=='invalid' else 2),0,0,0,0,0))
    else:
        commands.append((21,spec['opcode'],source,2 if mode=='invalid' else spec['axis']&MASK,0,0,0,0))
    if mode=='suffix':commands.append((125,n.wire.reference(1),0,0,0,0,0,0))
    return b'HTORCH03'+struct.pack('<3Q',count,end+len(data),0)+b''.join(struct.pack('<8Q',*cmd) for cmd in commands)+data


def controls(n,row,a,b,generator):
    spec=row['spec'];count=0;command=n.command(spec,a,b)
    for token in (0,1,generator):
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,token,b)]))
        n.check('zero raw or wrong-kind forward input rejects atomically',status,1);count+=1
    stale=n.create(0,[1],bytes(4));n.one(41,stale)
    status,_,_,_=n.issue(n.wire.packet([n.command(spec,stale,b)]))
    n.check('actually released forward input rejects atomically',status,1);count+=1
    if b is not None:
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,a,generator)]))
        n.check('second matmul input wrong kind rejects atomically',status,1);count+=1
    wrong=n.create(4,spec['a']['shape'],bytes(8*math.prod(spec['a']['shape'])))
    status,_,_,_=n.issue(n.wire.packet([n.command(spec,wrong,b)]))
    n.check('F64 is explicitly outside this F32 forward contract',status,7);count+=1
    n.one(41,wrong)
    if b is not None:
        wrong=n.create(4,spec['b']['shape'],bytes(8*math.prod(spec['b']['shape'])))
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,a,wrong)]))
        n.check('F64 second matmul input rejects atomically',status,7);count+=1;n.one(41,wrong)
    else:
        status,_,_,_=n.issue(n.wire.packet([(21,22,a,spec['axis'])]))
        n.check('unsupported softmax operation rejects atomically',status,4);count+=1
    maps=n.s.maps();status,_,_,_=n.issue(n.wire.packet([command,(125,a)]))
    n.check('successful forward prefix and invalid suffix roll back',status,4)
    n.check('failed forward prefix restores every mapping byte',n.s.maps(),maps);count+=1
    status,records,_,_=n.issue(transaction(n,row))
    n.check('actual factories and same-frame FE forward succeed',status,0)
    n.result('complete same-frame output',n.tensor(records[-1]['value'])[0],row['result'],spec)
    for record in records:n.one(41,record['value'])
    count+=1
    for mode,expected in (('suffix',4),('self',1),('future',1),('invalid',2)):
        maps=n.s.maps();status,_,_,_=n.issue(transaction(n,row,mode))
        n.check('same-frame '+mode+' rejects complete allocation prefix',status,expected)
        n.check('same-frame '+mode+' restores every mapping byte',n.s.maps(),maps);count+=1
    return count


def compare(n,row):
    spec=row['spec'];generator,rng=n.begin();a,b,handles,ar,br=n.inputs(spec)
    original_a,node_a=n.tensor(a);n.check('complete independent first input',original_a,n.expected(row['input_a']))
    original_b=node_b=None
    if b is not None:
        original_b,node_b=n.tensor(b);n.check('complete independent second input',original_b,n.expected(row['input_b']))
    status,records,before,after=n.issue(n.wire.packet([n.command(spec,a,b)]))
    n.check('independent Torch acceptance or explicitly named native capacity',status,expected_status(row))
    for label,handle,original in [('a',a,original_a),('b',b,original_b)]:
        if handle is not None:
            n.check('complete '+label+' source unchanged',n.tensor(handle)[0],original)
            if not spec['a']['requires_grad'] and (spec['b'] is None or not spec['b']['requires_grad']):
                n.check('all84 '+label+' no-grad source descriptor words unchanged',
                    Memory(before['bytes']).node(handle),Memory(after['bytes']).node(handle))
    control_count=0;elements=0
    if status==0:
        result=records[0]['value'];actual,node=n.tensor(result)
        n.result('complete independent output',actual,row['result'],spec)
        for label,handle,source_node in [('a',a,node_a),('b',b,node_b)]:
            if handle is not None:
                n.check('independent Tensor Storage Version from '+label,
                    [result!=handle,node[27]!=source_node[27],node[28]!=source_node[28]],[True]*3)
        if not actual['requires_grad']:
            edges=(5,6,7,45,48,49,50,51,83)
            n.check('no-grad result has no AD parents gradients or view edges',[node[word] for word in edges],[0]*len(edges))
        if spec['controls']:control_count=controls(n,row,a,b,generator)
        if 'seed' in row:
            seed=n.create(0,actual['shape'],seed_raw(actual['numel']))
            n.one(29,result,seed)
            for label,base in [('a',ar),('b',br)]:
                if 'gradient_'+label in row:
                    gradient=n.one(11,base);expected=row['gradient_'+label]
                    n.values('complete '+label+' actual Torch nonuniform VJP',gradient['payload'],
                        n.reference['blob'](expected['logical']),spec)
                    n.check('complete '+label+' actual Torch VJP count',gradient['value'],expected['numel'])
            n.one(41,seed)
        for token in handles:n.one(41,token)
        n.check('output survives releasing all input capabilities',n.tensor(result)[0],actual)
        n.one(41,result);elements=actual['numel']
    else:
        for token in handles:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],accepted=status==0,status=status,elements=elements,controls=control_count,
        gradient_checked='seed' in row and status==0,operation=spec['operation'],numerical=n.numerical,
        capacity_refusal=spec['native_limit'])


def allocation(n,row):
    spec=row['spec'];generator,rng=n.begin();a,b,handles,_,_=n.inputs(spec)
    original_a=n.tensor(a)[0];original_b=n.tensor(b)[0] if b is not None else None
    packet=n.wire.packet([n.command(spec,a,b)]);live=[];n.s.lower_limit(n.s.owner())
    for attempt in range(32):
        status,records,_,_=n.issue(packet)
        if status:
            n.check('real native forward allocation refusal',status,3);break
        live.append(records[0]['value'])
    else:raise ValueError('no genuine forward allocation refusal within32 attempts')
    n.s.restore_limit();status,records,_,_=n.issue(packet)
    n.check('identical forward request succeeds after resource restoration',status,0)
    result=records[0]['value'];live.append(result)
    actual=n.tensor(result)[0];n.result('complete recovered output',actual,row['result'],spec)
    n.check('complete first input unchanged across OOM and recovery',n.tensor(a)[0],original_a)
    if b is not None:n.check('complete second input unchanged across OOM and recovery',n.tensor(b)[0],original_b)
    for token in handles:n.one(41,token)
    n.check('recovered output survives releasing all inputs',n.tensor(result)[0],actual)
    for token in live:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],operation=spec['operation'],attempts=attempt+1,actual_allocation_failures=1,
        identical_request_recovered=True,numerical=n.numerical)


def recipes(reference):
    rows=reference['report']['rows']
    return [('case-%04d'%row['index'],compare,row) for row in rows]+[
        ('allocation-%04d'%row['index'],allocation,row) for row in rows if row['spec']['allocation']]
