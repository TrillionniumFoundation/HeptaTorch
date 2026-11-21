"""Real variadic F32 cat/stack packets and byte-exact physical replay."""
import math
import struct
from numeric_predicates.native import Observer as BaseObserver,Live,Replay,Memory,load_wire,dump
from join_forward.cases import raw_input,seed_raw

MASK=(1<<64)-1

class Observer(BaseObserver):
    def setup(self,spec):
        raw=raw_input(spec)
        if spec['strides'] is None:
            source=self.create(0,spec['shape'],raw,spec['requires_grad']);return source,[source],source
        backing=self.create(0,[spec['storage_count']],raw,spec['requires_grad'])
        shape,strides=spec['shape'],spec['strides']
        cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
        source=self.one(111,backing,self.wire.Payload(cfg),80)['value'];return source,[source,backing],backing

    def inputs(self,spec):
        sources=[];handles=[];bases=[]
        for item in spec['inputs']:
            source,owned,base=self.setup(item);sources.append(source);handles+=owned;bases.append(base)
        return sources,handles,bases

    def command(self,spec,sources):
        chosen=[sources[index] for index in spec['selection']]
        payload=self.wire.Payload(struct.pack('<'+'Q'*(len(chosen)+1),len(chosen),*chosen))
        return (spec['opcode'],spec['axis'],payload)


def expected_status(row):
    if row['spec']['native_limit'] is not None:
        assert row['accepted'] and row['spec']['native_limit']=='output_rank' and len(row['result']['shape'])==5
        return 2
    if row['accepted']:return 0
    error=row['error'];message=error['message']
    assert error['type'] in ('RuntimeError','IndexError','ValueError')
    if 'non-empty list of Tensors' in message or 'non-empty TensorList' in message:return 7
    assert any(part in message for part in ('Dimension out of range','zero-dimensional tensor',
        'Sizes of tensors must match','same number of dimensions','equal size')),(error,row['spec'])
    return 2


def transaction(n,row,mode='success'):
    spec=row['spec'];sources=spec['inputs'];count=len(sources)+1+(mode=='suffix');end=32+64*count
    commands=[];data=bytearray()
    for source in sources:
        at=end+len(data);raw=raw_input(source);shape=source['shape']
        assert source['strides'] is None and not source['requires_grad']
        commands.append((97,0,at,0,0,0,0,0))
        data+=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),at+64,len(raw))+raw
        while (end+len(data))%8:data.append(0)
    selected=[n.wire.reference(index+1) for index in spec['selection']]
    if mode in ('self','future'):selected[-1]=n.wire.reference(len(sources)+(1 if mode=='self' else 2))
    if mode=='empty':selected=[]
    axis=spec['axis'] if mode!='invalid' else len(sources[0]['shape'])+(spec['operation']=='stack')
    at=end+len(data);data+=struct.pack('<'+'Q'*(len(selected)+1),len(selected),*selected)
    commands.append((spec['opcode'],axis&MASK,at,0,0,0,0,0))
    if mode=='suffix':commands.append((125,n.wire.reference(1),0,0,0,0,0,0))
    return b'HTORCH03'+struct.pack('<3Q',count,end+len(data),0)+b''.join(struct.pack('<8Q',*cmd) for cmd in commands)+data


def controls(n,row,sources,generator):
    spec=row['spec'];count=0;command=n.command(spec,sources)
    for position in range(len(sources)):
        for token in (0,1,generator):
            changed=sources.copy();changed[position]=token
            status,_,_,_=n.issue(n.wire.packet([n.command(spec,changed)]))
            n.check('every list position rejects zero raw and wrong-kind handles',status,1);count+=1
    stale=n.create(0,[1],bytes(4));n.one(41,stale);changed=sources.copy();changed[-1]=stale
    status,_,_,_=n.issue(n.wire.packet([n.command(spec,changed)]))
    n.check('actually released list input rejects atomically',status,1);count+=1
    for position,item in enumerate(spec['inputs']):
        wrong=n.create(4,item['shape'],bytes(math.prod(item['shape'])*8));changed=sources.copy();changed[position]=wrong
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,changed)]))
        n.check('every list position rejects public F64 outside this F32 contract',status,7);count+=1;n.one(41,wrong)
    good=n.wire.packet([command]);offset=struct.unpack_from('<Q',good,48)[0]
    for at in (0,32,97,len(good)+8,(1<<64)-8):
        bad=bytearray(good);struct.pack_into('<Q',bad,48,at)
        status,_,_,_=n.issue(bytes(bad));n.check('list address range and alignment checked before access',status,100);count+=1
    bad=bytearray(good[:-8]);struct.pack_into('<Q',bad,16,len(bad))
    status,_,_,_=n.issue(bytes(bad));n.check('complete list payload extent required',status,100);count+=1
    for size,expected in ((0,7),(MASK,100)):
        bad=bytearray(good);struct.pack_into('<Q',bad,offset,size)
        status,_,_,_=n.issue(bytes(bad));n.check('empty and overflowing list counts reject atomically',status,expected);count+=1
    bad=bytearray(good);struct.pack_into('<Q',bad,40,len(spec['inputs'][0]['shape'])+(spec['operation']=='stack'))
    status,_,_,_=n.issue(bytes(bad));n.check('out-of-range join dimension rejects atomically',status,2);count+=1
    maps=n.s.maps();status,_,_,_=n.issue(n.wire.packet([command,(125,sources[0])]))
    n.check('valid join prefix and invalid suffix roll back',status,4)
    n.check('join prefix failure restores every actual mapping',n.s.maps(),maps);count+=1
    status,records,_,_=n.issue(transaction(n,row))
    n.check('three actual factories and same-frame FE join succeed',status,0)
    n.check('complete same-frame result equals actual Torch bytes and metadata',n.tensor(records[-1]['value'])[0],n.expected(row['result']))
    for record in records:n.one(41,record['value'])
    count+=1
    for mode,expected in (('suffix',4),('self',1),('future',1),('invalid',2),('empty',7)):
        maps=n.s.maps();status,_,_,_=n.issue(transaction(n,row,mode))
        n.check('same-frame '+mode+' rejects complete factory prefix',status,expected)
        n.check('same-frame '+mode+' restores every mapping byte',n.s.maps(),maps);count+=1
    return count


def compare(n,row):
    spec=row['spec'];generator,rng=n.begin();sources,handles,bases=n.inputs(spec)
    originals=[];nodes=[]
    for token,value in zip(sources,row['inputs']):
        actual,node=n.tensor(token);n.check('complete original independent input',actual,n.expected(value))
        originals.append(actual);nodes.append(node)
    status,records,before,after=n.issue(n.wire.packet([n.command(spec,sources)]))
    n.check('actual independent Torch acceptance or named native rank capacity',status,expected_status(row))
    for token,original in zip(sources,originals):
        n.check('every complete source unchanged',n.tensor(token)[0],original)
        if not any(item['requires_grad'] for item in spec['inputs']):
            n.check('all84 no-grad input descriptor words unchanged',Memory(before['bytes']).node(token),Memory(after['bytes']).node(token))
    control_count=0;elements=0;absent_gradients=0
    if status==0:
        result=records[0]['value'];actual,node=n.tensor(result)
        n.check('every output bit and full metadata equal actual Torch',actual,n.expected(row['result']))
        for token,parent in zip(sources,nodes):
            n.check('fresh result Tensor Storage Version independent of every input',
                [result!=token,node[27]!=parent[27],node[28]!=parent[28]],[True]*3)
        if not actual['requires_grad']:
            edges=(5,6,7,45,48,49,50,51,83)
            n.check('no-grad result has no AD parents gradients or view edges',[node[word] for word in edges],[0]*len(edges))
        if spec['controls']:control_count=controls(n,row,sources,generator)
        if 'seed' in row:
            seed=n.create(0,actual['shape'],seed_raw(actual['numel']));n.one(29,result,seed)
            for base,expected in zip(bases,row['gradients']):
                if expected is None:
                    _,state=n.tensor(base)
                    n.check('independently absent gradient remains genuinely undefined',[state[word] for word in (45,48,49)],[0,0,0])
                    absent_gradients+=1
                else:
                    gradient=n.one(11,base)
                    n.check('full independent nonuniform VJP bytes',gradient['payload'],n.reference['blob'](expected['logical']))
                    n.check('full independent VJP element count',gradient['value'],expected['numel'])
            n.one(41,seed)
        for token in handles:n.one(41,token)
        n.check('complete output survives releasing every public input',n.tensor(result)[0],actual)
        n.one(41,result);elements=actual['numel']
    else:
        for token in handles:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],accepted=status==0,status=status,elements=elements,controls=control_count,
        gradient_checked='seed' in row and status==0,absent_gradients_checked=absent_gradients,
        operation=spec['operation'],capacity_refusal=spec['native_limit'])


def allocation(n,row):
    spec=row['spec'];generator,rng=n.begin();sources,handles,_=n.inputs(spec)
    originals=[n.tensor(source)[0] for source in sources]
    packet=n.wire.packet([n.command(spec,sources)]);live=[];n.s.lower_limit(n.s.owner())
    for attempt in range(32):
        status,records,_,_=n.issue(packet)
        if status:
            n.check('real child join allocation failure',status,3);break
        live.append(records[0]['value'])
    else:raise ValueError('no genuine join allocation refusal within32 attempts')
    n.s.restore_limit();status,records,_,_=n.issue(packet)
    n.check('identical join request recovers after resource limit restoration',status,0)
    result=records[0]['value'];live.append(result);actual=n.tensor(result)[0]
    n.check('complete recovered result equals actual Torch',actual,n.expected(row['result']))
    n.check('every source survives OOM and recovery unchanged',[n.tensor(source)[0] for source in sources],originals)
    for token in handles:n.one(41,token)
    n.check('complete recovered output survives releasing all inputs',n.tensor(result)[0],actual)
    for token in live:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],operation=spec['operation'],attempts=attempt+1,actual_allocation_failures=1,
        identical_request_recovered=True)


def recipes(reference):
    rows=reference['report']['rows']
    return [('case-%04d'%row['index'],compare,row) for row in rows]+[
        ('allocation-%04d'%row['index'],allocation,row) for row in rows if row['spec']['allocation']]
