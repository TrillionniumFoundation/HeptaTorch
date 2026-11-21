"""Native execution predicates and byte-identical offline replay recipes."""
import math
import struct
from numeric_predicates.native import Observer as BaseObserver,Live,Replay,Memory,load_wire,dump
from indexed_f32.cases import raw_input,seed_raw

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
        a,ah,ar=self.setup(spec['a']);b=br=None;bh=[]
        if spec['b'] is not None:
            b,bh,br=(a,[],ar) if spec['same_input'] else self.setup(spec['b'])
        return a,b,ah+bh,ar,br

    def command(self,spec,a,b=None):
        words=[len(spec['index_shape'])]+spec['index_shape']+spec['indices']
        payload=self.wire.Payload(struct.pack('<'+'Q'*len(words),*(v&MASK for v in words)))
        if spec['operation']=='gather':return (31,a,spec['axis'],payload)
        return (spec['opcode'],a,b,spec['axis'],payload,spec['alpha_bits'] if spec['operation']=='index_add' else 0)

def expected_status(row):
    if row['accepted']:return 0
    message=row['error']['message']
    if message.startswith('Dimension out of range'):return 2
    if 'out of bounds' in message or 'out of range' in message:return 7
    raise ValueError('unmapped actual Torch indexed error: '+message)

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
    at=end+len(data);indices=spec['indices'].copy()
    if mode=='negative':indices[-1]=-1
    words=[len(spec['index_shape'])]+spec['index_shape']+indices
    data+=struct.pack('<'+'Q'*len(words),*(word&MASK for word in words))
    source=n.wire.reference(1)
    if mode in ('self','future'):source=n.wire.reference(len(sources)+(1 if mode=='self' else 2))
    if spec['operation']=='gather':commands.append((31,source,spec['axis']&MASK,at,0,0,0,0))
    else:commands.append((spec['opcode'],source,n.wire.reference(2),spec['axis']&MASK,at,
        spec['alpha_bits'] if spec['operation']=='index_add' else 0,0,0))
    if mode=='suffix':commands.append((125,n.wire.reference(1),0,0,0,0,0,0))
    return b'HTORCH03'+struct.pack('<3Q',count,end+len(data),0)+b''.join(struct.pack('<8Q',*cmd) for cmd in commands)+data

def controls(n,row,a,b,generator):
    spec=row['spec'];count=0;command=n.command(spec,a,b)
    for token in (0,1,generator):
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,token,b)]))
        n.check('wrong-kind or invalid indexed input handle rejects atomically',status,1);count+=1
    stale=n.create(0,[1],bytes(4));n.one(41,stale)
    status,_,_,_=n.issue(n.wire.packet([n.command(spec,stale,b)]))
    n.check('actual stale indexed input handle rejects atomically',status,1);count+=1
    if b is not None:
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,a,generator)]))
        n.check('second indexed input wrong-kind handle rejects atomically',status,1);count+=1
    good=n.wire.packet([command]);field=3 if b is None else 4
    payload=struct.unpack_from('<Q',good,32+field*8)[0]
    for at in (0,32,97,len(good)+8,(1<<64)-8):
        bad=bytearray(good);struct.pack_into('<Q',bad,32+field*8,at)
        status,_,_,_=n.issue(bytes(bad));n.check('indexed payload range and alignment validated before reads',status,100);count+=1
    bad=bytearray(good[:-8]);struct.pack_into('<Q',bad,16,len(bad))
    status,_,_,_=n.issue(bytes(bad));n.check('truncated shaped index data rejected atomically',status,100);count+=1
    bad=bytearray(good);struct.pack_into('<Q',bad,payload,5)
    status,_,_,_=n.issue(bytes(bad));n.check('index rank exceeds real native capacity',status,2);count+=1
    bad=bytearray(good);struct.pack_into('<Q',bad,payload+8,262145)
    status,_,_,_=n.issue(bytes(bad));n.check('index extent capacity checked before address arithmetic',status,3);count+=1
    maps=n.s.maps();status,_,_,_=n.issue(n.wire.packet([command,(125,a)]))
    n.check('successful forward prefix plus invalid opcode suffix rolls back',status,4)
    n.check('every mapping byte restored after successful-prefix failure',n.s.maps(),maps);count+=1
    status,records,_,_=n.issue(transaction(n,row))
    n.check('actual factory and same-frame FE indexed forward succeeds',status,0)
    n.check('complete same-frame result matches actual Torch',n.tensor(records[-1]['value'])[0],n.expected(row['result']))
    for record in records:n.one(41,record['value'])
    count+=1
    for mode,expected in (('suffix',4),('self',1),('future',1),('negative',7)):
        maps=n.s.maps();status,_,_,_=n.issue(transaction(n,row,mode))
        n.check('same-frame '+mode+' rejects complete allocated prefix',status,expected)
        n.check('same-frame '+mode+' restores every mapping byte',n.s.maps(),maps);count+=1
    return count

def compare(n,row):
    spec=row['spec'];generator,rng=n.begin();a,b,handles,ar,br=n.inputs(spec)
    original_a,node_a=n.tensor(a);n.check('complete independent first input',original_a,n.expected(row['input_a']))
    original_b=node_b=None
    if b is not None:
        original_b,node_b=n.tensor(b);n.check('complete independent second input',original_b,n.expected(row['input_b']))
    status,records,before,after=n.issue(n.wire.packet([n.command(spec,a,b)]))
    n.check('actual Torch indexed acceptance and error',status,expected_status(row))
    for label,handle,original in [('a',a,original_a),('b',b,original_b)]:
        if handle is not None:
            n.check('complete '+label+' source unchanged',n.tensor(handle)[0],original)
            if not spec['a']['requires_grad'] and (spec['b'] is None or not spec['b']['requires_grad']):
                n.check('all84 '+label+' no-grad source descriptor words unchanged',Memory(before['bytes']).node(handle),Memory(after['bytes']).node(handle))
    control_count=0;elements=0
    if status==0:
        result=records[0]['value'];actual,node=n.tensor(result)
        n.check('every output bit and all metadata equal independently observed Torch',actual,n.expected(row['result']))
        for label,handle,source_node in [('a',a,node_a),('b',b,node_b)]:
            if handle is not None:
                n.check('independent Tensor Storage Version from '+label,[result!=handle,node[27]!=source_node[27],node[28]!=source_node[28]],[True]*3)
        if not actual['requires_grad']:
            edge_words=(5,6,7,45,48,49,50,51,83)
            n.check('no-grad result has no AD parent gradient or view edges',[node[word] for word in edge_words],[0]*len(edge_words))
        if spec['controls']:control_count=controls(n,row,a,b,generator)
        if 'seed' in row:
            seed=n.create(0,actual['shape'],seed_raw(actual['numel']))
            n.one(29,result,seed)
            for label,base in [('a',ar),('b',br)]:
                if 'gradient_'+label in row:
                    gradient=n.one(11,base);expected=row['gradient_'+label]
                    n.check('complete '+label+' actual Torch backward bits',gradient['payload'],n.reference['blob'](expected['logical']))
                    n.check('complete '+label+' actual Torch backward count',gradient['value'],expected['numel'])
            n.one(41,seed)
        for token in handles:n.one(41,token)
        n.check('output survives releasing every input capability',n.tensor(result)[0],actual)
        n.one(41,result);elements=actual['numel']
    else:
        for token in handles:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],accepted=status==0,status=status,elements=elements,controls=control_count,
        gradient_checked='seed' in row,operation=spec['operation'])

def allocation(n,row):
    spec=row['spec'];generator,rng=n.begin();a,b,handles,_,_=n.inputs(spec)
    original_a=n.tensor(a)[0];original_b=n.tensor(b)[0] if b is not None else None
    packet=n.wire.packet([n.command(spec,a,b)]);live=[];n.s.lower_limit(n.s.owner())
    for attempt in range(32):
        status,records,_,_=n.issue(packet)
        if status:
            n.check('real child indexed allocation failure',status,3);break
        live.append(records[0]['value'])
    else:raise ValueError('no real indexed allocation failure within32 attempts')
    n.s.restore_limit();status,records,_,_=n.issue(packet)
    n.check('identical indexed request succeeds after resource limit restoration',status,0)
    result=records[0]['value'];live.append(result)
    n.check('complete recovered result equals actual Torch',n.tensor(result)[0],n.expected(row['result']))
    n.check('complete first input unchanged across OOM and recovery',n.tensor(a)[0],original_a)
    if b is not None:n.check('complete second input unchanged across OOM and recovery',n.tensor(b)[0],original_b)
    for token in handles:n.one(41,token)
    n.check('recovered result survives releasing all inputs',n.tensor(result)[0],n.expected(row['result']))
    for token in live:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],operation=spec['operation'],attempts=attempt+1,actual_allocation_failures=1,
        identical_request_recovered=True)

def recipes(reference):
    rows=reference['report']['rows']
    return [('case-%04d'%row['index'],compare,row) for row in rows]+[
        ('allocation-%04d'%row['index'],allocation,row) for row in rows if row['spec']['allocation']]
