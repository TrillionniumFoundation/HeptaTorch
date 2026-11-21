"""Actual spatial observations and identical predicates for physical archive replay."""
import math
import struct
from numeric_predicates.native import Observer as BaseObserver,Live,Replay,Memory,load_wire,dump
from spatial_forward.cases import raw_input,seed_raw

MASK=(1<<64)-1
pair=lambda value:(value[0]<<32)|value[1]


class Observer(BaseObserver):
    def setup(self,spec):
        if spec is None:return 0,[],None
        raw=raw_input(spec)
        if spec['strides'] is None:
            source=self.create(0,spec['shape'],raw,spec['requires_grad'])
            return source,[source],source
        backing=self.create(0,[spec['storage_count']],raw,spec['requires_grad'])
        shape,strides=spec['shape'],spec['strides']
        cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
        source=self.one(111,backing,self.wire.Payload(cfg),80)['value']
        return source,[source,backing],backing

    def inputs(self,spec):
        sources=[];handles=[];bases=[]
        for value in spec['inputs']:
            source,owned,base=self.setup(value);sources.append(source);handles.extend(owned);bases.append(base)
        return sources,handles,bases

    def command(self,spec,values):
        op,args=spec['operation'],spec['args']
        if op=='conv2d':return (22,*values,pair(args['stride']),pair(args['padding']),pair(args['dilation']),args['groups'])
        if op=='pool2d':return (23,23 if args['average'] else 24,*values,pair(args['kernel']),pair(args['stride']),pair(args['padding']),int(args['include']))
        if op=='reduce_axis':return (20,19 if args['mean'] else 18,*values,args['axis'],int(args['keep']))
        if op=='layernorm':return (26,*values,args['epsilon_bits'])
        if op=='batchnorm_eval':return (28,*values,args['epsilon_bits'])
        if op=='linear':return (15,*values)
        if op=='cross_entropy_probs':return (27,*values)
        raise ValueError(op)

    def numerical_bytes(self,label,actual,expected,spec):
        self.check(label+' complete byte count',len(actual),len(expected))
        self.check(label+' complete F32 extent',len(actual)%4,0)
        actual_values=struct.unpack('<'+'f'*(len(actual)//4),actual)
        expected_values=struct.unpack('<'+'f'*(len(expected)//4),expected)
        for index,(left,right) in enumerate(zip(actual_values,expected_values)):
            # Preserve the original spatial probe's math.isclose predicate and
            # its per-operation rtol/atol. These are observed Torch values.
            close=math.isnan(left) and math.isnan(right) if math.isnan(right) else math.isclose(left,right,rel_tol=spec['rtol'],abs_tol=spec['atol'])
            self.check(label+' actual Torch F32 value '+str(index),close,True)

    def numerical_tensor(self,label,actual,expected,spec):
        self.check(label+' complete metadata',{k:v for k,v in actual.items() if k not in ('logical','storage')},
                   {k:v for k,v in expected.items() if k not in ('logical','storage')})
        for field in ('logical','storage'):
            self.numerical_bytes(label+' '+field,actual[field],expected[field],spec)


def transaction(n,row,mode='success'):
    spec=row['spec'];sources=[value for value in spec['inputs'] if value is not None]
    count=len(sources)+1+(mode=='suffix');end=32+64*count
    commands=[];data=bytearray();values=[];ordinal=0
    for source in spec['inputs']:
        if source is None:values.append(0);continue
        ordinal+=1;values.append(n.wire.reference(ordinal))
        at=end+len(data);raw=raw_input(source);shape=source['shape']
        assert source['strides'] is None and not source['requires_grad']
        commands.append((97,0,at,0,0,0,0,0))
        data+=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),at+64,len(raw))+raw
        while (end+len(data))%8:data.append(0)
    if mode in ('self','future'):values[0]=n.wire.reference(len(sources)+(1 if mode=='self' else 2))
    command=n.command(spec,values);commands.append(tuple(command)+(0,)*(8-len(command)))
    if mode=='suffix':commands.append((125,n.wire.reference(1),0,0,0,0,0,0))
    return b'HTORCH03'+struct.pack('<3Q',count,end+len(data),0)+b''.join(struct.pack('<8Q',*(word&MASK for word in cmd)) for cmd in commands)+data


def controls(n,row,values,generator):
    spec=row['spec'];count=0;command=n.command(spec,values)
    for token in (0,1,generator):
        bad=values.copy();bad[0]=token
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,bad)]))
        n.check('invalid or wrong-kind forward input rejects atomically',status,1);count+=1
    stale=n.create(0,[1],bytes(4));n.one(41,stale);bad=values.copy();bad[0]=stale
    status,_,_,_=n.issue(n.wire.packet([n.command(spec,bad)]))
    n.check('stale forward input rejects atomically',status,1);count+=1
    if len(values)>1:
        bad=values.copy();bad[1]=generator
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,bad)]))
        n.check('wrong-kind second forward input rejects atomically',status,1);count+=1
    maps=n.s.maps();status,_,_,_=n.issue(n.wire.packet([command,(125,values[0])]))
    n.check('successful forward prefix plus unsupported suffix rejects',status,4)
    n.check('all mappings restored after forward-prefix failure',n.s.maps(),maps);count+=1
    status,records,_,_=n.issue(transaction(n,row))
    n.check('actual factory and same-frame forward succeeds',status,0)
    n.numerical_tensor('complete same-frame result',n.tensor(records[-1]['value'])[0],n.expected(row['result']),spec)
    for record in records:n.one(41,record['value'])
    count+=1
    for mode,expected in (('suffix',4),('self',1),('future',1)):
        maps=n.s.maps();status,_,_,_=n.issue(transaction(n,row,mode))
        n.check('same-frame '+mode+' rejects allocated prefix',status,expected)
        n.check('same-frame '+mode+' restores every mapping byte',n.s.maps(),maps);count+=1
    return count


def compare(n,row):
    spec=row['spec'];generator,rng=n.begin();values,handles,bases=n.inputs(spec)
    originals=[];source_nodes=[]
    for ordinal,(token,expected) in enumerate(zip(values,row['inputs'])):
        original,node=n.tensor(token) if token else (None,None)
        n.check('complete independent input '+str(ordinal),original,n.expected(expected) if expected is not None else None)
        originals.append(original);source_nodes.append(node)
    status,records,before,after=n.issue(n.wire.packet([n.command(spec,values)]))
    n.check('actual Torch forward acceptance',status,0)
    n.check('one matching forward response',[r['opcode'] for r in records],[n.command(spec,values)[0]])
    for ordinal,(token,original) in enumerate(zip(values,originals)):
        if token:
            n.check('complete input unchanged '+str(ordinal),n.tensor(token)[0],original)
            if not any(item is not None and item['requires_grad'] for item in spec['inputs']):
                n.check('all84 no-grad input descriptor words unchanged '+str(ordinal),Memory(before['bytes']).node(token),Memory(after['bytes']).node(token))
    result=records[0]['value'];actual,node=n.tensor(result)
    n.numerical_tensor('actual independent forward output',actual,n.expected(row['result']),spec)
    for ordinal,(token,source) in enumerate(zip(values,source_nodes)):
        if token:
            n.check('fresh Tensor Storage Version from input '+str(ordinal),
                    [result!=token,node[27]!=source[27],node[28]!=source[28]],[True]*3)
    if not actual['requires_grad']:
        edges=(5,6,7,45,48,49,50,51,83)
        n.check('no-grad output has no AD parent gradient or view edges',[node[word] for word in edges],[0]*len(edges))
    control_count=controls(n,row,values,generator) if spec['controls'] else 0
    gradient_count=0
    if 'seed' in row:
        seed=n.create(0,actual['shape'],seed_raw(actual['numel']))
        n.check('literal backward seed equals independently observed Torch',n.tensor(seed)[0],n.expected(row['seed']))
        n.one(29,result,seed)
        for ordinal,(base,expected) in enumerate(zip(bases,row['gradients'])):
            if expected is not None:
                gradient=n.one(11,base)
                n.check('complete actual backward count '+str(ordinal),gradient['value'],expected['numel'])
                n.numerical_bytes('complete actual Torch backward '+str(ordinal),gradient['payload'],n.reference['blob'](expected['logical']),spec)
                gradient_count+=1
        n.one(41,seed)
    for token in handles:n.one(41,token)
    n.check('output survives releasing every input capability',n.tensor(result)[0],actual)
    n.one(41,result);n.end(generator,rng)
    return dict(index=row['index'],operation=spec['operation'],accepted=True,status=0,elements=actual['numel'],
                controls=control_count,gradient_checked='seed' in row,gradient_tensors_checked=gradient_count)


def allocation(n,row):
    spec=row['spec'];generator,rng=n.begin();values,handles,_=n.inputs(spec)
    originals=[n.tensor(token)[0] if token else None for token in values]
    packet=n.wire.packet([n.command(spec,values)]);live=[];n.s.lower_limit(n.s.owner())
    for attempt in range(256):
        status,records,_,_=n.issue(packet)
        if status:
            n.check('actual child forward allocation refusal',status,3);break
        live.append(records[0]['value'])
    else:raise ValueError('no real forward allocation failure within256 attempts')
    n.s.restore_limit()
    n.check('complete generator state survives actual OOM',n.one(63,generator)['payload'],rng)
    status,records,_,_=n.issue(packet)
    n.check('identical forward request recovers after limit restoration',status,0)
    result=records[0]['value'];live.append(result);actual=n.tensor(result)[0]
    n.numerical_tensor('complete recovered result',actual,n.expected(row['result']),spec)
    for token,original in zip(values,originals):
        if token:n.check('complete input unchanged across OOM and recovery',n.tensor(token)[0],original)
    for token in handles:n.one(41,token)
    n.check('recovered result survives all input releases',n.tensor(result)[0],actual)
    for token in live:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],operation=spec['operation'],attempts=attempt+1,actual_allocation_failures=1,
                identical_request_recovered=True)


def recipes(reference):
    rows=reference['report']['rows']
    return [('case-%04d'%row['index'],compare,row) for row in rows]+[
        ('allocation-%04d'%row['index'],allocation,row) for row in rows if row['spec']['allocation']]
