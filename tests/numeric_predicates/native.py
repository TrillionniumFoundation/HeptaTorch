"""Native predicate observations, with complete physical owner/frame replay.

Python supplies literal inputs and reads metadata and bytes. Classification is
performed only by HIA; every expected result is an actual Torch observation.
"""

import math
import struct

from half_logical.native import Live, Replay, Memory, load_wire, dump, Observer as TransportObserver
from .cases import OPERATIONS, DTYPES, WIDTHS, raw_input


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Observer(TransportObserver):
    def create(self, dtype, shape, raw, requires=False):
        require(len(shape) <= 4, 'public rank capacity')
        cfg=struct.pack('<8Q',len(shape),int(requires),*shape,*([0]*(4-len(shape))),160,len(raw))
        packet=b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)
        packet+=struct.pack('<8Q',97,dtype,96,0,0,0,0,0)+cfg+raw
        status,records=self.wire.parse(self.s.exchange(packet))
        self.check('literal typed factory status',status,0)
        self.check('literal typed factory response',[r['opcode'] for r in records],[97])
        return records[0]['value']

    def setup(self,spec):
        raw=raw_input(spec)
        if spec['dtype']=='wrapped_float64':
            parent=self.create(0,[1],struct.pack('<I',0x3f800000),True)
            product=self.one(114,parent,struct.unpack('<Q',raw)[0])['value']
            source=self.one(107,product,1)['value']
            return source,[source,product,parent]
        if spec['strides'] is None:
            source=self.create(spec['native_dtype'],spec['shape'],raw,spec['requires_grad'])
            return source,[source]
        backing=self.create(spec['native_dtype'],[spec['input_count']],raw,spec['requires_grad'])
        shape,strides=spec['shape'],spec['strides']
        cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
        source=self.one(111,backing,self.wire.Payload(cfg),80)['value']
        return source,[source,backing]

    def command(self,spec,source):
        return (spec['opcode'],source,0,0,0,0,0,0)

    def tensor(self,token):
        meta=self.wire.meta(self.one(13,token))
        dtype,width,incarnation,leaf=struct.unpack('<4Q',self.one(96,token,0)['payload'])
        record=self.one(98,token,0,meta['numel'])
        raw=self.s.owner()['bytes'];memory=Memory(raw);node=memory.node(token)
        tensor_row=memory.row(1,token)
        storage,version=memory.row(2,node[27]),memory.row(3,node[28])
        require(dtype in WIDTHS and node[0]<=4 and node[4] in (0,1)
                and leaf in (0,1) and node[45] in (0,1),'public observed metadata domains')
        self.check('Tensor row binds actual Storage/Version/offset',list(tensor_row[2:5]),list(node[27:30]))
        require(storage[5]>0 and storage[6]>0 and storage[7]==0 and version[3]>0,
                'actual Storage and Version are live')
        require(node[29]%width==0 and tensor_row[7]<=3,'aligned offset and retained flags')
        self.check('absent gradient has no private binding',[node[48]==0,node[49]==0],[node[45]==0]*2)
        self.check('public dtype/width/incarnation',[dtype,width,incarnation],[node[72],WIDTHS[dtype],node[40]])
        start=memory.start+storage[2]
        require(start<=len(raw) and storage[3]<=len(raw)-start,'complete actual Storage extent')
        physical=raw[start:start+storage[3]]
        rank,count=node[:2];shape,strides=list(node[12:12+rank]),list(node[16:16+rank])
        logical=bytearray()
        for index in range(count):
            remainder,offset=index,node[29]
            for axis in range(rank-1,-1,-1):
                offset+=(remainder%shape[axis])*strides[axis]*width
                remainder//=shape[axis]
            require(0<=offset<=len(physical)-width,'actual logical address inside Storage')
            logical.extend(physical[offset:offset+width])
        self.check('public readback equals actual owned logical bytes',record['payload'],bytes(logical))
        self.check('complete public element count',record['value'],count)
        self.check('complete public geometry equals actual owner',
            [meta['rank'],meta['numel'],meta['shape'],meta['strides'],meta['storage_id'],meta['version_id'],
             meta['storage_offset'],meta['storage_bytes'],meta['version'],meta['requires_grad'],leaf],
            [rank,count,list(node[12:16]),list(node[16:20]),node[27],node[28],node[29],storage[3],
             version[2],node[4],int(node[5]==0)])
        self.check('complete source element census',count,math.prod(shape))
        name='torch.float64' if dtype==3 else 'torch.'+next(k for k,v in DTYPES.items() if v==dtype)
        result=dict(dtype=name,element_size=width,shape=shape,strides=strides,numel=count,
            storage_bytes=storage[3],storage_offset=node[29]//width,version=version[2],
            requires_grad=bool(node[4]),is_leaf=node[5]==0,is_view=bool(node[51]),
            history_present=node[5]!=0,retains_grad=bool(tensor_row[7]&1),
            logical=bytes(logical),storage=physical)
        return result,node

    def expected(self,value):
        return {key:self.reference['blob'](item) if key in ('logical','storage') else item
                for key,item in value.items()}


def compare_case(n,row):
    spec=row['spec'];generator,rng=n.begin();source,inputs=n.setup(spec)
    original,source_node=n.tensor(source)
    n.check('independent actual Torch input metadata and complete bytes',original,n.expected(row['input']))
    status,records,before,after=n.issue(n.wire.packet([n.command(spec,source)]))
    n.check('actual Torch predicate acceptance',status,0)
    n.check('one matching predicate response',[r['opcode'] for r in records],[spec['opcode']])
    n.check('all 84 input descriptor words unchanged',Memory(after['bytes']).node(source),Memory(before['bytes']).node(source))
    n.check('complete input unchanged',n.tensor(source)[0],original)
    result=records[0]['value'];actual,node=n.tensor(result)
    n.check('complete independently observed Torch result',actual,n.expected(row['result']))
    n.check('fresh Tensor/Storage/Version identities',
        [result!=source,node[27]!=source_node[27],node[28]!=source_node[28]],[True]*3)
    no_edges=(32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,
              584,592,600,608,616,624,640,648,656,664)
    n.check('no gradient Saved view or input ownership edges',[node[o//8] for o in no_edges],[0]*len(no_edges))
    n.check('result does not retain gradients',actual['retains_grad'],False)
    for token in inputs:n.one(41,token)
    n.check('complete result survives input release',n.tensor(result)[0],actual)
    n.one(41,result);n.end(generator,rng)
    return dict(index=row['index'],name=spec['name'],operation=spec['operation'],accepted=True)


def packet_controls(n,row):
    generator,rng=n.begin();source,inputs=n.setup(row['spec']);original=n.tensor(source)[0]
    controls=0
    for opcode in OPERATIONS.values():
        valid=[opcode,source,0,0,0,0,0,0]
        for position in range(2,8):
            bad=valid.copy();bad[position]=1
            status,_,_,_=n.issue(n.wire.packet([bad]));n.check('reserved word %d rejected'%position,status,7);controls+=1
        for handle in (0,1,generator):
            bad=valid.copy();bad[1]=handle
            status,_,_,_=n.issue(n.wire.packet([bad]));n.check('invalid zero raw or wrong-kind handle rejected',status,1);controls+=1
        bad=valid.copy();bad[7]=1
        status,_,_,_=n.issue(n.wire.packet([valid,bad]));n.check('allocated prefix and invalid suffix rollback',status,7);controls+=1
    # The unchanged command/b117 contract reports unknown opcodes as status 4;
    # status 7 is reserved here for a malformed supported predicate command.
    status,_,_,_=n.issue(n.wire.packet([(125,source)]));n.check('closed dispatch rejects next opcode',status,4);controls+=1
    n.check('all packet failures preserve complete input',n.tensor(source)[0],original)
    for token in inputs:n.one(41,token)
    for opcode in OPERATIONS.values():
        status,_,_,_=n.issue(n.wire.packet([(opcode,source)]));n.check('stale source rejected',status,1);controls+=1
    n.end(generator,rng)
    n.check('complete additional packet control census',controls,56)
    return dict(controls=controls,complete_owner_rollback=True)


def factory_transaction(n,row,invalid_suffix=False):
    spec=row['spec'];raw=raw_input(spec);shape=spec['shape']
    require(spec['strides'] is None and not spec['requires_grad'],'literal dense transaction fixture')
    count=3 if invalid_suffix else 2;end=32+64*count
    body=b'HTORCH03'+struct.pack('<3Q',count,end+64+len(raw),0)
    body+=struct.pack('<8Q',97,spec['native_dtype'],end,0,0,0,0,0)
    body+=struct.pack('<8Q',spec['opcode'],n.wire.reference(1),0,0,0,0,0,0)
    if invalid_suffix:body+=struct.pack('<8Q',spec['opcode'],n.wire.reference(1),0,0,0,0,0,1)
    body+=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),end+64,len(raw))+raw
    return body


def transaction_control(n,row):
    generator,rng=n.begin()
    status,records,_,_=n.issue(factory_transaction(n,row))
    n.check('same-frame factory predicate success',status,0)
    n.check('same-frame records match',[r['opcode'] for r in records],[97,row['spec']['opcode']])
    n.check('same-frame result equals actual Torch',n.tensor(records[1]['value'])[0],n.expected(row['result']))
    n.one(41,records[1]['value']);n.one(41,records[0]['value'])
    maps=n.s.maps();status,_,_,_=n.issue(factory_transaction(n,row,True))
    n.check('same-frame allocated prefix and invalid suffix fail',status,7)
    n.check('failed transaction restores every actual mapping',n.s.maps(),maps)
    n.end(generator,rng)
    return dict(reference_index=row['index'],successful_FE_transactions=1,failed_FE_transactions=1,
                complete_owner_rollback=True,complete_mappings_restored=True)


def allocation_control(n,row):
    generator,rng=n.begin();source,inputs=n.setup(row['spec']);original=n.tensor(source)[0]
    n.check('OOM input equals actual Torch observation',original,n.expected(row['input']))
    n.s.lower_limit(n.s.owner());live=[];command=n.wire.packet([n.command(row['spec'],source)])
    for attempt in range(32):
        status,records,_,_=n.issue(command)
        if status:
            n.check('actual native allocation refusal',status,3)
            break
        live.append(records[0]['value'])
    else:raise ValueError('no genuine allocation refusal within the fixed 32-attempt budget')
    n.check('complete RNG survives allocation failure',n.one(63,generator)['payload'],rng)
    n.s.restore_limit();status,records,_,_=n.issue(command)
    n.check('identical request succeeds after real limit restore',status,0)
    result=records[0]['value'];live.append(result)
    n.check('retry result equals independently observed Torch',n.tensor(result)[0],n.expected(row['result']))
    n.check('OOM and retry preserve complete source',n.tensor(source)[0],original)
    for token in live+inputs:n.one(41,token)
    n.end(generator,rng)
    return dict(reference_index=row['index'],attempts=attempt+1,allocation_refusals=1,
                complete_owner_rollback=True,identical_request_succeeded_after_limit_restore=True)


def recipes(reference_data):
    rows=reference_data['report']['rows']
    result=[('case-%03d'%row['index'],compare_case,(row,)) for row in rows]
    dense=[row for row in rows if row['spec']['name']=='float32/special-bits']
    require(len(dense)==5,'all five independent float32 transaction fixtures')
    result.append(('packet-controls',packet_controls,(dense[0],)))
    result.extend(('transaction-%d'%row['spec']['opcode'],transaction_control,(row,)) for row in dense)
    allocation=[row for row in rows if row['spec']['name']=='float16/all-65536-encodings']
    require(len(allocation)==5,'all five independent exhaustive Half allocation fixtures')
    result.extend(('allocation-%d'%row['spec']['opcode'],allocation_control,(row,)) for row in allocation)
    return result
