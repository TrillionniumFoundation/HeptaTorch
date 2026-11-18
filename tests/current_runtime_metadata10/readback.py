"""Reconstruct fixed observer observations solely from saved native bytes."""
import itertools
import json
from pathlib import Path
import re
import struct


def frames(raw,magic,require):
    out=[];at=0
    while at<len(raw):
        require(at+16<=len(raw),'complete original outer frame header')
        tag,size=struct.unpack_from('<8sQ',raw,at);at+=16
        require(tag==magic and 32<=size<=1<<20 and at+size<=len(raw),'complete original outer frame extent')
        out.append(raw[at:at+size]);at+=size
    require(at==len(raw),'no trailing frame bytes')
    return out


class Reader:
    def __init__(self,common,wire,owner,node_function,folder,trace,events,row,inputs):
        self.c=common;self.w=wire;self.owner=owner;self.node_function=node_function
        self.folder=folder;self.trace=trace;self.events=events;self.row=row;self.inputs=inputs
        self.require=common.require;self.at=0;self.tensor_at=0;self.owner_at=0;self.issue_at=0
        self.checks=0;self.latest=None;self.latest_sha=None;self.used_actual_owners=set();self.used_trace_owners=set()
        self.observations=json.loads(inputs.bind(folder/'observations.json'))
        self.ledger=json.loads(inputs.bind(folder/'raw/transport.json'))
        self.requests=frames(inputs.bind(folder/'raw/requests.frames.bin'),b'HTOWNR01',self.require)
        self.responses=frames(inputs.bind(folder/'raw/responses.frames.bin'),b'HTOWNS01',self.require)
        self.require(len(self.requests)==len(self.responses)==self.ledger['frames']==row['frames']==self.observations['frames'],
                     'all original request/response/ledger frame counts')
        self.require(self.ledger['pid']==row['process_id'] and self.ledger['exit_code']==row['exit_code']==0
            and self.ledger['responses_before_stdin_eof'] is True and row['stdout_EOF'] is True
            and row['stderr_EOF'] is True and row['forced_cleanup'] is False
            and self.observations['exit_code']==0 and self.observations['stdout_EOF'] is True
            and self.observations['stderr_EOF'] is True,'complete original native process and finish report')
        self.require(len(self.ledger['rows'])==len(self.responses),'all response ledger rows')
        for index,(body,item) in enumerate(zip(self.responses,self.ledger['rows']),1):
            self.require(item==dict(frame=index,pid=self.ledger['pid'],response_bytes=len(body),stdin_open=True),
                         'exact original response before stdin EOF ledger')
        self.tensors=[event for event in events if event['method']=='Native.tensor']
        self.finishes=[event for event in events if event['method']=='OwnerSession.finish']
        self.pointers={}
        for item in row['owner_observations']:self.pointers.setdefault(item['raw_sha256'],set()).add(item['pointer'])
        for event in self.tensors:self.pointers.setdefault(event['owner']['sha256'],set()).add(event['owner']['pointer'])

    def packet(self,body):
        self.require(self.at<len(self.requests) and self.requests[self.at]==body,
                     'every original recipe command and payload byte must match the real request')
        response=self.responses[self.at];self.at+=1
        status,records=self.w.parse(response)
        self.require(status==0 and len(records)==1,'one actual successful response for every fixed command')
        self.require(records[0]['opcode']==struct.unpack_from('<Q',body,32)[0],'exact request/response opcode')
        return records[0],response

    def one(self,*command):return self.packet(self.w.packet([command]))[0]

    def create(self,shape,data,requires):
        cfg=struct.pack('<8Q',len(shape),requires,*shape,*([0]*(4-len(shape))),160,len(data))
        body=b'HTORCH03'+struct.pack('<3Q',1,160+len(data),0)+struct.pack('<8Q',97,0,96,0,0,0,0,0)+cfg+data
        self.checks+=2
        return self.packet(body)[0]['value']

    def physical(self,item,trace=False):
        if trace:
            name=item['file'];self.require(re.fullmatch(r'owners/[0-9a-f]{64}\.owner\.xz',name) is not None,'exact traced owner path')
            path=self.trace/name;size=item['bytes'];digest=item['sha256'];self.used_trace_owners.add(name)
        else:
            name=item['file'];self.require(re.fullmatch(r'[0-9a-f]{64}\.owner\.xz',name) is not None,'exact original owner path')
            path=self.folder/name;size=item['raw_bytes'];digest=item['raw_sha256'];self.used_actual_owners.add(name)
        self.require(Path(name).name==digest+'.owner.xz','owner filename is its complete raw hash')
        compressed=self.inputs.bind(path,item['compressed_sha256']);raw=self.c.single_xz(compressed)
        self.require(len(raw)==size and self.c.sha(raw)==digest,'entire physical owner bytes')
        decoded=self.owner.decode_owner(raw,item['pointer']);verified=self.owner.verify_snapshot(decoded)
        if not trace:self.require(verified==item['verified'],'all original full owner equations recomputed')
        return raw,decoded,verified

    def owner_observation(self,label):
        self.require(self.owner_at<len(self.row['owner_observations']),'all original labelled owner observations')
        item=self.row['owner_observations'][self.owner_at];self.owner_at+=1
        self.require(item['label']==label,'exact original owner observation order')
        raw,decoded,verified=self.physical(item)
        self.latest=decoded;self.latest_sha=item['raw_sha256']
        return raw,decoded

    def live(self,snapshot,kind,token):
        slot=token&0xffffff
        self.require(token>>24&255==kind and 0<slot<=snapshot['header']['slots'],'actual typed owner identity')
        row=snapshot['tables'][str(kind)][slot-1]
        self.require(row[0]==token>>32 and row[1]==1,'actual typed owner generation and liveness')
        return row

    def tensor(self,token):
        start=self.at
        metadata_record=self.one(13,token);meta=self.w.meta(metadata_record)
        typed=self.one(96,token,0)
        self.require(len(typed['payload'])==32,'complete actual dtype/incarnation/leaf observation')
        dtype,width,incarnation,leaf=struct.unpack('<4Q',typed['payload'])
        data=self.one(98,token,0,meta['numel'])['payload']
        self.require(self.tensor_at<len(self.tensors),'one passive full owner for each original tensor read')
        event=self.tensors[self.tensor_at];self.tensor_at+=1
        self.require(event['token']==token and event['start_frame']==start and event['end_frame']==self.at
            and event['end_frame']==start+3 and event['pid']==self.ledger['pid'],
            'the same token and exact 13/96/98 frames at this normal tensor return')
        raw,snapshot,verified=self.physical(event['owner'],trace=True)
        node=self.node_function(raw,token)
        self.require(list(node)==event['descriptor_words'],'all 84 original descriptor words from this same physical owner')
        self.require(len(data)==meta['numel']*width and node[72]==dtype and node[40]==incarnation
            and node[28]==meta['version_id'] and node[27]==meta['storage_id'],
            'all five unchanged Native.tensor descriptor/data predicates')
        trow=self.live(snapshot,1,token);storage=self.live(snapshot,2,meta['storage_id']);version=self.live(snapshot,3,meta['version_id'])
        self.require(trow[2]==meta['storage_id'] and trow[3]==meta['version_id']
            and trow[4]==node[29]==meta['storage_offset'] and version[2]==meta['version']
            and storage[3]==meta['storage_bytes'] and width==4 and dtype==0
            and meta['storage_offset']%width==0,'actual F32 Tensor/Storage/Version relationships')
        self.require(node[0]==meta['rank'] and node[1]==meta['numel'] and node[4]==meta['requires_grad']
            and list(node[12:16])==meta['shape'] and list(node[16:20])==meta['strides'],
            'all metadata geometry comes from this actual descriptor')
        # Copy logical element bytes from the observed Storage; do not compute
        # Tensor arithmetic or manufacture a reference value.
        base=139264+144+552*snapshot['header']['slots']+storage[2]+meta['storage_offset']
        logical=[]
        if meta['numel']:
            for indices in itertools.product(*(range(n) for n in meta['shape'][:meta['rank']])):
                offset=base+width*sum(i*s for i,s in zip(indices,meta['strides']))
                self.require(0<=offset and offset+width<=len(raw),'physical logical data extent')
                logical.append(raw[offset:offset+width])
        self.require(b''.join(logical)==data,'the complete actual GetData payload must equal observed Storage bytes')
        value=dict(dtype='torch.float32',shape=meta['shape'][:meta['rank']],strides=meta['strides'][:meta['rank']],
            numel=meta['numel'],element_size=width,storage_bytes=meta['storage_bytes'],
            storage_offset=meta['storage_offset']//width,version=meta['version'],requires_grad=bool(meta['requires_grad']),
            is_leaf=bool(leaf),is_view=bool(node[51]),grad_fn=None if node[5]==0 else 'NATIVE_HISTORY_PRESENT',logical_bytes_hex=data.hex())
        self.require(value==event['tensor'],'all passive returned metadata must reconstruct from native frames and raw owner')
        self.checks+=5;self.latest=snapshot;self.latest_sha=event['owner']['sha256']
        return value,snapshot,node

    def snapshot(self,token,label):
        value,tensor_owner,unused=self.tensor(token)
        raw,before=self.owner_observation(label+'/before-grad-read')
        self.require(before==tensor_owner,'no native command separates tensor read and before-grad observation')
        index=self.owner.public_node(before,token);gradient_index=before['nodes'][str(index)][48]
        public_before=bool(gradient_index and before['nodes'][str(gradient_index)][25])
        gradient=self.one(91,token)['value']
        self.require(self.one(59,token)['value']==bool(gradient),'original gradient presence consistency')
        if gradient:
            observed,gradient_owner,n=self.tensor(gradient)
            self.require(gradient_index and self.owner.public_node(gradient_owner,gradient)==gradient_index,
                         'this getter token resolves to the actual private gradient node at this sampling time')
            value['gradient']=observed
            if not public_before:self.one(41,gradient)
        else:
            self.require(gradient_index==0,'no private gradient when original getter returns none')
            value['gradient']=None
        raw,after=self.owner_observation(label+'/after-grad-read')
        self.require(after['nodes'][str(index)][48:50]==before['nodes'][str(index)][48:50],
                     'temporary gradient publication/release retains the original private gradient relationship')
        return value

    def capture(self,label,tokens):return [self.snapshot(token,label+'/'+str(index)) for index,token in enumerate(tokens)]

    def issue(self,command,label):
        body=self.w.packet([command]);record,response=self.packet(body)
        journal=self.observations['journal'];self.require(self.issue_at<len(journal),'original operation journal')
        row=journal[self.issue_at];self.issue_at+=1
        self.require(row['label']==label and row['status']==0
            and row['request_sha256']==self.c.sha(body) and row['response_sha256']==self.c.sha(response)
            and row['owner_before_sha256']==self.latest_sha,'exact original operation packet and preceding physical owner')
        for key in ('owner_before_sha256','owner_after_sha256'):
            digest=row[key];self.require(digest in self.pointers,'operation owner must also have an actual pointer observation')
            pointer=next(iter(self.pointers[digest]));path=self.folder/(digest+'.owner.xz')
            compressed=self.inputs.bind(path);raw=self.c.single_xz(compressed)
            self.require(self.c.sha(raw)==digest,'complete original operation owner bytes')
            self.owner.verify_snapshot(self.owner.decode_owner(raw,pointer));self.used_actual_owners.add(path.name)
        self.latest_sha=row['owner_after_sha256']
        return record

    def replay(self,recipe):
        actual={};tensors=[]
        for i,shape in enumerate(recipe['shapes']):
            values=recipe.get('values',{}).get(str(i))
            if values is None:self.require(0 in shape,'literal empty creation');raw=b''
            else:
                self.require(shape==[2,1] and values==[[1.0],[2.0]],'literal nonempty creation')
                raw=struct.pack('<2f',1.0,2.0)
            tensors.append(self.create(shape,raw,1))
        actual['inputs_before_forward']=self.capture('inputs_before_forward',tensors)
        repeated=[tensors[i] for i,count in recipe['runs'] for unused in range(count)]
        payload=self.w.Payload(struct.pack('<Q',len(repeated))+struct.pack('<'+'Q'*len(repeated),*repeated))
        output=self.issue((34,recipe['dim'],payload),'actual-cat')['value'];actual['forward_status']=0
        actual['forward_output']=self.snapshot(output,'output-after-forward')
        actual['inputs_after_forward']=self.capture('inputs_after_forward',tensors)
        seed=None
        if recipe['backward']:
            output_meta,unused,unused2=self.tensor(output)
            seed=self.create(output_meta['shape'],struct.pack('<f',1.0)*output_meta['numel'],0)
            actual['seed_before']=self.snapshot(seed,'seed-before')
            record=self.issue((58,output,seed,0),'actual-backward')
            self.require(record['value']==0,'actual complete original backward response')
            actual['backward_status']=0
            actual['seed_after']=self.snapshot(seed,'seed-after')
            actual['output_after_backward']=self.snapshot(output,'output-after-backward')
        actual['inputs_after_backward_or_skip']=self.capture('inputs_after_backward_or_skip',tensors)
        self.one(41,output)
        if seed is not None:self.one(41,seed)
        actual['inputs_after_releasing_output_and_seed']=self.capture('inputs_after_releasing_output_and_seed',tensors)
        for token in tensors:self.one(41,token)
        unused,final=self.owner_observation('after-releasing-inputs')
        self.require(self.at==len(self.requests) and self.tensor_at==len(self.tensors)
            and self.owner_at==len(self.row['owner_observations'])
            and self.issue_at==len(self.observations['journal'])
            and self.checks==self.observations['checks'],'all original frames, metadata calls, owner observations and assertions')
        self.require(self.used_actual_owners=={p.name for p in self.folder.glob('*.owner.xz')},'complete original owner archive census')
        self.require(len(self.finishes)==1,'one exact original successful finish per native child')
        finish=self.finishes[0]
        descriptors=finish['entry_descriptors']
        self.require(set(descriptors)=={'input','output','stderr'}
            and all(type(item['fd']) is int and item['fd']>=0 and type(item['device']) is int
                and type(item['inode']) is int and type(item['mode']) is int for item in descriptors.values())
            and len({item['fd'] for item in descriptors.values()})==3
            and finish['same_entry_descriptors_closed']==dict(input=True,output=True,stderr=True),
            'all three original transport descriptors were open at entry and those same descriptors closed at return')
        self.require(finish['start_frame']==finish['end_frame']==self.at
            and finish['pid']==self.ledger['pid'] and finish['expected_exit']==finish['exit_code']==0
            and finish['actual_binary_sha256']==self.ledger['executable_sha256']
            and finish['stderr_hex']==finish['stdout_tail_hex']==''
            and all(finish[key] is True for key in ('returned_None','stdin_closed','stdout_closed','stderr_closed','requests_closed','responses_closed')),
            'both actual empty reads and every closed original transport descriptor')
        for key,value in actual.items():self.require(self.row[key]==value,'reported metadata must equal full physical reconstruction: '+key)
        return actual,dict(frames=self.at,tensor_observations=self.tensor_at,owner_observations=self.owner_at,
            checks=self.checks,pid=self.ledger['pid'],original_owner_archives=len(self.used_actual_owners),
            trace_owner_archives=sorted(self.used_trace_owners),native_exit_code=0,native_EOFs=2)
