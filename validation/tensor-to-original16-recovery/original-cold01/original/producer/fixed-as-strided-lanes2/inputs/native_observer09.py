"""Transport and read-only native-owned-byte observation; no Tensor evaluator."""
import hashlib
import importlib.util
from pathlib import Path
import re
import struct

PROTOCOL='7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
TAG=1<<63
BASE=139264
VIEW_BASE_FIELD=640
MASK=(1<<64)-1
sha=lambda b:hashlib.sha256(b).hexdigest()


def wire(path):
    assert sha(path.read_bytes())==PROTOCOL
    spec=importlib.util.spec_from_file_location('int64_test_transport',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Native:
    def __init__(self,binary,source,protocol,output):
        self.w=wire(protocol);self.s=self.w.Session(binary.resolve(),directory=output/'raw')
        raw=source.read_bytes();self.observer_bodies={}
        for name in ('object_address','so_data_base','so_slot_at','dptr'):
            self.observer_bodies[name]=re.search(rb'^fn '+name.encode()+rb'\(.*?^end\n',raw,re.M|re.S).group()
        assert b'{"value":139264}' in self.observer_bodies['object_address'] and b'{"value":552}' in self.observer_bodies['so_data_base'] and b'{"value":672}' in self.observer_bodies['dptr']

    def one(self,*cmd):return self.s.run([cmd])[0]

    def create(self,dtype,shape,raw,requires=0):
        cfg=struct.pack('<8Q',len(shape),requires,*shape,*([0]*(4-len(shape))),160,len(raw))
        packet=b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,dtype,96,0,0,0,0,0)+cfg+raw
        status,records=self.w.parse(self.s.exchange(packet));assert status==0 and len(records)==1,('create',status)
        return records[0]['value']

    def view(self,token,shape,strides,offset):
        cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),offset)
        return self.one(69,token,self.w.Payload(cfg),80)['value']

    def memory(self):return Memory(self.s.owner()['bytes'])

    def tensor(self,token):
        meta=self.w.meta(self.one(13,token));dtype,width,inc,leaf=struct.unpack('<4Q',self.one(96,token,0)['payload'])
        logical=self.one(98,token,0,meta['numel'])['payload'];memory=self.memory();node=memory.public_node(token);observed=memory.tensor(node)
        assert observed['logical_bytes_hex']==logical.hex() and observed['native_incarnation']==inc and observed['native_storage_id']==meta['storage_id'] and observed['native_version_id']==meta['version_id']
        assert observed['element_size']==width and observed['is_leaf']==bool(leaf) and observed['version']==meta['version']
        assert observed['storage_offset_elements']*width==meta['storage_offset'] and observed['shape']==meta['shape'][:meta['rank']] and observed['strides']==meta['strides'][:meta['rank']]
        observed['native_public_token']=token
        return observed

    def finish(self):return self.s.finish()


class Memory:
    def __init__(self,raw):
        self.raw=raw;assert len(raw)>BASE+128
        assert raw[64:72] in (b'HTTOCP08',) and raw[BASE:BASE+8] in (b'HTOTOCP8',)
        self.slots=self.q(BASE+16);self.pool=self.q(BASE+24);self.start=BASE+144+552*self.slots
        assert self.start+self.pool==len(raw)

    def q(self,at):
        assert 0<=at<=len(self.raw)-8
        return struct.unpack_from('<Q',self.raw,at)[0]

    def row(self,kind,token):
        slot=token&0xffffff;assert token>>24&255==kind and 0<slot<=self.slots
        at=BASE+128+((kind-1)*self.slots+slot-1)*64;value=struct.unpack_from('<8Q',self.raw,at)
        assert value[0]==token>>32 and value[1]==1
        return value

    def public_node(self,token):
        row=self.row(1,token);node=row[6];d=self.node(node);assert d[24]==token and d[25]==1 and d[30]==1
        return node

    def node(self,index):
        assert 0<index<self.q(8)
        address=self.q(4096+(index>>8)*8);assert address&TAG
        at=self.start+(address&(TAG-1))+(index&255)*672
        assert at<=len(self.raw)-672
        return struct.unpack_from('<84Q',self.raw,at)

    def tensor(self,index):
        node=self.node(index);assert node[38]==1 and node[30]==1
        self.row(1,node[24]);storage=self.row(2,node[27]);version=self.row(3,node[28])
        assert storage[5]>0 and storage[7]==0
        rank,n=node[0],node[1];shape=list(node[12:12+rank]);strides=list(node[16:16+rank]);dtype=node[72];width={0:4,1:1,2:8}[dtype]
        assert node[29]%width==0;offset=node[29];physical_at=self.start+storage[2]
        assert physical_at<=len(self.raw) and storage[3]<=len(self.raw)-physical_at
        physical=self.raw[physical_at:physical_at+storage[3]];logical=bytearray()
        for i in range(n):
            remaining=i;address=offset
            for axis in range(rank-1,-1,-1):address+=(remaining%shape[axis])*strides[axis]*width;remaining//=shape[axis]
            assert address<=len(physical) and width<=len(physical)-address
            logical.extend(physical[address:address+width])
        root=None
        if node[83] != 0:
            rootnode=self.node(node[80]);assert rootnode[38] and rootnode[40]==node[81];root=rootnode[40]
        return {'dtype':{0:'torch.float32',1:'torch.bool',2:'torch.int64'}[dtype],'element_size':width,
                'shape':shape,'strides':strides,'numel':n,'storage_bytes':storage[3],
                'storage_offset_elements':offset//width,'version':version[2],
                'requires_grad':bool(node[4]),'is_leaf':node[5]==0,'is_view':bool(node[51]),
                'grad_fn':None if node[5]==0 else 'NATIVE_HISTORY_PRESENT',
                'logical_bytes':len(logical),'logical_bytes_hex':logical.hex(),'storage_bytes_hex':physical.hex(),
                'native_node':index,'native_incarnation':node[40],'native_storage_id':node[27],
                'native_version_id':node[28],'native_base_incarnation':root}
