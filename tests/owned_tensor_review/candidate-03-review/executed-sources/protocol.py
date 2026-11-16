"""Independent tensor wire observer/encoder; no tensor operation execution."""
import math
import struct


def bits(value):
    return struct.unpack('<I',struct.pack('<f',float(value)))[0]


class Program:
    def __init__(self,next_slot=1,generation=0,keep=0,version=2):
        self.next_slot,self.generation,self.keep,self.version=next_slot,generation,keep,version
        self.commands=[];self.payloads=[]

    def command(self,code,*args):
        assert len(args)<=7
        row=[code,*args,*([0]*(7-len(args)))]
        self.commands.append(row)
        return len(self.commands)-1

    def make(self,code,*args):
        result=self.next_slot if self.next_slot<=self.keep else (self.generation<<(16 if self.version==2 else 8))|self.next_slot
        self.next_slot+=1
        self.command(code,*args)
        return result

    def tensor(self,shape,values,requires=False):
        assert len(shape)<=4 and len(values)==math.prod(shape)
        result=self.make(1,len(shape),int(requires),*shape,*([0]*(4-len(shape))),0)
        self.payloads.append((len(self.commands)-1,7,b''.join(struct.pack('<I',bits(v)) for v in values),4))
        return result

    def full(self,shape,value=0,requires=False):
        assert self.version==2 and len(shape)<=4
        return self.make(38,len(shape),int(requires),*shape,*([0]*(4-len(shape))),bits(value))

    def binary(self,op,a,b):return self.make(2,op,a,b)
    def matmul(self,a,b):return self.make(32,a,b)
    def transpose(self,a,x,y):return self.make(6,a,x,y)
    def select(self,a,axis,indices):
        result=self.make(30,a,axis%(1<<64),len(indices),0)
        self.payloads.append((len(self.commands)-1,4,b''.join(struct.pack('<q',v) for v in indices),8))
        return result
    def gather(self,a,axis,shape,indices):
        assert len(indices)==math.prod(shape)
        result=self.make(31,a,axis%(1<<64),0)
        raw=struct.pack('<Q',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+b''.join(struct.pack('<q',v) for v in indices)
        self.payloads.append((len(self.commands)-1,3,raw,8))
        return result
    def read(self,t,gradient=False,start=0,count=None):
        if count is None:
            return self.command(11 if gradient else 10,t)
        assert self.version==2
        return self.command(40 if gradient else 39,t,start,count)
    def meta(self,t):return self.command(13,t)
    def backward(self,t,seed=0):return self.command(29,t,seed)
    def sgd(self,t,lr):return self.command(9,t,bits(lr))
    def reset(self,keep):
        row=self.command(33,keep)
        self.generation+=1;self.keep=keep;self.next_slot=keep+1
        return row
    def encode(self):
        commands=[r[:] for r in self.commands]
        end=32+64*len(commands);payload=bytearray()
        for row,index,data,alignment in self.payloads:
            payload.extend(bytes((-(end+len(payload)))%alignment))
            commands[row][index]=end+len(payload)
            payload.extend(data)
        raw=b''.join(struct.pack('<8Q',*r) for r in commands)+payload
        return (b'HTORCH02' if self.version==2 else b'HTORCH01')+struct.pack('<3Q',len(commands),32+len(raw),0)+raw


def parse(raw,version=2):
    magic,v,count,status=struct.unpack_from('<8s3Q',raw)
    assert magic==(b'HTORSP02' if version==2 else b'HTORSP01') and v==version
    if status and version==2:
        assert count==0 and len(raw)==32
        return status,[]
    records=[];at=32
    for _ in range(count):
        assert at+32<=len(raw)
        code,error,result,size=struct.unpack_from('<4Q',raw,at);at+=32
        assert at+size<=len(raw)
        records.append({'code':code,'status':error,'result':result,'payload':raw[at:at+size]});at+=size
    assert at==len(raw)
    return status,records


def words(record):
    raw=record['payload'];assert len(raw)%8==0
    return struct.unpack('<'+'Q'*(len(raw)//8),raw)


def f32bits(record):
    raw=record['payload'];assert len(raw)%4==0
    return struct.unpack('<'+'I'*(len(raw)//4),raw)
