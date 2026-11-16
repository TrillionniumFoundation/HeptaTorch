"""Independent CPython/std::mt19937 qualification model; never production."""
import random
import struct
import subprocess

MASK=(1<<64)-1
f32=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
f64=lambda x:struct.unpack('<Q',struct.pack('<d',x))[0]

class Reference:
    def __init__(self,cpp,seed=0,state=None):
        self.cpp=cpp
        if state is None:self.seed(seed)
        else:self.restore(state)

    def seed(self,seed):
        p=subprocess.run([str(self.cpp),str(seed),str(0)],capture_output=True,check=True)
        assert not p.stderr and len(p.stdout)==625*8
        words=struct.unpack('<625Q',p.stdout)
        self.rng=random.Random();self.rng.setstate((3,tuple(words),None))
        self.seed_value=seed;self.fresh=True
        self.double_cache=0;self.double_valid=0;self.float_cache=0;self.float_valid=0

    def restore(self,state):
        assert len(state)==5056
        self.seed_value,left,seeded,index=struct.unpack_from('<QIIQ',state)
        index&=0xffffffff
        assert seeded and 1<=left<=624 and index<=624
        assert (left,index)==(1,0) or index and left+index==625
        words=[x&0xffffffff for x in struct.unpack_from('<624Q',state,24)]
        self.rng=random.Random();self.rng.setstate((3,tuple(words+[index if index else 624]),None))
        self.fresh=index==0
        self.double_valid=struct.unpack_from('<I',state,5040)[0];self.float_valid=state[5052]
        assert self.double_valid in (0,1) and self.float_valid in (0,1)
        self.double_cache=struct.unpack_from('<Q',state,5024)[0] if self.double_valid else 0
        self.float_cache=struct.unpack_from('<I',state,5048)[0] if self.float_valid else 0

    def state(self):
        words=self.rng.getstate()[1];index=0 if self.fresh else words[-1]
        raw=bytearray(5056);struct.pack_into('<QIIQ',raw,0,self.seed_value,1 if self.fresh else 625-index,1,index)
        struct.pack_into('<624Q',raw,24,*words[:624]);struct.pack_into('<Q',raw,5024,self.double_cache)
        struct.pack_into('<I',raw,5040,self.double_valid);struct.pack_into('<I',raw,5048,self.float_cache);raw[5052]=self.float_valid
        return bytes(raw)

    def word(self):self.fresh=False;return self.rng.getrandbits(32)
    def double_word(self):return (self.word()<<32)|self.word()

    def draw(self,op,count,a=0,b=0):
        values=[]
        for i in range(count):
            if op in ('advance','raw32','rand_f32','rand_f16','rand_bf16'):v=self.word()
            elif op in ('raw64','rand_f64'):v=self.double_word()
            elif op=='rand_int64':
                width=b-a;v=(self.double_word() if width>=1<<28 else self.word())%width+a
            else:raise ValueError(op)
            if op in ('rand_f32','rand_f16','rand_bf16'):
                unit=(v&((1<<24)-1))/(1<<24)
                if op=='rand_f32':v=f32(unit)
                elif op=='rand_f16':
                    v=struct.unpack('<H',struct.pack('<e',unit))[0]
                    if v==0x3c00:v=0
                else:
                    bits=f32(unit);v=(bits+0x7fff+((bits>>16)&1))>>16
                    if v==0x3f80:v=0
            elif op=='rand_f64':v=f64((v&((1<<53)-1))/(1<<53))
            if op!='advance':values.append(v&MASK)
        return values

def untemper(value):
    x=value
    for _ in range(3):x=value^(x>>18)
    y=x
    for _ in range(3):x=y^((x<<15)&0xefc60000)
    y=x
    for _ in range(5):x=y^((x<<7)&0x9d2c5680)
    y=x
    for _ in range(4):x=y^(x>>11)
    return x&0xffffffff
