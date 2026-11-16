"""Existing HIA ownership-source witnesses, without the earlier libc harness.

Extracted complete definitions from independent fixture author SHA256
d87017d97627eb6bd06372f4120f833886455cb337d0f7b82c39484c52047c2f
This file only authors HIA source text; it executes no owner operations.
"""

HEADER = ("hia-asm.v1 hepta.hia.program.v8 selfhost_elf\n"
          "fn f0(v0:readbuf,v1:mutbuf,v2:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)\n"
          "block b0(v0:readbuf,v1:mutbuf,v2:mutbuf)\n")

def alloc(size, optional=False, prior=False):
    prefix = ('  v20:usize = const() {"value":0}\n'
              '  v21:u8 = const() {"value":99}\n'
              '  _ = buffer.store(v1,v20,v21)\n') if prior else ''
    body = f'  v3:usize = const() {{"value":{size}}}\n'
    body += ('  v4:buffer,v5:bool = buffer.try_new(v3)\n' if optional
             else '  v4:buffer = buffer.new(v3)\n')
    body += ('  v6:usize = buffer.len(v4)\n  v7:readbuf = borrow.read(v4)\n'
             '  v8:usize = const() {"value":8}\n'
             '  v9:bool = compare.lt(v6,v8)\n'
             '  v10:usize = select(v9,v6,v8)\n'
             '  _ = buffer.copy(v1,v7,v10)\n  _ = loan.end(v7)\n'
             '  _ = owner.drop(v4)\n'
             '  v11:u32 = int.cast(v6) {"mode":"wrap"}\n')
    if optional:
        body += ('  v12:u32 = const() {"value":4294967295}\n'
                 '  v13:u32 = select(v5,v11,v12)\n  ret(v13)\nend\n')
    else:
        body += '  ret(v11)\nend\n'
    return HEADER + prefix + body

MOVE = HEADER + '''  v3:usize = const() {"value":8}
  v4:buffer = buffer.new(v3)
  v5:buffer = owner.move(v4)
  v6:mutbuf = borrow.mut(v5)
  v7:u8 = const() {"value":173}
  _ = buffer.fill(v6,v7)
  _ = loan.end(v6)
  v8:readbuf = borrow.read(v5)
  _ = buffer.copy(v1,v8,v3)
  _ = loan.end(v8)
  _ = owner.drop(v5)
  v9:u32 = const() {"value":17}
  ret(v9)
end
'''

RETURN = alloc(8, True).replace('buffer.try_new(v3)', 'call(v3) {"function":"make"}') + '''fn make(v0:usize) -> (buffer,bool) effects(alloc,read,write,trap) entry(b0)
block b0(v0:usize)
  v1:buffer,v2:bool = buffer.try_new(v0)
  ret(v1,v2)
end
'''

CHAIN = HEADER + '''  v3:usize = const() {"value":8}
  v4:buffer = buffer.new(v3)
  v5:mutbuf = borrow.mut(v4)
  v6:u8 = const() {"value":93}
  _ = buffer.fill(v5,v6)
  _ = loan.end(v5)
  v7:chain = chain.empty()
  v8:chain,v9:buffer,v10:bool = chain.try_push(v7,v4)
  _ = owner.drop(v9)
  v11:chain,v12:buffer,v13:bool = chain.pop(v8)
  _ = owner.drop(v11)
  v14:usize = buffer.len(v12)
  v15:readbuf = borrow.read(v12)
  _ = buffer.copy(v1,v15,v14)
  _ = loan.end(v15)
  _ = owner.drop(v12)
  v16:u32 = int.cast(v14) {"mode":"wrap"}
  ret(v16)
end
'''

CFG = HEADER + '''  v3:usize = const() {"value":8}
  v4:buffer = buffer.new(v3)
  v5:bool = const() {"value":true}
  branch v5 b1(v4,v1) b2(v4,v1)
block b1(v6:buffer,v7:mutbuf)
  jump b3(v6,v7)
block b2(v8:buffer,v9:mutbuf)
  jump b3(v8,v9)
block b3(v10:buffer,v11:mutbuf)
  v12:readbuf = borrow.read(v10)
  v13:usize = buffer.len(v12)
  _ = buffer.copy(v11,v12,v13)
  _ = loan.end(v12)
  _ = owner.drop(v10)
  v14:u32 = int.cast(v13) {"mode":"wrap"}
  ret(v14)
end
'''

THREE = HEADER + '''  v3:buffer,v4:chain,v5:buffer = call() {"function":"make3"}
  v6:usize = buffer.len(v3)
  v7:usize = buffer.len(v5)
  v8:usize = int.add.wrap(v6,v7)
  _ = owner.drop(v3)
  _ = owner.drop(v4)
  _ = owner.drop(v5)
  v9:u32 = int.cast(v8) {"mode":"wrap"}
  ret(v9)
end
fn make3() -> (buffer,chain,buffer) effects(alloc,read,write,trap) entry(b0)
block b0()
  v0:usize = const() {"value":8}
  v1:buffer = buffer.new(v0)
  v2:chain = chain.empty()
  v3:buffer = buffer.new(v0)
  ret(v1,v2,v3)
end
'''

CONSUME = HEADER + '''  v3:usize = const() {"value":8}
  v4:buffer = buffer.new(v3)
  v5:u32 = call(v4) {"function":"take"}
  ret(v5)
end
fn take(v0:buffer) -> (u32) effects(alloc,read,write,trap) entry(b0)
block b0(v0:buffer)
  v1:usize = buffer.len(v0)
  _ = owner.drop(v0)
  v2:u32 = int.cast(v1) {"mode":"wrap"}
  ret(v2)
end
'''
