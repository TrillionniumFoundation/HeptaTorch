import hashlib,json,struct,sys
from pathlib import Path
H,W,P,E,D=map(Path,sys.argv[1:]);sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
w=load_wire(W)
def verdict(path,output):
 s=w.Session(path,directory=output)
 def fr(*a):
  status,x=w.parse(s.exchange(w.packet(a)));return status,x
 t,r=w.parse(s.exchange(b''.join(factory(4,[0],[]))));assert t==0
 h=r[0]['value']
 vals=[]
 empty=w.parse(s.exchange(b''.join(factory(4,[0],[]))))[1][0]['value']
 mask=w.parse(s.exchange(b''.join(factory(1,[0],[]))))[1][0]['value']
 status,o=fr((129,mask,h,empty));assert status==0
 fresh=o[0]['value']
 for origin,handle in [('constructor',h),('ordinary-same-dtype-where',fresh)]:
  vals.append((origin,fr((13,handle))[0]))
  for opcode,shape in ((112,[0]),(113,[0,1])):
   config=w.Payload(struct.pack('<5Q',len(shape),*shape,*[0]*(4-len(shape))),8)
   b=s.owner();status,r=fr((opcode,handle,config,40));assert status==7 and not r and s.owner()==b
   vals.append((origin,opcode,status))
 for a in ((112,h,w.Payload(struct.pack('<5Q',1,0,0,0,0),8),40),(113,h,w.Payload(struct.pack('<5Q',2,0,1,0,0),8),40),
           (19,h),(98,h,0,1),(12,h,0,0x3f800000),
           (111,h,w.Payload(struct.pack('<10Q',1,0,0,0,0,1,0,0,0,0),8),80)):
  b=s.owner(); status,r=fr(a); z=s.owner()
  if status: assert b==z
  vals.append((a[0],status))
 s.finish();return vals
x=verdict(P,D/'diagnose-parent-review');y=verdict(E,D/'diagnose-candidate-review')
assert x==y
print(x,y)
