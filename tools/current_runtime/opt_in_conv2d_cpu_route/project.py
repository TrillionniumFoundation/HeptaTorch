"""Add a restricted native selector to the exact explicit-bias-order image."""
import argparse
import importlib.util
from pathlib import Path

_upstream = Path(__file__).resolve().parents[1] / "opt_in_conv2d_bias_first/project.py"
_spec = importlib.util.spec_from_file_location("conv_route_project_base", _upstream)
_up = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_up)
mapping, one, sha = _up.mapping, _up.one, _up.sha
PARENT = "318508540c44cdc60c18cb3a876c13b06a6cb60a34153ad4be621e1a964df4f8"


def geometry():
    """Require exact dimensional coverage before calling either native kernel."""
    text = """fn conv_cpu_route_shape(a:mutbuf,x:u64,w:u64,b:u64) -> (u64) effects(read,write,trap) entry(b0)
block b0(a:mutbuf,x:u64,w:u64,b:u64)
  zero:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  two:u64 = const() {"value":2}
  three:u64 = const() {"value":3}
  four:u64 = const() {"value":4}
  dx:u64 = call(a,x) {"function":"dptr"}
  dw:u64 = call(a,w) {"function":"dptr"}
  db:u64 = call(a,b) {"function":"dptr"}
  rx:u64 = call(a,dx) {"function":"getw"}
  rw:u64 = call(a,dw) {"function":"getw"}
  rb:u64 = call(a,db) {"function":"getw"}
  rokx:bool = compare.eq(rx,four)
  rokw:bool = compare.eq(rw,four)
  rokb:bool = compare.eq(rb,one)
  rok1:bool = bool.and(rokx,rokw)
  rok:bool = bool.and(rok1,rokb)
"""
    for i in range(4):
        text += f'  o{i}:u64 = const() {{"value":{96+8*i}}}\n'
    last = "rok"
    for node, expected in (("x",("batch","two","four","four")),
                           ("w",("two","two","three","three")),("b",("two",))):
        for i, want in enumerate(expected):
            name = f"{node}{i}"
            text += f'  {name}_ptr:u64 = int.add.wrap(d{node},o{i})\n'
            text += f'  {name}:u64 = call(a,{name}_ptr) {{"function":"getw"}}\n'
            if want == "batch":
                text += f'  {name}_one:bool = compare.eq({name},one)\n'
                text += f'  {name}_two:bool = compare.eq({name},two)\n'
                text += f'  {name}_ok:bool = bool.or({name}_one,{name}_two)\n'
            else:
                text += f'  {name}_ok:bool = compare.eq({name},{want})\n'
            text += f'  {name}_all:bool = bool.and({last},{name}_ok)\n'
            last = name + "_all"
    text += f"""  branch {last} b1(x0) b2()
block b1(n:u64)
  ret(n)
block b2()
  absent:u64 = const() {{"value":0}}
  ret(absent)
end
"""
    return text.encode()


ROUTE = b"""fn conv_cpu_route(a:mutbuf,xpub:u64,wpub:u64,bpub:u64,st:u64,pad:u64,dil:u64,mkldnn:u64) -> (u64) effects(read,write,trap) entry(b0)
block b0(a:mutbuf,xpub:u64,wpub:u64,bpub:u64,st:u64,pad:u64,dil:u64,mkldnn:u64)
  onepair:u64 = const() {"value":4294967297}
  zero:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  st_ok:bool = compare.eq(st,onepair)
  pad_ok:bool = compare.eq(pad,zero)
  dil_ok:bool = compare.eq(dil,onepair)
  flag0:bool = compare.eq(mkldnn,zero)
  flag1:bool = compare.eq(mkldnn,one)
  flag_ok:bool = bool.or(flag0,flag1)
  first_ok:bool = bool.and(st_ok,pad_ok)
  rest_ok:bool = bool.and(dil_ok,flag_ok)
  accepts:bool = bool.and(first_ok,rest_ok)
  branch accepts b1(a,xpub,wpub,bpub,st,pad,dil,mkldnn) b2(a)
block b1(ar:mutbuf,ix:u64,iw:u64,ib:u64,s:u64,p:u64,d:u64,en:u64)
  x:u64 = call(ar,ix) {"function":"public_f32"}
  w:u64 = call(ar,iw) {"function":"public_f32"}
  b:u64 = call(ar,ib) {"function":"public_f32"}
  z:u64 = const() {"value":0}
  vx:bool = compare.ne(x,z)
  vw:bool = compare.ne(w,z)
  vb:bool = compare.ne(b,z)
  vxw:bool = bool.and(vx,vw)
  v:bool = bool.and(vxw,vb)
  branch v b3(ar,ix,iw,ib,s,p,d,en,x,w,b) b4()
block b2(ea:mutbuf)
  unsupported:u64 = const() {"value":10}
  refused:u64 = call(ea,unsupported) {"function":"terror"}
  ret(refused)
block b3(ga:mutbuf,gi:u64,gw:u64,gb:u64,gs:u64,gp:u64,gd:u64,ge:u64,gx:u64,gy:u64,gz:u64)
  batch:u64 = call(ga,gx,gy,gz) {"function":"conv_cpu_route_shape"}
  no_batch:u64 = const() {"value":0}
  known:bool = compare.ne(batch,no_batch)
  branch known b5(ga,gi,gw,gb,gs,gp,gd,ge,batch) b2(ga)
block b4()
  no_handle:u64 = const() {"value":0}
  ret(no_handle)
block b5(ra:mutbuf,ri:u64,rw:u64,rb:u64,rs:u64,rp:u64,rd:u64,re:u64,rn:u64)
  enabled:u64 = const() {"value":1}
  n2:u64 = const() {"value":2}
  pref:bool = compare.eq(re,enabled)
  many:bool = compare.eq(rn,n2)
  post:bool = bool.and(pref,many)
  branch post b6(ra,ri,rw,rb,rs,rp,rd) b7(ra,ri,rw,rb,rs,rp,rd)
block b6(pa:mutbuf,pi:u64,pw:u64,pb:u64,ps:u64,pp:u64,pd:u64)
  pg:u64 = const() {"value":1}
  po:u64 = call(pa,pi,pw,pb,ps,pp,pd,pg) {"function":"conv_bias_forward"}
  ret(po)
block b7(fa:mutbuf,fi:u64,fw:u64,fb:u64,fs:u64,fp:u64,fd:u64)
  fg:u64 = const() {"value":1}
  fo:u64 = call(fa,fi,fw,fb,fs,fp,fd,fg) {"function":"conv_bias_first"}
  ret(fo)
end
"""


def project(parent):
    if sha(parent) != PARENT:
        raise ValueError("unexpected exact explicit-order parent")
    f = mapping(parent)
    cmd = f[b"command"]
    anchor = b'  any_bias:bool = bool.or(is_bias,is_first)\n'
    new = one(cmd, anchor, anchor + b"""  route_opcode:u64 = const() {"value":127}
  is_route:bool = compare.eq(v6,route_opcode)
  all_bias:bool = bool.or(any_bias,is_route)
""")
    new = one(new, b"  branch any_bias b132(", b"  branch all_bias b132(")
    anchor = b"""  pre_code:u64 = const() {"value":126}
  do_pre:bool = compare.eq(bop,pre_code)
  branch do_pre b137(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg) b138(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg)
"""
    new = one(new, anchor, b"""  route_code:u64 = const() {"value":127}
  do_route:bool = compare.eq(bop,route_code)
  branch do_route b140(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg) b141(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg)
block b140(qr:readbuf,qw:mutbuf,qa:mutbuf,qo:u64,qp:u64,qx:u64,qy:u64,qb:u64,qs:u64,qpad:u64,qd:u64,qe:u64)
  selected:u64 = call(qa,qx,qy,qb,qs,qpad,qd,qe) {"function":"conv_cpu_route"}
  jump b139(qr,qw,qa,qo,qp,selected)
block b141(tr:readbuf,tw:mutbuf,ta:mutbuf,to:u64,tp:u64,tx:u64,ty:u64,tb:u64,ts:u64,tpad:u64,td:u64,tg:u64)
  pre_code:u64 = const() {"value":126}
  do_pre:bool = compare.eq(tp,pre_code)
  branch do_pre b137(tr,tw,ta,to,tp,tx,ty,tb,ts,tpad,td,tg) b138(tr,tw,ta,to,tp,tx,ty,tb,ts,tpad,td,tg)
""")
    out = one(parent, cmd, new) + geometry() + ROUTE
    fn = mapping(out)
    if list(fn) != [*f, b"conv_cpu_route_shape", b"conv_cpu_route"]:
        raise ValueError("unexpected altered function set")
    if [k for k in f if f[k] != fn[k]] != [b"command"]:
        raise ValueError("unexpected modified native function")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = project(args.parent.read_bytes())
    with args.output.open("xb") as stream:
        stream.write(result)
    print(sha(result))
