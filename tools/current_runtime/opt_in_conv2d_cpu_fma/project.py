"""Add explicit (kh, kw, ic) fused Conv2d orders to the exact slow-only image."""
import argparse
import importlib.util
from pathlib import Path

_parent = Path(__file__).resolve().parents[1] / "opt_in_conv2d_cpu_route_slow_only/project.py"
_spec = importlib.util.spec_from_file_location("conv_cpu_slow_project", _parent)
_prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior)
one, mapping, sha = _prior.one, _prior.mapping, _prior.sha
SOURCE_PARENT = "07eab895d403e780eb52737720c36f6fd8bd7e2bb5ee7db1e9f64087c4d89084"


def project(parent):
    if sha(parent)!=SOURCE_PARENT:
        raise ValueError("unexpected exact slow-only source")
    f=mapping(parent)
    cmd=f[b"command"]
    anchor=b'  all_bias:bool = bool.or(any_bias,is_route)\n'
    patched=one(cmd,anchor,anchor+b"""  fma_opcode:u64 = const() {"value":128}
  is_fma:bool = compare.eq(v6,fma_opcode)
  new_bias:bool = bool.or(all_bias,is_fma)
""")
    patched=one(patched,b"  branch all_bias b132(",b"  branch new_bias b132(")
    anchor=b"""  route_code:u64 = const() {"value":127}
  do_route:bool = compare.eq(bop,route_code)
  branch do_route b140(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg) b141(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg)
"""
    patched=one(patched,anchor,b"""  f_code:u64 = const() {"value":128}
  do_fma:bool = compare.eq(bop,f_code)
  branch do_fma b142(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg) b143(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg)
block b142(fr:readbuf,fw0:mutbuf,fa:mutbuf,fo:u64,fp:u64,fx:u64,fy:u64,fb:u64,fs:u64,fpad:u64,fd:u64,fm:u64)
  f_result:u64 = call(fa,fx,fy,fb,fs,fpad,fd,fm) {"function":"conv_fma_explicit"}
  jump b139(fr,fw0,fa,fo,fp,f_result)
block b143(ur:readbuf,uw:mutbuf,ua:mutbuf,uo:u64,up:u64,ux:u64,uy:u64,ub:u64,us:u64,upad:u64,ud:u64,ug:u64)
  route_code:u64 = const() {"value":127}
  do_route:bool = compare.eq(up,route_code)
  branch do_route b140(ur,uw,ua,uo,up,ux,uy,ub,us,upad,ud,ug) b141(ur,uw,ua,uo,up,ux,uy,ub,us,upad,ud,ug)
""")
    kernel=Path(__file__).with_name("kernel.hepta").read_bytes()
    output=one(parent,cmd,patched)+kernel
    end=mapping(output)
    if list(end)!=[*f,*mapping(kernel)] or [k for k in f if f[k]!=end[k]]!=[b"command"]:
        raise ValueError("unqualified source change")
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    opts=p.parse_args()
    data=project(opts.parent.read_bytes())
    with opts.output.open("xb") as stream:
        stream.write(data)
    print(sha(data))
