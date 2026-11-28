"""Extend the exact selector to N=3 only when oneDNN is disabled."""
import argparse
import importlib.util
from pathlib import Path

_parent = Path(__file__).resolve().parents[1] / "opt_in_conv2d_cpu_route/project.py"
_spec = importlib.util.spec_from_file_location("conv_cpu_route_previous_project", _parent)
_prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior)
one, mapping, sha = _prior.one, _prior.mapping, _prior.sha
SOURCE_PARENT = "4cd14a332afc3cad5c67f9f12412591985021fa51af84836a558242e4339f8db"


def project(parent):
    if sha(parent) != SOURCE_PARENT:
        raise ValueError("unexpected exact selector source")
    functions = mapping(parent)
    shape = functions[b"conv_cpu_route_shape"]
    route = functions[b"conv_cpu_route"]
    more_shape = one(shape, b"  x0_ok:bool = bool.or(x0_one,x0_two)\n", b"""  x0_three:bool = compare.eq(x0,three)
  x0_small:bool = bool.or(x0_one,x0_two)
  x0_ok:bool = bool.or(x0_small,x0_three)
""")
    more_route = one(route, b"""  post:bool = bool.and(pref,many)
  branch post b6(ra,ri,rw,rb,rs,rp,rd) b7(ra,ri,rw,rb,rs,rp,rd)
""", b"""  post:bool = bool.and(pref,many)
  n3:u64 = const() {"value":3}
  large:bool = compare.eq(rn,n3)
  unqualified:bool = bool.and(pref,large)
  branch unqualified b2(ra) b8(ra,ri,rw,rb,rs,rp,rd,post)
block b8(next_a:mutbuf,next_i:u64,next_w:u64,next_b:u64,next_s:u64,next_p:u64,next_d:u64,post_ok:bool)
  branch post_ok b6(next_a,next_i,next_w,next_b,next_s,next_p,next_d) b7(next_a,next_i,next_w,next_b,next_s,next_p,next_d)
""")
    output = one(one(parent,shape,more_shape),route,more_route)
    final = mapping(output)
    if list(final) != list(functions) or [k for k in functions if final[k]!=functions[k]] != [
            b"conv_cpu_route_shape",b"conv_cpu_route"]:
        raise ValueError("unqualified native change")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=project(args.parent.read_bytes())
    with args.output.open("xb") as f:
        f.write(data)
    print(sha(data))
