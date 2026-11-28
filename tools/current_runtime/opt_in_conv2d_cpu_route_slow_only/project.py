"""Keep automatic conv2d routing on the scoped slow eager CPU path only."""
import argparse
import importlib.util
from pathlib import Path

_parent = Path(__file__).resolve().parents[1] / "opt_in_conv2d_cpu_route_n3/project.py"
_spec = importlib.util.spec_from_file_location("conv_cpu_route_n3_project", _parent)
_prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior)
one, mapping, sha = _prior.one, _prior.mapping, _prior.sha
SOURCE_PARENT = "bf90089e0cbd96f05852b6ba8488b5f3416156298822609d65bc6a4df1fbd79f"


def project(parent):
    if sha(parent) != SOURCE_PARENT:
        raise ValueError("unexpected exact N3 route source")
    functions = mapping(parent)
    existing = functions[b"conv_cpu_route"]
    start = existing.index(b"block b5(")
    end = existing.index(b"block b7(",start)
    new = b"""block b5(ra:mutbuf,ri:u64,rw:u64,rb:u64,rs:u64,rp:u64,rd:u64,re:u64,rn:u64)
  enabled:u64 = const() {"value":1}
  pref:bool = compare.eq(re,enabled)
  multiple_batches:bool = compare.ne(rn,enabled)
  unqualified:bool = bool.and(pref,multiple_batches)
  branch unqualified b2(ra) b7(ra,ri,rw,rb,rs,rp,rd)
"""
    output = one(parent,existing,existing[:start]+new+existing[end:])
    final = mapping(output)
    if list(final) != list(functions) or [k for k in functions if final[k]!=functions[k]] != [b"conv_cpu_route"]:
        raise ValueError("unqualified native change")
    if b"conv_bias_forward" in final[b"conv_cpu_route"]:
        raise ValueError("auto route still reaches explicit post-sum")
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
