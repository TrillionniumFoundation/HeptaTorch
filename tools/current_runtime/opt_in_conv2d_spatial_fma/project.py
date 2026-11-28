"""Combine exact #102 spatial/AD ownership and the explicit CPU FMA source."""
import argparse
import importlib.util
from pathlib import Path

_p=Path(__file__).resolve().parents[1]/"opt_in_conv2d_cpu_fma/project.py"
_spec=importlib.util.spec_from_file_location("explicit_cpu_fma_project",_p)
_fma=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fma)
one,mapping,sha=_fma.one,_fma.mapping,_fma.sha
PARENT="139eff54155623e29b8ad31f8d13167d66e1b607acf60cd6db39290c9fdd71b3"
SPATIAL="f58637bb9eda972f9a335c26f286538211390ea8b09a63eaf66c2e508e48f91e"
COMMON="318508540c44cdc60c18cb3a876c13b06a6cb60a34153ad4be621e1a964df4f8"
REPLACED=(b"unary",b"partial_apply_node",b"gradient_direct",b"record_body",
          b"to_original_incoming_describe",b"entry",b"predicate_command",
          b"conv2d",b"conv_first_geometry")
ADDED=(b"opt_in_square_f32",b"opt_in_square_f32_backward",
       b"opt_in_square_f32_gradient_word",b"opt_in_unsqueeze_command",
       b"opt_in_unsqueeze",b"opt_in_unsqueeze_geometry")


def project(parent, spatial, common):
    if (sha(parent),sha(spatial),sha(common))!=(PARENT,SPATIAL,COMMON):
        raise ValueError("unexpected exact runtime source")
    base, contrib, shared = map(mapping,(parent,spatial,common))
    if {k for k in shared if contrib[k]!=shared[k]} != set(REPLACED):
        raise ValueError("unexpected spatial modifications")
    if set(contrib)-set(shared)!=set(ADDED):
        raise ValueError("unexpected spatial additions")
    out=parent
    for name in REPLACED:
        if base[name]!=shared[name]:
            raise ValueError(f"conflicting CPU and spatial edits in {name!r}")
        out=one(out,base[name],contrib[name])
    for name in ADDED:
        if name in base:
            raise ValueError(f"unexpected competing definition {name!r}")
        out+=contrib[name]
    finish=mapping(out)
    if list(finish)!=[*base,*ADDED] or {k for k in base if finish[k]!=base[k]}!=set(REPLACED):
        raise ValueError("unqualified composition output")
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("parent","spatial","common","output"):
        p.add_argument("--"+k,type=Path,required=True)
    opts=p.parse_args()
    data=project(opts.parent.read_bytes(),opts.spatial.read_bytes(),opts.common.read_bytes())
    with opts.output.open("xb") as stream:
        stream.write(data)
    print(sha(data))
