"""Pin a backend-conditioned Conv2d bias rounding mismatch, not a parity pass."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import tempfile

_BASE = Path(__file__).resolve().parents[1] / "opt_in_conv2d_bias" / "qualify.py"
_spec = importlib.util.spec_from_file_location("biased_conv_qualification", _BASE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
wire, custody = base.wire, base.custody
# Little-endian F32 terms and bias, exact first-word counter-observations.
# Name, first weight, last weight, bias, expected sum, expected pre, expected post.
CASES = (
    ("half-ulp", "0000803f", "00008033", "000080bf", "0000803f", "00008033", "00000000"),
    ("decimal-cancellation", "cdcccc3d", "cdcc4c3e", "9a9999be", "9a99993e", "000080b2", "00000000"),
    ("large-cancellation", "0050c347", "7e454a3c", "0050c3c7", "0250c347", "7e454a3c", "0000803c"),
    ("small-decimal", "3333333f", "95bfd632", "333333bf", "3333333f", "95bfd632", "00000000"),
    ("negative-small-decimal", "7b146ec0", "498b7cb2", "7b146e40", "7b146ec0", "498b7cb2", "00000000"),
    ("one-ulp", "cdcccc3d", "cdcc4c3e", "cdcc4cbe", "9a99993e", "cdcccc3d", "cecccc3d"),
    ("agreement-control", "cdcccc3d", "cdcc4c3e", "77cc2b32", "9a99993e", "9a99993e", "9a99993e"),
)


def value(hexword):
    return struct.unpack("<f", bytes.fromhex(hexword))[0]


def observe(torch, x, w, bias):
    with torch.autograd.profiler.profile(use_cpu=True) as prof:
        direct = torch.nn.functional.conv2d(x, w, bias)
    keys = {event.key for event in prof.key_averages()}
    families = keys.intersection({"aten::_slow_conv2d_forward", "aten::mkldnn_convolution"})
    wire.need(len(families)==1, "unexpected eager backend in actual trace")
    return direct, families.pop()


def run(binary, wheel, install):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    wire.need(torch.__version__=="2.15.0+cpu" and torch.backends.cpu.get_cpu_capability()=="AVX2",
              "requires the locked Torch reference")
    wire.need(torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled,
              "oneDNN must be present and enabled initially")
    reference = custody.verify_torch_custody(wheel, install, torch)
    rows = []
    with tempfile.TemporaryDirectory(prefix="heptatorch-bias-rounding-") as tmp:
        exe = wire.stage_executable(Path(tmp), wire.pinned_bytes(binary,base.ELF,wire.ELF_CAP))
        for n in (1,2):
            fixture = ((n,2,4,4),(2,2,3,3),(1,1),(0,0),(1,1),1)
            xs,ws = fixture[:2]
            target = (n,2,2,2)
            count = n*8
            for label, first, last, bval, sumword, preword, postword in CASES:
                x = torch.ones(xs, dtype=torch.float32)
                w = torch.zeros(ws, dtype=torch.float32)
                w[:,0,0,0] = value(first)
                w[:,1,2,2] = value(last)
                b = torch.full((2,), value(bval), dtype=torch.float32)
                raw = tuple(custody.values(t) for t in (x,w,b))
                candidate = base.measure(exe,fixture,raw,target)
                candidate_words = candidate.pop("output")
                nobias = base.previous.run_native(exe,fixture,raw[:2],(False,False),None,target)
                wire.need(nobias["output"]==bytes.fromhex(sumword)*count,
                          "unexpected native bias-free reduction")
                wire.need(candidate_words==bytes.fromhex(postword)*count,
                          "unexpected native post-sum bias output")
                # All products except the two named terms are exact zero.
                pre = ((b+w[:,0,0,0])+w[:,1,2,2]).view(1,2,1,1).expand(target)
                wire.need(custody.values(pre)==bytes.fromhex(preword)*count,
                          "pre-sum discriminator changed")
                binding = torch.backends.mkldnn.enabled
                try:
                    for enabled in (True,False):
                        torch.backends.mkldnn.enabled = enabled
                        direct,family = observe(torch,x,w,b)
                        is_mkldnn = n==2 and enabled
                        expected_family = (
                            "aten::mkldnn_convolution" if is_mkldnn else "aten::_slow_conv2d_forward")
                        wire.need(family==expected_family, f"expected {expected_family}, got {family}")
                        eager_sum = torch.nn.functional.conv2d(x,w)
                        eager_post = eager_sum+b.view(1,2,1,1)
                        wire.need(custody.values(eager_sum)==nobias["output"],
                                  "case does not isolate bias addition from convolution reduction")
                        wire.need(custody.values(eager_post)==candidate_words,
                                  "post-sum comparator changed")
                        actual=custody.values(direct)
                        expected=postword if is_mkldnn else preword
                        wire.need(actual==bytes.fromhex(expected)*count,
                                  "direct eager bit pattern changed")
                        wire.need(tuple(custody.values(t) for t in (x,w,b))==raw,"eager input mutation")
                        rows.append(dict(name=label,geometry=fixture,one_dnn_enabled=enabled,
                                         eager_profiler_kernel=family,shape=target,words=count,
                                         term_and_bias_le_hex=[first,last,bval],
                                         source_sha256=list(map(custody.sha,raw)),
                                         native_nobias_le_hex=sumword,native_post_bias_le_hex=postword,
                                         eager_pre_bias_le_hex=preword,eager_direct_le_hex=expected,
                                         observed_mismatch=actual!=candidate_words,
                                         native_biased_frames=candidate["native_frames"],
                                         native_bias_free_frames=nobias["frames"]))
                finally:
                    torch.backends.mkldnn.enabled=binding
    return dict(schema="heptatorch.conv2d-bias-rounding-blocker.v1",status="OBSERVED_MISMATCH",
                native_sha256=base.ELF,whole_f32_bias_bit_parity=False,whole_pytorch_qualification=False,
                reference=dict(torch=str(torch.__version__),cpu="AVX2",threads=1,**reference),observations=rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","wheel","install"):
        parser.add_argument("--"+name,type=Path,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),sort_keys=True,indent=2))
