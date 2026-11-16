"""Pure PyTorch diagnostic; run only after the parent's oracle provenance gate."""
import json
import sys
from pathlib import Path
import torch


def metadata(tensor):
    return {'shape':list(tensor.shape),'stride':list(tensor.stride()),
            'numel':tensor.numel(),'contiguous':tensor.is_contiguous(),
            'requires_grad':tensor.requires_grad}


def probe():
    rows=[]
    shapes=[(0,),(2,0,3),(0,2,3),(2,3,0),
            (0,262144,262144,262144),(262144,0,262144,262144)]
    for shape in shapes:
        value=torch.empty(shape,dtype=torch.float32,device='cpu')
        rows.append({'case':'empty-'+str(shape),'metadata':metadata(value)})
    for shape in [(2,0,3),(0,2)]:
        a=torch.empty(shape,dtype=torch.float32,device='cpu',requires_grad=True)
        b=torch.empty(shape,dtype=torch.float32,device='cpu',requires_grad=True)
        for axis in [0,-1]:
            for kind in ['cat','stack']:
                out=getattr(torch,kind)([a,b],dim=axis)
                out.backward(torch.empty_like(out))
                rows.append({'case':kind+'-'+str(shape)+'-axis'+str(axis),
                             'output':metadata(out),'first_gradient':metadata(a.grad),
                             'second_gradient':metadata(b.grad)})
                a.grad=None;b.grad=None
    for axis in [-(1<<63),-999,-2,-1,0,1,999,(1<<63)-1]:
        a=torch.empty(0,dtype=torch.float32,device='cpu',requires_grad=True)
        b=torch.empty(0,dtype=torch.float32,device='cpu')
        try:
            out=torch.cat([a,b,a],dim=axis)
            out.backward(torch.empty_like(out))
            rows.append({'case':'cat-all-(0,)-axis'+str(axis),'accepted':True,
                         'output':metadata(out),'gradient':metadata(a.grad)})
        except Exception as exc:
            rows.append({'case':'cat-all-(0,)-axis'+str(axis),'accepted':False,
                         'exception_type':type(exc).__name__,'exception':str(exc)})
    return {'torch_version':str(torch.__version__),'torch_git_commit':torch.version.git_version,
            'rows':rows}


if __name__=='__main__':
    expected=json.loads(Path('compatibility/pytorch-oracle-lock.json').read_text())['oracle']
    if str(torch.__version__)!=expected['version'] or torch.version.git_version!=expected['git_commit']:
        raise RuntimeError('Empty tensor probe requires the pinned PyTorch oracle')
    report=probe()
    report.update(schema='heptatorch.pytorch-empty-semantics.v1',status='OBSERVED',cases=len(report['rows']))
    target=Path(sys.argv[1])
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
