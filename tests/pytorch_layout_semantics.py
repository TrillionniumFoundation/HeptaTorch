"""Pure locked-PyTorch metadata probe for layouts expressible by transpose."""
import json
import sys
from pathlib import Path
import torch


def meta(x):
    return {'shape':list(x.shape),'strides':list(x.stride()),
            'contiguous':x.is_contiguous(),
            'channels_last':x.is_contiguous(memory_format=torch.channels_last)}


def probe():
    rows=[]
    base=torch.arange(120,dtype=torch.float32,device='cpu').reshape(2,3,4,5)/8
    channels=base.transpose(1,3).transpose(2,3)
    ambiguous=torch.arange(6,dtype=torch.float32,device='cpu').reshape(2,3,1,1).transpose(1,3)
    cube=torch.arange(6,dtype=torch.float32,device='cpu').reshape(2,3,1).transpose(1,2)
    nontrivial=torch.arange(24,dtype=torch.float32,device='cpu').reshape(2,3,4).transpose(1,2)
    cases=[('cat-all-channels-axis1',[channels,channels],1,False),
           ('cat-all-channels-axis3',[channels,channels],3,False),
           ('cat-channels-with-skipped-empty',[torch.empty(0),channels],1,False),
           ('cat-channels-with-contiguous',[channels,channels.contiguous()],1,False),
           ('cat-ambiguous-singletons',[ambiguous,ambiguous],1,False),
           ('stack-last-axis-singleton-channel',[cube,cube],3,True),
           ('stack-middle-axis-singleton-channel',[cube,cube],1,True),
           ('stack-last-axis-nontrivial-channel',[nontrivial,nontrivial],3,True)]
    for name,inputs,axis,stack in cases:
        output=(torch.stack if stack else torch.cat)(inputs,dim=axis)
        rows.append({'case':name,'inputs':[meta(x) for x in inputs],
                     'axis':axis,'stack':stack,'output':meta(output),
                     'logical_values':output.reshape(-1).tolist()})
    return {'torch_version':str(torch.__version__),'torch_git_commit':torch.version.git_version,
            'rows':rows}


if __name__=='__main__':
    expected=json.loads(Path('compatibility/pytorch-oracle-lock.json').read_text())['oracle']
    if str(torch.__version__)!=expected['version'] or torch.version.git_version!=expected['git_commit']:
        raise RuntimeError('Layout probe requires the pinned PyTorch oracle')
    report=probe()
    report.update(schema='heptatorch.pytorch-layout-semantics.v1',status='OBSERVED',cases=len(report['rows']))
    target=Path(sys.argv[1]);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
