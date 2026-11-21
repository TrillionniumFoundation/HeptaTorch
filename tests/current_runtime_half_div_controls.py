"""Six exact division successors in the original312/1006 arithmetic observers.

The independent1196 reference is read in full. No new expected values are
computed here. Every old non-division request and numerical check stays in
its original observer; only its exact operation4 control receives this path.
"""
import importlib.util
from pathlib import Path


LABELS={
    'half_mul':tuple(f'promotion-{dtype}-{dtype}-3' for dtype in (5,6)),
    'half_addsub':tuple(f'mul-control-promotion-{dtype}-{dtype}-3:operation-{operation}'
        for dtype in (5,6) for operation in (1,2)),
}


def require(ok,message):
    if not ok:raise ValueError(message)


def read_reference(repository,directory,execution,inputs):
    root=Path(repository).resolve(strict=True)
    inputs.bind(Path(__file__).resolve())
    path=root/'tests/half_div/run.py';inputs.bind(path)
    spec=importlib.util.spec_from_file_location('half_div_control_reference',path)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    result=reader.read_reference(directory,execution,inputs)
    require(result['report']['cases']==1196,'complete independent1196 division reference')
    return result


def compare(n,row,left,right,reference,group):
    from numeric_predicates.native import Memory
    source=row['spec'];dtype=source['left']['dtype']
    require(group in LABELS and source['name'] in LABELS[group]
        and source.get('controls') is True and dtype in (5,6)
        and source['right']['dtype']==dtype,'only the six exact original division controls')
    rows=[item for item in reference['report']['rows']
        if item['spec']['name']==f'promotion-{dtype}-{dtype}-3']
    require(len(rows)==1 and rows[0]['accepted'] is True and rows[0]['spec']['operation']==4
        and rows[0]['spec']['left']==source['left'] and rows[0]['spec']['right']==source['right'],
        'the entire original literal operands match one actual division observation')
    successor=rows[0]
    def expected(value):
        return {key:reference['blob'](item) if key in ('logical','storage') else item
            for key,item in value.items()}
    before_values=[n.tensor(left)[0],n.tensor(right)[0]]
    n.check('original division-control operands equal actual independent Torch inputs',
        before_values,[expected(value) for value in successor['inputs']])
    status,records,before,after=n.issue(n.wire.packet([(2,4,left,right)]))
    n.check('original division refusal now matches actual Torch success',status,0)
    n.check('exact original division response',[record['opcode'] for record in records],[2])
    left_node,right_node=Memory(before['bytes']).node(left),Memory(before['bytes']).node(right)
    n.check('division preserves all84 left descriptor words',Memory(after['bytes']).node(left),left_node)
    n.check('division preserves all84 right descriptor words',Memory(after['bytes']).node(right),right_node)
    result=records[0]['value'];actual,node=n.tensor(result)
    n.check('all division result bytes and metadata equal independent Torch',actual,expected(successor['result']))
    n.check('division owns fresh Tensor Storage Version identities',
        [result not in (left,right),node[27] not in (left_node[27],right_node[27]),
            node[28] not in (left_node[28],right_node[28])],[True]*3)
    no_edges=(32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)
    n.check('division retains no AD Saved input or view edges',
        [node[offset//8] for offset in no_edges],[0]*len(no_edges))
    n.one(41,result)
    n.check('both complete operands survive division-result release',
        [n.tensor(left)[0],n.tensor(right)[0]],before_values)
    return actual['numel']
