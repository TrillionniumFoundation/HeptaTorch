"""Derive separate oracle/native paths from one immutable combined observer."""
import sys,hashlib,ast,textwrap
from pathlib import Path
src,out=map(Path,sys.argv[1:])
PIN='18466b25cd0cb3efa4321f49e5edefeb88a9fb641b71bafba96c0c73ac4a277b'
raw=src.read_bytes()
assert hashlib.sha256(raw).hexdigest()==PIN
s=raw.decode(); tree=ast.parse(s)
keys={'positive':('xx=create','vals'), 'self_positive':('h=create','inverted'),'numeric_positive':('raw_one=','want'),'numeric_self':('h=create','want1,want2')}
original={}
for node in tree.body:
 if not isinstance(node,ast.FunctionDef) or node.name not in keys:continue
 lines=s.splitlines(keepends=True); f=''.join(lines[node.lineno-1:node.end_lineno]); cut,payload=keys[node.name]
 start=f.index('\n')+1
 cut=f.index('    '+cut)
 original[node.name]=(f[:start],f[start:cut],f[cut:],payload)
pick="""def case_identity(name,parameters):
    return hashlib.sha256(json.dumps([name,parameters],separators=(',',':')).encode()).hexdigest()
def lookup(label,name,params):
    assert label not in used
    item=fixture['rows'][label]
    assert item['input_sha256']==case_identity(name,params)
    used.add(label)
    return tuple(bytes.fromhex(x) for x in item['payload'])
"""
fixtures=''; transplanted=s
for name,(header,oracle,native,payload) in original.items():
 args=header[header.index('(')+1:header.index(')')].split(',')[1:]
 params='('+','.join(args)+',)'
 pack=('('+payload+',)' if ',' not in payload else '('+payload+')')
 unpack=payload if ',' in payload else payload+','
 fixture_fn=header+oracle+f"    assert label not in results\n    results[label]=dict(input_sha256=case_identity({name!r},{params}),payload=[item.hex() for item in {pack}])\n"
 fixtures+=fixture_fn.replace('def '+name+'(','def fixture_'+name+'(',1)
 extra={'positive':'    n=len(words)\n','self_positive':'','numeric_positive':'    n=len(values)\n','numeric_self':"    w=WIDTHS[dt]; n=len(values)\n    init=b''.join(v.to_bytes(w,'little') for v in values)\n"}[name]
 # Every subsequent native assertion remains byte-for-byte as in the original.
 transplanted=transplanted.replace(header+oracle+native,header+extra+f"    {unpack}=lookup(label,{name!r},{params})\n"+native,1)
matrix=s.split('        for dt in sorted(set(WIDTHS)-{3}):',1)[1].split('    summary=dict',1)[0]
matrix='        for dt in sorted(set(WIDTHS)-{3}):'+matrix
matrixfn='def matrix(positive,self_positive,numeric_positive,numeric_self):\n'+textwrap.indent(textwrap.dedent(matrix),'    ')
transplanted=transplanted.replace(matrix,'        matrix(positive,self_positive,numeric_positive,numeric_self)\n')
transplanted=transplanted.replace("choices=images","choices=[*images,'fixture']").replace("required=arg not in ('oracle-lock','install-report')","required=arg not in ('oracle-lock','install-report','binary')").replace('a=p.parse_args()',"""p.add_argument('--fixture',type=Path)
p.add_argument('--fixture-sha256')
a=p.parse_args()""")
transplanted=transplanted.replace('mode,image,root=a.mode,a.binary.resolve(strict=True),a.output\npin=images[mode]','mode,root=a.mode,a.output')
start=transplanted.index('assert digest(image)==pin')
end=transplanted.index('s=wire.Session',start)
transplanted=transplanted[:start]+pick+fixtures+matrixfn+"""
if mode=='fixture':
    assert a.oracle_lock is not None and a.install_report is not None
    torch,oracle=load_oracle(a.oracle_lock,a.install_report)
    results={}
    matrix(fixture_positive,fixture_self_positive,fixture_numeric_positive,fixture_numeric_self)
    assert len(results)==79
    root.write_text(json.dumps(dict(oracle=oracle,rows=results),sort_keys=True,separators=(',',':'))+'\\n')
    sys.exit(0)
assert a.binary is not None
image=a.binary.resolve(strict=True)
pin=images[mode]
assert digest(image)==pin
wire=load_wire(a.protocol)
if mode=='candidate':
    assert a.fixture is not None and a.fixture_sha256 is not None
    raw_fixture=a.fixture.read_bytes()
    assert hashlib.sha256(raw_fixture).hexdigest()==a.fixture_sha256
    fixture=json.loads(raw_fixture)
    assert len(fixture['rows'])==79
    oracle=fixture['oracle']; used=set()
    assert 'torch' not in sys.modules
"""+transplanted[end:]
transplanted=transplanted.replace("    if mode!='baseline': summary['oracle']=oracle","    if mode!='baseline':\n        assert used==set(fixture['rows']) and 'torch' not in sys.modules\n        summary.update(oracle=oracle,fixture_sha256=a.fixture_sha256,separate_no_torch_native_process=True)")
compile(transplanted,str(out),'exec')
out.write_text(transplanted)
