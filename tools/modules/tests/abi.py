from pathlib import Path
import argparse,hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Observe native flat-ABI output atomicity and guarded buffers.')
for name in ('compiler-image','linker-image','boundary-output','runtime-request','runtime-executable','output'):
 parser.add_argument('--'+name,type=Path,required=True)
options=parser.parse_args()
O=options.output.resolve();O.mkdir(parents=True,exist_ok=False)
B=options.boundary_output.resolve()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();M=1<<20
cases=[]
def add(name,image,request,capacity,workspace,fill,expected,digest=None):cases.append((name,image,request,capacity,workspace,fill,expected,digest))
link_image=options.linker_image.resolve();compiler_image=options.compiler_image.resolve()
assert sha(link_image)=='03a9ba5b2220da6c00cdac54786a5d26493af0254c465581ebb4dc2429c8fa4d'
assert sha(compiler_image)=='3382734ac6e05196bd01ef395729710d783f69fc1302830b27318d4fe8405832'
request=options.runtime_request.resolve();linked=options.runtime_executable.resolve()
for fill in (0,90):add('full-runtime-link-workspace-'+str(fill),link_image,request,M,16*M,fill,linked.stat().st_size,sha(linked))
for name in ('corrupt-abi-version','link-buffer-mode-mismatch','cross-module-cycle-refused','overlapping-relocations-refused','unresolved-import','corrupt-relocation-outside-text'):
 add(name+'-atomic',link_image,B/ (name+'.request'),M,16*M,90,0xffffffff)
add('short-output-atomic',link_image,request,linked.stat().st_size-1,16*M,90,0xfffffffe)
add('short-workspace-atomic',link_image,request,M,16*M-1,90,0xfffffffe)
for name in ('mut-caller','interfaces-256'):
 for fill in (0,90):add(name+'-compiler-workspace-'+str(fill),compiler_image,B/(name+'.hepta'),M,2*M,fill,(B/(name+'.o')).stat().st_size,sha(B/(name+'.o')))
for name in ('mutable-alias-refused','ended-loan-call-refused','provider-effect-understatement-refused','export-borrowed-result-refused'):
 add(name+'-compiler-atomic',compiler_image,B/(name+'.hepta'),M,2*M,90,0xffffffff)
rows=[]
for name,image,request,capacity,workspace,fill,expected,digest in cases:
 req={'image':str(image),'image_sha256':sha(image),'request':str(request),'request_sha256':sha(request),'capacity':capacity,'workspace':workspace,'workspace_fill':fill}
 path=O/(name+'.request.json');path.write_text(json.dumps(req,indent=2)+'\n')
 p=subprocess.run([sys.executable,'-I','-S','-B',str(R/'observe_flat.py'),str(path)],capture_output=True,text=True,timeout=25)
 assert p.returncode==0,(name,p.returncode,p.stderr)
 row=json.loads(p.stdout)
 assert row['result']==expected,(name,row,expected)
 assert row['source_unchanged'] and row['output_guards_unchanged'] and row['workspace_guards_unchanged'] and not row['forbidden_modules'],(name,row)
 if expected in (0xffffffff,0xfffffffe):assert row['output_unchanged'],(name,row)
 else:assert row['produced_sha256']==digest,(name,row,digest)
 if name=='short-workspace-atomic':assert row['workspace_unchanged'],row
 rows.append({'case':name,'status':'PASS','image_sha256':sha(image),'request_sha256':sha(request),'capacity':capacity,'workspace':workspace,'workspace_fill':fill,**row})
report={'schema':'heptatorch.borrowed-modules-flat-abi-review.v1','status':'PASS','cases':len(rows),'rows':rows,'observer':'separate stdlib-only ctypes/mmap child, no HIA compiler/linker modules loaded','purpose':'real buffer guards, dirty-state initialization and output failure atomicity'}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
