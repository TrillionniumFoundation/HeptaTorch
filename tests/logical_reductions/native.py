#!/usr/bin/env python3
"""Execute the HIA image and compare literal frozen PyTorch observations.

This observer transports bytes and reads Tensor/Storage metadata. Expected
logical-reduction values come only from reference.py's actual Torch execution.
"""
import argparse
import hashlib
import importlib.util
import json
import lzma
import os
import re
import resource
import struct
import sys
import traceback
from pathlib import Path

from cases import CASES, CASE_SHA256, DTYPES, MASK, OPERATIONS, raw_input

sha=lambda b:hashlib.sha256(b).hexdigest()
PROTOCOL_SHA='7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e'
BASE=139264


def require(ok,label,**facts):
    if not ok:raise AssertionError(dict(rule=label,**facts))


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_wire(repo):
    path=repo/'tests/differentiable_view_native_review/protocol.py'
    require(sha(path.read_bytes())==PROTOCOL_SHA,'fixed-protocol')
    spec=importlib.util.spec_from_file_location('logical_wire',path)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    return wire


def node(raw,token):
    """Read the immutable published descriptor, independently of runtime kernels."""
    q=lambda off:struct.unpack_from('<Q',raw,off)[0]
    require(raw[64:72]==b'HTSINT02' and raw[BASE:BASE+8]==b'HTOSIN02','owner-format')
    slots=q(BASE+16);pool=q(BASE+24);start=BASE+144+552*slots
    require(start+pool==len(raw),'complete-owner')
    slot=token&0xffffff
    require(token>>24&255==1 and 0<slot<=slots,'public-handle-kind-slot')
    row=struct.unpack_from('<8Q',raw,BASE+128+(slot-1)*64)
    require(row[0]==token>>32 and row[1]==1,'live-public-handle')
    index=row[6];address=q(4096+(index>>8)*8)
    require(address>>63==1,'descriptor-page-address')
    at=start+(address&((1<<63)-1))+(index&255)*672
    values=struct.unpack_from('<84Q',raw,at)
    require(values[24]==token and values[25]==1 and values[30]==1,'public-descriptor-binding')
    return values


class Native:
    def __init__(self,args,wire,folder):
        self.wire=wire;self.folder=folder;folder.mkdir(parents=True,exist_ok=False)
        self.s=wire.Session(args.binary,directory=folder/'raw')
        self.checks=0;self.journal=[]

    def check(self,label,actual,expected):
        self.checks+=1;require(actual==expected,label,actual=actual,expected=expected)

    def one(self,*cmd):
        return self.s.run([cmd])[0]

    def create(self,dtype,shape,raw,requires=0):
        cfg=struct.pack('<8Q',len(shape),requires,*shape,*([0]*(4-len(shape))),160,len(raw))
        packet=b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,dtype,96,0,0,0,0,0)+cfg+raw
        status,records=self.wire.parse(self.s.exchange(packet))
        self.check('typed-create-status',status,0);self.check('typed-create-records',len(records),1)
        return records[0]['value']

    def view(self,token,spec):
        shape,strides=spec['shape'],spec['strides']
        cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
        return self.one(111,token,self.wire.Payload(cfg),80)['value']

    def setup(self,spec):
        raw=raw_input(spec)
        if spec['dtype']=='float64':
            parent=self.create(0,[1],struct.pack('<I',0x3f800000),1)
            product=self.one(114,parent,spec['words'][0])['value']
            source=self.one(107,product,1)['value']
            return source,[source,product,parent]
        backing=self.create(DTYPES[spec['dtype']],[len(spec['words'])],raw,int(spec['requires_grad']))
        source=self.view(backing,spec)
        return source,[source,backing]

    def tensor(self,token):
        meta=self.wire.meta(self.one(13,token))
        dtype,width,incarnation,leaf=struct.unpack('<4Q',self.one(96,token,0)['payload'])
        data=self.one(98,token,0,meta['numel'])['payload']
        owner=self.s.owner();n=node(owner['bytes'],token)
        self.check('typed-data-width',len(data),meta['numel']*width)
        self.check('descriptor-dtype',n[72],dtype)
        self.check('descriptor-incarnation',n[40],incarnation)
        self.check('descriptor-version-object',n[28],meta['version_id'])
        self.check('descriptor-storage-object',n[27],meta['storage_id'])
        return dict(dtype={0:'torch.float32',1:'torch.bool',2:'torch.int64',3:'torch.float64'}[dtype],
                    shape=meta['shape'][:meta['rank']],strides=meta['strides'][:meta['rank']],
                    numel=meta['numel'],element_size=width,storage_bytes=meta['storage_bytes'],
                    storage_offset=meta['storage_offset']//width,version=meta['version'],
                    requires_grad=bool(meta['requires_grad']),is_leaf=bool(leaf),is_view=bool(n[51]),
                    grad_fn=None if n[5]==0 else 'NATIVE_HISTORY_PRESENT',logical_bytes_hex=data.hex()),n

    def command(self,spec,source):
        axes=spec['axes']
        payload=0 if not axes else self.wire.Payload(struct.pack('<'+'q'*len(axes),*axes))
        return (OPERATIONS[spec['operation']],source,MASK if axes is None else len(axes),payload,int(spec['keepdim']),1,0,0)

    def issue(self,commands,label):
        before=self.s.owner();request=self.wire.packet(commands)
        response=self.s.exchange(request);status,records=self.wire.parse(response);after=self.s.owner()
        for endpoint in (before,after):
            digest=sha(endpoint['bytes']);path=self.folder/(digest+'.owner.xz')
            if not path.exists():path.write_bytes(lzma.compress(endpoint['bytes'],preset=1))
        row=dict(label=label,status=status,request_sha256=sha(request),response_sha256=sha(response),
                 owner_before_sha256=sha(before['bytes']),owner_after_sha256=sha(after['bytes']))
        if status:
            self.check(label+'/full-owner-rollback',after,before)
            row['complete_owner_rollback']=True
        self.journal.append(row)
        return status,records,before,after

    def finish(self):
        self.s.finish()
        transport=json.loads((self.folder/'raw/transport.json').read_text())
        # finish() reads to both actual EOFs and rejects nonempty final streams.
        require(transport['exit_code']==0,'native-exit-zero')
        dump(self.folder/'observations.json',dict(checks=self.checks,frames=self.s.frames,journal=self.journal,
                                                exit_code=0,stdout_EOF=True,stderr_EOF=True))


def expected_status(row):
    if row['accepted']:return 0
    error=row['error']
    if error['type']=='RuntimeError' and error['message']=='Autograd not support dtype: Bool':return 5
    if error['type']=='IndexError' and error['message'].startswith('Dimension out of range'):return 2
    if error['type']=='RuntimeError' and re.fullmatch(r'dim [0-9]+ appears multiple times in the list of dims',error['message']):return 7
    raise AssertionError(dict(rule='unmapped-reference-error',error=error))


def execute_case(args,wire,row,folder):
    n=Native(args,wire,folder);spec=row['spec'];record={}
    try:
        gen=n.one(60,1729)['value'];rng=n.one(63,gen)['payload']
        n.check('complete-rng-width',len(rng),5056)
        source,inputs=n.setup(spec);before,_=n.tensor(source)
        # Tensor values and geometry are independently observed; nonleaf class names differ.
        for key,value in row['input'].items():
            if key!='grad_fn':n.check('input/'+key,before[key],value)
        status,records,bowner,aowner=n.issue([n.command(spec,source)],'logical-reduction')
        n.check('actual-reference-error-status',status,expected_status(row))
        n.check('input-all-84-descriptor-words',node(aowner['bytes'],source),node(bowner['bytes'],source))
        after,_=n.tensor(source);n.check('input-complete-readback-unchanged',after,before)
        if status==0:
            n.check('operation-records',len(records),1);result=records[0]['value']
            actual,rnode=n.tensor(result);n.check('complete-reference-result',actual,row['result'])
            snode=node(aowner['bytes'],source)
            n.check('fresh-Tensor',result!=source,True)
            n.check('fresh-Storage',rnode[27]!=snode[27],True)
            n.check('fresh-Version',rnode[28]!=snode[28],True)
            no_edges=(32,40,48,56,72,80,88,176,184,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)
            n.check('no-gradient-Saved-view-or-input-edges',[rnode[o//8] for o in no_edges],[0]*len(no_edges))
            for token in inputs:n.one(41,token)
            retained,_=n.tensor(result);n.check('result-survives-input-release',retained,actual)
            n.one(41,result)
        else:
            for token in inputs:n.one(41,token)
        n.check('complete-RNG-unchanged',n.one(63,gen)['payload'],rng)
        n.one(66,gen)
        record.update(status='PASS',name=spec['name'],checks=n.checks,frames=n.s.frames,
                      accepted=status==0,reference_error_status=expected_status(row))
    except BaseException as error:
        record.update(status='FAIL',name=spec['name'],error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        try:n.finish()
        except BaseException as error:
            record.update(status='FAIL',finish_error=dict(type=type(error).__name__,message=str(error)))
    dump(folder/'report.json',record)
    return record


def controls(args,wire,folder):
    n=Native(args,wire,folder);rows=[]
    try:
        source=n.create(0,[2],struct.pack('<2I',0,0x80000000),1)
        for op in (118,119):
            valid=[op,source,MASK,0,0,1,0,0]
            for name,index,value,error in [('reserved48',6,1,7),('reserved56',7,1,7),('invalid-dtype',5,0,7),
                                          ('invalid-keepdim',4,2,7),('too-many-axes',2,5,7),('invalid-source',1,1,1)]:
                cmd=valid.copy();cmd[index]=value
                status,_,_,_=n.issue([cmd],name+str(op));n.check('control/'+name,status,error);rows.append(name+str(op))
            cmd=valid.copy();cmd[2]=1;cmd[3]=4096
            status,_,_,_=n.issue([cmd],'missing-axis-payload'+str(op));n.check('missing-axis-payload',status,2);rows.append('missing-axis-payload'+str(op))
            # A successful allocation followed by an invalid operation rolls back the whole frame.
            bad=valid.copy();bad[6]=1
            status,_,_,_=n.issue([valid,bad],'second-command-rollback'+str(op));n.check('multi-command-rollback',status,7);rows.append('second-command-rollback'+str(op))
        stale=source;n.one(41,source)
        for op in (118,119):
            status,_,_,_=n.issue([(op,stale,MASK,0,0,1,0,0)],'stale-source'+str(op));n.check('stale-source',status,1);rows.append('stale-source'+str(op))
        return dict(status='PASS',controls=len(rows),names=rows,checks=n.checks,frames=n.s.frames)
    finally:n.finish()


def allocation_control(args,wire,folder,op):
    n=Native(args,wire,folder);prior=None;attempts=[]
    try:
        source=n.create(1,[262144],bytes(262144))
        gen=n.one(60,1729)['value'];rng=n.one(63,gen)['payload']
        before=n.s.owner()
        pages=int(Path('/proc/%d/statm'%n.s.pid).read_text().split()[0]);virtual=pages*os.sysconf('SC_PAGE_SIZE')
        prior=resource.prlimit(n.s.pid,resource.RLIMIT_AS)
        page=lambda n: (n+4095)&~4095
        limit=virtual+page(before['length']+16)+page(5056+16)+65536
        require(prior[1]==resource.RLIM_INFINITY or limit<=prior[1],'allocation-limit-within-original-hard-limit')
        resource.prlimit(n.s.pid,resource.RLIMIT_AS,(limit,prior[1]))
        # A same-capacity read succeeds. Retain successful Bool outputs until the
        # same bounded owner must grow; no descriptor/shape limit is approached.
        n.one(96,source,0)
        command=(op,source,0,0,0,1,0,0)
        for attempt in range(32):
            status,records,before,after=n.issue([command],'allocation-attempt-%d'%attempt)
            attempts.append(dict(status=status,owner_bytes=before['length']))
            if status:
                n.check('actual-allocation-failure',status,3)
                break
        else:raise AssertionError('failed to exercise an actual owner-growth allocation failure')
        n.check('allocation-failure-RNG',n.one(63,gen)['payload'],rng)
        resource.prlimit(n.s.pid,resource.RLIMIT_AS,prior);prior=None
        status,records,_,_=n.issue([command],'identical-request-after-limit-restore')
        n.check('allocation-retry-success',status,0)
        actual,_=n.tensor(records[0]['value'])
        n.check('allocation-retry-empty-axes-copy',actual['logical_bytes_hex'],'00'*262144)
        n.check('allocation-retry-shape',actual['shape'],[262144])
        n.check('allocation-retry-RNG',n.one(63,gen)['payload'],rng)
        return dict(status='PASS',opcode=op,attempts=attempts,limit=limit,checks=n.checks,frames=n.s.frames,
                    full_owner_rollback=True,same_request_after_restore=True)
    finally:
        if prior is not None and n.s.poll() is None:resource.prlimit(n.s.pid,resource.RLIMIT_AS,prior)
        n.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.repository=args.repository.resolve();builddir=args.build_directory.resolve()
    args.binary=builddir/'heptatorch';args.source=builddir/'runtime.hepta';args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    require(not sys.flags.optimize,'assertions-enabled')
    paths=[Path(__file__).resolve(),Path(__file__).with_name('cases.py').resolve(),
           Path(__file__).with_name('reference.py').resolve(),args.reference.resolve(),args.binary,args.source,builddir/'build.json',
           args.repository/'tests/differentiable_view_native_review/protocol.py',args.repository/'tests/differentiable_view_native_review/owner_client.py']
    inputs={str(p):sha(p.read_bytes()) for p in paths}
    report=dict(schema='heptatorch.logical-reductions.native.v1',status='RUNNING',case_sha256=CASE_SHA256,
                expected_cases=len(CASES),rows=[],inputs=inputs,native_executed=False,torch_executed=False,
                full_pytorch_rewrite_complete=False)
    code=0
    try:
        build=json.loads((builddir/'build.json').read_bytes())
        require(build['status']=='PASS_NATIVE_BUILD_ONLY' and build['objects_reused'] is False and
                build['source_sha256']==inputs[str(args.source)] and build['native_sha256']==inputs[str(args.binary)] and
                build['compiler_sha256']==COMPILER_SHA and build['linker_sha256']==LINKER_SHA and
                len(build['processes'])==len(build['units'])+1 and
                all(p['exit_code']==0 and p['stdout_EOF'] and p['stderr_EOF'] for p in build['processes']),
                'exact-complete-cold-build')
        reference=json.loads(args.reference.read_bytes())
        require(reference['status']=='OBSERVED' and reference['torch_executed'] is True and
                reference['inputs_unchanged'] is True and reference['rng_unchanged'] is True and
                reference['case_sha256']==CASE_SHA256 and reference['cases']==reference['expected_cases']==len(CASES) and
                reference['source_sha256']==inputs[str(Path(__file__).with_name('reference.py').resolve())] and
                reference['oracle_provenance']['status']=='PASS' and len(reference['oracle_provenance']['checks'])==11 and
                all(reference['oracle_provenance']['checks'].values()),'complete-actual-reference')
        require([r['spec'] for r in reference['rows']]==list(CASES),'ordered-complete-case-list')
        wire=load_wire(args.repository);report['native_executed']=True
        for i,row in enumerate(reference['rows']):
            observed=execute_case(args,wire,row,args.output/('case-%03d'%i));report['rows'].append(observed)
        report['packet_controls']=controls(args,wire,args.output/'packet-controls')
        report['allocation_controls']=[allocation_control(args,wire,args.output/('allocation-%d'%op),op) for op in (118,119)]
        require(all(row['status']=='PASS' for row in report['rows']),'all-reference-cases-pass',
                failed=[row['name'] for row in report['rows'] if row['status']!='PASS'])
        require(all(sha(p.read_bytes())==inputs[str(p)] for p in paths),'all-bound-inputs-unchanged')
        report.update(status='PASS',cases=len(report['rows']),checks=sum(r['checks'] for r in report['rows']),
                      frames=sum(r['frames'] for r in report['rows'])+report['packet_controls']['frames']+
                      sum(r['frames'] for r in report['allocation_controls']),inputs_unchanged=True,
                      source_sha256=inputs[str(args.source)],native_sha256=inputs[str(args.binary)])
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));code=1
    dump(args.output/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','cases','expected_cases','checks','frames','error')}))
    return code

if __name__=='__main__':raise SystemExit(main())
