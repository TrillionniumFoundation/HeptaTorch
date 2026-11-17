"""Pure lifetime-qualified labels for raw Torch addresses in the fixed reference26.

The original reference and numerical comparisons are immutable. Only a Saved
rank-zero float64 scalar, observed solely through MulBackward._saved_other and
proven released by the successful default backward, starts a new address epoch.
No case number, address value, gradient value, or native result grants an epoch.
"""
import argparse,copy,hashlib,json,re
from pathlib import Path
REFERENCE='fecf1956790f78a9b1238f535c1c7cce9f1237b211c389ac0736d88687bbb9c9'
PROBE='4dd374419e4b21a9c814c855d846481730a12ea57a30dcf229a6b962f07021c5'
SHA=lambda b:hashlib.sha256(b).hexdigest()
FIELDS=('tensor_identity','storage_identity','base_tensor_identity')
PREFIX='observed_grad_storage_'
def require(ok,rule,**facts):
 if not ok:raise AssertionError({'identity_epoch_rule':rule,**facts})
def tensors(value,path=()):
 if isinstance(value,dict):
  if {'tensor_identity','storage_identity','dtype','element_size'}<=set(value):
   yield path,value
  for k,v in value.items():yield from tensors(v,path+(k,))
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from tensors(v,path+(i,))
def freed(event):
 return (event.get('accepted') is False and event.get('outcome')=='RAISED'
         and event.get('error',{}).get('type')=='RuntimeError'
         and ('saved tensors have already been freed' in event['error']['message']
              or 'backward through the graph a second time' in event['error']['message']))
def retirement_proofs(row,event):
 before,after=event['before'],event['after'];groups={}
 for name,attrs in before['saved_getters_after_tensor_getters'].items():
  e=attrs.get('_saved_other',{})
  t=e.get('value',{}).get('tensor') if isinstance(e.get('value'),dict) else None
  if e.get('accepted') is True and isinstance(t,dict) and t.get('dtype')=='torch.float64':
   key=t['tensor_identity'],t['storage_identity']
   groups.setdefault(key,[]).append((name,t))
 result=[]
 for (tid,sid),entries in groups.items():
  released=[after['saved_getters_after_tensor_getters'].get(name,{}).get('_saved_other',{}) for name,t in entries]
  if not any(freed(e) for e in released):continue
  require(all(freed(e) for e in released),'every-Saved-owner-released',tensor=tid,storage=sid)
  match=re.fullmatch(r'backward_phase_([0-9]+)_(.+)',event['label'])
  require(event['accepted'] is True and bool(match),'successful-default-backward')
  phase=int(match.group(1))
  require(phase==len(row['seed_phases'])-1 and row['seed_phases'][phase]['name']==match.group(2),'last-phase-has-retain-false')
  require(tid.isdecimal() and sid.isdecimal(),'raw-address-strings')
  allowed={('saved_getters_after_tensor_getters',name,'_saved_other','value','tensor') for name,t in entries}
  for name,t in entries:
   require(t['dtype']=='torch.float64' and t['element_size']==8 and t['numel']==1
      and t['shape']==[] and t['strides']==[] and t['storage_bytes']==8
      and t['storage_offset_elements']==t['storage_offset_bytes']==t['version']==0
      and t['requires_grad'] is False and t['is_view'] is False
      and t['base_tensor_identity'] is None,'original-rank0-Scalar64-role',owner=name)
   require(before['tensors'][name]['grad_fn_getter']['node']['kind']=='MulBackward0','real-MulBackward-owner',owner=name)
  old_occurrences=[]
  for path,t in tensors(before):
   require(t['base_tensor_identity']!=tid,'Scalar-is-not-a-live-base',path=list(path))
   if t['tensor_identity']==tid or t['storage_identity']==sid:
    require(path in allowed and t['tensor_identity']==tid and t['storage_identity']==sid,
       'no-live-public-gradient-or-other-Saved-alias',path=list(path))
    old_occurrences.append(list(path))
  require(len(old_occurrences)==len(entries),'all-old-Scalar-observations-covered')
  require(PREFIX+sid not in before['expired'],'no-observed-gradient-Weak-for-retired-Scalar')
  later=[]
  for path,t in tensors(after):
   if t['tensor_identity']==tid or t['storage_identity']==sid:
    require(t['dtype']!='torch.float64','released-Scalar-not-observable-afterward',path=list(path))
    later.append({'path':list(path),'tensor_identity':t['tensor_identity'],'storage_identity':t['storage_identity'],'dtype':t['dtype'],'shape':t['shape'],'storage_bytes':t['storage_bytes']})
  result.append({'event_label':event['label'],'tensor_address':tid,'storage_address':sid,
      'old_observation_paths':old_occurrences,'owner_names':[x[0] for x in entries],
      'actual_Saved_release_errors':[copy.deepcopy(e['error']) for e in released],
      'same_address_new_observations_immediately_after':later,
      'proof_scope':'Pinned probe has only live named strong roots, local primitive-only Saved getter copies, and no Weak for this original scalar; original default-backward Saved-release errors are required.'})
 return result
def annotate(value,t_epochs,s_epochs,path=(),changes=None):
 if isinstance(value,dict):
  result={}
  for k,v in value.items():
   p=path+(k,)
   if k in FIELDS and v is not None:
    require(isinstance(v,str) and v.isdecimal(),'identity-field-is-raw-address',path=list(p))
    epoch=(s_epochs if k=='storage_identity' else t_epochs).get(v,0)
    new=v+('@epoch%d'%epoch if epoch else '')
    result[k]=new
    if new!=v:changes.append({'path':list(p),'before':v,'after':new})
   elif k in ('expired','final_expired'):
    require(isinstance(v,dict),'Weak-expiry-map')
    target={}
    for name,item in v.items():
     epoch=s_epochs.get(name[len(PREFIX):],0) if name.startswith(PREFIX) else 0
     new=name+('@epoch%d'%epoch if epoch else '')
     require(new not in target,'unique-Weak-epoch-name')
     target[new]=item
     if name!=new:changes.append({'path':list(p),'key_before':name,'key_after':new})
    result[k]=target
   else:result[k]=annotate(v,t_epochs,s_epochs,p,changes)
  return result
 if isinstance(value,list):return [annotate(v,t_epochs,s_epochs,path+(i,),changes) for i,v in enumerate(value)]
 return value
def inverse(value):
 if isinstance(value,dict):
  out={}
  for k,v in value.items():
   if k in FIELDS and isinstance(v,str):out[k]=re.sub(r'@epoch[1-9][0-9]*$','',v)
   elif k in ('expired','final_expired'):
    out[k]={re.sub(r'@epoch[1-9][0-9]*$','',n):x for n,x in v.items()}
    require(len(out[k])==len(v),'inverse-Weak-name-unique')
   else:out[k]=inverse(v)
  return out
 if isinstance(value,list):return [inverse(x) for x in value]
 return value
def adapt_data(original):
 result=copy.deepcopy(original);proof_rows=[];all_changes=[]
 require(original['schema']=='heptatorch.concrete-view-reshape-reference26.v1','fixed-reference-schema')
 require(original['cases']==len(original['rows'])==26,'complete-reference26')
 for row,out in zip(original['rows'],result['rows']):
  te,se={},{};proofs=[];changes=[];final_seen=False;events=[]
  for i,event in enumerate(row['events']):
   if event['label'].startswith('final_drop_') and not final_seen:
    out['before_final_drops']=annotate(row['before_final_drops'],te,se,('before_final_drops',),changes);final_seen=True
   before=annotate(event['before'],te,se,('events',i,'before'),changes)
   retired=retirement_proofs(row,event)
   for p in retired:
    tid,sid=p['tensor_address'],p['storage_address']
    te[tid]=te.get(tid,0)+1;se[sid]=se.get(sid,0)+1
    proofs.append({**p,'event_index':i,'tensor_new_epoch':te[tid],'storage_new_epoch':se[sid]})
   # Before retains the preceding lifetime; every other event field is observed
   # after the operation. Saved-release errors and all numeric values are copied.
   remainder={k:v for k,v in event.items() if k!='before'}
   new=annotate(remainder,te,se,('events',i),changes)
   new['before']=before
   events.append({k:new[k] for k in event})
  out['events']=events
  if not final_seen:out['before_final_drops']=annotate(row['before_final_drops'],te,se,('before_final_drops',),changes)
  final=annotate({'final_expired':row['final_expired']},te,se,(),changes)
  out['final_expired']=final['final_expired']
  require(inverse(out)==row,'all-nonidentity-fields-and-inverse-exact',case=row['index'])
  if not proofs:require(out==row,'unaffected-row-identical',case=row['index'])
  proof_rows.append({'index':row['index'],'name':row['name'],'retirements':proofs,
      'changed_identity_fields_or_Weak_keys':len(changes),'all_nonidentity_fields_unchanged':True,
      'whole_inverse_equal':True,'changes':changes})
  all_changes.extend(changes)
 require(inverse(result)==original,'entire-reference-inverse-exact')
 report={'schema':'heptatorch.reference26.identity-epochs.v1','status':'PASS_PURE_LIFETIME_QUALIFIED_LABELS',
   'original_reference_sha256':REFERENCE,'probe_sha256':PROBE,
   'policy':'An address label changes only after the original unexposed rank0 Scalar64 has no public/gradient/base/Weak/other-Saved owner and every old _saved_other getter reports release after successful last-phase backward. The immutable native bijection, values, Versions, edges, Weak checks and EOF predicates are not edited.',
   'identity_scope':'Stable and injective within overlapping lifetimes; raw allocator addresses are not lifetime-global identity. Native incarnations/generation tokens must still be globally distinct through the unchanged reverse maps.',
   'death_observation_scope':'This proof uses original Saved-release errors and pinned probe root ownership; it is not a newly observed PyTorch Scalar StorageWeakRef expiration.',
   'rows':proof_rows,'retirements':sum(len(r['retirements']) for r in proof_rows),
   'identity_annotations':len(all_changes),'unaffected_rows':sum(not r['retirements'] for r in proof_rows),
   'original_reference_changed':False,'all_nonidentity_fields_unchanged':True,
   'all_original_comparator_methods_changed':False,'whole_inverse_equal':True,
   'new_native_executions':0,'new_Torch_executions':0,'whole_pytorch_gate':'HOLD'}
 return result,report
def admit(raw,probe):
 require(SHA(raw)==REFERENCE and SHA(probe)==PROBE,'exact-immutable-reference-and-probe')
 original=json.loads(raw)
 require(original['probe_sha256']==PROBE and original['source_files_unchanged'] is True,'original-probe-custody')
 require(original['status']=='OBSERVED' and original['probe_error_count']==0,'original-observation-status')
 return adapt_data(original)
def negative_controls(original,plan):
 rowproof=next(r for r in plan['rows'] if r['retirements'])
 row=original['rows'][rowproof['index']];proof=rowproof['retirements'][0];event=row['events'][proof['event_index']]
 tid,sid=proof['tensor_address'],proof['storage_address'];owner=proof['owner_names'][0]
 scalar=event['before']['saved_getters_after_tensor_getters'][owner]['_saved_other']['value']['tensor']
 controls=[]
 def reject(name,mutate,wanted):
  ev=copy.deepcopy(event);mutate(ev)
  try:retirement_proofs(row,ev)
  except AssertionError as e:
   require(e.args[0]['identity_epoch_rule']==wanted,'negative-exact-rule',case=name,got=e.args[0])
   controls.append({'name':name,'status':'PASS_EXPECTED_REJECTION','rule':wanted})
  else:raise AssertionError({'identity_epoch_rule':'negative-was-admitted','case':name})
 reject('simultaneous-public-alias',lambda e:e['before']['tensors'].__setitem__('extra_live',{'before_grad_fn_getter':copy.deepcopy(scalar)}),'no-live-public-gradient-or-other-Saved-alias')
 reject('previous-gradient-Weak',lambda e:e['before']['expired'].__setitem__(PREFIX+sid,False),'no-observed-gradient-Weak-for-retired-Scalar')
 reject('Scalar-still-observable-after',lambda e:e['after']['tensors'].__setitem__('extra_live',{'before_grad_fn_getter':copy.deepcopy(scalar)}),'released-Scalar-not-observable-afterward')
 reject('failed-backward',lambda e:e.__setitem__('accepted',False),'successful-default-backward')
 reject('retained-phase',lambda e:e.__setitem__('label','backward_phase_0_'+row['seed_phases'][0]['name']),'last-phase-has-retain-false')
 ev=copy.deepcopy(event)
 for name in proof['owner_names']:
  ev['after']['saved_getters_after_tensor_getters'][name]['_saved_other']=copy.deepcopy(ev['before']['saved_getters_after_tensor_getters'][name]['_saved_other'])
 require(retirement_proofs(row,ev)==[],'live-Saved-never-retired')
 controls.append({'name':'live-Saved-no-epoch','status':'PASS_NO_RETIREMENT'})
 return controls
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for n in ('reference','probe','output'):ap.add_argument('--'+n,type=Path,required=True)
 a=ap.parse_args();raw=a.reference.read_bytes();probe=a.probe.read_bytes()
 adapted,report=admit(raw,probe);report['copied_negative_controls']=negative_controls(json.loads(raw),report)
 a.output.mkdir(parents=True,exist_ok=False)
 out=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
 (a.output/'identity-epoch-plan.json').write_bytes(out)
 derived=(json.dumps(adapted,indent=2,allow_nan=False)+'\n').encode()
 (a.output/'derived-identity-labels.json').write_bytes(derived)
 require(a.reference.read_bytes()==raw and a.probe.read_bytes()==probe,'input-originals-unchanged')
 print(json.dumps({'status':report['status'],'retirements':report['retirements'],'unaffected_rows':report['unaffected_rows'],'identity_annotations':report['identity_annotations'],'copied_controls':len(report['copied_negative_controls']),'plan_sha256':SHA(out),'derived_sha256':SHA(derived),'new_native_executions':0,'new_Torch_executions':0}))
if __name__=='__main__':main()
