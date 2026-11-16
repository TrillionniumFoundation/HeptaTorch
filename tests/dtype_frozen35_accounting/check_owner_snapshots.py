"""Independent negative controls for the read-only PR28 node observer.

Inputs are copied JSON snapshots from an already-executed journal. This program
does not run a native application, import Torch, or open /proc memory.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, importlib.util, io, json, traceback

EXPECTED = '4cd9c3d1fafd1eee04ac1b2b8bb720a099d53dc821b25b9cf4380e6c67b36a71'

def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def w(node, offset): return node[offset//8]
def put(node, offset, value): node[offset//8]=value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observer',type=Path,required=True)
    p.add_argument('--journals',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-profile',choices=['dtype-native-dropout-523'],required=True)
    args=p.parse_args()
    for key in ('observer','journals','output'): setattr(args,key,getattr(args,key).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    executed=Path(__file__).read_bytes()
    (args.output/'executed-controls.py').write_bytes(executed)
    source=args.observer.read_bytes()
    assert sha(source)==EXPECTED, sha(source)
    (args.output/'observer-under-test.py').write_bytes(source)
    spec=importlib.util.spec_from_file_location('copied_snapshot_observer_under_test',args.observer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report={
      'schema':'heptatorch.pr28.private-node-observer-independent-negative-controls.v1',
      'status':'NOT_RUN','controls_sha256':sha(executed),'observer_sha256':EXPECTED,
      'native_executed':False,'Torch_executed':False,'native_memory_read':False,
      'native_memory_written':False,'expected_outcomes_come_from_named_invariants_not_target_return_values':True,
      'input_files':{},'positive_rows':[],'negative_rows':[],
      'scope':'Observer admission only, using copies of committed owner snapshots; no native or API qualification is claimed.',
    }
    try:
        chosen={}
        def bindings(event):
            result=[]
            for key,node in event['after']['nodes'].items():
                if w(node,304) and w(node,384):
                    g=str(w(node,384))
                    if g in event['after']['nodes']:
                        result.append((key,g))
            return result
        predicates={
            'materialization':lambda e:e['expected_status']==0
                and not any(c[0]==33 for c in e['commands']) and e['result'].get('created_private_G',0)>0,
            'compound-construction':lambda e:e['expected_status']==0
                and any(t['count']==3 for t in e['tickets']),
            'reset-with-private-gradient':lambda e:e['expected_status']==0
                and any(c[0]==33 for c in e['commands']) and e['result'].get('live_gradient_nodes',0)>0,
            'failed-frame-rollback':lambda e:e['expected_status']!=0,
        }
        # Event result fields select candidates only. Every selected positive is
        # independently rerun through the entire pure verifier below. Read one
        # journal at a time and retain only the four required event snapshots.
        for path in sorted(args.journals.rglob('events.json')):
            raw=path.read_bytes()
            report['input_files'][str(path)]={'sha256':sha(raw),'bytes':len(raw)}
            rows=json.loads(raw)
            selections={}
            for ordinal,event in enumerate(rows):
                if not event.get('result') or event.get('error'):
                    continue
                for name,predicate in predicates.items():
                    if name not in chosen and name not in selections and predicate(event):
                        selections[name]=(ordinal,event)
            if not selections:
                continue
            archive=path.with_name('snapshots.jsonl.gz')
            packed=archive.read_bytes()
            report['input_files'][str(archive)]={'sha256':sha(packed),'bytes':len(packed)}
            wanted={e[key] for _,e in selections.values() for key in ('before','after')}
            snapshots={};seen=set()
            with gzip.GzipFile(fileobj=io.BytesIO(packed),mode='rb') as stream:
                for line in stream:
                    item=json.loads(line);data=canonical(item['snapshot'])
                    assert sha(data)==item['sha256'],('journal snapshot hash',path)
                    assert item['sha256'] not in seen,('duplicate journal snapshot',path)
                    seen.add(item['sha256'])
                    if item['sha256'] in wanted:
                        snapshots[item['sha256']]=item['snapshot']
            assert set(snapshots)==wanted,('missing selected snapshots',path)
            for name,(ordinal,event) in selections.items():
                entry={'path':str(path),'ordinal':ordinal,
                    'before':snapshots[event['before']],'after':snapshots[event['after']],
                    'tickets':event['tickets'],'commands':event['commands'],
                    'status':event['expected_status'],'recorded_result':event['result']}
                assert entry['before']['profile']==entry['after']['profile']==args.expected_profile
                if name=='materialization':
                    assert any(w(entry['after']['nodes'][g],320)>entry['before']['header']['incarnation']
                               for _,g in bindings(entry))
                elif name=='reset-with-private-gradient':
                    assert bindings(entry)
                elif name=='failed-frame-rollback':
                    if not entry['before']['length']:
                        continue
                    assert entry['before']['full_sha256']==entry['after']['full_sha256']
                chosen[name]=entry
            if set(chosen)==set(predicates):
                break
        assert set(chosen)==set(predicates),('missing actual positive sample',sorted(set(predicates)-set(chosen)))
        material=chosen['materialization']
        report['expected_profile']=args.expected_profile
        report['selection_policy']='First actual eligible event of each fixed category; event result is only a selector, never a substitute for baseline verification.'
        baseline_bytes={name:canonical(e) for name,e in chosen.items()}
        for name,event in chosen.items():
            got=module.verify_transition(event['before'],event['after'],event['tickets'],event['commands'],event['status'])
            assert got==event['recorded_result'], (name,'baseline differs from recorded observation')
            report['positive_rows'].append({'case':name,'status':'PASS','journal':event['path'],
                'event':event['ordinal'],'result':got,'input_sha256':sha(canonical(event))})
        parent,gid=next((a,g) for a,g in bindings(material)
            if w(material['after']['nodes'][g],320)>material['before']['header']['incarnation'])
        def run(name,base,mutate,expected_rule):
            original=chosen[base]
            # Content-addressed journals may share the same dict for equal
            # before/after snapshots. JSON round-trip makes the two sides
            # independent; deepcopy would preserve that alias.
            changed=json.loads(canonical(original))
            assert changed['before'] is not changed['after']
            mutate(changed)
            assert canonical(changed['before'])==canonical(original['before']), 'mutation changed before snapshot'
            assert canonical(changed)!=canonical(original)
            location=args.output/'negative-inputs'/ (name+'.json.gz')
            location.parent.mkdir(exist_ok=True)
            data=canonical(changed)
            with location.open('wb') as f:
                with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(data)
            try:
                module.verify_transition(changed['before'],changed['after'],changed['tickets'],
                                         changed['commands'],changed['status'])
            except AssertionError as error:
                details=error.args[0] if error.args else None
                rule=details.get('observer_rule') if isinstance(details,dict) else None
                assert rule==expected_rule, (name,'wrong rejection',rule,expected_rule,repr(error))
                row={'case':name,'status':'PASS_EXPECTED_REJECTION','expected_rule':expected_rule,
                     'actual_rule':rule,'details':details,'input_sha256':sha(data),
                     'source_event':{'journal':original['path'],'event':original['ordinal']},
                     'compressed_input':str(location),'compressed_sha256':sha(location.read_bytes())}
            else:
                raise AssertionError((name,'mutated snapshot incorrectly accepted',expected_rule))
            assert canonical(original)==baseline_bytes[base], 'mutation escaped the deep copy'
            report['negative_rows'].append(row)

        def node_change(offset,delta=1,gradient=False):
            def f(e):
                key=gid if gradient else parent
                n=e['after']['nodes'][key];put(n,offset,w(n,offset)+delta)
            return f
        run('extra-incarnation-not-attributed','materialization',
            lambda e:e['after']['header'].__setitem__('incarnation',e['after']['header']['incarnation']+1),
            'unexplained-incarnation-delta')
        run('wrong-gradient-incarnation','materialization',node_change(392),'gradient-incarnation')
        run('wrong-gradient-incoming-reference-count','materialization',node_change(400,gradient=True),
            'gradient-reference-count')
        run('wrong-node-reference-count','materialization',node_change(208),'node-reference-count')
        run('wrong-accumulator-reference-count','materialization',node_change(88),'accumulator-reference-count')
        run('private-gradient-incorrectly-public','materialization',node_change(200,gradient=True),
            'unexpected-public-or-nonleaf-gradient')
        def absent_presence(e): put(e['after']['nodes'][parent],360,0)
        run('gradient-binding-without-presence','materialization',absent_presence,'gradient-not-defined')
        def wrong_T_mapping(e):
            g=e['after']['nodes'][gid];slot=w(g,192)&0xffffff
            e['after']['tables']['1'][slot-1][6]=int(parent)
        run('gradient-T-record-points-to-wrong-node','materialization',wrong_T_mapping,'T-row-node-mapping')
        run('committed-incoming-temporary-not-cleared','materialization',node_change(448),'descriptor-transient')
        run('hidden-gradient-node-entry','materialization',lambda e:e['after']['nodes'].pop(gid),'node-index-coverage')
        def duplicate_inc(e): put(e['after']['nodes'][gid],320,w(e['after']['nodes'][parent],320))
        run('duplicate-live-node-incarnation','materialization',duplicate_inc,'node-incarnation-unique')
        def orphan(e):
            g=e['after']['nodes'][gid]
            for n in e['after']['nodes'].values():
                if w(n,384)==int(gid):
                    put(n,384,0);put(n,392,0);put(n,360,0)
            put(g,208,w(g,208)-w(g,400));put(g,400,0)
        run('extra-unrooted-live-gradient-cannot-explain-allocation','materialization',orphan,'anonymous-live-node')
        def collapse_compound(e):
            next(t for t in e['tickets'] if t['count']==3)['count']=1
        run('compound-three-constructors-miscounted-as-one','compound-construction',collapse_compound,
            'unexplained-incarnation-delta')
        def duplicate_ticket(e): e['tickets'].append(copy.deepcopy(e['tickets'][0]))
        run('duplicate-program-creation-ticket','compound-construction',duplicate_ticket,'duplicate-creation-ticket')
        def wrong_reset_highwater(e):
            n=e['after']['header']['next'];e['after']['nodes'][str(n)]=[0]*80
            e['after']['header']['next']=n+1
        run('reset-high-water-includes-unexplained-dead-slot','reset-with-private-gradient',wrong_reset_highwater,
            'reset-high-water')
        run('unrequested-generation-change','materialization',
            lambda e:e['after']['header'].__setitem__('epoch',e['after']['header']['epoch']+1),'unrequested-reset')
        run('failed-frame-raw-owner-digest-changed','failed-frame-rollback',
            lambda e:e['after'].__setitem__('full_sha256','0'*64),'failed-frame-full-owner-rollback')
        run('failed-frame-owner-pointer-changed','failed-frame-rollback',
            lambda e:e['after'].__setitem__('pointer',e['after']['pointer']+4096),'failed-frame-structural-rollback')
        assert len(report['positive_rows'])==4 and len(report['negative_rows'])==18
        for name,event in chosen.items():
            assert canonical(event)==baseline_bytes[name]
        for path,metadata in report['input_files'].items():
            raw=Path(path).read_bytes()
            assert len(raw)==metadata['bytes'] and sha(raw)==metadata['sha256'],('input changed',path)
        assert args.observer.read_bytes()==source
        report.update(status='PASS_OBSERVER_CONTROLS_ONLY',positive_cases=4,negative_cases=18,
                      original_journals_unchanged=True,observer_source_unchanged=True,
                      failures=0)
        code=0
    except Exception as error:
        report.update(status='FAIL_OBSERVER_CONTROLS',error=type(error).__name__+': '+str(error),
                      traceback=traceback.format_exc())
        code=1
    destination=args.output/'report.json'
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'sha256':sha(destination.read_bytes()),
                      'positive_cases':len(report['positive_rows']),'negative_cases':len(report['negative_rows']),
                      'path':str(destination)}))
    return code

if __name__=='__main__': raise SystemExit(main())
