"""Pure physical readback of the unchanged Half47 and separate Half12 controls.

The original numerical and owner predicates consume recorded native frames.
No native constructor, Torch expression, compiler, process, prlimit, or /proc
read is performed. Original files are never rewritten by these APIs.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import sys
import textwrap
import types
import zlib

SOURCE = '0061acfaa8e886885053e8f539cf08187be1c1e10265c19e52df07435a71594b'
IMAGE = '955ffc996b8870ef67752f45647f5c1ca9a093f41ff402487091f03223372293'
BRIDGE = 'f3b19a8dcc1a004323e033f414e801a52f12b0f0d556e6cbc005169d26fdef9e'
READER = '1f67d27b8a5af133955d8a8cdfdb3094415efe31d1d9ee7ab09437f683ecef06'
REFERENCE = 'ae6079c8f0960db53d26e056cc67fbe0a8eee7de04aa0644f4475afc368a75fe'
CONTROL12 = '6461ea5959a864741c73eb35164a59d55418bcbd3fa9aafac5a8cdbfc6082f4f'
MASK = (1 << 64) - 1


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw,object_pairs_hook=unique)


def ordinary(value):
    """Compare all original JSON values, preserving ordered sequences."""
    return json.loads(json.dumps(value,allow_nan=False))


class Bound:
    def __init__(self, inputs):
        self.inputs, self.rows = inputs, {}

    def read(self, path, expected=None, size=None, mode=None):
        path = Path(path).absolute()
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_nlink==1, 'unique ordinary physical evidence: '+str(path))
        raw = self.inputs.bind(path,expected)
        after = path.lstat()
        require(all(getattr(before,k)==getattr(after,k) for k in
            ('st_dev','st_ino','st_size','st_mode','st_mtime_ns','st_ctime_ns')), 'evidence changed while reading')
        row = dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row['sha256']==expected, 'physical evidence SHA differs')
        require(size is None or row['bytes']==size, 'physical evidence length differs')
        require(mode is None or row['mode']==mode, 'physical evidence mode differs')
        require(str(path) not in self.rows or self.rows[str(path)]==row, 'physical input changed after binding')
        self.rows[str(path)] = row
        return raw

    def json(self, path, expected=None):
        return parse(self.read(path,expected))

    def recheck(self):
        for path,row in list(self.rows.items()):
            self.read(path,row['sha256'],row['bytes'],row['mode'])


def module(raw, name, path):
    value = types.ModuleType(name)
    value.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),value.__dict__)
    return value


def method_binding(function, bound):
    path = Path(function.__globals__['__file__']).resolve(strict=True)
    raw = bound.read(path)
    root = compile(raw,str(path),'exec')
    matches = []
    def visit(code):
        for value in code.co_consts:
            if isinstance(value,types.CodeType):
                if (value.co_qualname==function.__code__.co_qualname
                        and value.co_firstlineno==function.__code__.co_firstlineno):
                    matches.append(value)
                visit(value)
    visit(root)
    require(len(matches)==1 and matches[0]==function.__code__
            and function.__code__.co_filename==str(path), 'exact original physical method code')
    return dict(source_path=str(path),source_sha256=sha(raw),qualname=function.__code__.co_qualname,
        first_line=function.__code__.co_firstlineno,actual_method_code_equals_recompiled_original=True)


def frame_bodies(raw, magic):
    result, at = [], 0
    while at<len(raw):
        require(at+16<=len(raw), 'complete physical frame header')
        observed,size = struct.unpack_from('<8sQ',raw,at)
        require(observed==magic and at+16+size<=len(raw) and size<=1<<20, 'physical frame extent and magic')
        result.append((raw[at:at+16+size],raw[at+16:at+16+size]))
        at += 16+size
    require(at==len(raw), 'all physical frame bytes consumed')
    return result


def recorded_inputs(report, bound):
    require(isinstance(report['inputs'],dict) and report['inputs'], 'original bound input table')
    for path,row in report['inputs'].items():
        bound.read(path,row['sha256'],row['bytes'],row['mode'])


def outer_execution(kind, report, repository, build, build_sha, directory,
                    reader_directory, reference_directory, execution_directory, bound):
    receipt = bound.json(execution_directory/'runner.json')
    require(receipt['exit_code']==0 and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
        and receipt['forced_cleanup'] is False and isinstance(receipt['pid'],int) and receipt['pid']>0,
        'successful actual outer process and complete two-stream EOF')
    for flag in ('timed_out','execution_error','error','cleanup_errors','running_descendants_after_cleanup','terminated_process_groups'):
        require(not receipt.get(flag), 'outer process failure or cleanup: '+flag)
    if 'started' in receipt:
        require(receipt['started'] is True, 'outer process actually started')
    argv = receipt['argv']
    require(isinstance(argv,list) and argv[:4]==[argv[0],'-I','-S','-B'], 'actual isolated original worker flags')
    python = Path(argv[0]).resolve(strict=True)
    bound.read(python)
    expected = [argv[0],'-I','-S','-B']
    if kind=='half47':
        wrapper=repository/'tests/current_runtime_half_fixtures.py'
        bound.read(wrapper)
        successor=report['entry_groups'][0]['details']['multiplication_successor']
        negation=report['entry_groups'][0]['details']['negation_successor']
        addsub=report['entry_groups'][0]['details']['addsub_successor']
        division=report['entry_groups'][0]['details']['division_successor']
        expected += [str(wrapper),'--reader-directory',str(reader_directory),
            '--candidate',str(repository),'--build-directory',str(build),
            '--reference',str(reference_directory/'report.json'),'--expected-source-sha256',SOURCE,
            '--expected-native-sha256',IMAGE,'--expected-build-sha256',build_sha,
            '--expected-reference-sha256',REFERENCE,
            '--half-mul-reference-directory',successor['reference_directory'],
            '--half-mul-reference-execution',successor['reference_execution'],
            '--half-neg-reference-directory',negation['reference_directory'],
            '--half-neg-reference-execution',negation['reference_execution'],
            '--half-addsub-reference-directory',addsub['reference_directory'],
            '--half-addsub-reference-execution',addsub['reference_execution'],
            '--half-div-reference-directory',division['reference_directory'],
            '--half-div-reference-execution',division['reference_execution'],
            '--output',str(directory)]
        bound.read(reader_directory/'run.py')
    else:
        require(len(argv)>4, 'actual control entry script')
        control_path = Path(argv[4]).resolve(strict=True)
        bound.read(control_path,CONTROL12)
        require(str(control_path) in report['inputs'] and report['inputs'][str(control_path)]['sha256']==CONTROL12,
                'the actual12 report bound its complete original control source')
        expected += [str(control_path),'--bridge-directory',str(reader_directory),'--repository',str(repository),
            '--build-directory',str(build),'--expected-build-sha256',build_sha,'--output',str(directory)]
    require(argv==expected, 'complete actual outer argv and physical route')
    legacy12 = kind=='half12' and 'stdout' in receipt and 'stderr' in receipt
    if legacy12:
        intent = bound.json(execution_directory/'intent.json')
        require(intent==dict(argv=argv,source_sha256=CONTROL12), 'original12 outer pre-execution intent')
        names = [('stdout','stdout'),('stderr','stderr')]
    else:
        require(receipt.get('started') is True, 'standard outer process start receipt')
        names = [('stdout','runner.stdout'),('stderr','runner.stderr')]
    streams = {}
    for name,filename in names:
        raw = bound.read(execution_directory/filename)
        row = dict(bytes=len(raw),sha256=sha(raw))
        expected_row = receipt[name] if legacy12 else dict(bytes=receipt[name+'_bytes'],sha256=receipt[name+'_sha256'])
        require(row==expected_row and (name!='stderr' or raw==b''), 'entire actual outer '+name)
        streams[name]=row
    summary = parse(bound.read(execution_directory/names[0][1]))
    if kind=='half47':
        require(summary=={k:report[k] for k in ('status','expected_counts','native_executed')}, 'actual worker stdout summary')
    else:
        require(summary=={k:report.get(k) for k in ('status','native_children','frames','checks','readback_refusal_frames','native_executed')},
                'actual12 stdout summary')
    return dict(argv=argv,receipt=receipt,streams=streams,resolved_interpreter=str(python),
        resolved_interpreter_readback=bound.rows[str(python)],
        original_report_bound_interpreter=str(python) in report['inputs'],
        original_interpreter_SHA_inferred_from_later_readback=False)


class CasePath:
    """Original output writes become comparisons to already existing evidence."""
    def __init__(self, root, bound, relative=''):
        self.root,self.bound,self.relative = root,bound,relative

    def __truediv__(self, name):
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'bounded original output path')
        return CasePath(self.root,self.bound,str(Path(self.relative)/p))

    def __fspath__(self):
        return str(self.root/self.relative)

    def write_bytes(self, value):
        require(self.bound.read(self.root/self.relative)==value, 'original output bytes differ: '+self.relative)
        return len(value)


class ReadOnlyJournal:
    def __init__(self, rows):
        self.rows,self.at,self.closed = rows,0,False

    def write(self, text):
        require(not self.closed and self.at<len(self.rows), 'one original journal record per physical frame')
        require(parse(text)==self.rows[self.at], 'every original frame-owner predicate and endpoint agrees')
        self.at += 1
        return len(text)

    def flush(self):
        require(not self.closed, 'record journal is open during original exchange')


class RecordedCase:
    def __init__(self, folder, row, native, owner, bound):
        self.folder,self.row,self.native,self.owner,self.bound = folder,row,native,owner,bound
        physical_report = bound.json(folder/('control-report.json' if 'group' in row else 'report.json'))
        require(physical_report==row, 'top-level and physical original child report agree completely')
        receipt = bound.json(folder/'raw/transport.json')
        require(receipt==row['transport'] and row['status']=='PASS' and receipt['status']=='PASS'
            and receipt['exit_code']==0 and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
            and receipt['selected_image_unchanged'] is True and receipt['forced_cleanup'] is False
            and receipt['errors']==[] and receipt['actual_proc_exe_sha256']==receipt['selected_native_sha256']==IMAGE,
            'actual child image, success, no cleanup, and both real EOF')
        require(isinstance(receipt['pid'],int) and receipt['pid']>0
            and isinstance(receipt['start_ticks'],int) and receipt['start_ticks']>0,
            'actual native child PID and starttime')
        require(not row.get('error') and not row.get('EOF_error') and not row.get('finish_error'), 'no original child error')
        for filename,prefix in [('stderr.bin','stderr'),('extra-stdout-after-frames.bin','extra_stdout')]:
            raw = bound.read(folder/'raw'/filename)
            require(raw==b'' and len(raw)==receipt[prefix+'_bytes'] and sha(raw)==receipt[prefix+'_sha256'],
                    'entire actual child '+prefix)
        request_raw = bound.read(folder/'raw/requests.frames.bin',receipt['request_stream_sha256'])
        response_raw = bound.read(folder/'raw/responses.frames.bin',receipt['response_stream_sha256'])
        self.requests = frame_bodies(request_raw,b'HTOWNR01')
        self.responses = frame_bodies(response_raw,b'HTOWNS01')
        self.events = [parse(line) for line in bound.read(folder/'owner-frames.jsonl').splitlines()]
        self.intents = [parse(line) for line in bound.read(folder/'raw/request-intents.jsonl').splitlines()]
        require(len(self.requests)==len(self.responses)==len(self.events)==len(self.intents)==row['frames']==receipt['frames'],
                'every original request, response, intent, and owner frame accounted for')
        for i,((request,body),(_,response),event,intent) in enumerate(zip(self.requests,self.responses,self.events,self.intents)):
            require(intent==dict(frame=i+1,intended_bytes=len(request),intended_sha256=sha(request)), 'exact completed original write intent')
            require(event['index']==i and event['request_bytes']==len(body) and event['request_sha256']==sha(body)
                and event['response_bytes']==len(response) and event['response_sha256']==sha(response)
                and not event.get('error'), 'all actual journal frame bytes')
            status,records=native.WIRE.parse(response)
            require(status==event['status'] and len(records)==event['records'], 'original public decoder status and records')
        self.manifest = bound.json(folder/'owner-blob-manifest.json')
        require(isinstance(self.manifest,dict) and self.manifest, 'complete physical owner manifest')
        self.blob_bytes={};self.used=set();self.snapshot_calls=0;self.tensor_calls=0
        for digest,meta in self.manifest.items():
            require(meta['sha256']==digest and meta['path']=='complete-owner/'+digest+'.bin.gz', 'canonical complete owner blob path')
            packed=bound.read(folder/meta['path'],meta['compressed_sha256'],meta['compressed_bytes'],meta['mode'])
            decoder=zlib.decompressobj(31)
            data=decoder.decompress(packed)+decoder.flush()
            require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
                and sha(data)==digest and len(data)==meta['bytes'], 'complete single gzip owner and exact original bytes')
            self.blob_bytes[digest]=data
        require({p.name for p in (folder/'complete-owner').iterdir()}=={Path(r['path']).name for r in self.manifest.values()},
                'no missing or additional owner archive')
        self.receipt=receipt
        self.s=types.SimpleNamespace(pid=receipt['pid'],frames=0)
        self.current=self.events[0]['before']
        self.s.exchange=self.exchange_recorded
        self.journal=ReadOnlyJournal(self.events)
        self.boundaries=[]
        self.directory=CasePath(folder,bound)
        self.checks=[]

    def endpoint(self, reference, label):
        digest=reference['sha256'];self.used.add(digest)
        require(digest in self.blob_bytes, 'owner endpoint has original physical bytes')
        raw=self.blob_bytes[digest]
        require(reference['length']==len(raw) and isinstance(reference['pointer'],int), 'physical owner extent and pointer')
        snapshot=self.owner.decode_owner(raw,reference['pointer'])
        checks=self.owner.verify_snapshot(snapshot);self.snapshot_calls+=1
        require(ordinary(checks)==reference['checks'], 'complete original owner snapshot predicates')
        endpoint=dict(label=label,pointer=reference['pointer'],length=len(raw),sha256=digest,checks=checks)
        return snapshot,endpoint,raw

    def capture(self, label):
        return self.endpoint(self.current,label)

    def exchange_recorded(self, body):
        i=self.s.frames
        require(i<len(self.requests) and self.requests[i][1]==body, 'original recipe request differs from actual native bytes')
        event=self.events[i]
        require(all(self.current[k]==event['before'][k] for k in ('sha256','pointer','length')),
                'owner is continuous between physical command frames')
        self.current=event['after'];self.s.frames+=1
        return self.responses[i][1]

    def finish(self):
        require(self.s.frames==self.journal.at==len(self.events)==len(self.boundaries), 'every physical frame and owner rule consumed exactly once')
        require(self.used==set(self.manifest), 'every complete owner blob consumed by original predicates')
        return self.receipt

    def native_object(self):
        recorded=self
        methods={name:getattr(self.native.Native,name) for name in ('one','exchange','create','view')}
        methods['factory_packet']=staticmethod(self.native.Native.factory_packet)
        methods['capture']=lambda obj,label:recorded.capture(label)
        methods['finish']=lambda obj:recorded.finish()
        def tensor(obj, token):
            recorded.tensor_calls+=1
            return recorded.native.Native.tensor(obj,token)
        methods['tensor']=tensor
        cls=type('RecordedOriginalNative',(),methods)
        obj=cls()
        obj.owner=self.owner;obj.directory=self.directory;obj.s=self.s
        obj.journal=self.journal;obj.boundaries=self.boundaries
        return obj


class SavedIO:
    """Read original recorded resource I/O; never access /proc or call prlimit."""
    RLIMIT_AS=9
    RLIM_INFINITY=-1

    def __init__(self, case, driver):
        self.case,self.driver=case,driver
        self.map_label=None;self.map_at=0;self.limit_at=0
        self.reads=[];self.writes=[];self.limit_calls=[]
        self.limit=None

    def path(self, spelling):
        spelling=str(spelling)
        prefix='/proc/%d/'%self.case.s.pid
        require(spelling.startswith(prefix), 'only this recorded native PID resource paths')
        name=spelling[len(prefix):]
        require(name in ('maps','status','limits'), 'only original resource observation paths')
        context=self
        class Read:
            def read_bytes(self):
                if name=='maps':
                    require(context.map_label is not None and context.map_at<2, 'exact pair of original mapping reads')
                    filename=context.map_label+('.maps-before' if context.map_at==0 else '.maps-after')
                    context.map_at+=1
                else:
                    filename='before-limit.'+name
                raw=context.case.bound.read(context.case.folder/filename)
                context.reads.append(dict(original_path=spelling,saved_file=filename,bytes=len(raw),sha256=sha(raw)))
                return raw
        return Read()

    def dump(self, path, value):
        require(isinstance(path,CasePath) and path.root==self.case.folder, 'only compare existing case evidence')
        raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        path.write_bytes(raw)
        self.writes.append(dict(path=path.relative,bytes=len(raw),sha256=sha(raw)))

    def prlimit(self, pid, kind, replacement=None):
        require(pid==self.case.s.pid and kind==self.RLIMIT_AS, 'only the original child address-space limit record')
        if self.limit is None:
            self.limit=self.case.bound.json(self.case.folder/'allocation-limit.json')
            raw=self.case.bound.read(self.case.folder/'before-limit.limits')
            line=next(x for x in raw.decode().splitlines() if x.startswith('Max address space'))
            values=line[len('Max address space'):].split()
            parsed=[-1 if x=='unlimited' else int(x) for x in values[:2]]
            require(parsed==self.limit['old_limit'] and values[2]=='bytes', 'actual saved original RLIMIT_AS values')
        old=tuple(self.limit['old_limit']);soft=(self.limit['applied_soft_limit'],old[1])
        expected=[None,soft,old,None]
        require(self.limit_at<4 and replacement==expected[self.limit_at], 'original prlimit read/set/restore/read sequence')
        result=old if replacement is None else None
        if self.limit_at==3:
            rows=[x for x in self.case.row['checks'] if x['label']=='original-RLIMIT-restored']
            require(len(rows)==1 and rows[0]['passed'] is True, 'one original restoration observation')
            result=tuple(rows[0]['actual'])
            require(result==old and rows[0]['expected']==list(old), 'actual recorded original limit restoration')
        self.limit_calls.append(dict(pid=pid,resource=kind,replacement=replacement,recorded_result=result,
            original_setter_return_value_was_unused=replacement is not None))
        self.limit_at+=1
        return result

    def clone(self, function, globals_map):
        copied=types.FunctionType(function.__code__,globals_map,function.__name__,function.__defaults__,function.__closure__)
        copied.__kwdefaults__=function.__kwdefaults__
        require(copied.__code__ is function.__code__, 'unchanged complete original resource code object')
        return copied

    def controls(self, original):
        namespace=dict(original.__dict__)
        namespace.update(Path=self.path,resource=self,dump=self.dump)
        reject=self.clone(original.reject,namespace)
        def scoped_reject(driver,label,body,expected,**kwargs):
            self.map_label,self.map_at=label,0
            result=reject(driver,label,body,expected,**kwargs)
            require(self.map_at==(2 if kwargs.get('mappings',False) else 0), 'all original mapping observations consumed')
            return result
        namespace['reject']=scoped_reject
        namespace['resources']=self.clone(original.resources,namespace)
        return dict(reject=scoped_reject,entry_controls=self.clone(original.entry_controls,namespace),
            allocation_control=self.clone(original.allocation_control,namespace))


def driver_for(case, old_run, args, reference, compare):
    driver=object.__new__(old_run.Driver)
    def factory(binary,source,owner,directory,image):
        require(binary==args.build_directory/'heptatorch' and source==args.build_directory/'runtime.hepta'
            and owner is case.owner and directory is case.directory and image==IMAGE, 'unchanged original native-constructor arguments')
        return case.native_object()
    namespace=dict(old_run.Driver.__init__.__globals__);namespace['Native']=factory
    init=types.FunctionType(old_run.Driver.__init__.__code__,namespace,'recorded_original_Driver_init')
    require(init.__code__ is old_run.Driver.__init__.__code__, 'complete original Driver initialization code')
    init(driver,args,case.owner,case.directory,reference,compare)
    return driver


def admission(report, repository, build, build_sha, reader_directory, bound):
    require(re.fullmatch('[0-9a-f]{64}',build_sha) is not None, 'explicit actual current build digest')
    require(report['status'] in ('PASS','PASS_DIRECT6_AND_FE6_UNSUPPORTED_HALF_CONTROLS')
        and report['source_sha256']==SOURCE and report['native_sha256']==IMAGE
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is False and report['compiler_executed'] is False
        and report['all_inputs_unchanged'] is True and report['failures']==[]
        and report['full_project_status']=='HOLD', 'actual current Half report identity and bounded status')
    recorded_inputs(report,bound)
    bound.read(reader_directory/'source-manifest.json',READER)
    bound.read(reader_directory/'bridge.py',BRIDGE)
    wrapper=repository/'tests/current_runtime_half_fixtures.py'
    current=module(bound.read(wrapper),'half_current_identity_adapter',wrapper)
    half_mul=half_neg=half_addsub=half_div=None
    if report['status']=='PASS':
        successor=report['entry_groups'][0]['details']['multiplication_successor']
        half_mul=dict(root=Path(successor['reference_directory']),
                      execution=Path(successor['reference_execution']))
        negation=report['entry_groups'][0]['details']['negation_successor']
        addsub=report['entry_groups'][0]['details']['addsub_successor']
        division=report['entry_groups'][0]['details']['division_successor']
        half_neg=dict(root=Path(negation['reference_directory']),
                      execution=Path(negation['reference_execution']))
        half_addsub=dict(root=Path(addsub['reference_directory']),
                         execution=Path(addsub['reference_execution']))
        half_div=dict(root=Path(division['reference_directory']),
                      execution=Path(division['reference_execution']))
    bridge=current.current_bridge(repository,reader_directory,half_mul,half_neg,half_addsub,half_div)
    require(bridge.SOURCE==SOURCE and bridge.IMAGE==IMAGE and bridge.REFERENCE==REFERENCE,
            'current-image wrapper around the complete unchanged original bridge')
    return bridge


def prepare_originals(bridge, originals, old_run, startup, report, repository, build, build_sha, bound):
    inputs=originals['common'].Inputs()
    adapter=bridge.Adapter(originals,old_run,startup)
    require(adapter.bind_package(inputs)==report['reader_manifest_sha256']==READER, 'entire original146 reader package')
    source=adapter.candidate_admission(repository,inputs)
    cold=adapter.cold_admission(build,IMAGE,build_sha,repository,inputs)
    require(ordinary(source)==report['source_admission'] and ordinary(cold)==report['cold_admission'],
            'complete original source and actual73 cold admission readback')
    owner=adapter.make_owner(repository,IMAGE,inputs)
    methods={
        'Native.capture':originals['native'].Native.capture,
        'Native.exchange':originals['native'].Native.exchange,
        'Native.one':originals['native'].Native.one,
        'Native.factory_packet':originals['native'].Native.factory_packet,
        'Native.create':originals['native'].Native.create,
        'Native.view':originals['native'].Native.view,
        'Native.tensor':originals['native'].Native.tensor,
        'transport.exchange':originals['native'].CapturingSession.exchange,
        'transport.finish':originals['native'].CapturingSession.finish,
        'owner.verify_snapshot':owner.verify_snapshot,
        'owner.verify_frame_boundary':owner.verify_frame_boundary,
        'Driver.__init__':old_run.Driver.__init__, 'Driver.check':old_run.Driver.check,
        'Driver.observe':old_run.Driver.observe,'Driver.mark':old_run.Driver.mark,
        'Driver.cleanup':old_run.Driver.cleanup,'numeric_case':old_run.numeric_case,
        'boundary_case':old_run.boundary_case,'controls.reject':originals['controls'].reject,
        'controls.entry_controls':originals['controls'].entry_controls,
        'controls.resources':originals['controls'].resources,
        'controls.allocation_control':originals['controls'].allocation_control}
    bindings={name:method_binding(fn,bound) for name,fn in methods.items()}
    return inputs,adapter,owner,bindings,cold


def finish_inputs(original, bound):
    require(original.unchanged() is True, 'all original pure admission inputs unchanged')
    for path,row in original.rows.items():
        bound.read(path,row['sha256'],row['bytes'],row['mode'])
    bound.recheck()


def complete_original_input_table(original, report, extras, bound):
    """A supplied input table must equal the original reconstructed admission."""
    reconstructed=dict(original.rows)
    for path in extras:
        path=Path(path).resolve(strict=True)
        bound.read(path)
        reconstructed[str(path)]=bound.rows[str(path)]
    require(set(reconstructed)==set(report['inputs']), 'complete original input-binding census, without omissions or additions')
    for path,row in reconstructed.items():
        require(all(row[key]==report['inputs'][path][key] for key in ('bytes','sha256','mode')),
                'reconstructed original input identity differs: '+path)
    return len(reconstructed)


def qualify_half47(report, repository, build, build_sha, directory, reader_directory,
                   reference_directory, inputs, execution_directory):
    repository,build,directory,reader_directory,reference_directory,execution_directory = [
        Path(p).resolve(strict=True) for p in (repository,build,directory,reader_directory,reference_directory,execution_directory)]
    bound=Bound(inputs)
    bound.read(Path(__file__).resolve())
    require(bound.json(directory/'report.json')==report, 'exact actual47 report argument')
    bridge=admission(report,repository,build,build_sha,reader_directory,bound)
    execution=outer_execution('half47',report,repository,build,build_sha,directory,reader_directory,reference_directory,execution_directory,bound)
    results=[]
    with bridge.original_modules() as (originals,old_run,startup):
        original,adapter,owner,bindings,cold=prepare_originals(bridge,originals,old_run,startup,report,repository,build,build_sha,bound)
        plan=adapter.groups(repository,original)
        compare,comparison_proof=adapter.comparisons(repository,original)
        require(comparison_proof==report['original_comparison_inverse'], 'complete original numerical predicate inverse')
        reference=old_run.Reference(reference_directory/'report.json',REFERENCE,original,plan,bridge.PRODUCER_MANIFEST,comparison_proof)
        complete_original_input_table(original,report,[],bound)
        require(report['reference_sha256']==REFERENCE and report['reference_producer_manifest_sha256']==bridge.PRODUCER_MANIFEST
            and report['reference_producer_source_sha256']==bridge.PRODUCER_SOURCE, 'historical reference producer kept distinct from actual736')
        require(report['expected_counts']==old_run.TASK_COUNTS and report['boundary_contract']==old_run.BOUNDARY_CONTRACT
            and report['case_plan']==[old_run.public_plan(g) for g in plan]
            and report['original_numeric_groups']==26 and report['original_input_values']==908216
            and report['original_comparison_predicates_modified'] is False
            and report['original_standalone_refusals_not_transferred']==12
            and report['ordinary_half_AD_supported'] is False and report['full_dtype_gate_closed'] is False
            and report['full_transition_recomputed'] is False, 'all original47 domains, census and scope statements')
        categories=('numeric_groups','boundary_cases','entry_groups','allocation_controls')
        require([len(report[k]) for k in categories]==[26,16,1,4], 'original26+16+1+4 denominator')
        args=types.SimpleNamespace(build_directory=build,expected_native_sha256=IMAGE)
        jobs=[]
        for i,(group,wanted) in enumerate(zip(plan,reference.data['groups'])):
            jobs.append(('numeric_groups','numeric-%02d'%i,report['numeric_groups'][i],group,wanted))
        for i,(spec,wanted) in enumerate(zip(old_run.CASES,reference.data['boundary_cases'])):
            jobs.append(('boundary_cases','boundary-%02d'%i,report['boundary_cases'][i],spec,wanted))
        jobs.append(('entry_groups','entry-00',report['entry_groups'][0],None,None))
        for i,(spec,wanted) in enumerate(zip(old_run.OOM_SPECS,reference.data['allocation_recovery'])):
            jobs.append(('allocation_controls','allocation-%02d'%i,report['allocation_controls'][i],spec,wanted))
        for category,name,row,spec,wanted in jobs:
            require(row['native_started'] is True, 'original child creation recorded')
            expected_name='public5-6-private3-and-noAD-refusals' if category=='entry_groups' else spec['name']
            require(row['name']==expected_name, 'exact original child recipe identity')
            case=RecordedCase(directory/name,row,originals['native'],owner,bound)
            driver=driver_for(case,old_run,args,reference,compare)
            saved=SavedIO(case,driver);controls=saved.controls(originals['controls'])
            if category=='numeric_groups':
                details=old_run.numeric_case(driver,spec,wanted)
            elif category=='boundary_cases':
                details=old_run.boundary_case(driver,spec,wanted)
            elif category=='entry_groups':
                details=adapter.half_div_successor.wrap_entry(controls['entry_controls'])(driver)
                require(details['actual_refusals']==151 and details['actual_successful_multiplications']==22
                    and details['actual_successful_negations']==2 and details['actual_successful_addsub']==42
                    and details['actual_successful_single_command_addsub']==40
                    and details['actual_successful_FE_addsub']==2 and details['actual_successful_divisions']==22
                    and details['actual_successful_additions']==24 and details['actual_successful_subtractions']==18
                    and details['original_entry_requests']==len(driver.operations)==239,
                    'all original239 requests:151 refusals and22/2/40/2/22 exact arithmetic successors')
            else:
                details=controls['allocation_control'](driver,spec,wanted)
                require(saved.limit_at==4 and details['actual_status_sequence']==[3,0], 'original resource failure and actual recovery sequence')
            cleanup=driver.cleanup()
            driver.n.finish()
            require(ordinary(details)==row['details'] and ordinary(cleanup)==row['cleanup_owner']
                and ordinary(driver.checks)==row['checks'] and ordinary(driver.marks)==row['observations']
                and ordinary(driver.operations)==row['operations'], 'complete original cases, numerical loops, owner relations and checks replay exactly')
            results.append(dict(category=category,directory=name,name=row['name'],pid=case.receipt['pid'],
                start_ticks=case.receipt['start_ticks'],frames=case.s.frames,checks=len(driver.checks),
                tensor_public_storage_reconstructions=case.tensor_calls,owner_blobs=len(case.blob_bytes),
                complete_snapshot_predicate_calls=case.snapshot_calls,frame_predicates=len(case.boundaries),
                original_details=details,resource_readback=dict(reads=saved.reads,existing_output_comparisons=saved.writes,
                    original_recorded_prlimit_sequence=saved.limit_calls,new_prlimit_or_proc_read_executed=False)))
        require(len(results)==report['native_children']==47 and len({x['pid'] for x in results})==47
            and sum(x['frames'] for x in results)==report['frames'], 'all47 distinct actual children and exact actual frame denominator')
        finish_inputs(original,bound)
    return dict(schema='heptatorch.current736.half47-physical-qualification.v1',status='PASS_PHYSICAL_HALF47',
        actual_report_sha256=bound.rows[str(directory/'report.json')]['sha256'],source_sha256=SOURCE,native_sha256=IMAGE,
        actual_build_sha256=build_sha,cases=47,original_counts=report['expected_counts'],
        frames=sum(x['frames'] for x in results),checks=sum(x['checks'] for x in results),
        original_input_values=908216,original_entry_requests=239,original_refusals=239,
        retained_refusals=151,accepted_multiplication_successors=22,accepted_division_successors=22,
        accepted_negation_successors=2,accepted_addsub_successors=42,
        accepted_single_command_addsub_successors=40,accepted_FE_addsub_successors=2,
        accepted_addition_successors=24,accepted_subtraction_successors=18,
        retained_actual_Bool_subtraction_errors=4,retained_private3_addsub_protocol_refusals=8,
        retained_private3_division_protocol_refusals=4,
        original_OOM_controls=4,
        complete_snapshot_predicate_calls=sum(x['complete_snapshot_predicate_calls'] for x in results),
        tensor_public_storage_reconstructions=sum(x['tensor_public_storage_reconstructions'] for x in results),
        owner_blobs=sum(x['owner_blobs'] for x in results),native_child_EOFs=94,
        original_bound_inputs=len(report['inputs']),strict_cold_physical_inputs=cold['physical_inputs'],
        exact_original_method_bindings=bindings,actual_outer_execution=execution,children=results,
        numerical_expected_values_computed_by_host=False,original_recipe_code_preserved=True,
        native_executed_by_qualifier=False,Torch_executed_by_qualifier=False,compiler_executed_by_qualifier=False,
        new_proc_reads_or_resource_limit_changes=False,all_physical_inputs_unchanged=True,physical_inputs=bound.rows,
        historical_plan_fields_are_not_actual_execution_facts=True,
        original_worker_bound_interpreter=False,full_dtype_gate_closed=False,full_project_status='HOLD')


def original12_recipe(control_path, bound):
    """Wrap the exact contiguous original case statements without editing them."""
    raw=bound.read(control_path,CONTROL12)
    original=module(raw,'fixed_original_half12_source',control_path)
    tree=ast.parse(raw)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    candidates=[]
    for node in ast.walk(main):
        if isinstance(node,ast.Try):
            begin=next((i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
                    and t.value.id=='driver' and t.attr=='gen' for t in n.targets)),None)
            if begin is not None:
                candidates.append((node,begin))
    require(len(candidates)==1, 'one exact original12 per-child recipe body')
    node,begin=candidates[0]
    final=node.body[-1]
    require(isinstance(final,ast.Expr) and isinstance(final.value,ast.Call)
        and isinstance(final.value.func,ast.Attribute) and isinstance(final.value.func.value,ast.Name)
        and final.value.func.value.id=='row' and final.value.func.attr=='update', 'original case ends at its complete observation update')
    lines=raw.decode().splitlines(keepends=True)
    fragment=''.join(lines[node.body[begin].lineno-1:final.end_lineno])
    body=textwrap.dedent(fragment)
    wrapped='def replay_original_case(driver, row, group, dtype, opcode, wire):\n'+textwrap.indent(body,'    ')
    parsed=ast.parse(wrapped).body[0]
    require(ast.dump(ast.Module(body=node.body[begin:],type_ignores=[]),include_attributes=False)
        ==ast.dump(ast.Module(body=parsed.body,type_ignores=[]),include_attributes=False),
        'every original case statement and assertion preserved')
    proof=dict(source_path=str(control_path),source_sha256=CONTROL12,
        original_start_line=node.body[begin].lineno,original_end_line=final.end_lineno,
        original_fragment_bytes=len(fragment.encode()),original_fragment_sha256=sha(fragment.encode()),
        wrapped_source_bytes=len(wrapped.encode()),wrapped_source_sha256=sha(wrapped.encode()),
        only_new_wrapper_and_indentation=True,complete_original_statement_AST_equal=True,
        original_numerical_owner_refusal_and_cleanup_statements_changed=False)
    return original,wrapped,proof


def qualify_half12(report, repository, build, build_sha, directory, reader_directory,
                   inputs, execution_directory):
    repository,build,directory,reader_directory,execution_directory = [Path(p).resolve(strict=True)
        for p in (repository,build,directory,reader_directory,execution_directory)]
    bound=Bound(inputs)
    bound.read(Path(__file__).resolve())
    require(bound.json(directory/'report.json')==report, 'exact actual12 report argument')
    bridge=admission(report,repository,build,build_sha,reader_directory,bound)
    execution=outer_execution('half12',report,repository,build,build_sha,directory,reader_directory,None,execution_directory,bound)
    control_path=Path(execution['argv'][4]).resolve(strict=True)
    original12,wrapped,recipe_proof=original12_recipe(control_path,bound)
    require(report['schema']=='heptatorch.current736.half-direct6-and-FE6-refusals.v1'
        and report['required_direct_cases']==report['required_FE_transaction_cases']==6
        and report['required_refusal_frames']==report['readback_refusal_frames']==12
        and report['original47_and239_modified_or_rerun'] is False
        and all(report[k] is True for k in ('complete_owner_byte_rollback_readback','original_full_snapshot_predicates_rechecked',
            'original_frame_predicates_for_direct6','dedicated_failed_FE_frame_scope_for_FE6','actual_stdin_transactions_and_both_EOF')),
        'separate direct6/FE6 scope and complete original12 denominator')
    expected=[(group,dtype,operation,opcode) for group in ('direct','FE-transaction') for dtype in (5,6)
        for operation,opcode in (('any',118),('all',119),('sum-Bool',115))]
    require(len(report['cases'])==12, 'all twelve explicit controls retained')
    results=[]
    with bridge.original_modules() as (originals,old_run,startup):
        original,adapter,owner,bindings,cold=prepare_originals(bridge,originals,old_run,startup,report,repository,build,build_sha,bound)
        fixed_methods=('Native.capture','Native.exchange','owner.verify_snapshot','transport.exchange','transport.finish','controls.reject')
        require(report['original_method_bindings']=={name:bindings[name] for name in fixed_methods},
                'all six actual original method bindings retained')
        bindings['dedicated_failed_FE_frame']=method_binding(original12.reject_failed_half_fe_transaction,bound)
        complete_original_input_table(original,report,[control_path,Path(execution['resolved_interpreter'])],bound)
        cls=type('Original12Driver',(),dict(check=old_run.Driver.check,rng_unchanged=old_run.Driver.rng_unchanged))
        for (group,dtype,operation,opcode),row in zip(expected,report['cases']):
            name='%s-dtype%d-%s'%(group,dtype,operation)
            require(row['name']==name and row['group']==group and row['dtype']==dtype
                and row['opcode']==opcode and row['expected_status']==7 and row['native_executed'] is True,
                'exact original12 case identity and execution fact')
            case=RecordedCase(directory/name,row,originals['native'],owner,bound)
            driver=cls();driver.checks=[];driver.operations=[];driver.directory=case.directory;driver.n=case.native_object()
            saved=SavedIO(case,driver);controls=saved.controls(originals['controls'])
            namespace=dict(original12.__dict__)
            namespace['Path']=saved.path
            exact_FE=saved.clone(original12.reject_failed_half_fe_transaction,namespace)
            def failed_FE(d,wire,body,dt,op):
                saved.map_label,saved.map_at='same-frame-factory-FE-refusal',0
                result=exact_FE(d,wire,body,dt,op)
                require(saved.map_at==2, 'two actual failed-FE mapping records')
                return result
            namespace.update(reject=controls['reject'],reject_failed_half_fe_transaction=failed_FE)
            exec(compile(wrapped,'<exact-original-half12-case-fragment>','exec'),namespace)
            observed={}
            namespace['replay_original_case'](driver,observed,group,dtype,opcode,originals['native'].WIRE)
            driver.n.finish()
            require(all(ordinary(value)==row[key] for key,value in observed.items())
                and ordinary(driver.checks)==row['checks'] and ordinary(driver.operations)==row['refusals'],
                'all original12 assertions and recorded comparisons replay exactly')
            errors=[event for event in case.events if event['status']]
            require(len(errors)==1 and errors[0]['status']==7 and errors[0]['records']==0, 'one actual status7 empty result per control')
            error=errors[0]
            if group=='direct':
                require(len(error['commands'])==1 and error['commands'][0][0]==opcode
                    and 'frame_admission' not in error, 'unchanged original single-frame owner admission for direct6')
            else:
                require(error['frame_admission']=='new-explicit-failed-Half-FE-transaction-only'
                    and error['owner_check']==dict(original_full_snapshot_predicates_before_and_after=True,
                        actual_complete_owner_bytes_equal=True,actual_owner_pointer_and_length_equal=True,
                        old_single_sum_frame_admission_invoked=False,intermediate_success_state_inferred=False),
                    'FE6 retains its explicit failed-transaction-only scope')
            results.append(dict(name=name,group=group,dtype=dtype,opcode=opcode,pid=case.receipt['pid'],
                start_ticks=case.receipt['start_ticks'],frames=case.s.frames,checks=len(driver.checks),
                tensor_public_storage_reconstructions=case.tensor_calls,owner_blobs=len(case.blob_bytes),
                complete_snapshot_predicate_calls=case.snapshot_calls,
                original_frame_predicates=sum('frame_admission' not in e for e in case.events),
                dedicated_failed_FE_frames=sum('frame_admission' in e for e in case.events),
                mappings_readback=saved.reads))
        require(len(results)==report['native_children']==len({x['pid'] for x in results})==12
            and sum(x['frames'] for x in results)==report['frames']==180
            and sum(x['checks'] for x in results)==report['checks']==222,
            'all distinct actual12 children and complete physical frames/checks')
        finish_inputs(original,bound)
    return dict(schema='heptatorch.current736.half12-physical-qualification.v1',status='PASS_PHYSICAL_HALF12',
        actual_report_sha256=bound.rows[str(directory/'report.json')]['sha256'],source_sha256=SOURCE,native_sha256=IMAGE,
        actual_build_sha256=build_sha,cases=12,direct_cases=6,failed_FE_cases=6,frames=180,checks=222,
        native_child_EOFs=24,actual_status7_empty_record_frames=12,
        owner_blobs=sum(x['owner_blobs'] for x in results),
        complete_snapshot_predicate_calls=sum(x['complete_snapshot_predicate_calls'] for x in results),
        tensor_public_storage_reconstructions=sum(x['tensor_public_storage_reconstructions'] for x in results),
        original_bound_inputs=len(report['inputs']),strict_cold_physical_inputs=cold['physical_inputs'],
        exact_original_method_bindings=bindings,original_case_source_derivation=recipe_proof,
        actual_outer_execution=execution,children=results,all_physical_inputs_unchanged=True,physical_inputs=bound.rows,
        direct6_original_frame_predicates_preserved=True,FE6_separate_failed_transaction_scope=True,
        original47_and239_modified_or_rerun=False,native_executed_by_qualifier=False,
        Torch_executed_by_qualifier=False,compiler_executed_by_qualifier=False,
        new_proc_reads_or_resource_limit_changes=False,full_project_status='HOLD')


class Inputs:
    """Standalone CLI custody; CI may supply its existing shared Inputs object."""
    def __init__(self):
        self.rows={}

    def bind(self, path, expected=None):
        path=Path(path).absolute()
        before=path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_nlink==1, 'ordinary standalone evidence input')
        raw=path.read_bytes();after=path.lstat()
        require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mode','st_mtime_ns','st_ctime_ns')),
                'standalone input changed during read')
        row=dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row['sha256']==expected, 'standalone input SHA')
        require(str(path) not in self.rows or self.rows[str(path)]==row, 'standalone input changed')
        self.rows[str(path)]=row
        return raw


def main():
    import argparse
    import traceback
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('half47','half12'),required=True)
    for name in ('repository','build-directory','directory','reader-directory','execution-directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path)
    parser.add_argument('--expected-build-sha256',required=True)
    args=parser.parse_args()
    require(not sys.flags.optimize, 'all original assertions remain enabled')
    require(not args.output.exists(), 'new pure readback output only')
    inputs=Inputs()
    try:
        report=parse(inputs.bind(args.directory/'report.json'))
        if args.suite=='half47':
            require(args.reference_directory is not None, 'the original full physical reference is required')
            result=qualify_half47(report,args.repository,args.build_directory,args.expected_build_sha256,
                args.directory,args.reader_directory,args.reference_directory,inputs,args.execution_directory)
        else:
            require(args.reference_directory is None, 'the unsupported12 protocol controls have no Torch reference')
            result=qualify_half12(report,args.repository,args.build_directory,args.expected_build_sha256,
                args.directory,args.reader_directory,inputs,args.execution_directory)
    except BaseException as error:
        result=dict(status='FAIL_PHYSICAL_HALF_READBACK',error=repr(error),traceback=traceback.format_exc(),
            native_executed_by_qualifier=False,Torch_executed_by_qualifier=False,compiler_executed_by_qualifier=False,
            physical_inputs=inputs.rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','cases','frames','checks') if key in result}))
    return 0 if result['status'].startswith('PASS_PHYSICAL_') else 1


if __name__=='__main__':
    raise SystemExit(main())
