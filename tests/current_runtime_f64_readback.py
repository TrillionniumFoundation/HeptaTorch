"""Pure physical qualification of current773 F64 48 observations.

Complete original numerical, boundary, resource and owner functions consume
the recorded native frames. Two malformed factory controls retain their reviewed
status changes; the original F64 negation request is checked against the same
actual405 Torch reference used by wide_neg. All87 original requests remain.
No native constructor, process, Torch, compiler, prlimit, /proc read, or rewrite
of original evidence is performed.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import sys
import types
import zlib

SOURCE = '4722c53b248780534a846a472784a565c4454ab738db75d655a7801b6623f07f'
IMAGE = '5cb190cfc4b920bb03e78803e79296e2e046ec90eb6e07b2d5ba6eebfb534edd'
READER = '1deb2fa84d9ea9653ca56769a5527f0102571ba2a4690f8deb85f5d4bdc038b3'
ENTRY = '32196771085c4dbf24751ce9fa3b052a35655bd9c7cddb392d019174ffa50a44'
REFERENCE = 'b1dafeed270b4dd226c349dece9d7f6213c2dabd34681685d97ec13c068e9b11'
PRODUCER = '23d09ddb6414eca8610e80ff17519129dc6175c02789f8c693a8e1ef36bd590e'
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



def outer_execution(report, repository, build, build_sha, directory,
                    reader_directory, reference_directory, execution_directory, bound):
    receipt = bound.json(execution_directory / 'runner.json')
    require(receipt['started'] is True and receipt['exit_code'] == 0
            and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
            and receipt['forced_cleanup'] is False
            and isinstance(receipt['pid'], int) and receipt['pid'] > 0,
            'actual original48 outer process succeeded with complete two-stream EOF')
    for flag in ('timed_out', 'execution_error', 'error', 'cleanup_errors',
                 'running_descendants_after_cleanup', 'terminated_process_groups'):
        require(not receipt.get(flag), 'outer process failure or cleanup: ' + flag)
    argv = receipt['argv']
    require(isinstance(argv, list) and argv[:4] == [argv[0], '-I', '-S', '-B'],
            'actual isolated current48 worker flags')
    wrapper = repository / 'tests/current_runtime_f64_fixtures.py'
    bound.read(wrapper)
    expected = [argv[0], '-I', '-S', '-B', str(wrapper),
        '--reader-directory', str(reader_directory),
        '--candidate', str(repository), '--build-directory', str(build),
        '--reference', str(reference_directory / 'report.json'), '--expected-source-sha256', SOURCE,
        '--expected-native-sha256', IMAGE, '--expected-build-sha256', build_sha,
        '--expected-reference-sha256', REFERENCE,
        '--wide-neg-reference-directory', report['source_admission']['wide_neg_successor']['reference_directory'],
        '--wide-neg-reference-execution', report['source_admission']['wide_neg_successor']['reference_execution'],
        '--public-f64-arithmetic-reference-directory', report['source_admission']['public_f64_arithmetic_successor']['reference_directory'],
        '--public-f64-arithmetic-reference-execution', report['source_admission']['public_f64_arithmetic_successor']['reference_execution'],
        '--output', str(directory)]
    require(argv == expected, 'complete actual current48 argv and physical source route')
    python = Path(argv[0]).resolve(strict=True)
    bound.read(python)
    intent = bound.json(execution_directory / 'intent.json')
    require(set(intent) == {'argv', 'observer_python'} and intent['argv'] == argv,
            'complete pre-execution intent uses the actual argv')
    identity = intent['observer_python']
    require(set(identity) == {'path', 'bytes', 'sha256', 'mode'}
            and identity['path'] == str(python)
            and {key: identity[key] for key in ('bytes', 'sha256', 'mode')} == bound.rows[str(python)],
            'actual observer Python was explicitly bound before native verification')
    streams = {}
    for name in ('stdout', 'stderr'):
        raw = bound.read(execution_directory / ('runner.' + name))
        row = dict(bytes=len(raw), sha256=sha(raw))
        require(row == dict(bytes=receipt[name + '_bytes'], sha256=receipt[name + '_sha256'])
                and (name != 'stderr' or raw == b''), 'complete actual outer ' + name)
        streams[name] = row
    summary = parse(bound.read(execution_directory / 'runner.stdout'))
    require(summary == {key: report[key] for key in ('status', 'expected_numeric_groups',
                                                   'expected_boundary_cases', 'native_executed')},
            'the actual outer stdout is the original48 summary')
    return dict(argv=argv, receipt=receipt, streams=streams, observer_python_pre_execution_intent=intent,
        intent_sha256=bound.rows[str(execution_directory / 'intent.json')]['sha256'],
        resolved_interpreter=str(python), resolved_interpreter_readback=bound.rows[str(python)],
        observer_python_bound_before_execution=True,
        original_worker_bound_interpreter=str(python) in report['inputs'],
        historical_Torch_interpreter_inferred=False)


class RecordedAppend:
    """Compare original progress writes to existing bytes without opening output."""
    def __init__(self, state):
        self.state, self.closed = state, False

    def __enter__(self):
        return self

    def write(self, text):
        require(not self.closed and isinstance(text, str), 'original progress append is open text')
        raw = text.encode()
        at = self.state['at']
        require(self.state['raw'][at:at + len(raw)] == raw,
                'original numeric progress write differs from physical evidence')
        self.state['at'] += len(raw)
        self.state['writes'] += 1
        return len(text)

    def __exit__(self, *unused):
        self.closed = True


class CasePath:
    """Original output writes become comparisons to already existing evidence."""
    def __init__(self, root, bound, relative='', appends=None):
        self.root,self.bound,self.relative = root,bound,relative
        self.appends = {} if appends is None else appends

    def __truediv__(self, name):
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'bounded original output path')
        return CasePath(self.root,self.bound,str(Path(self.relative)/p),self.appends)

    def __fspath__(self):
        return str(self.root/self.relative)

    def write_bytes(self, value):
        require(self.bound.read(self.root/self.relative)==value, 'original output bytes differ: '+self.relative)
        return len(value)

    def open(self, mode):
        require(mode == 'a' and self.relative == 'numeric-progress.jsonl',
                'only original numeric progress appends may be replayed')
        if self.relative not in self.appends:
            self.appends[self.relative] = dict(raw=self.bound.read(self.root / self.relative), at=0, writes=0)
        return RecordedAppend(self.appends[self.relative])


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
        require(all(row['at'] == len(row['raw']) for row in self.directory.appends.values()),
                'every original numeric progress byte consumed')
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
    require(re.fullmatch('[0-9a-f]{64}', build_sha) is not None, 'explicit actual current build digest')
    require(report['schema'] == 'heptatorch.f64.tensor-native736.v1' and report['status'] == 'PASS'
            and report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
            and report['actual_build_sha256'] == build_sha and report['native_executed'] is True
            and report['Torch_executed'] is False and report['compiler_executed'] is False
            and report['all_inputs_unchanged'] is True and report['failures'] == []
            and report['full_project_status'] == 'HOLD',
            'actual current F6448 report identity and limited qualification scope')
    recorded_inputs(report, bound)
    bound.read(reader_directory / 'source-manifest.json', READER)
    bound.read(reader_directory / 'current48.py', ENTRY)
    wrapper = repository / 'tests/current_runtime_f64_fixtures.py'
    reader = module(bound.read(wrapper), 'current_f64_48_identity_adapter', wrapper)
    require(reader.CURRENT_SOURCE == SOURCE and reader.CURRENT_IMAGE == IMAGE,
            'the same current-image adapter for execution and independent readback')
    return reader


def prepare_originals(adapter, report, repository, build, build_sha, bound):
    originals, old_run = adapter.old, adapter.run
    inputs = originals['common'].Inputs()
    require(adapter.bind_package(inputs) == report['reader_manifest_sha256'] == READER,
            'complete reviewed109-payload reader package')
    source = adapter.candidate_admission(repository, inputs)
    cold = adapter.cold_admission(build, IMAGE, build_sha, inputs)
    require(ordinary(source) == report['source_admission'] and ordinary(cold) == report['cold_admission'],
            'complete original source inverse, two-status derivation and actual73 cold admission')
    owner = adapter.make_owner(repository, IMAGE, inputs)
    methods = {
        'Native.capture': originals['native'].Native.capture,
        'Native.exchange': originals['native'].Native.exchange,
        'Native.one': originals['native'].Native.one,
        'Native.factory_packet': originals['native'].Native.factory_packet,
        'Native.create': originals['native'].Native.create,
        'Native.view': originals['native'].Native.view,
        'Native.tensor': originals['native'].Native.tensor,
        'transport.exchange': originals['native'].CapturingSession.exchange,
        'transport.finish': originals['native'].CapturingSession.finish,
        'owner.verify_snapshot': owner.verify_snapshot,
        'owner.verify_frame_boundary': owner.verify_frame_boundary,
        'Driver.__init__': old_run.Driver.__init__, 'Driver.check': old_run.Driver.check,
        'Driver.observe': old_run.Driver.observe, 'Driver.mark': old_run.Driver.mark,
        'Driver.cleanup': old_run.Driver.cleanup, 'numeric_case': old_run.numeric_case,
        'boundary_case': old_run.boundary_case, 'original_controls.reject': originals['controls'].reject,
        'original_controls.entry_controls': originals['controls'].entry_controls,
        'original_controls.resources': originals['controls'].resources,
        'original_controls.allocation_control': originals['controls'].allocation_control,
        'current_controls.reject': adapter.current_controls.reject,
        'current_controls.entry_controls': adapter.current_controls.entry_controls,
        'current_controls.resources': adapter.current_controls.resources,
        'current_controls.allocation_control': adapter.current_controls.allocation_control,
    }
    bindings = {name: method_binding(function, bound) for name, function in methods.items()}
    return inputs, owner, bindings, cold


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



def qualify_f64_48(report, repository, build, build_sha, directory, reader_directory,
                   reference_directory, inputs, execution_directory):
    repository, build, directory, reader_directory, reference_directory, execution_directory = [
        Path(path).resolve(strict=True) for path in
        (repository, build, directory, reader_directory, reference_directory, execution_directory)]
    bound = Bound(inputs)
    bound.read(Path(__file__).resolve())
    require(bound.json(directory / 'report.json') == report, 'exact actual48 report argument')
    reader = admission(report, repository, build, build_sha, reader_directory, bound)
    execution = outer_execution(report, repository, build, build_sha, directory,
                                reader_directory, reference_directory, execution_directory, bound)
    results = []
    negation = report['source_admission']['wide_neg_successor']
    shared = dict(root=negation['reference_directory'], execution=negation['reference_execution'])
    arithmetic = report['source_admission']['public_f64_arithmetic_successor']
    mixed = dict(root=arithmetic['reference_directory'], execution=arithmetic['reference_execution'])
    with reader.current_reader(repository, reader_directory, shared, mixed) as adapter:
        originals, old_run = adapter.old, adapter.run
        original, owner, bindings, cold = prepare_originals(adapter, report, repository, build, build_sha, bound)
        mapping = parse(original.bind(reader_directory / 'frozen-candidate719/original55-mapping.json', old_run.MAPPING))
        plan = old_run.groups(original, mapping)
        compare, comparison_proof = old_run.comparator(original)
        require(comparison_proof == report['original_comparison_inverse'], 'complete original numerical predicate inverse')
        reference = adapter.reference(reference_directory / 'report.json', REFERENCE, original, plan, PRODUCER)
        require(old_run.Reference.__init__.__globals__['SOURCE'] ==
                'e2eedf870e306572672817cfc98bcf49d9aeacc0657c04dbaf50e5e6f5ceb66d',
                'historical719 reference namespace remains separate from actual736')
        complete_original_input_table(original, report, [], bound)
        require(report['reference_sha256'] == REFERENCE
                and report['case_plan'] == [{key: value for key, value in group.items() if key != 'pairs'} for group in plan]
                and report['boundary_contract'] == old_run.BOUNDARY_CONTRACT
                and report['expected_numeric_groups'] == 33 and report['expected_boundary_cases'] == 12
                and report['expected_entry_groups'] == 1 and report['expected_allocation_controls'] == 2
                and report['original_strict55_modified'] is False and report['original_strict55_failed_groups'] == 4
                and report['original_strict55_mismatches'] == 4008 and report['original_sqrt_groups_not_exposed'] == 6
                and report['original_standalone_refusals_not_transferred'] == 16
                and report['full_dtype_gate_closed'] is False and report['full_transition_recomputed'] is False,
                'all original48 census, recipe, comparison and unclosed-scope statements')
        categories = ('numeric_groups', 'boundary_cases', 'entry_groups', 'allocation_controls')
        require([len(report[key]) for key in categories] == [33, 12, 1, 2], 'complete33+12+1+2 denominator')
        require(sum(group['input_values'] for group in plan) == 1798174, 'all original numeric input values')
        args = types.SimpleNamespace(build_directory=build, expected_native_sha256=IMAGE)
        jobs = []
        for index, (group, wanted) in enumerate(zip(plan, reference.data['groups'])):
            jobs.append(('numeric_groups', 'numeric-%02d' % index, report['numeric_groups'][index], group, wanted))
        for index, (spec, wanted) in enumerate(zip(old_run.CASES, reference.data['boundary_cases'])):
            jobs.append(('boundary_cases', 'boundary-%02d' % index, report['boundary_cases'][index], spec, wanted))
        jobs.append(('entry_groups', 'entry-00', report['entry_groups'][0], None, None))
        for index, stage in enumerate(('cast', 'binary')):
            jobs.append(('allocation_controls', 'allocation-' + stage, report['allocation_controls'][index], stage, None))
        for category, name, row, spec, wanted in jobs:
            require(row['native_started'] is True, 'original native child creation recorded')
            expected_name = ('explicit-dtype-raw-and-noAD-boundaries' if category == 'entry_groups' else
                             'F64-' + spec + '-actual-OOM-recovery' if category == 'allocation_controls' else spec['name'])
            require(row['name'] == expected_name, 'exact original child recipe identity')
            case = RecordedCase(directory / name, row, originals['native'], owner, bound)
            driver = driver_for(case, old_run, args, reference, compare)
            saved = SavedIO(case, driver)
            controls = saved.controls(adapter.current_controls)
            if category == 'numeric_groups':
                details = old_run.numeric_case(driver, spec, wanted)
                require(set(case.directory.appends) == {'numeric-progress.jsonl'}
                        and case.directory.appends['numeric-progress.jsonl']['writes'] == len(details['batches']),
                        'all original numeric batch progress writes compared')
            elif category == 'boundary_cases':
                details = old_run.boundary_case(driver, spec, wanted)
            elif category == 'entry_groups':
                details = adapter.public_f64_arithmetic_successor.wrap_entry(controls['entry_controls'])(driver)
                require(details['original_entry_requests'] == len(driver.operations) == 87
                        and details['actual_refusals'] == 63 and details['actual_successful_negations'] == 1
                        and details['actual_successful_mixed_arithmetic'] == 23
                        and details['actual_successful_FE_arithmetic'] == 1,
                        'all original87 requests,63 retained refusals,one checked negation and23 checked arithmetic results')
                changed = [operation for operation in driver.operations if operation['label'] in
                           ('public-factory-dtype-5', 'public-factory-dtype-6')]
                require([operation['label'] for operation in changed] ==
                        ['public-factory-dtype-5', 'public-factory-dtype-6']
                        and all(operation['expected_status'] == operation['actual_status'] == 100
                                and operation['record_count'] == 0 for operation in changed),
                        'exact two malformed Half factory status changes remain rejections')
            else:
                details = controls['allocation_control'](driver, spec)
                require(saved.limit_at == 4 and details['actual_status_sequence'] == [3, 0],
                        'original resource failure and actual identical-request recovery sequence')
            if category != 'numeric_groups':
                require(case.directory.appends == {}, 'no additional original numeric output route')
            cleanup = driver.cleanup()
            driver.n.finish()
            require(ordinary(details) == row['details'] and ordinary(cleanup) == row['cleanup_owner']
                    and ordinary(driver.checks) == row['checks'] and ordinary(driver.marks) == row['observations']
                    and ordinary(driver.operations) == row['operations'],
                    'complete original cases, numerical loops, owner relations and assertions replay exactly')
            results.append(dict(category=category, directory=name, name=row['name'], pid=case.receipt['pid'],
                start_ticks=case.receipt['start_ticks'], frames=case.s.frames, checks=len(driver.checks),
                tensor_public_storage_reconstructions=case.tensor_calls, owner_blobs=len(case.blob_bytes),
                complete_snapshot_predicate_calls=case.snapshot_calls, frame_predicates=len(case.boundaries),
                original_details=details, numeric_progress_bytes=sum(len(item['raw']) for item in case.directory.appends.values()),
                resource_readback=dict(reads=saved.reads, existing_output_comparisons=saved.writes,
                    original_recorded_prlimit_sequence=saved.limit_calls, new_prlimit_or_proc_read_executed=False)))
        require(len(results) == report['native_children'] == 48
                and len({row['pid'] for row in results}) == 48
                and sum(row['frames'] for row in results) == report['frames'],
                'all48 distinct actual children and complete actual frame denominator')
        numeric = [row['original_details'] for row in results if row['category'] == 'numeric_groups']
        require(sum(row['exact_non_nan'] for row in numeric) == sum(row['non_nan_attempted'] for row in numeric)
                and all(row['mismatches'] == [] for row in numeric), 'every original non-NaN and NaN predicate passed')
        finish_inputs(original, bound)
    return dict(schema='heptatorch.current736.f64-48.physical-qualification.v1', status='PASS_PHYSICAL_F64_48',
        actual_report_sha256=bound.rows[str(directory / 'report.json')]['sha256'], source_sha256=SOURCE, native_sha256=IMAGE,
        actual_build_sha256=build_sha, cases=48, original_counts=dict(numeric=33, boundary=12, entry=1, allocation=2),
        frames=sum(row['frames'] for row in results), checks=sum(row['checks'] for row in results),
        original_input_values=1798174, original_refusal_recipes=87, changed_factory_status_expectations=2,
        unchanged_other_refusal_expectations=61, retained_refusals=63,
        actual_numeric_successors=24, wide_neg_successor=negation, original_OOM_controls=2,
        public_f64_arithmetic_successor=arithmetic, retained_bool_subtraction_refusals=2,
        actual_mixed_arithmetic_successors=23, actual_FE_arithmetic_successors=1,
        exact_non_nan=sum(row['exact_non_nan'] for row in numeric),
        nan_classification=sum(row['nan_classification'] for row in numeric),
        complete_snapshot_predicate_calls=sum(row['complete_snapshot_predicate_calls'] for row in results),
        tensor_public_storage_reconstructions=sum(row['tensor_public_storage_reconstructions'] for row in results),
        owner_blobs=sum(row['owner_blobs'] for row in results), native_child_EOFs=96,
        original_bound_inputs=len(report['inputs']), strict_cold_physical_inputs=cold['physical_inputs'],
        exact_original_method_bindings=bindings, actual_outer_execution=execution, children=results,
        numerical_expected_values_computed_by_host=False, original_recipe_code_preserved=True,
        native_executed_by_qualifier=False, Torch_executed_by_qualifier=False, compiler_executed_by_qualifier=False,
        new_proc_reads_or_resource_limit_changes=False, all_physical_inputs_unchanged=True, physical_inputs=bound.rows,
        original_worker_bound_interpreter=False, observer_python_pre_execution_intent_verified=True,
        historical_Torch_interpreter_identity_inferred=False,
        original_strict55_failed_groups=4, original_strict55_mismatches=4008,
        full_dtype_gate_closed=False, full_project_status='HOLD')


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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'build-directory', 'directory', 'reader-directory',
                 'reference-directory', 'execution-directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    args = parser.parse_args()
    require(not sys.flags.optimize, 'all original assertions remain enabled')
    require(not args.output.exists(), 'new pure readback output only')
    inputs = Inputs()
    try:
        report = parse(inputs.bind(args.directory / 'report.json'))
        result = qualify_f64_48(report, args.repository, args.build_directory, args.expected_build_sha256,
            args.directory, args.reader_directory, args.reference_directory, inputs, args.execution_directory)
    except BaseException as error:
        result = dict(status='FAIL_PHYSICAL_F64_48', error=repr(error), traceback=traceback.format_exc(),
            native_executed_by_qualifier=False, Torch_executed_by_qualifier=False, compiler_executed_by_qualifier=False,
            physical_inputs=inputs.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'cases', 'frames', 'checks') if key in result}))
    return 0 if result['status'] == 'PASS_PHYSICAL_F64_48' else 1


if __name__ == '__main__':
    raise SystemExit(main())
