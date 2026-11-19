"""Read actual Torch observations and their supervised physical process proof."""

import hashlib
import json
from pathlib import Path
import stat

from .cases import CASES, CASE_SHA256

SUPPORT='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(raw,mode):
    return dict(bytes=len(raw),mode=mode,sha256=sha(raw))


def read(repository,directory,execution,bind):
    directory,execution=Path(directory).resolve(),Path(execution).resolve()
    raw=bind(directory/'report.json');report=json.loads(raw)
    process=json.loads(bind(execution/'process.json'))
    require(report['schema']=='heptatorch.numeric-predicates.reference.v1'
        and report['status']=='OBSERVED_NUMERIC_PREDICATES'
        and report['cases']==report['expected_cases']==len(CASES)==485
        and report['case_sha256']==CASE_SHA256 and len(report['rows'])==len(CASES)
        and report['failures']==[] and report['Torch_executed'] is True
        and report['native_executed'] is False and report['compiler_executed'] is False
        and report['host_expected_values_computed'] is False
        and report['inputs_unchanged'] is True and report['rng_unchanged'] is True
        and report['all_observed_blobs_unchanged'] is True,'complete independent actual Torch reference')
    provenance=report['oracle_provenance']
    require(provenance['status']=='PASS' and len(provenance['checks'])==11
            and all(value is True for value in provenance['checks'].values())
            and provenance['lock_sha256']==LOCK,'all eleven original pinned-wheel provenance checks')
    bind(repository/'tests/pytorch_oracle.py',SUPPORT)
    lock=json.loads(bind(repository/'compatibility/pytorch-oracle-lock.json',LOCK))
    require(provenance['expected']==lock['oracle'],'actual reference uses the exact fixed official wheel lock')
    require(process['started'] is True and process['exit_code']==0
            and process['stdout_EOF'] is True and process['stderr_EOF'] is True
            and process['forced_cleanup'] is False and not process.get('timed_out')
            and not process.get('error') and not process.get('execution_error')
            and not process['cleanup_errors'] and not process['running_descendants_after_cleanup']
            and not process['terminated_process_groups'],'actual reference exit, dual EOF and normal completion')
    argv=process['argv']
    require(len(argv)==7 and argv[1:]==['-B',str(repository/'tests/numeric_predicates/reference.py'),
            '--repository',str(repository),'--output',str(directory)]
            and report['actual_process']['argv']==argv and report['actual_process']['pid']==process['pid'],
            'exact supervised original Torch argv and PID')
    interpreter=Path(argv[0]).resolve(strict=True)
    python=report['actual_process']['python']
    require(python['path']==str(interpreter)
            and identity(bind(interpreter),stat.S_IMODE(interpreter.stat().st_mode))==
            {key:python[key] for key in ('bytes','mode','sha256')},'actual exact interpreter bytes and mode')
    for channel in ('stdout','stderr'):
        stream=bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete actual reference '+channel)
    require(bind(execution/'runner.stderr')==b'','clean actual Torch reference stderr')
    require(json.loads(bind(execution/'runner.stdout'))==dict(status='OBSERVED_NUMERIC_PREDICATES',
        cases=485,expected_cases=485,error=None),'actual reference stdout reports the complete observation')
    expected_files={str(p.resolve()) for p in [repository/'tests/numeric_predicates/reference.py',
        repository/'tests/numeric_predicates/__init__.py',repository/'tests/numeric_predicates/cases.py',
        repository/'tests/pytorch_oracle.py',repository/'compatibility/pytorch-oracle-lock.json',
        repository/'compatibility/upstream-lock.json',repository/'build/pytorch-install.json',interpreter]}
    require(set(report['inputs'])==expected_files,'complete exact original reference input census')
    for path,saved in report['inputs'].items():
        file=Path(path);observed=identity(bind(file,saved['sha256']),stat.S_IMODE(file.stat().st_mode))
        require(saved==dict(path=path,**observed),'all original reference input bytes and modes unchanged')
    def blob(item):
        require(set(item)=={'path','bytes','sha256'} and item['path']=='blobs/'+item['sha256']+'.bin'
                and len(item['sha256'])==64,'canonical complete observed blob reference')
        data=bind(directory/item['path'],item['sha256'])
        require(len(data)==item['bytes'],'complete independently observed blob extent')
        return data
    for index,(row,spec) in enumerate(zip(report['rows'],CASES)):
        require(row==json.loads(bind(directory/('case-%03d.json'%index)))
                and row['index']==index and row['spec']==spec and row['accepted'] is True
                and all(row[field] is True for field in ('fresh_storage','fresh_tensor','input_unchanged','rng_unchanged')),
                'unchanged complete independent reference row and literal recipe')
        for value in (row['input'],row['result']):
            blob(value['logical']);blob(value['storage'])
    return dict(root=directory,execution=execution,report=report,report_sha256=sha(raw),blob=blob)
