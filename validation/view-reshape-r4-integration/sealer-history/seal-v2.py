#!/usr/bin/env python3
"""Preserve complete R4 CI evidence with original file bytes and POSIX modes."""
import argparse, hashlib, io, json, lzma, os, stat, sys, tarfile, traceback
from pathlib import Path
DIRECTORIES=(
 'build/differentiable-view-recovered',
 'build/differentiable-view-r4-shape','build/differentiable-view-r4-scalar07b',
 'build/view-reshape-r4-reference26','build/view-reshape-r4-reference32',
 'build/differentiable-view-reference14','build/differentiable-view-input4-reference',
 'build/view-reshape-r4-native','build/view-reshape-r4-resource3','build/view-reshape-r4-backward-resource',
 'tools/differentiable_view_r4_build_review','tools/differentiable_view_shape_r4',
 'tools/differentiable_view_scalar_r4','tests/differentiable_view_shape_review',
 'tests/differentiable_view_shape_reference_review','tests/differentiable_view_native_review',
 'tests/differentiable_view_input4_review','tests/wrapped_scalar_resource_review',
 'tests/backward_private_cache_resource_review')
REPORTS=tuple('build/view-reshape-r4-'+x+'.json' for x in
 ('reference26','reference32','shape-build','scalar07b-build','view26','scalar32','view14','input4','resource3','backward-resource'))
INPUT_LISTS=(
 'build/differentiable-view-r4-shape/inputs-before.json',
 'build/differentiable-view-r4-scalar07b/inputs-before.json',
 'build/view-reshape-r4-resource3/inputs-before.json',
 'build/view-reshape-r4-backward-resource/inputs-before.json')
SOURCE_FILES=('Makefile','.github/workflows/native.yml','tools/seal_r4_ci_evidence.py')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def dump(path,obj):
    with Path(path).open('xb') as f:f.write((json.dumps(obj,indent=2,sort_keys=True)+'\n').encode())
def info(path):
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():raise ValueError(('regular file required',str(path)))
    raw=path.read_bytes();after=path.stat()
    attrs=('st_dev','st_ino','st_size','st_mode','st_mtime_ns','st_ctime_ns')
    if any(getattr(before,k)!=getattr(after,k) for k in attrs):raise ValueError(('file changed',str(path)))
    return raw,{'bytes':len(raw),'sha256':digest(raw),'mode':stat.S_IMODE(before.st_mode)}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--stack-build',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();repo=a.repo.resolve();stack=a.stack_build.resolve();out=a.output.resolve()
    if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('source commit must be a full Git SHA')
    if not out.is_relative_to(repo/'build'):raise ValueError('output must be below repository build')
    out.mkdir(parents=True,exist_ok=False)
    report={'schema':'heptatorch.r4.ci-byte-mode-custody.v2','status':'RUNNING',
            'source_commit':a.source_commit,'repository':str(repo),'stack_build':str(stack),
            'native_executed':False,'Torch_executed':False,'compiler_executed':False,
            'case_admissions_executed_here':0,'closed_schema_count':0,'whole_pytorch_gate':'HOLD',
            'scope':'File-byte and original POSIX-mode custody only; original reports keep their own statuses.',
            'required_reports':list(REPORTS),'required_directories':list(DIRECTORIES),
            'missing':[],'errors':[]}
    selected={};required_input_bindings={};declared_input_bindings=[];verified_input_paths=set();input_list_bindings={}
    def problem(stage,path,error):
        report['errors'].append({'stage':stage,'path':str(path),'error':repr(error)})
    def add(path):
        path=Path(path)
        if path.is_symlink():raise ValueError(('symlink forbidden',str(path)))
        resolved=path.resolve()
        if resolved.is_relative_to(out):raise ValueError('cannot archive output recursively')
        if resolved.is_relative_to(repo):name=resolved.relative_to(repo).as_posix()
        elif resolved.is_relative_to(stack):name='external/stack-toolchain/'+resolved.relative_to(stack).as_posix()
        else:raise ValueError(('undeclared external input',str(resolved)))
        if name in selected and selected[name]!=resolved:raise ValueError(('member collision',name))
        selected[name]=resolved
    code=1
    try:
        for name in DIRECTORIES:
            directory=repo/name
            if not directory.is_dir() or directory.is_symlink():report['missing'].append(name);continue
            paths=[]
            try:
                for path in directory.rglob('*'):paths.append(path)
            except BaseException as error:problem('directory-traversal',directory,error)
            for path in sorted(paths):
                try:
                    if path.is_symlink():raise ValueError(('symlink in evidence',str(path)))
                    if path.is_file():add(path)
                    elif not path.is_dir():raise ValueError(('nonregular evidence',str(path)))
                except BaseException as error:problem('directory-member',path,error)
        for name in (*REPORTS,*SOURCE_FILES):
            path=repo/name
            if not path.is_file():report['missing'].append(name)
            else:
                try:add(path)
                except BaseException as error:problem('required-file',path,error)
        for pattern in ('view-reshape-r4-*.make.stdout','view-reshape-r4-*.make.stderr'):
            paths=[]
            try:
                for path in (repo/'build').glob(pattern):paths.append(path)
            except BaseException as error:problem('make-stream-traversal',repo/'build',error)
            for path in sorted(paths):
                try:add(path)
                except BaseException as error:problem('make-stream',path,error)
        for name in INPUT_LISTS:
            path=repo/name
            if not path.is_file():report['missing'].append(name);continue
            try:
                raw,list_meta=info(path);input_list_bindings[str(path)]=list_meta;items=json.loads(raw)
                if not isinstance(items,dict) or not items:raise ValueError(('empty input custody',name))
            except BaseException as error:
                problem('input-list',path,error);continue
            for spelling,expected in items.items():
                declared_input_bindings.append({'input_list':name,'source_path':spelling,'expected':expected})
                try:
                    source=Path(spelling)
                    if not source.is_absolute() or str(source.resolve())!=spelling:raise ValueError(('noncanonical bound input',spelling))
                    if spelling in required_input_bindings and required_input_bindings[spelling]!=expected:
                        problem('input-declaration',source,ValueError('conflicting expected metadata'))
                    else:required_input_bindings[spelling]=expected
                    add(source)
                    raw,current=info(source)
                    if current!=expected:raise ValueError(('input custody mismatch',spelling,current,expected))
                    verified_input_paths.add(spelling)
                except BaseException as error:problem('declared-input',spelling,error)
        if not selected:raise ValueError('no actual evidence available')
        members={}
        archive=out/'complete.tar.xz'
        with tarfile.open(archive,'x:xz',format=tarfile.PAX_FORMAT,preset=1) as tf:
            for name,path in sorted(selected.items()):
                try:raw,meta=info(path)
                except BaseException as error:
                    verified_input_paths.discard(str(path));problem('archive-read',path,error);continue
                if str(path) in input_list_bindings and meta!=input_list_bindings[str(path)]:
                    problem('input-list-at-archive',path,ValueError(('parsed input list changed before archive',meta,input_list_bindings[str(path)])))
                for declared in declared_input_bindings:
                    if declared['source_path']==str(path) and meta!=declared['expected']:
                        verified_input_paths.discard(str(path))
                        problem('input-at-archive',path,ValueError(('archived actual metadata differs from declared input',meta,declared['expected'])))
                entry=tarfile.TarInfo(name);entry.size=len(raw);entry.mode=meta['mode'];entry.mtime=0
                tf.addfile(entry,io.BytesIO(raw));members[name]={**meta,'source_path':str(path)}
        original_archive,archive_meta=info(archive)
        decoder=lzma.LZMADecompressor()
        with archive.open('rb') as f:
            while not decoder.eof:
                piece=f.read(65536) if decoder.needs_input else b''
                if not piece and decoder.needs_input:raise ValueError('truncated XZ')
                decoder.decompress(piece,max_length=1048576)
            if decoder.unused_data or f.read(1):raise ValueError('trailing or concatenated XZ')
        with tarfile.open(archive,'r:xz') as tf:
            rows=tf.getmembers()
            if len(rows)!=len(members) or len({m.name for m in rows})!=len(members) or {m.name for m in rows}!=set(members):raise ValueError('archive member denominator')
            for m in rows:
                if not m.isfile():raise ValueError(('nonregular archive member',m.name))
                raw=tf.extractfile(m).read();expected=members[m.name]
                if len(raw)!=m.size or len(raw)!=expected['bytes'] or digest(raw)!=expected['sha256'] or m.mode!=expected['mode']:raise ValueError(('archived bytes or mode',m.name))
            end=max(m.offset_data+((m.size+511)//512)*512 for m in rows)
            tf.fileobj.seek(end);tail=tf.fileobj.read()
            if len(tail)<1024 or any(tail):raise ValueError('TAR zero end padding')
        for name,archived in members.items():
            path=selected[name]
            try:
                _,meta=info(path)
                if meta!={k:archived[k] for k in ('bytes','sha256','mode')}:raise ValueError(('evidence changed',name))
            except BaseException as error:
                verified_input_paths.discard(str(path));problem('post-archive-custody',path,error)
        manifest={'schema':'heptatorch.r4.ci-complete-members.v2','source_commit':a.source_commit,
                  'repository':str(repo),'stack_build':str(stack),
                  'member_count':len(members),'member_bytes':sum(m['bytes'] for m in members.values()),
                  'members':members,'archive':archive_meta,
                  'external_path_mapping':{'prefix':'external/stack-toolchain/','original_root':str(stack)},
                  'required_input_bindings':required_input_bindings,
                  'declared_input_bindings':declared_input_bindings,
                  'input_list_bindings':input_list_bindings}
        dump(out/'file-manifest.json',manifest)
        report.update(member_count=len(members),member_bytes=manifest['member_bytes'],archive=archive_meta,
                      manifest=info(out/'file-manifest.json')[1],all_selected_files_unchanged=not report['errors'],
                      complete_single_XZ_EOF=True,complete_TAR_zero_padding=True,all_original_member_modes_preserved=True,
                      bound_input_count=len(verified_input_paths),declared_input_count=len(required_input_bindings),
                      preserved_readable_member_count=len(members),unarchived_selected_members=sorted(set(selected)-set(members)))
        report['missing']=sorted(set(report['missing']))
        report['status']='FAIL' if report['missing'] or report['errors'] else 'PASS'
        code=int(report['status']!='PASS')
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
    dump(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','member_count','member_bytes','missing','error')}))
    return code
if __name__=='__main__':raise SystemExit(main())
