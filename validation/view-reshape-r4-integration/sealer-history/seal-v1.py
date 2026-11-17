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
    report={'schema':'heptatorch.r4.ci-byte-mode-custody.v1','status':'RUNNING',
            'source_commit':a.source_commit,'repository':str(repo),'stack_build':str(stack),
            'native_executed':False,'Torch_executed':False,'compiler_executed':False,
            'case_admissions_executed_here':0,'closed_schema_count':0,'whole_pytorch_gate':'HOLD',
            'scope':'File-byte and original POSIX-mode custody only; original reports keep their own statuses.',
            'required_reports':list(REPORTS),'required_directories':list(DIRECTORIES),
            'missing':[],'errors':[]}
    selected={};required_input_bindings={}
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
            for path in sorted(directory.rglob('*')):
                if path.is_symlink():raise ValueError(('symlink in evidence',str(path)))
                if path.is_file():add(path)
        for name in (*REPORTS,*SOURCE_FILES):
            path=repo/name
            if not path.is_file():report['missing'].append(name)
            else:add(path)
        for pattern in ('view-reshape-r4-*.make.stdout','view-reshape-r4-*.make.stderr'):
            for path in sorted((repo/'build').glob(pattern)):add(path)
        for name in INPUT_LISTS:
            path=repo/name
            if not path.is_file():report['missing'].append(name);continue
            raw,_=info(path);items=json.loads(raw)
            if not isinstance(items,dict) or not items:raise ValueError(('empty input custody',name))
            for spelling,expected in items.items():
                source=Path(spelling)
                if not source.is_absolute() or str(source.resolve())!=spelling:raise ValueError(('noncanonical bound input',spelling))
                raw,current=info(source)
                if current!=expected:raise ValueError(('input custody mismatch',spelling,current,expected))
                if spelling in required_input_bindings and required_input_bindings[spelling]!=expected:raise ValueError(('conflicting bound input',spelling))
                required_input_bindings[spelling]=expected;add(source)
        if not selected:raise ValueError('no actual evidence available')
        members={}
        archive=out/'complete.tar.xz'
        with tarfile.open(archive,'x:xz',format=tarfile.PAX_FORMAT,preset=1) as tf:
            for name,path in sorted(selected.items()):
                raw,meta=info(path)
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
        for name,path in selected.items():
            _,meta=info(path)
            if meta!={k:members[name][k] for k in ('bytes','sha256','mode')}:raise ValueError(('evidence changed',name))
        manifest={'schema':'heptatorch.r4.ci-complete-members.v1','source_commit':a.source_commit,
                  'repository':str(repo),'stack_build':str(stack),
                  'member_count':len(members),'member_bytes':sum(m['bytes'] for m in members.values()),
                  'members':members,'archive':archive_meta,
                  'external_path_mapping':{'prefix':'external/stack-toolchain/','original_root':str(stack)},
                  'required_input_bindings':required_input_bindings}
        dump(out/'file-manifest.json',manifest)
        report.update(member_count=len(members),member_bytes=manifest['member_bytes'],archive=archive_meta,
                      manifest=info(out/'file-manifest.json')[1],all_selected_files_unchanged=True,
                      complete_single_XZ_EOF=True,complete_TAR_zero_padding=True,all_original_member_modes_preserved=True,
                      bound_input_count=len(required_input_bindings))
        report['missing']=sorted(set(report['missing']))
        report['status']='FAIL' if report['missing'] else 'PASS'
        code=int(report['status']!='PASS')
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
    dump(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','member_count','member_bytes','missing','error')}))
    return code
if __name__=='__main__':raise SystemExit(main())
