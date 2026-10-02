"""Create six deliverables from an explicit mapping; verify ZIP contents and hashes."""
import argparse
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from common import atomic_json, digest, evidence, inside, load, utc

ARCHIVES={'evidence':'evidence.zip','harbor_task':'harbor_task.zip'}
FILES={'pilot':'小规模试跑结论.md','codex':'trajectory_codex.json','seed':'trajectory_seed.json','checklist':'自检checklist.md'}
DENIED={'.env','id_rsa','id_ed25519','credentials.json','auth.json'}


def resolve_mapping(plan,root):
    if set(plan['archives'])!=set(ARCHIVES) or set(plan['files'])!=set(FILES):
        raise ValueError('mapping must cover exactly two archives and four standalone files')
    result={}
    for label in ARCHIVES:
        selected={};items=plan['archives'][label]
        if not items:raise ValueError('empty archive mapping')
        for item in items:
            src=inside(root,item['source']);target=PurePosixPath(item['target'])
            if target.is_absolute() or '..' in target.parts or not target.parts or '\\' in item['target'] or ':' in item['target']:
                raise ValueError('unsafe archive target')
            key=target.as_posix()
            if key.casefold() in {x.casefold() for x in selected}:raise ValueError('duplicate/case-colliding archive name')
            if any(p in DENIED or p=='.git' or p.endswith('.pem') for p in src.parts):
                raise ValueError('credential/git path rejected; inspect mapping')
            selected[key]=src
        result[label]=selected
    if 'task.toml' not in result['harbor_task'] or 'instruction.md' not in result['harbor_task']:
        raise ValueError('Harbor ZIP must explicitly map task.toml/instruction.md at root')
    for label in FILES:result[label]=inside(root,plan['files'][label])
    return result


def make_zip(path,files):
    expected={}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for target,source in sorted(files.items()):
            sha=digest(source);info=zipfile.ZipInfo(target,date_time=(2026,1,1,0,0,0))
            executable=source.suffix=='.sh' or bool(source.stat().st_mode & stat.S_IXUSR)
            info.create_system=3;info.external_attr=(stat.S_IFREG | (0o755 if executable else 0o644))<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            with source.open('rb') as src,archive.open(info,'w',force_zip64=True) as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            expected[target]=sha
    import hashlib
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC check failed')
        if set(archive.namelist())!=set(expected):raise ValueError('ZIP member mismatch')
        for target,sha in expected.items():
            h=hashlib.sha256()
            with archive.open(target) as src:
                for chunk in iter(lambda:src.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=sha:raise ValueError('ZIP content mismatch')
    return expected


def bundle(plan,root,out):
    root=Path(root).resolve();out=Path(out).resolve()
    if out==root or root in out.parents:raise ValueError('output must be outside submitted source tree')
    if out.exists():raise ValueError('output already exists; use a new destination')
    qa=evidence(root,plan['qa_report']);mapping=resolve_mapping(plan,root)
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='delivery-',dir=out.parent))
    try:
        members={}
        for key,name in ARCHIVES.items():members[name]=make_zip(temporary/name,mapping[key])
        for key,name in FILES.items():shutil.copyfile(mapping[key],temporary/name)
        files={name:{'sha256':digest(temporary/name),'bytes':(temporary/name).stat().st_size} for name in list(ARCHIVES.values())+list(FILES.values())}
        report={'status':'PACKAGED_NOT_PLATFORM_ACCEPTED','created_at':utc(),'files':files,'archive_members':members,
                'qa_report_sha256':digest(qa),'boundary':'Packaging verifies paths/hashes only. QA status, secrets and semantic completeness require review.'}
        # Manifest is an internal packaging proof alongside, not a seventh required deliverable.
        atomic_json(temporary/'delivery_manifest.json',report)
        os.replace(temporary,out)
    except Exception:
        # Preserve staged outputs for diagnosis; no recursive cleanup or deletion of user materials.
        raise
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mapping',type=Path);p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    try:report=bundle(load(a.mapping),a.root,a.out)
    except (ValueError,KeyError,TypeError,OSError) as exc:p.exit(2,str(exc)+'\n')
    print(report['status'])
