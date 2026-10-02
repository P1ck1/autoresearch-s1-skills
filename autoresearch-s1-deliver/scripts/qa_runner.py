"""Run the unmodified supplied QA collector; a collection is not semantic final review."""
import argparse
import subprocess
import sys
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
    source=a.source.resolve();out=a.out.resolve()
    if source.is_dir() and (out==source or source in out.parents):p.exit(2,'QA output must be outside the reviewed tree\n')
    qa=Path(__file__).resolve().parents[1]/'references'/'supplied-qa'/'autoresearch-task-qa'
    argv=[sys.executable,str(qa/'scripts'/'audit_task.py'),str(source),'--out-dir',str(out),'--policy','implementation']
    if a.review:argv+=['--review',str(a.review.resolve())]
    elif (out/'review.json').exists():p.exit(2,'Existing review found: refusing to overwrite final report with collection-only run\n')
    raise SystemExit(subprocess.call(argv,cwd=qa))
