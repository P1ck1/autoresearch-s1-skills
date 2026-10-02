"""Bounded host probes. Does not install packages, build or start containers."""
import argparse
import platform
import shutil
import subprocess
from pathlib import Path
from common import atomic_json, utc

COMMANDS = {'docker_client': ['docker','--version'],
            'docker_daemon': ['docker','info','--format','{{.ServerVersion}}'],
            'gpu': ['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
            'harbor': ['harbor','--version'], 'codex': ['codex','--version']}


def inspect(root):
    checks={}
    for key, argv in COMMANDS.items():
        if not shutil.which(argv[0]):
            checks[key]={'status':'MISSING'}; continue
        try:
            proc=subprocess.run(argv, capture_output=True, text=True, timeout=20)
            checks[key]={'status':'OBSERVED' if proc.returncode==0 else 'FAILED',
                         'returncode':proc.returncode,'stdout':proc.stdout[-8000:], 'stderr':proc.stderr[-2000:]}
        except (subprocess.TimeoutExpired, OSError) as exc:
            checks[key]={'status':'FAILED','reason':str(exc)}
    disk=shutil.disk_usage(root)
    return {'created_at':utc(),'platform':platform.platform(),'cpu_count':__import__('os').cpu_count(),
            'disk_free_bytes':disk.free,'checks':checks,'status':'HOST_OBSERVATIONS_ONLY',
            'still_required':['actual GPU container test','two clean image builds','target Harbor schema and transfer',
                              'per-model adapter probe','candidate permissions and isolation','external controller smoke'],
            'boundary':'Version probes do not certify readiness, isolation or model access.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path.cwd()); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); atomic_json(a.out,inspect(a.root)); print('HOST_OBSERVATIONS_ONLY')
