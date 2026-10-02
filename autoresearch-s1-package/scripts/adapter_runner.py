"""Run an owner-controlled adapter without a shell; never executes candidate paths."""
import os
import signal
import subprocess
import time
from pathlib import Path
from common import atomic_json, load, utc


def invoke(argv, request, directory, timeout, stop_file=None, max_log_bytes=32*1024*1024):
    if not isinstance(argv,list) or not argv or any(not isinstance(x,str) or not x for x in argv):
        raise ValueError('adapter_argv must be a nonempty array of strings')
    if not Path(argv[0]).is_absolute():
        raise ValueError('adapter executable must be absolute')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    receipt=directory/'receipt.json'; req=directory/'request.json'
    atomic_json(req,request)
    started=utc(); begin=time.monotonic(); interrupted=None
    with (directory/'stdout.log').open('wb') as out, (directory/'stderr.log').open('wb') as err:
        proc=subprocess.Popen(argv+['--request',str(req.resolve()),'--receipt',str(receipt.resolve())],
                              stdin=subprocess.DEVNULL,stdout=out,stderr=err,shell=False,
                              start_new_session=(os.name=='posix'))
        while proc.poll() is None:
            if stop_file and Path(stop_file).exists(): interrupted='OWNER_STOP'
            elif time.monotonic()-begin>=timeout: interrupted='TIMEOUT'
            elif out.tell()+err.tell()>max_log_bytes: interrupted='LOG_LIMIT'
            if interrupted:
                if os.name=='posix': os.killpg(proc.pid,signal.SIGTERM)
                else: proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name=='posix': os.killpg(proc.pid,signal.SIGKILL)
                    else: proc.kill()
                    proc.wait(timeout=5)
                break
            time.sleep(0.1)
    ended=utc()
    result={'started_at':started,'ended_at':ended,'wall_seconds':time.monotonic()-begin,
            'exit_code':proc.returncode,'interrupted':interrupted,'receipt':None}
    # An interrupted adapter may leave its Harbor/container job alive. Do not auto-restart it.
    if interrupted: result['requires_resource_reconciliation']=True
    if receipt.exists() and not receipt.is_symlink(): result['receipt']=load(receipt)
    atomic_json(directory/'execution.json',result)
    return result
