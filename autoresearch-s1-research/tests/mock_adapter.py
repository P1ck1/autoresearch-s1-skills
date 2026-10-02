"""TEST FIXTURE ONLY. Refuses production; never calls a model or creates research evidence."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--request',type=Path);p.add_argument('--receipt',type=Path)
p.add_argument('--behavior',default='valid');a=p.parse_args();request=json.loads(a.request.read_text(encoding='utf-8'))
if request.get('mode')!='simulation':raise SystemExit('Mock adapter refuses production')
if a.behavior=='hang':time.sleep(20)
start=datetime.now(timezone.utc);time.sleep(0.16);end=datetime.now(timezone.utc)
root=a.receipt.parent
def write(name,value):
    file=root/name;file.write_text(json.dumps(value),encoding='utf-8')
    return {'path':name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
log=write('simulation-log.json',{'simulation':True,'start':start.isoformat(),'end':end.isoformat()})
receipt={key:request[key] for key in ('operation','mode','track','model','effort')}
receipt['no_live_job']=a.behavior!='live_job'
if request['operation']=='probe':
    receipt.update(text_response=True,tool_call=True,public_score=True,record_written=True,evidence=[log])
    if a.behavior=='wrong_model':receipt['model']='wrong'
else:
    receipt.update({key:request[key] for key in ('task_id','task_version','attempt_id')})
    score=write('public-score.json',{'metric':0.5});score['metric_path']=['metric']
    method=write('candidate.json',{'simulation':'not a real method'})
    row={'round':request['next_round'],'policy_name':'SIMULATION','method_summary':'Synthetic fixture only.',
         'status':'ok','score':0.5,'failure_reason':None,'retained_best':True,'time':end.strftime('%Y-%m-%d %H:%M:%S')}
    if a.behavior=='forged_score':row['score']=0.9
    interval={'start':start.isoformat(),'end':end.isoformat(),'reason':'synthetic test interval','evidence':log}
    if a.behavior=='future':interval['end']=(end+timedelta(hours=11)).isoformat()
    receipt.update(rounds=[row],round_evidence={str(row['round']):score},candidate_evidence=[method],active=[interval],excluded=[])
    if a.behavior=='no_progress':receipt.update(rounds=[],active=[])
a.receipt.write_text(json.dumps(receipt),encoding='utf-8')
