"""External two-track controller. Production needs an audited, host-specific trusted adapter."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from adapter_runner import invoke
from common import atomic_json, duration, evidence, load, number, timestamp, utc
from trajectory_check import validate_rounds

MODELS={'codex':('GPT-5.6 Sol','Max'),'seed':('Seed 2.1 Turbo','High')}


def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def read_journal(path):
    events=[];previous='0'*64
    if not Path(path).exists():return events
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row=json.loads(line); claimed=row.pop('hash')
        if row.get('seq')!=len(events)+1 or row.get('previous')!=previous:
            raise ValueError('journal sequence/chain mismatch')
        actual=hashlib.sha256(canonical(row)).hexdigest()
        if actual!=claimed:raise ValueError('journal hash mismatch')
        row['hash']=claimed;events.append(row);previous=claimed
    return events


def append(path,kind,data):
    events=read_journal(path)
    row={'seq':len(events)+1,'previous':events[-1]['hash'] if events else '0'*64,
         'at':utc(),'kind':kind,'data':data}
    row['hash']=hashlib.sha256(canonical(row)).hexdigest()
    with Path(path).open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return row


def validate_config(config,config_root,state):
    mode=config.get('mode')
    if mode not in ('production','simulation'):raise ValueError('explicit mode required')
    if set(config['tracks'])!=set(MODELS):raise ValueError('two fixed tracks required')
    target=39600 if mode=='production' else number(config['simulation_target_seconds'],'simulation target')
    if mode=='simulation' and not 0<target<=60:raise ValueError('simulation target must be 0..60s')
    wall=number(config['max_wall_seconds'],'max wall');slice_limit=number(config['slice_timeout_seconds'],'slice timeout')
    if wall<=target or not 0<slice_limit<=wall:raise ValueError('invalid timeout budget')
    if type(config['max_attempts']) is not int or not 1<=config['max_attempts']<=10000:raise ValueError('invalid attempts limit')
    if mode=='production':
        if not sys.platform.startswith('linux'):raise ValueError('production requires target Linux server')
        isolation=load(evidence(config_root,config['isolation_report']))
        if isolation.get('status')!='VERIFIED' or isolation.get('task_id')!=config['task_id']:
            raise ValueError('matching verified isolation report required')
        evidence(config_root,config['model_probe_report'])
        probes=load(evidence(config_root,config['model_probe_report']))
        if probes.get('status')!='PROBE_EVIDENCE_VALID':raise ValueError('real probe evidence required')
    resolved=[]
    for track,(model,effort) in MODELS.items():
        setting=config['tracks'][track]
        if (setting['model'],setting['effort'])!=(model,effort):raise ValueError('model/effort mismatch')
        candidate=Path(setting['candidate_root']).resolve();resolved.append(candidate)
        if candidate==state or candidate in state.parents or state in candidate.parents:
            raise ValueError('state must be separate from candidate trees')
        for item in setting.get('adapter_files',[]): evidence(config_root,item)
        if mode=='production' and not setting.get('adapter_files'):raise ValueError('adapter source hashes required')
    if resolved[0]==resolved[1] or resolved[0] in resolved[1].parents or resolved[1] in resolved[0].parents:
        raise ValueError('candidate trees must be separate')
    return target


def check_receipt(receipt,request,execution,directory,next_round):
    for key in ('task_id','task_version','track','model','effort','mode','attempt_id'):
        if receipt.get(key)!=request[key]:raise ValueError('receipt identity mismatch: '+key)
    if receipt.get('no_live_job') is not True:raise ValueError('adapter has not reconciled running jobs')
    rows=receipt.get('rounds',[])
    if rows: validate_rounds(rows,next_round)
    if rows and not receipt.get('candidate_evidence'):
        raise ValueError('method snapshot evidence required for completed rounds')
    for item in receipt.get('candidate_evidence',[]): evidence(directory,item)
    for row in rows:
        item=receipt['round_evidence'][str(row['round'])]
        raw=load(evidence(directory,item));cursor=raw
        for key in item['metric_path']:cursor=cursor[key]
        if cursor is not None:number(cursor,'raw public score')
        if cursor!=row['score']:raise ValueError('score disagrees with raw public scorer output')
    good,bad=[],[]
    start,end=timestamp(execution['started_at']),timestamp(execution['ended_at'])
    if end<start:raise ValueError('wall clock moved backwards')
    for field,target in (('active',good),('excluded',bad)):
        for interval in receipt.get(field,[]):
            a,b=timestamp(interval['start']),timestamp(interval['end'])
            if not start<=a<b<=end:raise ValueError('receipt interval outside observed adapter execution')
            if not isinstance(interval.get('reason'),str) or not interval['reason'].strip():raise ValueError('time reason missing')
            evidence(directory,interval['evidence']);target.append((a,b))
    secs=duration(good,bad)
    if secs>execution['wall_seconds']+0.25:raise ValueError('effective time exceeds monotonic observed duration')
    return {'rounds':rows,'active':good,'excluded':bad,'effective_seconds':secs,
            'evidence_directory':str(directory),'receipt_sha256':hashlib.sha256(canonical(receipt)).hexdigest()}


def rebuild(events,mode,target):
    state={'mode':mode,'status':'RUNNING','tracks':{key:{'rounds':[],'active':[],'excluded':[],'effective_seconds':0.0} for key in MODELS}}
    for event in events:
        if event['kind']=='accepted':
            d=event['data'];track=state['tracks'][d['track']]
            track['rounds'].extend(d['rounds']);track['active'].extend(d['active']);track['excluded'].extend(d['excluded'])
    for track in state['tracks'].values():
        track['effective_seconds']=duration(track['active'],track['excluded'])
        track['minimum_met']=mode=='production' and track['effective_seconds']>=36000
        track['target_met']=track['effective_seconds']>=target
    return state


def run(config_path,state_dir,resume=False):
    config_path=Path(config_path).resolve();config=load(config_path);state_dir=Path(state_dir).resolve()
    target=validate_config(config,config_path.parent,state_dir)
    if state_dir.exists() and not resume:raise ValueError('existing state requires --resume')
    state_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=state_dir/'controller.lock'
    # A stale lock is not auto-deleted: an owner must verify there is no live controller/job.
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    journal=state_dir/'events.jsonl';config_hash=hashlib.sha256(canonical(config)).hexdigest()
    try:
        events=read_journal(journal)
        if events:
            if events[0]['kind']!='created' or events[0]['data']['config_hash']!=config_hash:
                raise ValueError('resume config mismatch')
            if events[-1]['kind']=='attempt_started':raise ValueError('unreconciled interrupted attempt; owner intervention required')
            if events[-1]['kind'] in ('OWNER_STOP','INTERRUPTED','ADAPTER_INVALID','BUDGET_EXHAUSTED','NO_PROGRESS'):
                raise ValueError('terminal interruption requires explicit owner reconciliation, not automatic restart')
        else:
            append(journal,'created',{'config_hash':config_hash,'mode':config['mode'],'task_id':config['task_id']})
        while True:
            events=read_journal(journal);state=rebuild(events,config['mode'],target)
            attempts=sum(e['kind']=='attempt_started' for e in events)
            elapsed=timestamp(utc())-timestamp(events[0]['at'])
            status=None
            if (state_dir/'STOP').exists():status='OWNER_STOP'
            elif all(x['target_met'] and x['rounds'] for x in state['tracks'].values()):
                status='SIMULATION_COMPLETE' if config['mode']=='simulation' else 'READY_FOR_FINAL_REVIEW'
            elif elapsed<0:status='INTERRUPTED'
            elif elapsed>=config['max_wall_seconds'] or attempts>=config['max_attempts']:status='BUDGET_EXHAUSTED'
            if status:
                append(journal,status,{});state['status']=status;atomic_json(state_dir/'state.json',state);return state
            # Sequential slices; each adapter must restore only this track's legitimate session.
            pending=[k for k,v in state['tracks'].items() if not (v['target_met'] and v['rounds'])]
            track=min(pending,key=lambda k:state['tracks'][k]['effective_seconds'])
            setting=config['tracks'][track];attempt_id=f'{attempts+1:06d}-{track}'
            request={'operation':'research_slice','attempt_id':attempt_id,'task_id':config['task_id'],
                     'task_version':config['task_version'],'track':track,'model':setting['model'],
                     'effort':setting['effort'],'mode':config['mode'],
                     'next_round':len(state['tracks'][track]['rounds'])+1,
                     'candidate_root':setting['candidate_root'],
                     'previous_accepted_attempts':[e['data']['attempt_id'] for e in events if e['kind']=='accepted' and e['data']['track']==track]}
            append(journal,'attempt_started',{'attempt_id':attempt_id,'track':track})
            directory=state_dir/'attempts'/attempt_id
            try:
                execution=invoke(setting['adapter_argv'],request,directory,
                                 min(config['slice_timeout_seconds'],config['max_wall_seconds']-elapsed),state_dir/'STOP')
                if execution['interrupted'] or execution['exit_code']!=0:
                    raise ValueError('adapter interrupted or failed; reconcile any surviving jobs')
                accepted=check_receipt(execution['receipt'] or {},request,execution,directory,request['next_round'])
                if accepted['effective_seconds']<=0:raise ValueError('no evidence-backed progress')
                accepted.update(track=track,attempt_id=attempt_id)
                append(journal,'accepted',accepted)
                atomic_json(state_dir/f'trajectory_{track}.json',{'rounds':state['tracks'][track]['rounds']+accepted['rounds']})
                atomic_json(state_dir/'state.json',rebuild(read_journal(journal),config['mode'],target))
            except (ValueError,KeyError,TypeError,OSError) as exc:
                status='OWNER_STOP' if (state_dir/'STOP').exists() else 'ADAPTER_INVALID'
                append(journal,status,{'attempt_id':attempt_id,'reason':str(exc)})
                state['status']=status;state['reason']=str(exc);atomic_json(state_dir/'state.json',state);return state
    finally:
        lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path)
    p.add_argument('--state',required=True,type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try: state=run(a.config,a.state,a.resume)
    except (ValueError,KeyError,TypeError,OSError) as exc:p.exit(2,str(exc)+'\n')
    print(state['status']);raise SystemExit(0 if state['status']=='READY_FOR_FINAL_REVIEW' else 1)
