"""Validate eight-field trajectories and evidence-backed time intervals; no task execution."""
import argparse
import math
from datetime import datetime
from pathlib import Path
from common import atomic_json, duration, evidence, load, number, timestamp

FIELDS={'round','policy_name','method_summary','status','score','failure_reason','retained_best','time'}


def validate_rounds(rounds,start=1):
    if not isinstance(rounds,list) or not rounds: raise ValueError('nonempty rounds required')
    previous=None
    for expected,row in enumerate(rounds,start):
        if set(row)!=FIELDS: raise ValueError('each round must have exactly eight fields')
        if type(row['round']) is not int or row['round']!=expected: raise ValueError('round order mismatch')
        for key in ('policy_name','method_summary','status'):
            if not isinstance(row[key],str) or not row[key].strip(): raise ValueError(key+' is empty')
        if type(row['retained_best']) is not bool: raise ValueError('retained_best must be bool')
        if not isinstance(row['time'],str): raise ValueError('round time must be string')
        dt=datetime.strptime(row['time'],'%Y-%m-%d %H:%M:%S')
        if previous is not None and dt<previous: raise ValueError('round times reversed')
        previous=dt
        if row['score'] is not None: number(row['score'],'score')
        if row['status']=='ok':
            if row['score'] is None or row['failure_reason'] is not None:
                raise ValueError('successful round requires score and null failure_reason')
        elif not isinstance(row['failure_reason'],str) or not row['failure_reason'].strip() or row['retained_best']:
            raise ValueError('failed round requires reason and cannot be best')


def validate_time(doc,root):
    if doc.get('track') not in ('codex','seed'): raise ValueError('unknown track')
    observed_start=timestamp(doc['observed_start']); observed_end=timestamp(doc['observed_end'])
    if observed_end<=observed_start: raise ValueError('invalid observation window')
    good,bad=[],[]
    for key,target in (('active',good),('excluded',bad)):
        for interval in doc.get(key,[]):
            a,b=timestamp(interval['start']),timestamp(interval['end'])
            if not observed_start<=a<b<=observed_end: raise ValueError('interval outside observation window')
            if not isinstance(interval.get('reason'),str) or not interval['reason'].strip():
                raise ValueError('interval reason required')
            evidence(root,interval['evidence']);target.append((a,b))
    secs=duration(good,bad)
    return {'effective_seconds':secs,'minimum_met':secs>=36000,'target_met':secs>=39600,
            'observation_seconds':observed_end-observed_start}


def audit(trajectory,times,root):
    validate_rounds(trajectory['rounds'])
    value=validate_time(times,root)
    if times.get('mode') not in ('production','simulation'):raise ValueError('explicit time evidence mode required')
    status='SIMULATION_ONLY' if times['mode']=='simulation' else ('STATIC_RECORDS_PASS' if value['minimum_met'] else 'FAIL_DURATION')
    return {'status':status,
            'round_count':len(trajectory['rounds']),**value,
            'boundary':'Checks structure, interval arithmetic and hashes, not whether intervals describe genuine research.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory',type=Path);p.add_argument('--times',required=True,type=Path)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    try: report=audit(load(a.trajectory),load(a.times),a.root);atomic_json(a.out,report)
    except (ValueError,KeyError,TypeError,OSError) as exc:p.exit(2,str(exc)+'\n')
    print(report['status']);raise SystemExit(0 if report['status']=='STATIC_RECORDS_PASS' else 1)
