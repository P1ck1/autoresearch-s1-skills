"""Revalidate accepted controller receipts and export per-track time evidence."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from common import atomic_json, digest, load
from supervisor import canonical, check_receipt, read_journal


def export(state_dir):
    root=Path(state_dir).resolve();events=read_journal(root/'events.jsonl')
    mode=events[0]['data']['mode']
    tracks={key:{'track':key,'mode':mode,'observed_start':None,'observed_end':None,'active':[],'excluded':[]} for key in ('codex','seed')}
    counts={'codex':0,'seed':0}
    for row in events:
        if row['kind']!='accepted':continue
        data=row['data'];attempt=data['attempt_id'];track=data['track'];directory=root/'attempts'/attempt
        if directory.resolve().parent!=root/'attempts':raise ValueError('unsafe attempt path')
        request=load(directory/'request.json');receipt=load(directory/'receipt.json');execution=load(directory/'execution.json')
        verified=check_receipt(receipt,request,execution,directory,counts[track]+1)
        # Journal serialization turns interval tuples into JSON arrays. Compare the
        # canonical JSON representation so an unchanged receipt survives that
        # harmless representation change while any value change still fails.
        same_payload = canonical({
            'active': verified['active'],
            'excluded': verified['excluded'],
            'rounds': verified['rounds'],
        }) == canonical({
            'active': data['active'],
            'excluded': data['excluded'],
            'rounds': data['rounds'],
        })
        if verified['receipt_sha256']!=data['receipt_sha256'] or not same_payload:
            raise ValueError('journal differs from revalidated receipt')
        counts[track]+=len(verified['rounds']);doc=tracks[track]
        doc['observed_start']=doc['observed_start'] or execution['started_at'];doc['observed_end']=execution['ended_at']
        for key in ('active','excluded'):
            for original in receipt.get(key,[]):
                interval=dict(original);interval['evidence']=dict(original['evidence'])
                interval['evidence']['path']=(Path('attempts')/attempt/original['evidence']['path']).as_posix()
                doc[key].append(interval)
    for track,doc in tracks.items():
        if doc['observed_start'] is None:raise ValueError('no accepted evidence for '+track)
        atomic_json(root/f'time_{track}.json',doc)
    return tracks


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('state',type=Path);a=p.parse_args()
    try:export(a.state)
    except (ValueError,KeyError,TypeError,OSError) as exc:p.exit(2,str(exc)+'\n')
    print('Time evidence exported; simulation results remain non-production.')
