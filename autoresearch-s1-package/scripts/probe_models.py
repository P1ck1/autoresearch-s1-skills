"""Bounded probes via a configured, trusted Codex/Harbor adapter; not a provider client."""
import argparse
from pathlib import Path
from adapter_runner import invoke
from common import atomic_json, evidence, load, number

EXPECTED={'codex':('GPT-5.6 Sol','Max'),'seed':('Seed 2.1 Turbo','High')}


def probe(config,out):
    if set(config['tracks'])!=set(EXPECTED): raise ValueError('both fixed tracks are required')
    timeout=number(config['timeout_seconds'],'timeout_seconds')
    if not 0<timeout<=600: raise ValueError('probe timeout must be 1..600 seconds')
    root=Path(out); root.mkdir(parents=True,exist_ok=False)
    reports={}
    for track,(model,effort) in EXPECTED.items():
        setting=config['tracks'][track]
        if (setting['model'],setting['effort'])!=(model,effort): raise ValueError('model substitution prohibited')
        request={'operation':'probe','track':track,'model':model,'effort':effort,
                 'mode':config.get('mode','production')}
        result=invoke(setting['adapter_argv'],request,root/track,timeout)
        receipt=result.get('receipt') or {}
        errors=[]
        if result['exit_code']!=0 or result['interrupted']: errors.append('ADAPTER_FAILED')
        for field,value in request.items():
            if receipt.get(field)!=value: errors.append('IDENTITY_MISMATCH:'+field)
        for field in ('text_response','tool_call','public_score','record_written','no_live_job'):
            if receipt.get(field) is not True: errors.append('UNVERIFIED:'+field)
        try:
            items=receipt.get('evidence',[])
            if not items: raise ValueError('missing evidence')
            for item in items: evidence(root/track,item)
        except (ValueError,KeyError,OSError) as exc: errors.append(str(exc))
        reports[track]={'status':'PROBE_EVIDENCE_VALID' if not errors else 'FAIL','errors':errors}
    simulated=config.get('mode')=='simulation'
    report={'status':'SIMULATION_ONLY' if simulated else ('PROBE_EVIDENCE_VALID' if all(x['status']=='PROBE_EVIDENCE_VALID' for x in reports.values()) else 'FAIL'),
            'tracks':reports,'boundary':'Trusted adapter must prove actual provider routing/effort and tool execution. Evidence is not independent attestation.'}
    atomic_json(root/'report.json',report); return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    try: report=probe(load(a.config),a.out)
    except (ValueError,KeyError,TypeError,OSError) as exc: p.exit(2,str(exc)+'\n')
    print(report['status']); raise SystemExit(0 if report['status']=='PROBE_EVIDENCE_VALID' else 1)
