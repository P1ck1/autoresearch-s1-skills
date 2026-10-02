import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import atomic_json,duration
from supervisor import run,read_journal,MODELS
from export_evidence import export
from trajectory_check import audit,validate_rounds


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.config={'mode':'simulation','simulation_target_seconds':0.05,'task_id':'test','task_version':'fixed',
                     'max_wall_seconds':30,'slice_timeout_seconds':4,'max_attempts':10,
                     'tracks':{k:{'model':m,'effort':e,'candidate_root':str(self.root/k),'adapter_argv':[sys.executable,str(Path(__file__).parent/'mock_adapter.py')]} for k,(m,e) in MODELS.items()}}
        self.path=self.root/'config.json';self.state=self.root/'control'
    def execute(self):atomic_json(self.path,self.config);return run(self.path,self.state)
    def test_simulation_and_export(self):
        result=self.execute();self.assertEqual(result['status'],'SIMULATION_COMPLETE')
        self.assertFalse(result['tracks']['codex']['minimum_met'])
        docs=export(self.state);traj=json.loads((self.state/'trajectory_codex.json').read_text())
        self.assertEqual(audit(traj,docs['codex'],self.state)['status'],'SIMULATION_ONLY')
    def test_final_return_does_not_stop_early(self):
        self.config['simulation_target_seconds']=0.3;result=self.execute()
        self.assertGreaterEqual(len(result['tracks']['codex']['rounds']),2)
        self.assertGreaterEqual(len(result['tracks']['seed']['rounds']),2)
    def test_inflated_time_rejected(self):
        self.config['tracks']['codex']['adapter_argv']+=['--behavior','future']
        self.assertEqual(self.execute()['status'],'ADAPTER_INVALID')
    def test_forged_score_rejected(self):
        self.config['tracks']['codex']['adapter_argv']+=['--behavior','forged_score']
        self.assertEqual(self.execute()['status'],'ADAPTER_INVALID')
    def test_live_jobs_block_resume(self):
        self.config['tracks']['codex']['adapter_argv']+=['--behavior','live_job'];self.execute()
        with self.assertRaises(ValueError):run(self.path,self.state,True)
    def test_no_progress_blocks(self):
        self.config['tracks']['codex']['adapter_argv']+=['--behavior','no_progress']
        self.assertEqual(self.execute()['status'],'ADAPTER_INVALID')
    def test_budget_limit(self):
        self.config['max_attempts']=1;self.assertEqual(self.execute()['status'],'BUDGET_EXHAUSTED')
    def test_resume_identity(self):
        self.execute();self.config['task_version']='changed';atomic_json(self.path,self.config)
        with self.assertRaises(ValueError):run(self.path,self.state,True)
    def test_tampered_journal(self):
        self.execute();path=self.state/'events.jsonl';content=path.read_text();path.write_text(content.replace('created','changed',1))
        with self.assertRaises(ValueError):read_journal(path)
    def test_tampered_evidence(self):
        self.execute();next((self.state/'attempts').glob('*/public-score.json')).write_text('{}')
        with self.assertRaises(ValueError):export(self.state)
    def test_owner_stop(self):
        self.execute();(self.state/'STOP').write_text('owner')
        self.assertEqual(run(self.path,self.state,True)['status'],'OWNER_STOP')
    def test_shared_candidate_rejected(self):
        self.config['tracks']['seed']['candidate_root']=self.config['tracks']['codex']['candidate_root']
        with self.assertRaises(ValueError):self.execute()
    def test_overlap_and_exclusion(self):self.assertEqual(duration([(0,10),(5,15)],[(4,6),(5,9)]),10)
    def test_bad_round(self):
        row={'round':1,'policy_name':'x','method_summary':'x','status':'ok','score':None,'failure_reason':None,'retained_best':False,'time':'2026-10-02 10:00:00'}
        with self.assertRaises(ValueError):validate_rounds([row])

if __name__=='__main__':unittest.main()
