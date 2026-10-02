import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_models import probe,EXPECTED
from adapter_runner import invoke


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        adapter=Path(__file__).parent/'mock_adapter.py'
        self.config={'mode':'simulation','timeout_seconds':5,'tracks':{k:{'model':m,'effort':e,'adapter_argv':[sys.executable,str(adapter)]} for k,(m,e) in EXPECTED.items()}}
    def test_both_simulated_never_real_pass(self):
        r=probe(self.config,self.root/'out');self.assertEqual(r['status'],'SIMULATION_ONLY');self.assertTrue(all(t['status']=='PROBE_EVIDENCE_VALID' for t in r['tracks'].values()))
    def test_wrong_model_receipt_fails(self):
        self.config['tracks']['codex']['adapter_argv']+=['--behavior','wrong_model']
        r=probe(self.config,self.root/'out');self.assertEqual(r['tracks']['codex']['status'],'FAIL')
    def test_substitution_rejected(self):
        self.config['tracks']['seed']['effort']='Max'
        with self.assertRaises(ValueError):probe(self.config,self.root/'out')
    def test_timeout_is_interruption(self):
        result=invoke(self.config['tracks']['codex']['adapter_argv']+['--behavior','hang'],{'mode':'simulation'},self.root/'attempt',0.25)
        self.assertEqual(result['interrupted'],'TIMEOUT');self.assertTrue(result['requires_resource_reconciliation'])
    def test_shell_string_not_accepted(self):
        with self.assertRaises(ValueError):invoke('echo hello',{},self.root/'attempt',1)

if __name__=='__main__':unittest.main()
