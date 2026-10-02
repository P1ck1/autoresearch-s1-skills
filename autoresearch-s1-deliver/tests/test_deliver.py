import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import digest
from bundle_delivery import bundle


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name);self.root=self.base/'source';self.root.mkdir()
        for name in ('task.toml','instruction.md','pilot.md','codex.json','seed.json','check.md','qa.json'):(self.root/name).write_text('{}')
        self.plan={'archives':{'evidence':[{'source':'qa.json','target':'qa.json'}],
                             'harbor_task':[{'source':n,'target':n} for n in ('task.toml','instruction.md')]},
                   'files':dict(pilot='pilot.md',codex='codex.json',seed='seed.json',checklist='check.md'),
                   'qa_report':{'path':'qa.json','sha256':digest(self.root/'qa.json')}}
    def test_six_outputs_and_verified_archives(self):
        report=bundle(self.plan,self.root,self.base/'out');self.assertEqual(len(report['files']),6)
        self.assertEqual(report['status'],'PACKAGED_NOT_PLATFORM_ACCEPTED')
        with zipfile.ZipFile(self.base/'out'/'harbor_task.zip') as z:self.assertEqual(set(z.namelist()),{'task.toml','instruction.md'})
    def test_source_escape(self):
        self.plan['files']['pilot']='../outside'
        with self.assertRaises(ValueError):bundle(self.plan,self.root,self.base/'out')
    def test_target_escape(self):
        self.plan['archives']['evidence'][0]['target']='../escape'
        with self.assertRaises(ValueError):bundle(self.plan,self.root,self.base/'out')
    def test_credential_path(self):
        (self.root/'.env').write_text('secret');self.plan['archives']['evidence'][0]['source']='.env'
        with self.assertRaises(ValueError):bundle(self.plan,self.root,self.base/'out')
    def test_source_not_overwritten(self):
        with self.assertRaises(ValueError):bundle(self.plan,self.root,self.root/'out')
    def test_existing_destination(self):
        (self.base/'out').mkdir()
        with self.assertRaises(ValueError):bundle(self.plan,self.root,self.base/'out')
    def test_duplicate_case_path(self):
        self.plan['archives']['evidence'].append({'source':'qa.json','target':'QA.json'})
        with self.assertRaises(ValueError):bundle(self.plan,self.root,self.base/'out')

if __name__=='__main__':unittest.main()
