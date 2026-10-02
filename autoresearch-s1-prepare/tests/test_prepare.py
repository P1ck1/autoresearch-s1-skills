import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import digest, load
from paired_stats import analyze


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.doc={'direction':'maximize','evaluation_mode':'stochastic','formal_seeds':[1,2,3],
                  'upper_bound':20,'upper_bound_basis':'fixed before experiments','same_protocol':True,'paired_runs':[]}
        for i,b in enumerate([9,10,11],1):
            row={'seed':i,'baseline':b,'reference':b+3,'quality_valid':True}
            for role in ('baseline','reference'):
                file=self.root/f'{role}-{i}.json';file.write_text(json.dumps({'metric':row[role]}))
                row[role+'_evidence']={'path':file.name,'sha256':digest(file),'metric_path':['metric']}
            self.doc['paired_runs'].append(row)

    def test_three_sigma_boundary(self):
        r=analyze(self.doc,self.root);self.assertEqual(r['status'],'NUMERIC_CHECKS_PASS');self.assertEqual(r['baseline_sample_std'],1);self.assertAlmostEqual(r['normalized_reference'],0.3)
    def test_missing_seed(self):
        self.doc['paired_runs'].pop()
        with self.assertRaises(ValueError):analyze(self.doc,self.root)
    def test_duplicate_seed(self):
        self.doc['paired_runs'][1]['seed']=1
        with self.assertRaises(ValueError):analyze(self.doc,self.root)
    def test_bool_not_metric(self):
        self.doc['paired_runs'][0]['baseline']=True
        with self.assertRaises(ValueError):analyze(self.doc,self.root)
    def test_nonfinite(self):
        self.doc['upper_bound']=float('inf')
        with self.assertRaises(ValueError):analyze(self.doc,self.root)
    def test_bad_upper(self):
        self.doc['upper_bound']=10
        with self.assertRaises(ValueError):analyze(self.doc,self.root)
    def test_uncropped_score(self):
        self.doc['upper_bound']=12;r=analyze(self.doc,self.root)
        self.assertEqual(r['normalized_reference'],1.5);self.assertIn('NORMALIZED_OUT_OF_RANGE',r['failures'])
    def test_hash_mismatch(self):
        (self.root/'baseline-1.json').write_text('{}')
        with self.assertRaises(ValueError):analyze(self.doc,self.root)
    def test_source_metric_mismatch(self):
        self.doc['paired_runs'][0]['reference']=20
        with self.assertRaises(ValueError):analyze(self.doc,self.root)
    def test_path_traversal(self):
        self.doc['paired_runs'][0]['baseline_evidence']['path']='../other.json'
        with self.assertRaises(ValueError):analyze(self.doc,self.root)
    def test_determinism_needs_basis(self):
        self.doc['evaluation_mode']='deterministic'
        self.assertIn('DETERMINISM_BASIS_MISSING',analyze(self.doc,self.root)['failures'])
    def test_invalid_quality(self):
        self.doc['paired_runs'][0]['quality_valid']=False
        self.assertEqual(analyze(self.doc,self.root)['status'],'FAIL')
    def test_minimize(self):
        self.doc['direction']='minimize';self.doc['upper_bound']=0
        for row in self.doc['paired_runs']:
            row['reference']=row['baseline']-3;f=self.root/row['reference_evidence']['path'];f.write_text(json.dumps({'metric':row['reference']}));row['reference_evidence']['sha256']=digest(f)
        self.assertEqual(analyze(self.doc,self.root)['status'],'NUMERIC_CHECKS_PASS')
    def test_below_sigma_even_constant_difference(self):
        for row in self.doc['paired_runs']:
            row['reference']=row['baseline']+2;f=self.root/row['reference_evidence']['path'];f.write_text(json.dumps({'metric':row['reference']}));row['reference_evidence']['sha256']=digest(f)
        self.assertIn('BELOW_3_SIGMA',analyze(self.doc,self.root)['failures'])
    def test_json_duplicate_keys(self):
        f=self.root/'dup.json';f.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError):load(f)

if __name__=='__main__':unittest.main()
