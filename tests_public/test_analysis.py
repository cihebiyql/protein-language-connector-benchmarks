"""Portable stdlib checks; these tests never import torch or model code."""
import copy
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import analyze_results as analysis

ROOT=Path(__file__).resolve().parents[1]


def example_row(identifier="one",gold=2.):
    return {"sample_id":identifier,"pkd":gold,"split":"test",
            "groups":{"target_sequence_sha256":"target"}}


def example_prediction(identifier="one",text='{"pKD":1}',gold=2.):
    return {"sample_id":identifier,"split":"test","gold_pkd":gold,
            "raw_text":text,"parse":analysis.parse_answer(text)}


class ParserAndMetricTests(unittest.TestCase):
    def test_valid_numbers_are_not_clipped(self):
        for text,value in [(' {"pKD":5.2}\n',5.2),('{"pKD":-1000}',-1000.),('{"pKD":1e100}',1e100),('{"pKD":0}',0.)]:
            self.assertEqual(analysis.parse_answer(text),{"valid":True,"value":value,"reason":None})

    def test_adversarial_answers_rejected(self):
        invalid=['{"pKD":true}','{"pKD":false}','{"pKD":null}','{"pKD":"5"}',
                 '{"pKD":NaN}','{"pKD":Infinity}','{"pKD":-Infinity}','{"pKD":1e999}',
                 '{"pKD":1,"pKD":2}','{"pKD":5,"explanation":"ok"}','[]','{}','"5"',
                 'prefix {"pKD":5}','```json\n{"pKD":5}\n```','{"pKD":{"x":1}}',None,
                 '{"pKD":'+'9'*5000+'}']
        for value in invalid:
            with self.subTest(value=str(value)[:60]):
                parsed=analysis.parse_answer(value)
                self.assertIs(parsed['valid'],False);self.assertIsNone(parsed['value'])

    def test_exact_original_reason_categories(self):
        self.assertEqual(analysis.parse_answer('{"pKD":NaN}')['reason'],'invalid_json_or_nonfinite')
        self.assertEqual(analysis.parse_answer('[]')['reason'],'wrong_schema')
        self.assertEqual(analysis.parse_answer('{"pKD":false}')['reason'],'nonfinite_or_nonnumeric_pKD')
        with self.assertRaises(ValueError):analysis.strict_loads('{"a":1,"a":2}')

    def test_tied_ranks_and_undefined_spearman(self):
        self.assertEqual(analysis.average_ranks([10,10,20,30,30]),[1.5,1.5,3.,4.5,4.5])
        self.assertAlmostEqual(analysis.spearman([1,2,3],[1,1,2]),math.sqrt(3)/2)
        self.assertEqual(analysis.spearman([1,1,2,3],[4,4,5,6]),1.)
        self.assertEqual(analysis.spearman([1,1,2,3],[3,3,2,1]),-1.)
        self.assertIsNone(analysis.spearman([1,2],[4,4]));self.assertIsNone(analysis.spearman([1],[2]))
        with self.assertRaises(ValueError):analysis.spearman([1,float('inf')],[1,2])

    def test_complete_coverage_and_failed_denominator(self):
        rows=[example_row(),example_row('two',4.)]
        predictions=[example_prediction(),example_prediction('two','bad',4.)]
        metrics=analysis.generation_metrics(rows,predictions)
        self.assertEqual((metrics['n_total'],metrics['n_valid'],metrics['n_failed']),(2,1,1))
        self.assertIsNone(metrics['full_sample_mae']);self.assertEqual(metrics['valid_only_mae'],1.)
        with self.assertRaises(ValueError):analysis.generation_metrics(rows,predictions[:1])
        with self.assertRaises(ValueError):analysis.generation_metrics(rows,list(reversed(predictions)))
        changed=copy.deepcopy(predictions);changed[0]['gold_pkd']=2.1
        with self.assertRaises(ValueError):analysis.generation_metrics(rows,changed)

    def test_forged_saved_parse_is_rejected(self):
        prediction=example_prediction();prediction['parse']['value']=True
        with self.assertRaises(ValueError):analysis.generation_metrics([example_row()],[prediction])

    def test_three_seed_rule_never_averages_successful_subset(self):
        partial=analysis.seed_statistics({'17':1.,'42':2.,'2026':None})
        self.assertEqual(partial['coverage'],'2/3');self.assertEqual(partial['missing_invalid_seeds'],[2026])
        self.assertIsNone(partial['mean']);self.assertIsNone(partial['sample_SD'])
        full=analysis.seed_statistics({'17':1.,'42':2.,'2026':3.})
        self.assertEqual(full['mean'],2.);self.assertEqual(full['sample_SD'],1.)
        with self.assertRaises(ValueError):analysis.seed_statistics({'17':1.,'42':2.})
        with self.assertRaises(ValueError):analysis.seed_statistics({'17':True,'42':2.,'2026':3.})

    def test_result_complete_cannot_override_independent_acceptance(self):
        self.assertFalse(analysis.official_eligible('complete',False,True))
        self.assertFalse(analysis.official_eligible('complete','true',True))
        self.assertFalse(analysis.official_eligible('invalid_or_incomplete',True,True))
        self.assertFalse(analysis.official_eligible('complete',True,False))
        self.assertTrue(analysis.official_eligible('complete',True,True))

    def test_earliest_validation_tie_and_bool_numeric(self):
        audit=analysis.Audit(ROOT)
        history=[{'epoch':i,'optimizer_steps':516*i,'validation_answer_token_nll':v}for i,v in enumerate([.5,.5,.6],1)]
        selection={'best_epoch':1,'validation_answer_token_nll':.5,'all_training_epochs_completed':3,
                   'optimizer_steps':1548,'test_labels_used_for_selection':False}
        result={'best_epoch':1,'best_validation_answer_token_nll':.5,'actual_epochs':3,'optimizer_steps':1548}
        analysis.audit_training(audit,history,selection,result,'fixture');self.assertFalse(audit.errors)
        selection['best_epoch']=2;analysis.audit_training(audit,history,selection,result,'fixture')
        self.assertTrue(audit.errors)
        self.assertFalse(analysis.Audit(ROOT).same(True,1,'bool is not a numerical metric'))

    def test_output_refuses_frozen_and_existing_directories(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)/'repo';root.mkdir();(root/'results').mkdir()
            with self.assertRaises(ValueError):analysis.export_report({},root/'results'/'new',root)
            occupied=Path(directory)/'occupied';occupied.mkdir();(occupied/'keep').write_text('preserve')
            with self.assertRaises(ValueError):analysis.export_report({},occupied,root)
            self.assertEqual((occupied/'keep').read_text(),'preserve')


@unittest.skipUnless((ROOT/'results/matched_budget/final_summary.json').is_file(),
                     'Public snapshot not populated yet; validate must pass before release')
class PublicReleaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=analysis.audit_repository(ROOT)

    def test_all_real_attempts_and_official8910(self):
        self.assertTrue(self.report['validation_passed'],self.report['errors'][:10])
        self.assertEqual(self.report['counts'],{'expected':36,'raw_attempts':31,'official':30,'quarantined':1,
            'missing':5,'official_predictions':8910,'official_parse_valid':8910,'quarantined_predictions':152})
        self.assertEqual(len(self.report['attempts']),31)
        self.assertTrue(all(a['raw_consistent']for a in self.report['attempts']))

    def test_case31_is_numeric_consistent_but_not_official(self):
        quarantined=[a for a in self.report['attempts']if not a['included_in_official_tables']]
        self.assertEqual(len(quarantined),1);case=quarantined[0]
        self.assertEqual(case['case']['sender_mode'],'global16')
        self.assertIs(case['valid_for_acceptance'],False)
        self.assertTrue(case['raw_consistent']);self.assertEqual(case['raw_metrics']['n_total'],152)
        self.assertNotIn(case['case_id'],self.report['official_case_ids'])
        pools=[g for g in self.report['paired_contrasts']if g['name']=='balanced_pool_allocation']
        self.assertTrue(all(g['k']==0 and g['mean']is None and g['sample_SD']is None for g in pools))

    def test_all_seven_fixed_contrasts_and_seed_coverage(self):
        self.assertEqual(len(self.report['paired_contrasts']),14)
        self.assertEqual({g['name']for g in self.report['paired_contrasts']},set(analysis.CONTRASTS))
        gates=[g for g in self.report['paired_contrasts']if g['name']=='gate_zero_initialization']
        self.assertTrue(all(g['k']==3 for g in gates))

    def test_adversarial_promotion_of_case31_still_excluded(self):
        original=analysis.Audit.read
        def replace_acceptance(audit,relative,jsonl=False):
            value=original(audit,relative,jsonl)
            if str(relative).endswith('acceptance.json')and 'global16' in str(relative):
                value=copy.deepcopy(value);value['valid_for_acceptance']=True
            return value
        with patch.object(analysis.Audit,'read',replace_acceptance):report=analysis.audit_repository(ROOT)
        self.assertFalse(report['validation_passed'])
        self.assertEqual(len(report['official_case_ids']),30)
        self.assertTrue(any('quarantine' in e for e in report['errors']))

    def test_analyze_export_preserves_quarantine_and_all_missing_cells(self):
        with tempfile.TemporaryDirectory()as directory:
            output=Path(directory)/'analysis'
            analysis.export_report(self.report,output,ROOT)
            loaded=json.loads((output/'analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(loaded['counts']['official_predictions'],8910)
            with(output/'cases.csv').open(encoding='utf-8',newline='')as handle:rows=list(csv.DictReader(handle))
            quarantine=[r for r in rows if r['official']=='False']
            self.assertEqual(len(quarantine),1);self.assertEqual(quarantine[0]['official_full_sample_mae'],'')
            self.assertNotEqual(quarantine[0]['raw_full_sample_mae'],'')
            with(output/'paired_contrasts.csv').open(encoding='utf-8',newline='')as handle:contrasts=list(csv.DictReader(handle))
            self.assertEqual(len(contrasts),14)
            self.assertTrue(all(c['mean']=='' for c in contrasts if c['name']=='balanced_pool_allocation'))


if __name__=='__main__':
    unittest.main()
