import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bench_v02.contracts import ContractError
from bench_v02.interventions import matched_donors

def row(sample,target='ACDEFG',context='assay1',group=None):
    return {'sample_id':sample,'panel_id':'A1','task':'affinity','split':'test',
            'independent_group_id':group or sample,'peptide_sequence':'ACDEFGH',
            'target_sequence':target,'chemical_state':{'n_terminal':'Free','c_terminal':'Free','explicitly_representable':True},
            'experimental_context':{'comparable_condition_group':context}}

class MatchingTests(unittest.TestCase):
    def test_target_condition_state_and_family_are_all_matched(self):
        records=[row('a'),row('b'),row('other_target',target='MNPQRS'),
                 row('other_context',context='assay2'),row('same_group',group='a')]
        records.append(row('other_chemistry'))
        records[-1]['chemical_state']['n_terminal']='Acetyl'
        result={r['sample_id']:r for r in matched_donors(records)}
        self.assertEqual(result['a']['donor_ids'],['b'])
        self.assertFalse(result['a']['coverage_complete'])

    def test_unknown_conditions_do_not_match_merely_because_both_missing(self):
        result=matched_donors([row('a',context=None),row('b',context=None)])
        self.assertTrue(all(not r['donor_ids'] for r in result))
        self.assertIn('experimental_condition_comparability_unverified',result[0]['unmatchable_reasons'])

    def test_unknown_chemical_states_do_not_match(self):
        a,b=row('a'),row('b')
        a.pop('chemical_state');b.pop('chemical_state')
        result=matched_donors([a,b])
        self.assertEqual(result[0]['donor_ids'],[])
        self.assertIn('chemical_state_unverified',result[0]['unmatchable_reasons'])

    def test_order_stability_and_no_replacement_fill(self):
        records=[row(str(i)) for i in range(6)]
        first={r['sample_id']:r['donor_ids'] for r in matched_donors(records,n=3)}
        second={r['sample_id']:r['donor_ids'] for r in matched_donors(records[::-1],n=3)}
        self.assertEqual(first,second)
        self.assertTrue(all(len(set(v))==3 for v in first.values()))

    def test_unfrozen_groups_and_cross_split_are_not_donors(self):
        a,b,c=row('a'),row('b'),row('c')
        b['independent_group_id']=None
        c['split']='train'
        self.assertEqual(matched_donors([a,b,c])[0]['donor_ids'],[])
        with self.assertRaises(ContractError):
            matched_donors([a,a])

if __name__=='__main__':
    unittest.main()
