import importlib.util
import math
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/bench_v02/prepare_affinity.py'
SPEC = importlib.util.spec_from_file_location('prepare_v02_affinity', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AffinityPreparationTests(unittest.TestCase):
    def test_kd_units(self):
        result = MODULE.parse_affinity('Kd=10 nM')
        self.assertAlmostEqual(result['kd_molar'], 1e-8)
        self.assertAlmostEqual(result['pkd'], 8)
        self.assertTrue(result['label_exact'])

    def test_bound_is_not_exact(self):
        result = MODULE.parse_affinity('KD > 10 uM')
        self.assertEqual(result['relation'], '>')
        self.assertFalse(result['label_exact'])

    def test_scientific_notation_and_error(self):
        self.assertAlmostEqual(MODULE.parse_affinity('KD=2.2×10^-6 M')['kd_molar'], 2.2e-6)
        self.assertAlmostEqual(MODULE.parse_affinity('KD=10 ± 2 nM')['reported_error_molar'], 2e-9)

    def test_other_endpoints_and_ambiguity(self):
        for value in ['Ki(app)=2.2×10^-6 M', 'IC50=1 nM', 'Ka=10 uM', 'KD=10-20 nM', 'KD~10 nM', 'KD=0 nM']:
            self.assertIsNone(MODULE.parse_affinity(value)['pkd'], value)

    def base(self):
        return {'ID': 'x', 'Document_Type': 'Research Articles', 'Affinity_Parameters': 'Kd',
                'Affinity': 'KD=10 nM', 'uM': .01, 'Peptide_Sequence': 'ACDEFG',
                'Protein_Sequence': 'ACDEFGHIK', 'Linear/Cyclic': 'Linear', 'Cyclization_Method': 'None',
                'Cyclization_Way': 'None', 'Residue_Modification': 'No', 'Chemical_Modification': None,
                'Nterminal': 'Free', 'Cterminal': 'Amide', 'DOI': '10.test/source'}

    def normal(self, row, research=None):
        locks = {'locked_source_dois': [], 'locked_peptide_sequences': [], 'locked_parent_gene_series': []}
        return MODULE.normalize(row, row if research is None else research, locks)

    def test_broad_reference_target_keeps_missing_conditions(self):
        result = self.normal(self.base())
        self.assertTrue(result['eligibility']['A1_broad_reference_target'])
        self.assertFalse(result['eligibility']['A1_strict_comparable_pairs'])
        self.assertIsNone(result['experimental_context']['comparable_condition_group'])
        self.assertEqual(result['split_status'], 'pending_global_group_split')

    def test_raw_standardized_conflict_is_quarantined(self):
        row = self.base(); row['uM'] = 2
        self.assertFalse(self.normal(row)['eligibility']['A1_broad_reference_target'])

    def test_missing_target_is_never_replaced_by_id(self):
        row = self.base(); row['Protein_Sequence'] = None
        result = self.normal(row)
        self.assertIsNone(result['target_sequence'])
        self.assertFalse(result['eligibility']['A1_broad_reference_target'])

    def test_d_aa_and_cyclic_annotations_gate(self):
        row = self.base(); row['Peptide_Sequence'] = 'ACdEFG'
        self.assertFalse(self.normal(row)['eligibility']['A1_broad_reference_target'])
        row = self.base(); row['Cyclization_Method'] = 'disulfide bond'
        self.assertFalse(self.normal(row)['eligibility']['A1_broad_reference_target'])

    def test_research_rewrite_never_overwrites_main(self):
        row = self.base(); research = dict(row, Affinity='IC50=100 nM')
        result = self.normal(row, research)
        self.assertEqual(result['measurement_type'], 'KD')
        research['Peptide_Sequence'] = 'AAAAAAAA'
        self.assertFalse(self.normal(row, research)['eligibility']['A1_broad_reference_target'])


if __name__ == '__main__':
    unittest.main()
