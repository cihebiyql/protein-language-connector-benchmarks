import unittest
from src.mt_data import condition_tensors_values, load_task, load_catalog, load_manifest, sequence_sha
from pathlib import Path


class DataTests(unittest.TestCase):
    def test_missing_zero_and_field_separation(self):
        row = {'condition_values': ['same', 'same'] + [None] * 8}
        ids, observed = condition_tensors_values(row)
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(ids[2:], [0] * 8)
        self.assertEqual(observed, [True, True] + [False] * 8)
        self.assertTrue(all(0 <= x < 4096 for x in ids))

    def test_labels_identity_cannot_change_conditions(self):
        row = {'condition_values': ['rt'] + [None] * 9, 'target': 1, 'sample_id': 'A', 'positive_ids': ['class1']}
        first = condition_tensors_values(row)
        row.update(target=999999, sample_id='B', positive_ids=['class2'], group_id='new', split='test')
        self.assertEqual(first, condition_tensors_values(row))

    def test_invalid_values_rejected(self):
        for bad in ([], ['x'] * 9, ['x'] * 11, [42] + [None] * 9, [''] + [None] * 9):
            with self.assertRaises(ValueError):
                condition_tensors_values({'condition_values': bad})

    def test_real_prepared_views(self):
        root = Path(__file__).resolve().parents[1] / 'data'
        if not root.exists():
            self.skipTest('actual views not staged')
        manifest = load_manifest(root)
        self.assertEqual(manifest['expected_sequences'], 17279)
        self.assertEqual(len(load_catalog(root)), 31)
        self.assertEqual(len(load_task(root, 'meltome', 'test')), 1945)
        selected, calibrated = load_task(root, 'rt', 'selection'), load_task(root, 'rt', 'calibration')
        self.assertEqual((len(selected), len(calibrated)), (525, 530))
        self.assertFalse({x['sample_id'] for x in selected} & {x['sample_id'] for x in calibrated})
        human = load_task(root, 'meltome', 'train') + load_task(root, 'meltome', 'validation') + load_task(root, 'meltome', 'test')
        self.assertEqual(len(human), 10093)
        self.assertTrue(any('U' in row['sequence'] for row in human))
        self.assertEqual(human[0]['group_id'], sequence_sha(human[0]['sequence']))


if __name__ == '__main__':
    unittest.main()
