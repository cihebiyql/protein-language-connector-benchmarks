import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_b1_pilot import metrics


class RankingTests(unittest.TestCase):
    def test_known_positive_recall_and_mrr(self):
        rows = [{'sample_id':'a', 'targets':['T1','T4']}, {'sample_id':'b','targets':['T3']}]
        scores = np.array([[4,3,2,1],[4,3,2,1]])
        actual, predictions = metrics(scores, rows, ['T1','T2','T3','T4'])
        self.assertEqual(actual['known_positive_recall_at_3'], 0.75)
        self.assertAlmostEqual(actual['mrr'], (1+1/3)/2)
        self.assertEqual(predictions[0]['ranked_target_ids'], ['T1','T2','T3','T4'])


if __name__=='__main__':
    unittest.main()
