"""Check diagnostic matching without training or opening dataset images."""

import unittest
import numpy as np

from scripts.analyze_person import counts, iou


class PersonAnalysisTests(unittest.TestCase):
    def test_iou_empty_disjoint_identical_and_partial(self):
        self.assertEqual(iou([0, 0, 2, 2], []).tolist(), [])
        np.testing.assert_allclose(iou([0, 0, 2, 2], [[0, 0, 2, 2], [2, 2, 3, 3], [1, 0, 3, 2]]), [1, 0, 1/3])

    def test_one_ground_truth_cannot_match_duplicate_predictions(self):
        truth = [{'class_id': 2, 'box': [0, 0, 2, 2]}]
        predictions = [{'class_id': 2, 'confidence': c, 'box': [0, 0, 2, 2]} for c in (.9, .8)]
        self.assertEqual(counts(truth, predictions, .25), {'tp': 1, 'fp': 1, 'fn': 0})

    def test_confidence_boundary_and_wrong_class(self):
        truth = [{'class_id': 2, 'box': [0, 0, 2, 2]}]
        predictions = [{'class_id': 2, 'confidence': .25, 'box': [0, 0, 2, 2]},
                       {'class_id': 0, 'confidence': .9, 'box': [0, 0, 2, 2]}]
        self.assertEqual(counts(truth, predictions, .25), {'tp': 1, 'fp': 0, 'fn': 0})
        self.assertEqual(counts(truth, predictions, .251), {'tp': 0, 'fp': 0, 'fn': 1})

    def test_empty_truth_and_predictions(self):
        self.assertEqual(counts([], [], .25), {'tp': 0, 'fp': 0, 'fn': 0})
        self.assertEqual(counts([], [{'class_id': 2, 'confidence': .5, 'box': [0, 0, 2, 2]}], .25),
                         {'tp': 0, 'fp': 1, 'fn': 0})

    def test_confidence_order_and_second_unmatched_truth(self):
        truth = [{'class_id': 2, 'box': [0, 0, 3, 2]}, {'class_id': 2, 'box': [1, 0, 4, 2]}]
        predictions = [{'class_id': 2, 'confidence': .4, 'box': [0, 0, 4, 2]},
                       {'class_id': 2, 'confidence': .9, 'box': [0, 0, 3, 2]}]
        self.assertEqual(counts(truth, predictions, .25), {'tp': 2, 'fp': 0, 'fn': 0})


if __name__ == '__main__':
    unittest.main()
