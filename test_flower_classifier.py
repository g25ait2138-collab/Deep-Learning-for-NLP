"""Offline unit tests: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from src.flower_classifier import (DATA, MYSTERY_FLOWER, FlowerModel, sigmoid,
    train_model, binary_cross_entropy_from_logits)


class TestFlowerClassifier(unittest.TestCase):
    def test_sigmoid_extreme_inputs_and_array(self):
        with np.errstate(over='raise', invalid='raise'):
            z = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(z, [0.0, 0.5, 1.0])
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_repeatable_training(self):
        a, la = train_model(seed=42, epochs=1000)
        b, lb = train_model(seed=42, epochs=1000)
        np.testing.assert_allclose(a.weights, b.weights)
        self.assertEqual(a.bias, b.bias)
        np.testing.assert_array_equal(la, lb)

    def test_loss_drops_and_training_predictions(self):
        model, losses = train_model()
        self.assertEqual(len(losses), 6000)
        self.assertGreater(losses[0], losses[-1])
        pred = model.predict(DATA[:, :2])
        self.assertEqual(pred.shape, (8,))
        self.assertTrue(np.all(np.isin(pred, [0, 1])))
        self.assertGreaterEqual(np.mean(pred == DATA[:, 2]), 0.75)

    def test_mystery_prediction_is_probability(self):
        model, _ = train_model()
        score = model.predict_proba(MYSTERY_FLOWER)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_bce_works_for_large_logits(self):
        score = binary_cross_entropy_from_logits([1000, -1000], [1, 0])
        self.assertAlmostEqual(score, 0.0, places=10)

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            train_model(np.array([[1, 2, 2.0]]))
        with self.assertRaises(ValueError):
            train_model(epochs=0)


if __name__ == '__main__':
    unittest.main()
