import unittest

import torch

from ai.train import EarlyStopping


class EarlyStoppingTestCase(unittest.TestCase):
    def test_early_stopping_triggers_on_no_improve(self):
        model = torch.nn.Linear(2, 1)
        stopper = EarlyStopping(patience=2, min_delta=1e-4)
        losses = [1.0, 0.9, 0.95, 0.95, 0.95]

        stopped_epoch = None
        for epoch, loss in enumerate(losses, start=1):
            if stopper.step(loss, epoch, model):
                stopped_epoch = epoch
                break

        self.assertEqual(stopped_epoch, 4)
        self.assertEqual(stopper.best_epoch, 2)
        self.assertAlmostEqual(stopper.best_value, 0.9)

    def test_early_stopping_min_delta_respected(self):
        model = torch.nn.Linear(2, 1)
        stopper = EarlyStopping(patience=2, min_delta=0.01)

        stopper.step(1.0, 1, model)
        stopper.step(0.995, 2, model)

        self.assertEqual(stopper.best_epoch, 1)
        self.assertAlmostEqual(stopper.best_value, 1.0)
        self.assertEqual(stopper.epochs_since_improve, 1)

    def test_early_stopping_restore_best(self):
        model = torch.nn.Linear(2, 1)
        stopper = EarlyStopping(patience=2, min_delta=1e-4)

        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.5)
        stopper.step(1.0, 1, model)
        best_weight = model.weight.detach().clone()
        best_bias = model.bias.detach().clone()

        with torch.no_grad():
            model.weight.fill_(3.0)
            model.bias.fill_(2.0)
        stopper.step(1.2, 2, model)
        stopper.restore_best(model)

        self.assertTrue(torch.equal(model.weight, best_weight))
        self.assertTrue(torch.equal(model.bias, best_bias))

    def test_early_stopping_disabled_when_patience_zero(self):
        model = torch.nn.Linear(2, 1)
        stopper = EarlyStopping(patience=0, min_delta=1e-4)
        losses = [1.0, 1.1, 1.2, 1.3]

        flags = [stopper.step(loss, epoch, model) for epoch, loss in enumerate(losses, start=1)]

        self.assertFalse(any(flags))
        self.assertFalse(stopper.snapshot().enabled)


if __name__ == "__main__":
    unittest.main()
