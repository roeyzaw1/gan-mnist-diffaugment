import copy
import unittest

import torch

from diffaugment import D_train_aug, G_train_aug, diff_augment
from model import Discriminator, Generator
from utils import D_train, G_train


class DiffAugmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_disabled_policy_preserves_values_and_rng(self):
        x = torch.linspace(-1, 1, 4 * 784).reshape(4, 784)
        before = torch.get_rng_state().clone()
        self.assertIs(diff_augment(x), x)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_translation_and_cutout_preserve_background_shape_and_gradients(self):
        for policy in [('translation',), ('cutout',), ('translation', 'cutout')]:
            with self.subTest(policy=policy), torch.random.fork_rng(devices=[]):
                torch.manual_seed(7)
                background = torch.full((8, 784), -1.)
                self.assertTrue(torch.equal(diff_augment(background, policy), background))
                x = torch.linspace(-.9, .9, 8 * 784).reshape(8, 784).requires_grad_()
                output = diff_augment(x, policy)
                self.assertEqual(output.shape, x.shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(((output >= -1) & (output <= 1)).all())
                output.sum().backward()
                self.assertGreater(x.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_zero_augmentation_updates_match_original_training_bitwise(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            g, d = Generator(784), Discriminator(784)
            g_aug, d_aug = copy.deepcopy(g), copy.deepcopy(d)
            optimizers = [torch.optim.Adam(m.parameters(), lr=.0002) for m in [g, d, g_aug, d_aug]]
            x = torch.linspace(-1, 1, 8 * 784).reshape(8, 784)
            criterion = torch.nn.BCELoss()
            state = torch.get_rng_state().clone()
            d_loss = D_train(x, g, d, optimizers[1], criterion, torch.device('cpu'))
            g_loss = G_train(x, g, d, optimizers[0], criterion, torch.device('cpu'))
            torch.set_rng_state(state)
            da_loss, _ = D_train_aug(x, g_aug, d_aug, optimizers[3], criterion, torch.device('cpu'))
            ga_loss = G_train_aug(x, g_aug, d_aug, optimizers[2], criterion, torch.device('cpu'))
            self.assertEqual((d_loss, g_loss), (da_loss, ga_loss))
            for a, b in [(g, g_aug), (d, d_aug)]:
                self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))

    def test_augmented_generator_step_trains_G_without_changing_architecture(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            g, d = Generator(784), Discriminator(784)
            before = {k: v.clone() for k, v in g.state_dict().items()}
            optimizer = torch.optim.Adam(g.parameters(), lr=.0002)
            loss = G_train_aug(torch.zeros(8, 784), g, d, optimizer, torch.nn.BCELoss(),
                               torch.device('cpu'), policy=('translation', 'cutout'))
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in g.state_dict().items()))
            self.assertEqual(sum(p.numel() for p in g.parameters()), 1486352)


if __name__ == '__main__':
    unittest.main()
