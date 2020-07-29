import unittest

import torch
from src.pytorch import vae
from src.pytorch.train import fit


class TestShapeNet(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestShapeNet, self).__init__(*args, **kwargs)
        self.in_data = vae.Variable(torch.rand(1, 3, 250))
        self.num_cubes = 20 ** 3
        self.model = vae.VariationalAutoEncoder(num_cubes=self.num_cubes, threshold=0.0002, num_sample_cube=20)
        self.init_params = self.model.state_dict()
        self.vae_out, self.probs, self.mu_out, self.sigma_out = self.model(self.in_data)

    def test_weights_change(self):
        """
        Test all tensors are being optimize
        :return:
        """
        after_param = self.model.state_dict()
        for name in self.init_params:
            param = self.init_params[name]
            self.assertTrue(torch.all(param.eq(after_param[name])))

    def test_loss(self):
        loss_obj = vae.VAELoss()
        # prob_pred, prob_target, x_diff_pred, x_diff_target
        # loss = loss_obj(prob_pred, prob_target, x_diff_pred, x_diff_target)

    def test_reparameterize(self):
        pass


    def test_dims_VariationalAutoEncoder(self):
        # dim test
        self.assertEqual(self.vae_out.shape[0], 1)  # should be (1, 0<n<num_cubes*num_sample_cube, 3)
        self.assertEqual(self.vae_out.shape[2], 3)
        self.assertEqual(self.probs.shape, (1, self.num_cubes))
        self.assertEqual(self.mu_out.shape, (1, 3 * self.num_cubes))
        self.assertEqual(self.sigma_out.shape, (1, 3 * self.num_cubes))
        self.assertEqual(self.probs.flatten().sum(), 1.0)

    def test_encoder(self):
        pass


if __name__ == '__main__':
    unittest.main()
