import torch.nn as nn


class FilterLocalization(nn.Module):

    def __init__(self, coeff=2., threshold=0.01):
        super(FilterLocalization, self).__init__()

        self.bce_loss = nn.BCELoss(reduction='mean')
        self.threshold = threshold
        self.coeff = coeff
        self.loss = 0.0

    def forward(self, p, prob_target, samples):

        BCE = self.bce_loss(p[0], prob_target)
        self.loss = self.coeff * BCE

        # filter out low frequency cubes
        mask = p[0] > self.threshold  # in shape probs
        out = samples[0][mask]  # torch.Size([high_prob_cubes, 100, 3])

        return out.view(1, -1, 3)  # estimated diff point cloud in [bs, num_right_cubes * samples_in_cube, 3]
