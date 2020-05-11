import torch
import torch.nn as nn
import torchvision.models as models


class MixLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, mge_weight=1.0):
        r"""

        """
        super().__init__()

        self.register_buffer('mge_weight', torch.tensor(mge_weight))

        self.criterion = nn.MSELoss()

    def compute_gradient(self, test, truth):
        filter = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        f = filter.expand(1,3,3,3)

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

