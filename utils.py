import torch

ACTIVATION_to_ID = {'Linear':0, 'Swish':1, 'Softplus':2}
ID_to_ACTIVATION = {v:k for k, v in ACTIVATION_to_ID.items()}


class Swish(torch.nn.Module):
    def __init__(self, beta = 1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)