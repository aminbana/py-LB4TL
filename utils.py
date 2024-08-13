import torch

class Swish(torch.nn.Module):
    def __init__(self, beta = 1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)