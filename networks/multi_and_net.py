import torch

class MultiAndNet(torch.nn.Module):
    def __init__(self, networks:torch.nn.ModuleList):
        super(MultiAndNet, self).__init__()

        self.networks = networks


    def forward(self, X):
        pass
