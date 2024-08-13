import torch

class LinearPredicateNet(torch.nn.Module):
    def __init__(self, t:int, A:torch.Tensor, b:float):
        super(LinearPredicateNet, self).__init__()

        self.A = A.float()
        self.b = float(b)
        self.t = t


    def forward(self, X):
        t_ = self.t
        return torch.matmul(X[:,t_], self.A) + self.b
