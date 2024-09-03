import torch
from utils import Swish, ACTIVATION_to_ID, ID_to_ACTIVATION
from typing import List, Dict

class MultiLengthActivation(torch.nn.Module):
    def __init__(self, activations_list, approximate:bool = False, beta:float = 1.0):
        super(MultiLengthActivation, self).__init__()
        swish = Swish(beta = beta) if approximate else torch.nn.ReLU()
        softplus = torch.nn.Softplus(beta = beta) if approximate else torch.nn.ReLU()
        # No op activation
        linear = lambda x: x

        self.activations = []

        j = 0
        for (a, count) in activations_list:

            if a == 'Linear':
                a = linear
            elif a == 'Swish':
                a = swish
            elif a == 'Softplus':
                a = softplus
            else:
                raise ValueError(f'Activation {a} not recognized')

            self.activations.append((a, (j, j + count)))
            j += count


    def forward(self, x):
        new_x = []
        for a, (i, j) in self.activations:
            new_x.append(a(x[:, i:j]))
        new_x = torch.cat(new_x, dim = 1)
        return new_x

class SparseLinearLayer(torch.nn.Module):
    def __init__(self, W, b):
        super(SparseLinearLayer, self).__init__()
        W_ = W
        b_ = b.to_dense().unsqueeze(0)

        self.register_buffer('W', W_)
        self.register_buffer('b', b_)

    def forward(self, x):
        return x @ self.W + self.b

class NeuralNetwork(torch.nn.Module):
    def __init__(self, weights:List[torch.Tensor], biases:List[torch.Tensor], activations_list, approximate:bool = False, beta:float = 1.0, sparse = False):
        super(NeuralNetwork, self).__init__()
        num_layers = len(weights)
        layers = torch.nn.ModuleList()
        assert weights[-1].shape[-1] == 1, f'Output layer must have 1 output, got {weights[-1].shape[-1]}'

        for i in range(num_layers):
            if not sparse:
                linear_layer = torch.nn.Linear(weights[i].shape[0], weights[i].shape[1])
                linear_layer.weight.data = weights[i].T
                linear_layer.bias.data = biases[i]
            else:
                linear_layer = SparseLinearLayer(weights[i], biases[i])

            layers.append(linear_layer)
            if i < num_layers - 1:
                activation = MultiLengthActivation(activations_list[i], approximate = approximate, beta = beta)
                layers.append(activation)

        self.layers = torch.nn.Sequential(*layers)

        self.all_inputs = [weights, biases, activations_list, approximate, beta, sparse]


    def forward(self, x):
        assert len(x.shape) == 3, f'Input shape must be 3D, got {len(x.shape)}D'
        x = x.flatten(start_dim = 1)
        return self.layers(x).squeeze(-1)

    def save_to_disk(self, path:str):
        torch.save(self.all_inputs, path)

    @staticmethod
    def load_from_disk(path:str):
        all_inputs = torch.load(path)
        return NeuralNetwork(*all_inputs)




