from formulas.formula import Formula
import torch
from typing import Dict
from utils import ACTIVATION_to_ID, ID_to_ACTIVATION
from networks.neural_net import NeuralNetwork


def organize_tensor_network(layer_information:Dict, W1_first_dim:int):
    # reorder layers to group similar activations together
    W1_second_dim = layer_information['W1_width']

    W1 = torch.zeros((W1_first_dim, W1_second_dim))

    W2_first_dim = W1_second_dim
    W2_second_dim = layer_information['W2_width']

    W2 = torch.zeros((W2_first_dim, W2_second_dim))

    b1 = torch.zeros(W1_second_dim)

    for (i, j), value in layer_information['W1'].items():
        W1[i, j] = value

    for (i, j), value in layer_information['W2'].items():
        W2[i, j] = value

    for i, value in layer_information['b1'].items():
        b1[i] = value

    activations = layer_information['gates']
    activations_vector = torch.tensor([ACTIVATION_to_ID[act] for act in activations])


    # argsort activations_vector
    indices = torch.argsort(activations_vector)

    activations_sorted = activations_vector[indices]

    # count the number of each activation
    activation_counts = torch.zeros(len(ACTIVATION_to_ID), dtype = torch.int32)
    for i in range(len(activations)):
        activation_counts[ACTIVATION_to_ID[activations[i]]] += 1

    activation_counts = [(ID_to_ACTIVATION[i], activation_counts[i].item()) for i in range(len(ACTIVATION_to_ID))]

    activations_sorted = [ID_to_ACTIVATION[act.item()] for act in activations_sorted]



    W1_sorted = W1[:, indices]
    W2_sorted = W2[indices, :]
    b1_sorted = b1[indices]
    return W1_sorted, b1_sorted, activation_counts, W2_sorted

def generate_network(formula:Formula, Network:Dict = None, approximate:bool = False, beta:float = 1.0):
    F = formula.parse_to_PropLogic()
    depth = F.find_depth()

    if Network is None:
        Network = {}
        Network['filled_predicates'] = {}
        for i in range(depth):
            Network[i] = {}



            Network[i]['W1'] = {}
            Network[i]['b1'] = {}
            Network[i]['W1_width'] = 0
            Network[i]['gates'] = []

            Network[i]['W2'] = {}
            Network[i]['W2_width'] = 0
            Network[i]['boolean_expression'] = []

    _, idx = F.fill_neural_net(Network, expected_layer_to_output = depth)


    Weight_first_dim = formula.d_state * formula.T

    weights = []
    biases = []
    activations = []
    W_prev_layer = None
    for layer in range(depth):
        # print(f"Layer {layer}:", Network[layer]['boolean_expression'])

        W1, b1, activation, W2 = organize_tensor_network(Network[layer], W1_first_dim = Weight_first_dim)
        Weight_first_dim = W2.shape[-1]

        if W_prev_layer is not None:
            weights.append(W_prev_layer @ W1)
        else:
            weights.append(W1)
        biases.append(b1)

        activations.append(activation)

        W_prev_layer = W2

    W_prev_layer = W_prev_layer[:, idx].unsqueeze(-1)
    weights.append(W_prev_layer)
    biases.append(torch.zeros(W_prev_layer.shape[-1]))

    return NeuralNetwork(weights, biases, activations, approximate = approximate, beta = beta)


if __name__ == '__main__':


    from formula_factory import FormulaFactory
    args = {}
    args['T'] = 40
    args['d_state'] = 2
    args['Batch'] = 1
    args['approximation_beta'] = 200
    args['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args['detailed_str_mode'] = False

    FF = FormulaFactory(args)

    p0 = FF.LinearPredicate(torch.tensor([1, -1]).cuda(), -2, 0)
    p1 = FF.LinearPredicate(torch.tensor([2, 0]).cuda(), 0., 0)

    and_1 = FF.And([p0.at(0), p0.at(3)])
    F = FF.G(FF.Not(p1.at(2)), 0, 5)

    neural_net = generate_network(F, approximate = False, beta = 1.0).to(args['device'])
    torch.manual_seed(0)
    x = torch.randn(1, 40, 2).to(args['device'])
    print("x:", x[:, :10] * 2)
    import time
    start_time = time.time()

    r1 = neural_net(x)
    r2 = F.evaluate(x)[0]
    r3 = F.parse_to_PropLogic().evaluate(x)[0]
    print(r1, r2, r3)

    print(F.parse_to_PropLogic().abstract_str())