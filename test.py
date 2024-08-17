from formula_factory import FormulaFactory
import torch
import time

from networks.neural_net_generator import generate_network

T = 40
d_state = 2
Batch = 10
approximation_beta = 200
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
detailed_str_mode = False

args = {'T': T, 'd_state': d_state, 'Batch': Batch, 'approximation_beta': approximation_beta, 'device': device, 'detailed_str_mode': detailed_str_mode}

FF = FormulaFactory(args)

p0 = FF.LinearPredicate(torch.tensor([1, -1]), -2, 0)
p1 = FF.LinearPredicate(torch.tensor([2, 0]), 0., 0)

and_1 = FF.And([p0.at(0), p1.at(3)])
and_2 = FF.And([p0.at(1), p1.at(2)])
F = FF.F(FF.G(FF.Or([FF.Not(p0.at(2)), and_1, and_2]), 0, 25, 2), 2, 3)

neural_net = generate_network(F, approximate=False, beta=approximation_beta).to(args['device'])
x = torch.randn((Batch, T, d_state), device=args['device'])

print("Formula:", F)

start_time = time.time()
for _ in range(100):
    r = neural_net(x)
print("--- %s seconds ---" % (time.time() - start_time))
print("Answer from the neural network: ", r)#.mean())

start_time = time.time()
for _ in range(100):
    r = F.evaluate(x)[0]
print("--- %s seconds ---" % (time.time() - start_time))
print("Answer from the formula: ", r)#.mean())