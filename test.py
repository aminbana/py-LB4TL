
from formula_factory import FormulaFactory
import torch
from networks.neural_net_generator import generate_network
from networks.neural_net import NeuralNetwork

H = 1500  # horizon


print("The LB4TL is under construction.")

num_episodes = 10
beta = 15
detailed_str_mode = False
device = 'cuda'
d_state = 7
args = {'T': H+1, 'd_state': d_state, 'Batch': num_episodes, 'approximation_beta': beta, 'device': device, 'detailed_str_mode': detailed_str_mode}
FF = FormulaFactory(args)


### E_1
p1 = FF.LinearPredicate(torch.tensor([ 0, -1, 0, 0, 0, 0, 0]) , -9  )
p2 = FF.LinearPredicate(torch.tensor([ 0,  1, 0, 0, 0, 0, 0]) , -9  )
p3 = FF.LinearPredicate(torch.tensor([-1,  0, 0, 0, 0, 0, 0]) , -9  )
p4 = FF.LinearPredicate(torch.tensor([ 1,  0, 0, 0, 0, 0, 0]) , -9  )
p5 = FF.LinearPredicate(torch.tensor([ 0,  0, 1, 0, 0, 0, 0]) , -35 )



### E_2
p6 = FF.LinearPredicate(torch.tensor([-1,  0, 0, 0, 0, 0, 1]) ,  1  )
p7 = FF.LinearPredicate(torch.tensor([ 1,  0, 0, 0, 0, 0,-1]) ,  1  )
p8 = FF.LinearPredicate(torch.tensor([ 0, -1, 0, 0, 0, 0, 0]) ,  1  )
p9 = FF.LinearPredicate(torch.tensor([ 0,  1, 0, 0, 0, 0, 0]) ,  1  )
p10= FF.LinearPredicate(torch.tensor([ 0,  0,-1, 0, 0, 0, 0]) , 0.6 )
p11= FF.LinearPredicate(torch.tensor([ 0,  0, 1, 0, 0, 0, 0]) ,-0.11)
p12= FF.LinearPredicate(torch.tensor([ 0,  0, 0,-1, 0, 0, 0]) ,  2  )
p13= FF.LinearPredicate(torch.tensor([ 0,  0, 0, 1, 0, 0, 0]) ,  0  )
p14= FF.LinearPredicate(torch.tensor([ 0,  0, 0, 0,-1, 0, 0]) ,  1  )
p15= FF.LinearPredicate(torch.tensor([ 0,  0, 0, 0, 1, 0, 0]) ,  1  )
p16= FF.LinearPredicate(torch.tensor([ 0,  0, 0, 0, 0,-1, 0]) ,  1  )
p17= FF.LinearPredicate(torch.tensor([ 0,  0, 0, 0, 0, 1, 0]) ,  1  )


### E_3
p18= FF.LinearPredicate(torch.tensor([0,  0, 0, 0, 0, 0, 1]) , -9.5)



formula = FF.And([
                  FF.G(FF.Or( [
                              p1,p2,p3,p4,p5
                             ])
                      ,0   , H)
                 ,FF.F(FF.And([
                             p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17
                             ])
                      ,1100, H)
                 ,FF.G(
                      p18
                      ,0 , H)
                 ])

import time
x = torch.randn((num_episodes, H+1, d_state), device=args['device'], requires_grad=True)

neural_net_soft_sparse = generate_network(formula, approximate=True , beta=5, sparse=True).to(device)
t_start = time.time()

for _ in range(1):
    x.grad = None
    res = neural_net_soft_sparse(x)
    res.mean().backward()
print(res)
print("--- Sparse time: %s seconds ---" % (time.time() - t_start))


neural_net_soft_sparse = generate_network(formula, approximate=True , beta=5, sparse=True).to(device)
neural_net_soft_sparse.save_to_disk("neural_net_soft_sparse.pth")
print(" --------- Saved To disk -------- ")
loaded_net = NeuralNetwork.load_from_disk("neural_net_soft_sparse.pth").to(device)
print(loaded_net(x))
print(" --------- Loaded from disk -------- ")

# neural_net_exact_sparse = generate_network(formula, approximate=False , beta=1, sparse=True).to(device)
# for _ in range(1):
#     x.grad = None
#     res = neural_net_soft_sparse(x)
#     res.mean().backward()
# print(res)
# print("--- %s seconds ---" % (time.time() - t_start))

print(" ----------------- ")

neural_net_soft = generate_network(formula, approximate=True , beta=5, sparse=False).to(device)
t_start = time.time()

for _ in range(1):
    x.grad = None
    res = neural_net_soft(x)
    res.mean().backward()

print(res)
print("--- Dense time %s seconds ---" % (time.time() - t_start))

neural_net_soft.save_to_disk("neural_net_soft.pth")

loaded_net_dense = NeuralNetwork.load_from_disk("neural_net_soft.pth").to(device)
print(loaded_net_dense(x))
print(" ----------------- ")