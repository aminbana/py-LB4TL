from formula_factory import FormulaFactory
import torch
# measure time
import time

Batch = 1
T = 300_000
d = 1
approximation_beta = 2

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


X = torch.randn(Batch, T, d).cuda()
X.requires_grad = True


FF = FormulaFactory(T, d, approximation_beta, True)


p1 = FF.LinearPredicate(torch.tensor([1]).cuda(), -3)

F_ = FF.F(
FF.G (p1,0, 5)
, 0, 4)


start_time = time.time()

print(F_.evaluate(X, 0))

robustness = F_.approximate(X, 0)
print(robustness)
robustness.backward()

print("--- %s seconds ---" % (time.time() - start_time))
#print(X.grad)



