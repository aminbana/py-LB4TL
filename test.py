from formula_factory import FormulaFactory
import torch
# measure time
import time

Batch = 100
T = 300_000
d = 1
approximation_beta = 2

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


X = torch.randn(Batch, T, d).to(device)
X.requires_grad = True


FF = FormulaFactory(T, d, approximation_beta, device, True)


p1 = FF.LinearPredicate(torch.tensor([1]).cuda(), -0.1)

F_ = FF.F(
        FF.G (
            FF.Not(p1),
        0, 5),
    0, 4)


start_time = time.time()

print(F_.evaluate(X, 0))

robustness = F_.approximate(X, 0).mean()
print(robustness)
robustness.backward()

print("--- %s seconds ---" % (time.time() - start_time))
#print(X.grad)



