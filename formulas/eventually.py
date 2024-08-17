import formulas.formula as formula
from typing import List
import torch

from utils import Swish
from formulas.multi_or import Or

class F(formula.Formula):
    def __init__(self, args, f:formula.Formula, t0:int, t_init:int, t_final:int):
        super().__init__(args)


        self.f = f

        assert t_init >= 0, "Initial time must be non-negative"
        assert t_final >= t_init, "Final time must be greater than the initial time"

        self.t0 = t0
        self.t_init = t_init
        self.t_final = t_final

        self.approximation_gate = Swish(beta = self.approximation_beta)



    def detailed_str(self):
        t = self.t0
        st = ' ∨ '.join([self.f.at(t_).detailed_str() for t_ in range(t + self.t_init, t + self.t_final + 1)])
        return f'({st})'

    def abstract_str(self):
        t = self.t0
        st = f'F[{t+self.t_init},{t+self.t_final}]({self.f.abstract_str()})'
        return st

    def evaluate(self, X):

        t = self.t0

        v = []
        critical_indices = []

        for t_ in range(t + self.t_init, t + self.t_final + 1):
            ret = self.f.at(t_).evaluate(X)
            v.append(ret[0])
            critical_indices.append(ret[1])

        v = torch.stack(v, dim = -1)
        critical_indices = torch.stack(critical_indices, dim = -1)

        argmax = torch.argmax(v, dim = -1)

        return v[torch.arange(v.shape[0], device=self.device), argmax], critical_indices[torch.arange(v.shape[0]), argmax]


    def approximate(self, X:torch.Tensor):

        t = self.t0

        v = []

        for t_ in range(t + self.t_init, t + self.t_final + 1):
            ret = self.f.at(t_).approximate(X)
            v.append(ret)

        v = torch.stack(v, dim = -1)

        v_last = None
        while v.shape[-1] > 1:
            is_odd = v.shape[-1] % 2
            if is_odd:
                v_last = v[..., -1]
                v = v[..., :-1]

            v_even = v[..., ::2]
            v_odd = v[..., 1::2]

            v = self.approximation_gate(v_even - v_odd) + v_odd

            if is_odd:
                v = torch.cat([v, v_last.unsqueeze(-1)], dim = -1)


        return v.squeeze(-1)


    def parse_to_PropLogic(self):
        t = self.t0

        new_formulas = []
        for t_ in range(t + self.t_init, t + self.t_final + 1):
            new_formulas.append(self.f.at(t_).parse_to_PropLogic())

        while len(new_formulas) > 1:
            # extract head of the list
            f1 = new_formulas.pop(0)
            f2 = new_formulas.pop(0)
            new_formulas.append(Or(self.args, [f1, f2], t0 = 0))

        return new_formulas[0]

    def find_depth(self):
        assert False, "Call parse_to_PropLogic before calling find_depth to ensure the formula has only and/or operators."

    def at(self, t:int):
        t = t + self.t0
        return F(self.args, self.f, t, self.t_init, self.t_final)

    def fill_neural_net(self, net, expected_layer_to_output:int):
        assert False, "Call parse_to_PropLogic before calling fill_neural_net to ensure the formula has only and/or operators."
