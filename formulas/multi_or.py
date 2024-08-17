import formulas.formula as formula
from typing import List
import torch

from utils import Swish


class Or(formula.Formula):
    def __init__(self, args, formulas:List[formula.Formula], t0:int = 0):
        super().__init__(args)

        self.formulas = formulas
        assert len(self.formulas) >= 2, "Or operator must have at least two formula"

        self.approximation_gate = Swish(beta = self.approximation_beta)
        self.t0 = t0

    def detailed_str(self):
        t = self.t0
        st = ' ∨ '.join([f.at(t).detailed_str() for f in self.formulas])
        return f'({st})'

    def abstract_str(self):
        t = self.t0
        st = ' or '.join([f.at(t).abstract_str() for f in self.formulas])
        return f'({st})'

    def evaluate(self, X):
        t = self.t0

        v = []
        critical_indices = []

        for f in self.formulas:
            ret = f.at(t).evaluate(X)
            v.append(ret[0])
            critical_indices.append(ret[1])

        v = torch.stack(v, dim = -1)
        critical_indices = torch.stack(critical_indices, dim = -1)

        argmax = torch.argmax(v, dim = -1)

        return v[torch.arange(v.shape[0], device=self.device), argmax], critical_indices[torch.arange(v.shape[0]), argmax]

    def approximate(self, X:torch.Tensor):
        t = self.t0
        v = []

        for f in self.formulas:
            ret = f.at(t).approximate(X)
            v.append(ret)

        v = torch.stack(v, dim = -1)

        # implement min on array using only binary operations

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
        for f in self.formulas:
            new_formulas.append(f.at(t).parse_to_PropLogic())

        while len(new_formulas) > 1:
            # extract head of the list
            f1 = new_formulas.pop(0)
            f2 = new_formulas.pop(0)
            new_formulas.append(Or(self.args, [f1, f2], t0 = 0))

        return new_formulas[0]

    def find_depth(self):
        t = self.t0
        assert len(self.formulas) == 2, "First call parse_to_PropLogic to simplify the formula"
        return 1 + max([f.at(t).find_depth() for f in self.formulas])

    def at(self, t:int):
        t = t + self.t0
        return Or(self.args, self.formulas, t0 = t)


    def fill_neural_net(self, net, expected_layer_to_output:int):
        t = self.t0

        d = self.find_depth()

        assert len(self.formulas) == 2, "First call parse_to_PropLogic to simplify the formula"
        f1 = self.formulas[0].at(t)
        f2 = self.formulas[1].at(t)

        net, idx_1 = f1.fill_neural_net(net, expected_layer_to_output = d - 1)
        net, idx_2 = f2.fill_neural_net(net, expected_layer_to_output = d - 1)
        layer = d - 1

        net[layer]['W1_width'] += 2
        idx_middle = net[layer]['W1_width'] - 2

        net[layer]['W1'][(idx_1, idx_middle)] = 1
        if idx_1 == idx_2: #  means that f1 and f2 were both the same predicate
            net[layer]['W1'][(idx_2, idx_middle)] += -1
        else:
            net[layer]['W1'][(idx_2, idx_middle)] = -1

        net[layer]['W1'][(idx_2, idx_middle+1)] = 1

        net[layer]['gates'].append('Swish')
        net[layer]['gates'].append('Linear')

        name = self.abstract_str()
        net[layer]['boolean_expression'].append(name)
        net[layer]['W2_width'] += 1
        idx = net[layer]['W2_width'] - 1
        net[layer]['W2'][(idx_middle, idx)] = 1
        net[layer]['W2'][(idx_middle+1, idx)] = 1


        # fill the after part
        net, idx = self.fill_the_after_part(expected_layer_to_output, idx, net, d, name)

        return net, idx