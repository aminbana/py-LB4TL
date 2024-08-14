import formulas.formula as formula
from typing import List
import torch

from utils import Swish


class Or(formula.Formula):
    def __init__(self, T, d_state, approximation_beta, device, id, detailed_str_mode:bool, formulas:List[formula.Formula]):
        super().__init__(T, d_state, approximation_beta, device, id, detailed_str_mode)

        self.formulas = formulas
        assert len(self.formulas) >= 2, "Or operator must have at least two formula"

        self.approximation_gate = Swish(beta = self.approximation_beta)

    def detailed_str(self, t:int):
        st = ' ∨ '.join([f.detailed_str(t) for f in self.formulas])
        return f'({st})'

    def abstract_str(self, t:int):
        st = ' or '.join([f.abstract_str(t) for f in self.formulas])
        return f'({st})'

    def evaluate(self, X, t):

        v = []
        critical_indices = []

        for f in self.formulas:
            ret = f.evaluate(X, t)
            v.append(ret[0])
            critical_indices.append(ret[1])

        v = torch.stack(v, dim = -1)
        critical_indices = torch.stack(critical_indices, dim = -1)

        argmax = torch.argmax(v, dim = -1)

        return v[torch.arange(v.shape[0], device=self.device), argmax], critical_indices[torch.arange(v.shape[0]), argmax]

    def approximate(self, X:torch.Tensor, t:int):
        v = []

        for f in self.formulas:
            ret = f.approximate(X, t)
            v.append(ret)

        v = torch.stack(v, dim = -1)

        # implement min on array using only binary operations

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