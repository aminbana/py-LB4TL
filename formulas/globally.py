import formulas.formula as formula
from typing import List
import torch

class G(formula.Formula):

    PREDICATE_ID = -1

    def __init__(self, T, d_state, approximation_beta, detailed_str_mode:bool, f:formula.Formula, t_init:int, t_final:int):
        super().__init__(T, d_state, approximation_beta, detailed_str_mode)

        G.PREDICATE_ID += 1
        self.id = G.PREDICATE_ID

        self.f = f

        assert t_init >= 0, "Initial time must be non-negative"
        assert t_final > t_init, "Final time must be greater than the initial time"

        self.t_init = t_init
        self.t_final = t_final

        self.approximation_gate = torch.nn.Softplus(beta = self.approximation_beta)

    def detailed_str(self, t:int):
        st = ' ∧ '.join([self.f.detailed_str(t_) for t_ in range(t + self.t_init, t + self.t_final + 1)])
        return f'({st})'

    def abstract_str(self, t:int):
        st = ' and '.join([self.f.abstract_str(t_) for t_ in range(t + self.t_init, t + self.t_final + 1)])
        return f'({st})'

    def evaluate(self, X, t):

        v = []
        critical_indices = []

        for t_ in range(t + self.t_init, t + self.t_final + 1):
            ret = self.f.evaluate(X, t_)
            v.append(ret[0])
            critical_indices.append(ret[1])

        v = torch.stack(v, dim = -1)
        critical_indices = torch.stack(critical_indices, dim = -1)

        argmin = torch.argmin(v, dim = -1)

        return v[torch.arange(v.shape[0]), argmin], critical_indices[torch.arange(v.shape[0]), argmin]

    def approximate(self, X:torch.Tensor, t:int):
        v = []

        for t_ in range(t + self.t_init, t + self.t_final + 1):
            ret = self.f.approximate(X, t_)
            v.append(ret)

        v = torch.stack(v, dim = -1)

        while v.shape[-1] > 1:
            is_odd = v.shape[-1] % 2
            if is_odd:
                v_last = v[..., -1]
                v = v[..., :-1]

            v_even = v[..., ::2]
            v_odd = v[..., 1::2]

            v = v_even - self.approximation_gate(v_even - v_odd)

            if is_odd:
                v = torch.cat([v, v_last.unsqueeze(-1)], dim = -1)


        return v.squeeze(-1)