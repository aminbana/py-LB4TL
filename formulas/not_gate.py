import formulas.formula as formula
from formulas.predicate import LinearPredicate
from typing import List
import torch

class Not(formula.Formula):

    def __init__(self, T, d_state, approximation_beta, device, id, detailed_str_mode:bool, f:formula.Formula):
        super().__init__(T, d_state, approximation_beta, device, id, detailed_str_mode)

        assert isinstance(f, LinearPredicate), "Not gate can only be applied to a Predicate"

        self.f = f


    def detailed_str(self, t:int):
        st = f'¬ {self.f.detailed_str(t)}'
        return st
    def abstract_str(self, t:int):
        st = f'not {self.f.abstract_str(t)}'
        return st

    def evaluate(self, X, t):
        v = self.f.evaluate(X, t)
        return -v[0], v[1]

    def approximate(self, X, t):
        return -self.f.approximate(X, t)
