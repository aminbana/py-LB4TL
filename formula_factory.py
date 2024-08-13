import torch

from formulas.predicate import LinearPredicate
from formulas.multi_and import And
from formulas.multi_or import Or
from formulas.globally import G
from formulas.eventually import F
from formulas.formula import Formula

from typing import List

SUPPORTED_FORMULAS = []
class FormulaFactory:
    def __init__(self, T, d_state, approximation_beta, detailed_str_mode = False):
        # T is the horizon, i.e., the length of the trajectory
        # d_state is the dimension of the state space

        self.T = T
        self.d_state = d_state
        self.detailed_str_mode = detailed_str_mode
        self.approximation_beta = approximation_beta

    def LinearPredicate(self, A:torch.Tensor, b:float, t0 = 0):
       return LinearPredicate(self.T, self.d_state, self.approximation_beta, self.detailed_str_mode, t0, A, b)

    def And(self, formulas:List[Formula]):
        return And(self.T, self.d_state, self.approximation_beta, self.detailed_str_mode, formulas)

    def Or(self, formulas:List[Formula]):
        return Or(self.T, self.d_state, self.approximation_beta, self.detailed_str_mode, formulas)

    def G(self, formula:Formula, t_init:int, t_final:int):
        return G(self.T, self.d_state, self.approximation_beta, self.detailed_str_mode, formula, t_init, t_final)

    def F(self, formula:Formula, t_init:int, t_final:int):
        return F(self.T, self.d_state, self.approximation_beta, self.detailed_str_mode, formula, t_init, t_final)



    # def get_NeuralNet(self, formula:Formula , approximate:bool = False, ):
    #
    #     formula.get_neural_net(approximate)
    #
    #     return None
