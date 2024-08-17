import torch

from formulas.predicate import LinearPredicate
from formulas.multi_and import And
from formulas.multi_or import Or
from formulas.globally import G
from formulas.eventually import F
from formulas.formula import Formula
from formulas.not_gate import Not

from typing import List

SUPPORTED_FORMULAS = []
class FormulaFactory:
    def __init__(self, T, d_state, approximation_beta, device, detailed_str_mode = False):
        # T is the horizon, i.e., the length of the trajectory
        # d_state is the dimension of the state space

        self.T = T
        self.d_state = d_state
        self.detailed_str_mode = detailed_str_mode
        self.approximation_beta = approximation_beta
        self.device = device

        self.linear_predicate_id = -1
        self.and_id = -1
        self.or_id = -1
        self.G_id = -1
        self.F_id = -1

    def LinearPredicate(self, A:torch.Tensor, b:float, t0 = 0):
       return LinearPredicate(self.T, self.d_state, self.approximation_beta, self.device, id, self.detailed_str_mode, t0, A, b)

    def And(self, formulas:List[Formula]):
        return And(self.T, self.d_state, self.approximation_beta, self.device, id, self.detailed_str_mode, formulas)

    def Or(self, formulas:List[Formula]):
        return Or(self.T, self.d_state, self.approximation_beta, self.device, id, self.detailed_str_mode, formulas)

    def G(self, formula:Formula, t_init:int, t_final:int):
        return G(self.T, self.d_state, self.approximation_beta, self.device, id, self.detailed_str_mode, formula, t_init, t_final)

    def F(self, formula:Formula, t_init:int, t_final:int):
        return F(self.T, self.d_state, self.approximation_beta, self.device, id, self.detailed_str_mode, formula, t_init, t_final)
    
    def Ordered(self, formula1:Formula, formula2:Formula, t_init:int, t_final:int):
        or_list = []
        for i in range(t_init,t_final):
            or_list.append( self.And( [ self.F(formula1, i, i) , self.F( formula2, i+1 , t_final) ] ) )
                         
        return self.Or(or_list)
    
    def Until(self, formula1:Formula, formula2:Formula, t_init:int, t_final:int):
        or_list = []
        for i in range(t_init,t_final):
            or_list.append( self.And( [ self.F(formula1, i, i) , self.G( formula2, i+1 , t_final) ] ) )
                         
        return self.Or(or_list)
    
    def Release(self, formula1:Formula, formula2:Formula, t_init:int, t_final:int):
        and_list = []
        for i in range(t_init,t_final):
            and_list.append( self.Or( [ self.F(formula1, i, i) , self.G( formula2, i+1 , t_final) ] ) )
                         
        return self.And(and_list)

    def Not(self, formula:Formula):
        return Not(self.T, self.d_state, self.approximation_beta, self.device, id, self.detailed_str_mode, formula)

    # def get_NeuralNet(self, formula:Formula , approximate:bool = False, ):
    #
    #     formula.get_neural_net(approximate)
    #
    #     return None
