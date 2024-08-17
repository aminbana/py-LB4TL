import torch

from formulas.predicate import LinearPredicate
from formulas.multi_and import And
from formulas.multi_or import Or
from formulas.globally import G
from formulas.eventually import F
from formulas.formula import Formula
from formulas.not_gate import Not

from typing import List, Dict

SUPPORTED_FORMULAS = []
class FormulaFactory:
    def __init__(self, args:Dict):

        self.args = args

        self.linear_predicate_id_ = -1

    def LinearPredicate(self, A:torch.Tensor, b:float, t0 = 0):
        self.linear_predicate_id_ += 1
        id_ = self.linear_predicate_id_
        return LinearPredicate(self.args, id_, A, b, t0)

    def And(self, formulas:List[Formula], t0 : int = 0):
        return And(self.args, formulas, t0)

    def Or(self, formulas:List[Formula], t0:int = 0):
        return Or(self.args, formulas, t0)

    def G(self, formula:Formula, t_init:int, t_final:int, t0:int = 0):
        return G(self.args, formula, t0, t_init, t_final)

    def F(self, formula:Formula, t_init:int, t_final:int, t0:int = 0):
        return F(self.args, formula, t0, t_init, t_final)
    def Ordered(self, formula1:Formula, formula2:Formula, t_init:int, t_final:int):
        or_list = []
        for i in range(t_init,t_final):
            or_list.append( self.And( [ formula1.at(i) , self.F( formula2, i+1 , t_final) ] ) )

        return self.Or(or_list)
    
    def Until(self, formula1:Formula, formula2:Formula, t_init:int, t_final:int):
        or_list = []
        for i in range(t_init,t_final):
            or_list.append( self.And( [ formula1.at(i) , self.G( formula2, i+1 , t_final) ] ) )

        return self.Or(or_list)
    
    def Release(self, formula1:Formula, formula2:Formula, t_init:int, t_final:int):
        and_list = []
        for i in range(t_init,t_final):
            and_list.append( self.Or( [ formula1.at(i) , self.G( formula2, i+1 , t_final) ] ) )

        return self.And(and_list)


    def Not(self, formula:Formula):
        return Not(self.args, formula)
