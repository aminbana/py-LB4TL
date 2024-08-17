import formulas.formula as formula
from formulas.predicate import LinearPredicate
from typing import List
import torch

class Not(formula.Formula):

    def __init__(self, args, f:formula.Formula, t0:int = 0):
        super().__init__(args)

        assert isinstance(f, LinearPredicate), "Not gate can only be applied to a Predicate"

        self.f = f
        self.t0 = t0


    def detailed_str(self):
        t = self.t0
        st = f'¬ ({self.f.at(t).detailed_str()})'
        return st
    def abstract_str(self):
        t = self.t0
        st = f'not {self.f.at(t).abstract_str()}'
        return st

    def evaluate(self, X):
        t = self.t0
        v = self.f.at(t).evaluate(X)
        return -v[0], v[1]

    def approximate(self, X):
        t = self.t0
        return -self.f.at(t).approximate(X)


    def parse_to_PropLogic(self):
        t = self.t0
        return Not(self.args, self.f.at(t).parse_to_PropLogic(), t0 = 0)

    def at(self, t:int):
        t = t + self.t0
        return Not(self.args, self.f, t0 = t)

    def find_depth(self):
        return 1

    def fill_neural_net(self, net, expected_layer_to_output:int):
        t = self.t0
        d = self.d_state

        self_depth = self.find_depth()
        assert self_depth == 1

        # filling before part

        abstract_name_predicate = self.f.abstract_str()
        not_predicate_name = "¬" + abstract_name_predicate
        if not not_predicate_name in net['filled_predicates'].keys():
            if not abstract_name_predicate in net['filled_predicates'].keys():
                self.f.fill_neural_net(net, expected_layer_to_output = 1)
            (idx1, _) = net['filled_predicates'][abstract_name_predicate]

            net[0]['boolean_expression'].append(not_predicate_name)
            net[0]['W2_width'] += 1
            idx2 = net[0]['W2_width'] - 1

            net[0]['W2'][(idx1, idx2)] = -1

            net['filled_predicates'][not_predicate_name] = (idx1, idx2)

        (_, idx) = net['filled_predicates'][not_predicate_name]

        # filling the after part with pure linear
        return self.fill_the_after_part(expected_layer_to_output, idx, net, self_depth, not_predicate_name)

