import formulas.formula as formula
from typing import List
import torch

def format_number(num):
    # Format the number to 2 decimal places, and strip trailing zeros
    return ('{:.2f}'.format(num)).rstrip('0').rstrip('.')

def concat_with_sign(numbers):
    result = numbers[0]  # Start with the first number

    for num in numbers[1:]:
        if num.startswith('-'):
            result += ' - ' + num[1:]  # Directly concatenate negative numbers
        else:
            result += ' + ' + num  # Prepend + before positive numbers

    return result

class LinearPredicate(formula.Formula):

    def __init__(self, args, id_:int, A:torch.Tensor, b:float, t0:int):
        super().__init__(args)
        self.A = A.float()
        self.b = float (b)

        assert self.A.shape == torch.Size([self.d_state]), "Predicate specification does not match state space dimension"
        self.A = self.A.to(self.device)

        self.id_ = id_

        self.t0 = t0
        self._assert_time_bound(t0)



    def detailed_str(self):
        t_ = self.t0

        coeffs = []

        for i in range(self.d_state):
            if self.A[i] != 0:
                if self.A[i] == 1:
                    coeffs.append(f'X{t_}[{i}]')
                else:
                    coeffs.append(f'{format_number(self.A[i])} X{t_}[{i}]')



        if self.b == 0:
            if len(coeffs) == 0:
                return '0 >= 0'
        else:
            coeffs.append(format_number(self.b))

        return f'{concat_with_sign(coeffs)} >= 0'


    def abstract_str(self):
        t_ = self.t0
        return f'p{self.id_}(X{t_})'

    def _assert_time_bound(self, t:int):
        assert t < self.T, "Predicate time bound exceeded"

    def evaluate(self, X:torch.Tensor):
        t_ = self.t0
        self._assert_input_shape(X)

        return torch.matmul(X[:, t_, :], self.A) + self.b, torch.ones(X.shape[0], dtype = torch.long, device=self.device) * t_

    def _assert_input_shape(self, X:torch.Tensor):
        assert X.shape == torch.Size([X.shape[0], self.T, self.d_state]), "Predicate input shape mismatch"
        assert X.dtype == torch.float32, "Predicate input type mismatch"


    def approximate(self, X:torch.Tensor):
        return self.evaluate(X)[0]


    def parse_to_PropLogic(self):
        t = self.t0
        return LinearPredicate(self.args, self.id_, self.A, self.b, t)

    def find_depth(self):
        return 1

    def at(self, t:int):
        t = t + self.t0
        return LinearPredicate(self.args, self.id_, self.A, self.b, t)


    def fill_neural_net(self, net, expected_layer_to_output:int):
        t = self.t0
        d = self.d_state

        self_depth = self.find_depth()
        assert self_depth == 1

        # filling before part

        abstract_name = self.abstract_str()
        if not abstract_name in net['filled_predicates'].keys():
            net[0]['boolean_expression'].append(abstract_name)
            net[0]['W2_width'] += 1
            net[0]['W1_width'] += 1
            net[0]['gates'].append('Linear')

            idx1 = net[0]['W1_width'] - 1
            idx2 = net[0]['W2_width'] - 1

            for i in range(d):
                net[0]['W1'][(t * d + i, idx1)] = self.A[i].item()
                net[0]['b1'][idx1] = self.b

            net[0]['W2'][(idx1, idx2)] = 1

            net['filled_predicates'][abstract_name] = (idx1, idx2)

        (idx1, idx2) = net['filled_predicates'][abstract_name]
        idx = idx2

        # filling the after part with pure linear
        net, idx = self.fill_the_after_part(expected_layer_to_output, idx, net, self_depth, abstract_name)
        return net, idx







