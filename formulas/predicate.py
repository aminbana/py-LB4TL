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

    PREDICATE_ID = -1

    def __init__(self, T, d_state, approximation_beta, detailed_str_mode:bool, t0:int, A:torch.Tensor, b:float):
        super().__init__(T, d_state, approximation_beta, detailed_str_mode)
        self.A = A.float()
        self.b = float (b)

        self.t0 = t0

        assert self.A.shape == torch.Size([self.d_state]), "Predicate specification does not match state space dimension"

        LinearPredicate.PREDICATE_ID += 1
        self.id = LinearPredicate.PREDICATE_ID


    def detailed_str(self, t:int):
        self._assert_time_bound(t)
        t_ = t + self.t0

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


    def abstract_str(self, t:int):
        self._assert_time_bound(t)
        t_ = t + self.t0
        return f'p{self.id}(X{t_})'

    def _assert_time_bound(self, t:int):
        assert self.t0 + t < self.T, "Predicate time bound exceeded"

    def evaluate(self, X:torch.Tensor, t:int):
        self._assert_time_bound(t)
        self._assert_input_shape(X)

        t_ = t + self.t0

        return torch.matmul(X[:, t_, :], self.A) + self.b, torch.ones(X.shape[0], dtype = torch.long) * t_

    def _assert_input_shape(self, X:torch.Tensor):
        assert X.shape == torch.Size([X.shape[0], self.T, self.d_state]), "Predicate input shape mismatch"
        assert X.dtype == torch.float32, "Predicate input type mismatch"


    def approximate(self, X:torch.Tensor, t:int):
        self._assert_time_bound(t)
        self._assert_input_shape(X)

        return self.evaluate(X, t)[0]

