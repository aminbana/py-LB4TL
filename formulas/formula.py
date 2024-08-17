from abc import ABC, abstractmethod
import torch
from typing import Dict

class Formula(ABC):
    def __init__(self, args:Dict):
        # T is the horizon, i.e., the length of the trajectory
        # d_state is the dimension of the state space

        #T: int, d_state: int, approximation_beta: float, device: torch.device, id_: int, detailed_str_mode: bool

        self.args = args
        self.T:int = args['T']
        self.d_state:int = args['d_state']
        self.approximation_beta:float = args['approximation_beta']
        self.device:torch.device = args['device']
        self.detailed_str_mode:bool = args['detailed_str_mode']

    @abstractmethod
    def detailed_str(self) -> str:
        pass

    @abstractmethod
    def abstract_str(self) -> str:
        pass

    def __str__(self) -> str:
        if self.detailed_str_mode:
            return self.detailed_str()
        return self.abstract_str()

    @abstractmethod
    def at(self, t:int):
        pass

    @abstractmethod
    def evaluate(self, X:torch.Tensor) -> (torch.Tensor, torch.Tensor):
        pass

    @abstractmethod
    def approximate(self, X:torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def parse_to_PropLogic(self):
        pass

    @abstractmethod
    def find_depth(self) -> int:
        pass

    def fill_neural_net(self, net, expected_layer_to_output:int):
        pass

    def fill_the_after_part(self, expected_layer_to_output, idx, net, self_depth, name):
        idx_2 = idx
        for t in range(self_depth, expected_layer_to_output):
            net[t]['boolean_expression'].append(name)
            net[t]['W2_width'] += 1
            idx_2 = net[t]['W2_width'] - 1
            net[t]['W1_width'] += 1
            idx_1 = net[t]['W1_width'] - 1

            net[t]['gates'].append('Linear')

            net[t]['W1'][(idx, idx_1)] = 1

            net[t]['W2'][(idx_1, idx_2)] = 1

            idx = idx_2
        return net, idx_2
