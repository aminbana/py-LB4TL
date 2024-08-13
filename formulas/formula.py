from abc import ABC, abstractmethod
import torch

class Formula(ABC):
    def __init__(self, T:int, d_state:int, approximation_beta:float, detailed_str_mode:bool = False):
        # T is the horizon, i.e., the length of the trajectory
        # d_state is the dimension of the state space

        self.T = T
        self.d_state = d_state
        self.detailed_str_mode = detailed_str_mode
        self.approximation_beta = approximation_beta

    @abstractmethod
    def detailed_str(self, t:int) -> str:
        pass

    @abstractmethod
    def abstract_str(self, t:int) -> str:
        pass


    def __str__(self) -> str:
        if self.detailed_str_mode:
            return self.detailed_str(0)
        return self.abstract_str(0)

    @abstractmethod
    def evaluate(self, X:torch.Tensor, t:int) -> (torch.Tensor, torch.Tensor):
        pass

    @abstractmethod
    def approximate(self, X:torch.Tensor, t:int) -> torch.Tensor:
        pass

