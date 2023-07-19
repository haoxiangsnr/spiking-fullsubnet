import numpy as np


class EpochState:
    def __init__(self) -> None:
        self.epoch = 1

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = state_dict["epoch"]
        self.epoch += 1

    def state_dict(self) -> dict:
        return {"epoch": self.epoch}

    @property
    def value(self) -> int:
        return self.epoch

    @value.setter
    def value(self, epoch: int) -> None:
        self.epoch = epoch


class BestScoreState:
    def __init__(self, save_max_score) -> None:
        self.best_score = -np.inf if save_max_score else np.inf

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_score = state_dict["best_score"]

    def state_dict(self) -> dict:
        return {"best_score": self.best_score}

    @property
    def value(self) -> float:
        return self.best_score

    @value.setter
    def value(self, score: float) -> None:
        self.best_score = score


class WaitCountState:
    def __init__(self) -> None:
        self.wait_count = 0

    def load_state_dict(self, state_dict: dict) -> None:
        self.wait_count = state_dict["wait_count"]

    def state_dict(self) -> dict:
        return {"wait_count": self.wait_count}

    @property
    def value(self) -> int:
        return self.wait_count

    @value.setter
    def value(self, wait_count: int) -> None:
        self.wait_count = wait_count
