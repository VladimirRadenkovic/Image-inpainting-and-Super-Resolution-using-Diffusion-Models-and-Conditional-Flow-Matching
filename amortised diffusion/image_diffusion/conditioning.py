from typing import Type
import torch


class Conditioning:

    @classmethod
    def from_configdict(cls, config):
        return cls()


class Amortized(Conditioning):

    def __init__(self, p_cond: float, n_corrector: int, delta: float):
        self.p_cond = p_cond
        self.n_corrector = n_corrector
        self.delta = delta

    @classmethod
    def from_configdict(cls, config):
        return cls(
            p_cond=config['p_cond'],
            n_corrector=config['n_corrector'],
            delta=config['delta']
        )


class ReconstructionGuidance(Conditioning):

    def __init__(self, gamma: float, start_fraction: float, update_rule: str, n_corrector: int, delta: float) -> None:
        self.gamma = gamma
        self.start_fraction = start_fraction
        self.update_rule = update_rule
        self.n_corrector = n_corrector
        self.delta = delta

    @classmethod
    def from_configdict(cls, config):
        return cls(
            gamma=config['gamma'],
            start_fraction=config['start_fraction'],
            update_rule=config['update_rule'],
            n_corrector=config['n_corrector'],
            delta=config['delta']
        )


class Replacement(Conditioning):

    def __init__(self, delta: float, start_fraction: float, noise: bool, n_corrector: int) -> None:
        self.delta = delta
        self.start_fraction = start_fraction
        self.noise = noise
        self.n_corrector = n_corrector

    @classmethod
    def from_configdict(cls, config):
        return cls(
            delta=config['delta'],
            start_fraction=config['start_fraction'],
            noise=config['noise'],
            n_corrector=config['n_corrector'],
        )


# TODO(Vincent): use register as with datasets
def get_conditioning(type_: str) -> Type[Conditioning]:
    if type_.lower() == "amortized":
        print("Conditioning: Amortized")
        return Amortized
    elif type_.lower() == "reconstruction_guidance":
        print("Conditioning: Reconstruction")
        return ReconstructionGuidance
    elif type_.lower() == "replacement":
        print("Conditioning: Replacement")
        return Replacement
    else:
        raise NotImplementedError(f"Unknown conditioning {type_}")
