"""Configuration dataclass for the Hashi RL environment."""

from dataclasses import dataclass


@dataclass
class RLConfig:
    """Configuration for the HashiEnv oracle-aware RL environment.

    Parameters
    ----------
    mask_over_2 : bool
        Mask actions that would increment an edge above 2 bridges.
    mask_capacity : bool
        Mask actions that would cause a node to exceed its capacity.
    mask_crossing : bool
        Mask actions that would create a bridge crossing.
    reward_correct : float
        Per-step reward for a correct bridge increment.
    reward_solve : float
        Bonus reward added on top of reward_correct when the puzzle is solved.
    reward_failure : float
        Reward (negative) returned on any terminal failure.
    gamma : float
        Discount factor for REINFORCE return computation.
    """

    mask_over_2: bool = True
    mask_capacity: bool = False
    mask_crossing: bool = False
    reward_correct: float = 1.0
    reward_solve: float = 10.0
    reward_failure: float = -10.0
    gamma: float = 1.0
