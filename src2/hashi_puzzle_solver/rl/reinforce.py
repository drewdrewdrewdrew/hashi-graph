"""REINFORCE training loop for Hashi RL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.optim as optim

from hashi_puzzle_solver.rl.env import HashiEnv
from hashi_puzzle_solver.rl.rollout import collect_rollout
from hashi_puzzle_solver.utils.common import custom_collate_with_conflicts

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from hashi_puzzle_solver.rl.config import RLConfig


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted returns for a single trajectory.

    Parameters
    ----------
    rewards : list[float]
        Per-step rewards from a single trajectory.
    gamma : float
        Discount factor applied when accumulating future rewards.

    Returns
    -------
    list[float]
        Discounted return ``G_t`` for each time step ``t``.
    """
    G = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def reinforce_loss(
    trajectories: list[list[tuple[torch.Tensor, float]]],
    gamma: float = 1.0,
    ) -> torch.Tensor:
    r"""Compute REINFORCE policy gradient loss with batch-mean baseline.

    Collects all ``(log_prob, return)`` pairs from every trajectory, computes
    discounted returns via :func:`compute_returns`, subtracts the batch-mean
    return as a variance-reduction baseline, and returns

    .. math::

        \mathcal{L} = -\sum_{t} \log \pi(a_t \mid s_t) \cdot (G_t - \bar{G})

    Parameters
    ----------
    trajectories : list[list[tuple[Tensor, float]]]
        One list per puzzle; each element is a ``(log_prob, reward)`` pair
        where ``log_prob`` is a scalar tensor with a gradient.
    gamma : float
        Discount factor forwarded to :func:`compute_returns`.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor with a gradient.
    """
    all_log_probs: list[torch.Tensor] = []
    all_returns: list[float] = []

    for traj in trajectories:
        if not traj:
            continue
        rewards = [r for _, r in traj]
        returns = compute_returns(rewards, gamma)
        for (lp, _), g in zip(traj, returns, strict=True):
            all_log_probs.append(lp)
            all_returns.append(g)

    if not all_log_probs:
        return torch.zeros((), requires_grad=True)

    log_probs = torch.stack(all_log_probs)
    returns_t = torch.tensor(all_returns, dtype=torch.float32)

    # Batch-mean baseline for variance reduction
    advantages = returns_t - returns_t.mean()

    # Optionally normalise by std for training stability
    std = advantages.std()
    if std > 1e-8:
        advantages = advantages / std

    return -(log_probs * advantages).sum()


def train_one_update(
    puzzles: list[Data],
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    config: RLConfig,
    max_steps: int = 200,
) -> dict[str, float]:
    """Collect a rollout batch, compute REINFORCE loss, and apply one gradient update.

    Fresh :class:`~hashi_puzzle_solver.rl.env.HashiEnv` instances are created
    and reset from ``puzzles`` at the start of each call so the caller does
    not need to manage env state between updates.

    Parameters
    ----------
    puzzles : list[Data]
        Puzzle graphs; one env is created and reset per puzzle.
    model : torch.nn.Module
        Policy model (e.g. ``TransformerEdgeClassifier``).
    optimizer : Optimizer
        Optimizer to apply the gradient step.
    config : RLConfig
        Environment and reward configuration (also provides ``gamma``).
    max_steps : int
        Safety cap on total rollout iterations passed to
        :func:`~hashi_puzzle_solver.rl.rollout.collect_rollout`.

    Returns
    -------
    dict[str, float]
        ``{"loss": float}`` — the scalar REINFORCE loss for this update.
    """
    envs = [HashiEnv(config) for _ in puzzles]
    for env, puzzle in zip(envs, puzzles, strict=True):
        env.reset(puzzle)

    trajectories = collect_rollout(envs, model, max_steps=max_steps)
    loss = reinforce_loss(trajectories, gamma=config.gamma)

    # Restore training mode (collect_rollout sets model.eval())
    model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": float(loss.item())}


def _greedy_rollout(
    envs: list[HashiEnv],
    model: torch.nn.Module,
    max_steps: int = 200,
) -> list[dict[str, Any]]:
    """Deterministic greedy (argmax) rollout for evaluation.

    Parameters
    ----------
    envs : list[HashiEnv]
        Already-reset environments.
    model : torch.nn.Module
        Policy model evaluated in eval mode with no gradient.
    max_steps : int
        Safety cap on total rollout iterations.

    Returns
    -------
    list[dict[str, Any]]
        One dict per puzzle containing:

        - ``"rewards"``: ``list[float]``
        - ``"terminal_reason"``: ``str | None``
        - ``"steps"``: ``int``
    """
    n = len(envs)
    results: list[dict[str, Any]] = [
        {"rewards": [], "terminal_reason": None, "steps": 0} for _ in range(n)
    ]
    active: list[int] = list(range(n))

    with torch.no_grad():
        for _ in range(max_steps):
            if not active:
                break

            obs_list = [envs[i].get_obs() for i in active]
            edge_counts = [obs.edge_index.size(1) for obs in obs_list]
            batch_data = custom_collate_with_conflicts(obs_list)

            logits: torch.Tensor = model(
                batch_data.x,
                batch_data.edge_index,
                edge_attr=getattr(batch_data, "edge_attr", None),
                batch=getattr(batch_data, "batch", None),
                node_type=getattr(batch_data, "node_type", None),
            )

            logits_per_env = logits.split(edge_counts, dim=0)
            still_active: list[int] = []

            for local_idx, env_idx in enumerate(active):
                env = envs[env_idx]
                env_logits = logits_per_env[local_idx]

                fwd_idx = env.fwd_puzzle_indices
                assert fwd_idx is not None
                fwd_logits = env_logits[fwd_idx]

                if fwd_logits.dim() == 2:
                    scores = fwd_logits[:, 1] + fwd_logits[:, 2]
                else:
                    scores = fwd_logits  # pragma: no cover

                legal = env.legal_action_mask()
                scores = scores.masked_fill(~legal, float("-inf"))
                action = int(scores.argmax().item())

                _, reward, done, info = env.step(action)
                results[env_idx]["rewards"].append(float(reward))
                results[env_idx]["steps"] = int(results[env_idx]["steps"]) + 1

                if done:
                    results[env_idx]["terminal_reason"] = info["terminal_reason"]
                else:
                    still_active.append(env_idx)

            active = still_active

    return results


def evaluate(
    puzzles: list[Data],
    model: torch.nn.Module,
    config: RLConfig,
    max_steps: int = 200,
) -> dict[str, float]:
    """Evaluate model on puzzles using a greedy argmax policy.

    Parameters
    ----------
    puzzles : list[Data]
        Puzzle graphs to evaluate on.
    model : torch.nn.Module
        Policy model (placed in eval mode; no gradient is computed).
    config : RLConfig
        Environment configuration.
    max_steps : int
        Safety cap on greedy rollout steps per puzzle.

    Returns
    -------
    dict[str, float]
        Keys:

        - ``perfect_accuracy`` — fraction of puzzles solved (primary metric)
        - ``edge_acc`` — mean final edge-label accuracy across all puzzles
        - ``avg_episode_length`` — mean steps per episode
        - ``avg_solve_length`` — mean steps for solved episodes (0.0 if none solved)
        - ``avg_return`` — mean total reward per episode
        - ``oracle_failure_rate`` — fraction terminated by oracle violation
        - ``capacity_failure_rate`` — fraction terminated by capacity violation
        - ``crossing_failure_rate`` — fraction terminated by crossing violation
        - ``dead_end_unsolved_rate`` — fraction that reached a dead end without solving
        - ``max_steps_rate`` — fraction that hit the max-steps safety guard
    """
    model.eval()
    n = len(puzzles)

    zero_metrics: dict[str, float] = {
        "perfect_accuracy": 0.0,
        "edge_acc": 0.0,
        "avg_episode_length": 0.0,
        "avg_solve_length": 0.0,
        "avg_return": 0.0,
        "oracle_failure_rate": 0.0,
        "capacity_failure_rate": 0.0,
        "crossing_failure_rate": 0.0,
        "dead_end_unsolved_rate": 0.0,
        "max_steps_rate": 0.0,
    }
    if n == 0:
        return zero_metrics

    envs = [HashiEnv(config) for _ in puzzles]
    for env, puzzle in zip(envs, puzzles, strict=True):
        env.reset(puzzle)

    results = _greedy_rollout(envs, model, max_steps=max_steps)

    solved_count = 0
    oracle_fail = 0
    capacity_fail = 0
    crossing_fail = 0
    dead_end = 0
    max_steps_count = 0

    total_return = 0.0
    episode_lengths: list[int] = []
    solve_lengths: list[int] = []
    total_edge_correct = 0
    total_edges = 0

    for env, result in zip(envs, results, strict=True):
        tr: str | None = result["terminal_reason"]
        steps: int = result["steps"]
        rewards: list[float] = result["rewards"]

        episode_lengths.append(steps)
        total_return += sum(rewards)

        assert env.current_bridges is not None
        assert env.fwd_puzzle_indices is not None
        assert env.target_bridges is not None

        curr_fwd = env.current_bridges[env.fwd_puzzle_indices]
        n_edges = len(env.fwd_puzzle_indices)
        n_correct = int((curr_fwd == env.target_bridges).sum().item())
        total_edge_correct += n_correct
        total_edges += n_edges

        if tr == "solved":
            solved_count += 1
            solve_lengths.append(steps)
        elif tr == "oracle_failure":
            oracle_fail += 1
        elif tr == "capacity_failure":
            capacity_fail += 1
        elif tr == "crossing_failure":
            crossing_fail += 1
        elif tr == "dead_end_unsolved":
            dead_end += 1
        elif tr == "max_steps":
            max_steps_count += 1

    return {
        "perfect_accuracy": solved_count / n,
        "edge_acc": total_edge_correct / total_edges if total_edges > 0 else 0.0,
        "avg_episode_length": sum(episode_lengths) / n,
        "avg_solve_length": (
            sum(solve_lengths) / len(solve_lengths) if solve_lengths else 0.0
        ),
        "avg_return": total_return / n,
        "oracle_failure_rate": oracle_fail / n,
        "capacity_failure_rate": capacity_fail / n,
        "crossing_failure_rate": crossing_fail / n,
        "dead_end_unsolved_rate": dead_end / n,
        "max_steps_rate": max_steps_count / n,
    }


class ReinforceTrainer:
    """Wrapper class that manages REINFORCE updates and evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        Policy model.
    optimizer : Optimizer
        Parameter optimizer.
    config : RLConfig
        RL configuration (rewards, masking toggles, gamma).
    train_puzzles : list[Data]
        Puzzles used during training updates.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        config: RLConfig,
        train_puzzles: list[Data],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.train_puzzles = train_puzzles
        self.update_count: int = 0
        self.loss_history: list[float] = []

    def update(self, max_steps: int = 200) -> dict[str, float]:
        """Run one REINFORCE gradient update on all training puzzles.

        Parameters
        ----------
        max_steps : int
            Safety cap on rollout steps passed to :func:`train_one_update`.

        Returns
        -------
        dict[str, float]
            ``{"loss": float}``
        """
        metrics = train_one_update(
            self.train_puzzles,
            self.model,
            self.optimizer,
            self.config,
            max_steps=max_steps,
        )
        self.update_count += 1
        self.loss_history.append(metrics["loss"])
        return metrics

    def train(
        self,
        n_updates: int,
        max_steps: int = 200,
    ) -> list[dict[str, float]]:
        """Run ``n_updates`` consecutive REINFORCE gradient updates.

        Parameters
        ----------
        n_updates : int
            Number of gradient updates to perform.
        max_steps : int
            Safety cap on rollout steps per update.

        Returns
        -------
        list[dict[str, float]]
            Loss dict for each update, in order.
        """
        history = []
        for _ in range(n_updates):
            history.append(self.update(max_steps=max_steps))
        return history

    def evaluate(
        self,
        eval_puzzles: list[Data] | None = None,
        max_steps: int = 200,
    ) -> dict[str, float]:
        """Evaluate using greedy argmax policy.

        Parameters
        ----------
        eval_puzzles : list[Data] or None
            Puzzles to evaluate on; falls back to ``train_puzzles`` if ``None``.
        max_steps : int
            Safety cap on greedy rollout steps.

        Returns
        -------
        dict[str, float]
            Evaluation metrics dict from :func:`evaluate`.
        """
        puzzles = eval_puzzles if eval_puzzles is not None else self.train_puzzles
        return evaluate(puzzles, self.model, self.config, max_steps=max_steps)
