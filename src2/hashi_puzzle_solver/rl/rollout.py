"""Batched active-only rollout for RL policy collection."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributions import Categorical

from hashi_puzzle_solver.rl.env import HashiEnv
from hashi_puzzle_solver.utils.common import custom_collate_with_conflicts

_DEFAULT_MAX_STEPS = 200


def collect_rollout(
    envs: list[HashiEnv],
    model: torch.nn.Module,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> list[list[tuple[torch.Tensor, float]]]:
    """Collect model policy rollout trajectories using batched active-only stepping.

    For each rollout step, only the still-active environments are collated into
    a single batch and forwarded through ``model``.  Finished environments are
    dropped immediately and not stepped again.

    Action scores are derived from the model's 3-class edge logits as
    ``logits[:, 1] + logits[:, 2]``, expressing the model's belief that a
    given forward puzzle edge still needs at least one more bridge.

    Parameters
    ----------
    envs : list[HashiEnv]
        Already-reset environments.  Each env must have been reset before
        calling this function.
    model : torch.nn.Module
        ``TransformerEdgeClassifier`` (or compatible) producing per-edge logits
        of shape ``[num_edges, 3]``.
    max_steps : int
        Safety cap on total rollout steps (applied per global iteration, not per
        env).  All active envs are stepped once per iteration.

    Returns
    -------
    list[list[tuple[torch.Tensor, float]]]
        One list per input puzzle.  Each element is a ``(log_prob, reward)``
        pair where ``log_prob`` is a scalar Tensor with a gradient.
    """
    model.eval()
    n = len(envs)
    trajectories: list[list[tuple[torch.Tensor, float]]] = [[] for _ in range(n)]

    # Active env indices (into the original `envs` list)
    active: list[int] = list(range(n))

    for _ in range(max_steps):
        if not active:
            break

        obs_list = [envs[i].get_obs() for i in active]
        edge_counts = [obs.edge_index.size(1) for obs in obs_list]

        batch_data = custom_collate_with_conflicts(obs_list)

        # Forward pass — keep gradients so log_prob can flow back
        logits: torch.Tensor = model(
            batch_data.x,
            batch_data.edge_index,
            edge_attr=getattr(batch_data, "edge_attr", None),
            batch=getattr(batch_data, "batch", None),
            node_type=getattr(batch_data, "node_type", None),
        )

        # Split logits back into per-puzzle chunks (same order as `active`)
        logits_per_env: tuple[torch.Tensor, ...] = logits.split(edge_counts, dim=0)

        still_active: list[int] = []

        for local_idx, env_idx in enumerate(active):
            env = envs[env_idx]
            env_logits = logits_per_env[local_idx]  # [num_edges, 3]

            # Extract forward puzzle edges and derive a scalar score per action
            fwd_idx = env.fwd_puzzle_indices
            assert fwd_idx is not None
            fwd_logits = env_logits[fwd_idx]  # [num_fwd, 3]

            if fwd_logits.dim() == 2:
                # Score = model belief that this edge needs ≥ 1 more bridge
                scores = fwd_logits[:, 1] + fwd_logits[:, 2]
            else:
                scores = fwd_logits  # pragma: no cover

            # Mask illegal actions with -inf before sampling
            legal = env.legal_action_mask()
            scores = scores.masked_fill(~legal, float("-inf"))

            dist = Categorical(logits=scores)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            _, reward, done, _ = env.step(int(action.item()))
            trajectories[env_idx].append((log_prob, float(reward)))

            if not done:
                still_active.append(env_idx)

        active = still_active

    return trajectories


def random_policy_rollout(
    envs: list[HashiEnv],
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> list[dict[str, Any]]:
    """Collect rollout trajectories using a uniform-random legal-action policy.

    Samples uniformly over legal actions at every step.  Used as a sanity
    baseline against which the model policy is compared.

    Parameters
    ----------
    envs : list[HashiEnv]
        Already-reset environments.
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

    for _ in range(max_steps):
        if not active:
            break

        still_active: list[int] = []

        for env_idx in active:
            env = envs[env_idx]
            legal = env.legal_action_mask()
            legal_actions = legal.nonzero(as_tuple=True)[0]

            if legal_actions.numel() == 0:
                # No legal actions remain — treat as a stuck terminal
                results[env_idx]["terminal_reason"] = "no_legal_actions"
                continue

            # Uniform sample over legal actions
            pick = int(torch.randint(0, legal_actions.numel(), (1,)).item())
            action = int(legal_actions[pick].item())

            _, reward, done, info = env.step(action)
            results[env_idx]["rewards"].append(float(reward))
            results[env_idx]["steps"] = int(results[env_idx]["steps"]) + 1

            if done:
                results[env_idx]["terminal_reason"] = info["terminal_reason"]
            else:
                still_active.append(env_idx)

        active = still_active

    return results
