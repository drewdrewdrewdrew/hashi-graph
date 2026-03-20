"""Oracle-aware sequential RL environment for Hashi puzzle solving."""

from typing import Any

import torch
from torch_geometric.data import Data

from hashi_puzzle_solver.utils.train_utils import (
    get_unused_capacity_index,
    update_node_features,
)

from .config import RLConfig

_DEFAULT_MODEL_CONFIG: dict[str, bool] = {
    "use_capacity": True,
    "use_structural_degree": True,
    "use_unused_capacity": True,
}


class HashiEnv:
    """Oracle-aware sequential RL environment for a single Hashi puzzle.

    The agent selects one forward puzzle edge per step and increments its
    bridge count by one.  Episodes terminate on puzzle completion or on any
    rule or oracle violation.

    Parameters
    ----------
    config : RLConfig
        Reward values and action-masking toggles.
    model_config : dict[str, Any] or None
        Forwarded to ``update_node_features`` to locate the
        ``unused_capacity`` column in the node-feature matrix.  Defaults to
        ``{use_capacity, use_structural_degree, use_unused_capacity} = True``.
    """

    def __init__(
        self,
        config: RLConfig,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.model_config: dict[str, Any] = (
            model_config if model_config is not None else dict(_DEFAULT_MODEL_CONFIG)
        )

        # ── state fields ─────────────────────────────────────────────────
        # All are set / replaced in reset().  Until reset() is called the
        # env is considered done and no other method should be invoked.
        self.data: Data | None = None
        self._original_x: torch.Tensor | None = None
        self.current_bridges: torch.Tensor | None = None
        self.M: int = 0
        self.fwd_puzzle_indices: torch.Tensor | None = None
        self.target_bridges: torch.Tensor | None = None
        self.target_total_bridges: int = 0
        self.step_count: int = 0
        self.done: bool = True
        self.terminal_reason: str | None = None

    # ── public API ───────────────────────────────────────────────────────

    def reset(self, data: Data) -> tuple[Data, dict[str, int]]:
        """Reset the environment for a new puzzle episode.

        Parameters
        ----------
        data : Data
            Puzzle graph containing at least ``edge_index``, ``edge_mask``,
            ``y`` (target bridge counts), ``node_type``, and ``x``.

        Returns
        -------
        tuple[Data, dict[str, int]]
            ``(initial_obs, info)`` where *info* contains ``max_steps`` and
            ``num_puzzle_edges``.
        """
        self.data = data.clone()
        num_edges = self.data.edge_index.size(1)
        self.M = num_edges // 2
        self.current_bridges = torch.zeros(num_edges, dtype=torch.float)

        # Forward puzzle edges: j < M and edge_mask[j] is True
        fwd_mask = self.data.edge_mask[: self.M]
        self.fwd_puzzle_indices = torch.where(fwd_mask)[0]

        # Per-edge target counts for forward puzzle edges
        self.target_bridges = self.data.y[self.fwd_puzzle_indices].float()
        self.target_total_bridges = int(self.target_bridges.sum().item())

        # Cache original node features so update_node_features always
        # receives the clean initial state as its base.
        self._original_x = self.data.x.clone()

        # Initialise unused_capacity with zero bridges placed
        self.data.x = update_node_features(
            self._original_x,
            self.current_bridges,
            self.data.edge_index,
            self.data.node_type,
            self.model_config,
        )

        self.step_count = 0
        self.done = False
        self.terminal_reason = None

        return self.get_obs(), {
            "max_steps": self.target_total_bridges,
            "num_puzzle_edges": len(self.fwd_puzzle_indices),
        }

    def legal_action_mask(self) -> torch.Tensor:
        """Return a boolean mask over forward puzzle edges.

        Returns
        -------
        torch.Tensor
            Bool tensor of shape ``[num_fwd_puzzle_edges]``.  ``True`` means
            the action is currently selectable.
        """
        assert self.data is not None, "Call reset() before legal_action_mask()"
        assert self.current_bridges is not None
        assert self.fwd_puzzle_indices is not None

        k = len(self.fwd_puzzle_indices)
        mask = torch.ones(k, dtype=torch.bool)

        # ── always mask edges already at 2 bridges ────────────────────
        if self.config.mask_over_2:
            curr_fwd = self.current_bridges[self.fwd_puzzle_indices]
            mask &= curr_fwd < 2

        # ── optional capacity masking ─────────────────────────────────
        if self.config.mask_capacity and self.model_config.get(
            "use_unused_capacity", True
        ):
            unused_idx = get_unused_capacity_index(self.model_config)
            srcs = self.data.edge_index[0, self.fwd_puzzle_indices]
            dsts = self.data.edge_index[1, self.fwd_puzzle_indices]
            src_cap = self.data.x[srcs, unused_idx]
            dst_cap = self.data.x[dsts, unused_idx]
            mask &= (src_cap > 0) & (dst_cap > 0)

        # ── optional crossing masking ─────────────────────────────────
        if self.config.mask_crossing:
            conflicts = getattr(self.data, "edge_conflict_index", None)
            if conflicts is not None and conflicts.size(1) > 0:
                for idx in range(k):
                    if not mask[idx]:
                        continue
                    j_abs = int(self.fwd_puzzle_indices[idx].item())
                    involved = (conflicts[0] == j_abs) | (conflicts[1] == j_abs)
                    if not involved.any():
                        continue
                    partner_rows = conflicts[:, involved]
                    partners = torch.where(
                        partner_rows[0] == j_abs,
                        partner_rows[1],
                        partner_rows[0],
                    )
                    if self.current_bridges[partners].any():
                        mask[idx] = False

        return mask

    def step(self, action: int) -> tuple[Data, float, bool, dict[str, Any]]:
        """Increment the selected forward puzzle edge by one bridge.

        Parameters
        ----------
        action : int
            Index into the forward puzzle edges
            (``0 .. num_fwd_puzzle_edges - 1``).  Must satisfy
            ``legal_action_mask()[action] == True``.

        Returns
        -------
        tuple[Data, float, bool, dict[str, Any]]
            ``(next_obs, reward, done, info)`` where *info* contains
            ``terminal_reason`` (``str | None``) and ``step`` (``int``).
        """
        assert self.data is not None, "Call reset() before step()"
        assert not self.done, "Environment is done; call reset() to start a new episode"
        assert self.current_bridges is not None
        assert self.fwd_puzzle_indices is not None
        assert self._original_x is not None
        assert self.target_bridges is not None

        legal = self.legal_action_mask()
        assert legal[action], f"Action {action} is not in legal_action_mask"

        # Absolute index of the forward edge in the full edge_index
        j = int(self.fwd_puzzle_indices[action].item())

        # ── 2. Oracle check ───────────────────────────────────────────
        if self.current_bridges[j] >= self.target_bridges[action]:
            self.done = True
            self.terminal_reason = "oracle_failure"
            return (
                self.get_obs(),
                self.config.reward_failure,
                True,
                {"terminal_reason": self.terminal_reason, "step": self.step_count},
            )

        # ── 3. Capacity check (when masking is disabled) ──────────────
        if (
            not self.config.mask_capacity
            and self.model_config.get("use_unused_capacity", True)
        ):
            unused_idx = get_unused_capacity_index(self.model_config)
            src = int(self.data.edge_index[0, j].item())
            dst = int(self.data.edge_index[1, j].item())
            if self.data.x[src, unused_idx] <= 0 or self.data.x[dst, unused_idx] <= 0:
                self.done = True
                self.terminal_reason = "capacity_failure"
                return (
                    self.get_obs(),
                    self.config.reward_failure,
                    True,
                    {
                        "terminal_reason": self.terminal_reason,
                        "step": self.step_count,
                    },
                )

        # ── 4. Crossing check (when masking is disabled) ──────────────
        if not self.config.mask_crossing:
            conflicts = getattr(self.data, "edge_conflict_index", None)
            if conflicts is not None and conflicts.size(1) > 0:
                involved = (conflicts[0] == j) | (conflicts[1] == j)
                if involved.any():
                    partner_rows = conflicts[:, involved]
                    partners = torch.where(
                        partner_rows[0] == j,
                        partner_rows[1],
                        partner_rows[0],
                    )
                    if self.current_bridges[partners].any():
                        self.done = True
                        self.terminal_reason = "crossing_failure"
                        return (
                            self.get_obs(),
                            self.config.reward_failure,
                            True,
                            {
                                "terminal_reason": self.terminal_reason,
                                "step": self.step_count,
                            },
                        )

        # ── 5. Apply increment (bidirectional sync) ───────────────────
        self.current_bridges[j] += 1
        self.current_bridges[j + self.M] += 1

        # ── 6. Update unused_capacity in node features ────────────────
        self.data.x = update_node_features(
            self._original_x,
            self.current_bridges,
            self.data.edge_index,
            self.data.node_type,
            self.model_config,
        )

        self.step_count += 1

        # ── 7. Solved check ───────────────────────────────────────────
        curr_fwd = self.current_bridges[self.fwd_puzzle_indices]
        if torch.all(curr_fwd == self.target_bridges):
            self.done = True
            self.terminal_reason = "solved"
            reward = self.config.reward_correct + self.config.reward_solve
            return (
                self.get_obs(),
                reward,
                True,
                {"terminal_reason": self.terminal_reason, "step": self.step_count},
            )

        # Max-steps safety guard
        if self.step_count >= self.target_total_bridges:
            self.done = True
            self.terminal_reason = "max_steps"
            return (
                self.get_obs(),
                self.config.reward_correct,
                True,
                {"terminal_reason": self.terminal_reason, "step": self.step_count},
            )

        return (
            self.get_obs(),
            self.config.reward_correct,
            False,
            {"terminal_reason": None, "step": self.step_count},
        )

    def get_obs(self) -> Data:
        """Assemble the current state as a ``Data`` observation.

        Returns
        -------
        Data
            Clone of the current puzzle state with ``edge_attr[:, 3]`` set to
            the current per-edge bridge counts.
        """
        assert self.data is not None, "Call reset() before get_obs()"
        assert self.current_bridges is not None

        obs = self.data.clone()
        if obs.edge_attr is not None and obs.edge_attr.size(1) >= 1:
            obs.edge_attr = obs.edge_attr.clone()
            # Last column must match RLEdgeEncoder (bridge count in final dim).
            obs.edge_attr[:, -1] = self.current_bridges
        return obs
