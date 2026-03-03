import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.utils.diffusion_utils import inject_noise
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer

@pytest.fixture
def sample_puzzle_data():
    # Simple 2-node puzzle
    # capacity, structural_degree, unused_capacity
    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float) 
    # Bidirectional: 0-1 and 1-0
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # inv_dx, inv_dy, is_meta, bridge_label, is_labeled
    edge_attr = torch.zeros((2, 5), dtype=torch.float) 
    y = torch.tensor([1, 1], dtype=torch.long) # Ground truth: 1 bridge
    edge_mask = torch.tensor([True, True], dtype=torch.bool)
    node_type = torch.tensor([1, 1], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask, node_type=node_type)
    return data

def test_inject_noise_0(sample_puzzle_data):
    # 0 noise should mean input bridges == ground truth
    model_config = {
        "use_unused_capacity": True,
        "use_capacity": True,
        "use_structural_degree": True,
    }
    
    corrupted = inject_noise(
        sample_puzzle_data,
        noise_rate=0.0,
        bridge_label_idx=3,
        is_labeled_idx=4,
        model_config=model_config,
        device=torch.device("cpu")
    )
    
    assert torch.all(corrupted.edge_attr[:, 3] == sample_puzzle_data.y.float())
    assert torch.all(corrupted.edge_attr[:, 4] == 1.0)
    
    # unused_capacity should be original_capacity (2) - current_bridges = 2 - 1 = 1
    # Wait, in Hashi a bridge count of 1 means 1 bridge. 
    # unused_capacity = capacity - degree.
    # Island 0 has one neighbor 1 with 1 bridge. So degree is 1.
    # unused_capacity = 2 - 1 = 1.
    assert torch.all(corrupted.x[:, 2] == 1.0)

def test_inject_noise_1(sample_puzzle_data):
    # 1.0 noise should flip all original edges
    model_config = {
        "use_unused_capacity": True,
        "use_capacity": True,
        "use_structural_degree": True,
    }
    
    corrupted = inject_noise(
        sample_puzzle_data,
        noise_rate=1.0,
        bridge_label_idx=3,
        is_labeled_idx=4,
        model_config=model_config,
        device=torch.device("cpu")
    )
    
    # Ground truth was 1, so corrupted should be 0 or 2
    assert torch.all(corrupted.edge_attr[:, 3] != sample_puzzle_data.y.float())
    assert torch.all(torch.isin(corrupted.edge_attr[:, 3], torch.tensor([0.0, 2.0])))
    assert torch.all(corrupted.edge_attr[:, 4] == 1.0)

def test_diffusion_trainer_step(sample_puzzle_data):
    # Mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1) # dummy param
            self.use_verification_head = False
            self.use_noise_head = False

        def forward(
            self,
            x,
            edge_index,
            edge_attr=None,
            batch=None,
            node_type=None,
            **kwargs
        ):
            # Return dummy logits [num_edges, 3]
            num_edges = edge_index.size(1)
            logits = torch.randn(num_edges, 3, device=x.device, requires_grad=True)
            
            res = [logits]
            if kwargs.get("return_verification"):
                res.append(torch.randn(x.size(0) if batch is None else batch.max() + 1, 1, device=x.device, requires_grad=True))
            if kwargs.get("return_noise"):
                res.append(torch.randn(x.size(0) if batch is None else batch.max() + 1, 2, device=x.device, requires_grad=True))
            
            if len(res) == 1:
                return res[0]
            return tuple(res)

    config = {
        "data": {
            "root_dir": "dataset/",
        },
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_edge_labels_as_features": True,
            "use_verification_head": False,
            "use_noise_head": False,
        },
        "training": {
            "mode": "diff-discrete",
            "loss_weights": {"ce": 1.0, "degree": 0.0, "crossing": 0.0, "verify": 0.0},
            "num_inference_steps_training": 1,
            "learning_rate": 0.001,
            "batch_size": 1,
        }
    }
    
    device = torch.device("cpu")
    
    # BaseTrainer requires config, device
    trainer = DiffusionTrainer(config, device)
    trainer.model = MockModel().to(device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    
    batch = Batch.from_data_list([sample_puzzle_data])
    
    metrics = trainer.run_epoch(
        loader=[batch],
        epoch=1,
        total_epochs=1,
        training=True,
        noise_rate=0.5
    )
    
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] != 0


# ---------------------------------------------------------------------------
# Step-2 invariant: fresh_alphas must always be exactly zero in _prepare_mixed_batch
# ---------------------------------------------------------------------------

def _make_minimal_trainer():
    """Return a DiffusionTrainer with a mock model, no real data needed."""
    config = {
        "data": {"root_dir": "dataset/"},
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": False,
            "use_edge_labels_as_features": False,
            "use_continuous_edge_labels": True,
            "use_verification_head": False,
            "use_noise_head": False,
            "aux_predict_output_noise": False,
            "use_component_meta": False,
        },
        "training": {
            "mode": "diff-cont",
            "loss_weights": {"ce": 1.0, "degree": 0.0, "crossing": 0.0, "verify": 0.0, "noise": 0.0},
            "num_inference_steps_training": 1,
            "learning_rate": 0.001,
            "batch_size": 4,
            "sigma_max": 2.0,
            "scale_min": 4.0,
            "scale_max": 8.0,
            "zero_signal_prob": 1.0,
            "recursive_carryover": True,
        },
    }
    trainer = DiffusionTrainer(config, torch.device("cpu"))
    return trainer


def _make_cont_batch(n: int = 4):
    """Build a minimal PyG Batch of n puzzles with continuous logit edge attrs."""
    puzzles = []
    for _ in range(n):
        x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        # 8-channel edge attr: first 3 are logit slots, rest padding
        edge_attr = torch.zeros((2, 8), dtype=torch.float)
        y = torch.tensor([1, 1], dtype=torch.long)
        edge_mask = torch.tensor([True, True], dtype=torch.bool)
        node_type = torch.tensor([1, 1], dtype=torch.long)
        puzzles.append(
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                 edge_mask=edge_mask, node_type=node_type)
        )
    return Batch.from_data_list(puzzles)


def test_fresh_alphas_are_zero():
    """fresh_alphas returned by _prepare_mixed_batch must be exactly 0.0.

    Enforces the Step-2 sigma-invariant: every fresh puzzle enters with
    alpha=0 (pure noise start), never a stale random sample.
    """
    trainer = _make_minimal_trainer()
    # Monkey-patch bridge_logits_idx so inject_continuous_noise can run
    trainer.bridge_logits_idx = 0

    batch = _make_cont_batch(n=4)
    training_cfg = trainer.config["training"]

    _, alphas, sigmas, _, _ = trainer._prepare_mixed_batch(batch, training_cfg, training=True)

    assert torch.all(alphas == 0.0), (
        f"fresh_alphas should be exactly zero; got {alphas}"
    )


def test_fresh_sigmas_equal_sigma_max():
    """fresh_sigmas must equal sigma_max for every fresh puzzle."""
    trainer = _make_minimal_trainer()
    trainer.bridge_logits_idx = 0

    sigma_max = trainer.config["training"]["sigma_max"]
    batch = _make_cont_batch(n=4)
    training_cfg = trainer.config["training"]

    _, _, sigmas, _, _ = trainer._prepare_mixed_batch(batch, training_cfg, training=True)

    assert torch.all(sigmas == sigma_max), (
        f"fresh_sigmas should all equal sigma_max={sigma_max}; got {sigmas}"
    )


# ---------------------------------------------------------------------------
# Step-3 invariant: _refill_buffer must handle dict noise_pred gracefully
# ---------------------------------------------------------------------------

def _buffer_training_cfg():
    """Return a minimal training_cfg that allows _refill_buffer to populate."""
    return {
        "mode": "diff-cont",
        "loss_weights": {"ce": 1.0, "degree": 0.0, "crossing": 0.0, "verify": 0.0, "noise": 0.0},
        "zero_signal_prob": 0.0,  # carry-over fraction = 1.0, so buffer fills
        "sigma_max": 2.0,
        "scale_min": 4.0,
        "scale_max": 8.0,
        "recursive_carryover": True,
    }


def test_refill_buffer_tensor_noise_pred():
    """_refill_buffer must accept a plain tensor noise_pred without error."""
    trainer = _make_minimal_trainer()
    trainer.bridge_logits_idx = 0

    batch = _make_cont_batch(n=2)
    logits = torch.zeros(batch.edge_index.size(1), 3)
    scales = torch.ones(batch.num_graphs)
    noise_pred = torch.zeros(batch.num_graphs, 2)  # tensor case

    trainer._refill_buffer(batch, logits, scales, _buffer_training_cfg(), noise_pred=noise_pred)

    assert len(trainer.carry_over_buffer_train) > 0
    for _, p_noise, _ in trainer.carry_over_buffer_train:
        assert p_noise.shape == (2,)


def test_refill_buffer_dict_noise_pred():
    """_refill_buffer must extract noise_pred['global'] when model returns a dict.

    Reproduces the crash that occurred in hierarchical mode where
    `noise_pred.detach()` fails on a plain Python dict.
    """
    trainer = _make_minimal_trainer()
    trainer.bridge_logits_idx = 0

    batch = _make_cont_batch(n=2)
    logits = torch.zeros(batch.edge_index.size(1), 3)
    scales = torch.ones(batch.num_graphs)
    noise_pred = {
        "global": torch.zeros(batch.num_graphs, 2),
        "component": torch.zeros(batch.num_graphs, 4),
    }

    trainer._refill_buffer(batch, logits, scales, _buffer_training_cfg(), noise_pred=noise_pred)

    assert len(trainer.carry_over_buffer_train) > 0
    for _, p_noise, _ in trainer.carry_over_buffer_train:
        assert p_noise.shape == (2,), (
            f"Expected per-graph global noise of shape (2,), got {p_noise.shape}"
        )


def test_refill_buffer_dict_noise_pred_values():
    """Global noise values extracted from a dict must match the 'global' tensor."""
    trainer = _make_minimal_trainer()
    trainer.bridge_logits_idx = 0

    batch = _make_cont_batch(n=2)
    logits = torch.zeros(batch.edge_index.size(1), 3)
    scales = torch.ones(batch.num_graphs)

    global_tensor = torch.tensor([[0.5, 0.3], [0.1, 0.9]])
    noise_pred = {"global": global_tensor, "component": torch.zeros(2, 4)}

    trainer._refill_buffer(batch, logits, scales, _buffer_training_cfg(), noise_pred=noise_pred)

    collected = torch.stack([p_noise for _, p_noise, _ in trainer.carry_over_buffer_train])
    assert torch.allclose(collected, global_tensor), (
        f"Buffer noise should equal noise_pred['global']; got {collected}"
    )
