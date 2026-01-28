"""Configuration models for Hashi Puzzle Solver."""

from dataclasses import asdict, dataclass, field
import pathlib
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""

    root_dir: str = "dataset/"
    size: list[int] | None = None
    difficulty: int | None = None
    limit: int | None = None


@dataclass
class ModelConfig:
    """Configuration for GNN model architecture."""

    type: str = "transformer"
    node_embedding_dim: int = 64
    hidden_channels: int = 128
    num_layers: int = 4
    heads: int = 8
    dropout: float = 0.25

    # Legacy feature toggle (consider using use_structural_degree instead)
    use_degree: bool = False

    # Granular Embedding Dimensions
    edge_type_embedding_dim: int = 8
    capacity_embedding_dim: int = 16
    degree_embedding_dim: int = 8
    conflict_embedding_dim: int = 4
    unused_embedding_dim: int = 32
    closeness_embedding_dim: int = 16
    ap_embedding_dim: int = 4
    spectral_embedding_dim: int = 32
    distance_embedding_dim: int = 16
    logit_embedding_dim: int = 16

    # Noise Projection
    noise_embedding_dim: int = 16
    use_noise_in_message_passing: bool = False
    use_noise_in_prediction: bool = True
    use_noise_in_global_meta: bool = True

    # Edge MLP Multipliers
    edge_mlp_width_mult: float = 2.0
    edge_mlp_depth_mult: int = 2

    # Node Encoder Multipliers
    node_encoder_width_mult: float = 1.0
    node_encoder_depth_mult: int = 2

    # Noise Head Multipliers
    noise_mlp_width_mult: float = 1.0
    noise_mlp_depth_mult: int = 2

    # Meta Node Toggles
    use_global_meta_node: bool = True
    use_row_col_meta: bool = True
    use_meta_mesh: bool = True
    use_meta_row_col_edges: bool = True
    use_component_meta: bool = False
    use_hierarchical_component_meta: bool = False
    edge_concat_global_meta: bool = True
    edge_concat_component_meta: bool = False
    component_merge_margin: float = 0.5
    use_edge_features_in_prediction: bool = True

    # Edge Feature Toggles
    use_distance: bool = True
    use_conflict_edges: bool = True
    use_potential_crossing: bool = True
    use_categorical_edge_types: bool = True
    use_edge_labels_as_features: bool = False
    use_continuous_edge_labels: bool = True
    use_cut_edges: bool = True
    use_time_conditioning: bool = False
    time_noise_std: float = 0.0

    # Node encoder feature toggles
    use_structural_degree: bool = True
    use_structural_degree_nsew: bool = False
    use_capacity: bool = True
    use_unused_capacity: bool = True
    use_conflict_status: bool = True
    use_closeness_centrality: bool = True
    use_articulation_points: bool = True
    use_spectral_features: bool = True

    # Verification Head Toggles
    use_verification_head: bool = False
    verifier_use_puzzle_nodes: bool = False
    verifier_use_row_col_meta_nodes: bool = False
    use_noise_head: bool = True
    aux_predict_output_noise: bool = True


@dataclass
class LossWeightsConfig:
    """Weights for various loss components."""

    ce: float = 1.0
    degree: float = 0.0
    crossing: float = 0.0
    verify: float = 0.0
    noise: float = 0.0


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    monitor: str = "val_loss"
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for the training loop and optimizer."""

    mode: str = "one-shot"
    recursive_carryover: bool = False
    num_inference_steps_training: int = 1
    n_blocks: int = 1

    # Optimization Settings
    learning_rate: float = 0.001
    adam_epsilon: float = 1e-8
    weight_decay: float = 1e-5
    batch_size: int = 32
    accumulation_steps: int = 1
    epochs: int = 100
    device: str = "auto"
    num_workers: int = 0
    use_persistent_workers: bool = False

    # Diffusion specific settings
    sigma_max: float = 2.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    alpha_power: float = 1.0
    zero_signal_prob: float = 0.0
    diffusion_step_lr: float = 0.1
    flush_first_step: bool = False
    eval_rollout_interval: int = 10
    diffusion_max_steps: int = 20
    use_adaptive_sampler: bool = False

    # AR-specific training settings
    ar_max_steps: int = 100
    ar_k: int = 1
    ar_threshold: float = 0.5
    steps_per_epoch: int = 100
    gumbel_temperature: float = 1.0

    # Other settings
    model_dir: str = "models"

    # Nested configs
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    eval_interval: int = 1


@dataclass
class HashiModelConfig:
    """Root configuration object for Hashi Puzzle Solver."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "HashiModelConfig":
        """
        Create a HashiModelConfig from a nested dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns
        -------
            HashiModelConfig: Populated configuration object.
        """
        data_dict = config_dict.get("data", {})
        model_dict = config_dict.get("model", {})
        training_dict = config_dict.get("training", {})

        loss_weights_dict = training_dict.get("loss_weights", {})
        early_stopping_dict = training_dict.get("early_stopping", {})

        # Remove nested dicts from training_dict to build TrainingConfig
        training_base_dict = {
            k: v
            for k, v in training_dict.items()
            if k not in ["loss_weights", "early_stopping", "masking", "augmentation"]
        }

        return cls(
            data=DataConfig(**data_dict),
            model=ModelConfig(**model_dict),
            training=TrainingConfig(
                **training_base_dict,
                loss_weights=LossWeightsConfig(**loss_weights_dict),
                early_stopping=EarlyStoppingConfig(**early_stopping_dict),
            ),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | pathlib.Path) -> "HashiModelConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns
        -------
            HashiModelConfig: Populated configuration object.
        """
        with pathlib.Path(yaml_path).open() as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to a nested dictionary.

        Returns
        -------
            dict[str, Any]: Nested dictionary of configuration parameters.
        """
        return asdict(self)
