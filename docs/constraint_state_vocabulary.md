# Constraint State Vocabulary

## Motivation

The current `NodeEncoder` embeds `structural_degree`, `capacity`, and `unused_capacity` as independent features — two categorical `nn.Embedding` tables and one `nn.Linear(1, dim)` — then concatenates them and hopes the refiner MLP learns their interactions.

The problem: the *deductive implications* of these features are step-functions of their *combination*, not smooth functions of each feature independently. For example, a node with degree=1 and net_capacity=2 is **forced** to place a double bridge on its only neighbor. That constraint is a discrete fact arising from the pair (1, 2), not something that interpolates smoothly from nearby values. The `Linear` on unused_capacity is particularly wrong — it imposes a continuity prior on a quantity whose semantics change qualitatively at specific thresholds (positive → unsatisfied, zero → satisfied, negative → over-saturated).

## The Vocabulary

Replace the three separate embeddings with a single `nn.Embedding` over the joint state `(degree, net_capacity)`:

- **degree** `d ∈ {1, 2, 3, 4}` — number of potential bridge directions (static per island per puzzle)
- **net_capacity** `n` — remaining bridges needed (`original_capacity - bridges_placed`); integer-valued since all call sites discretize via `argmax` before `update_node_features`; can go negative on over-saturation

### Range and size

| net_capacity range | bins | × 4 degrees | total entries |
|--------------------|------|-------------|---------------|
| [-7, 7]            | 15   | 4           | **60**        |

Values outside the range clamp to the nearest bin. 60 entries is small enough that overfitting is not a concern.

### Index computation

```python
NC_MIN = -7
NC_MAX = 7
NC_BINS = NC_MAX - NC_MIN + 1  # 15

def constraint_vocab_index(degree: int, net_capacity: int) -> int:
    n = max(NC_MIN, min(NC_MAX, net_capacity))
    return (degree - 1) * NC_BINS + (n - NC_MIN)
```

### Encoding

```python
self.constraint_vocab = nn.Embedding(4 * NC_BINS, vocab_dim)
```

This replaces `capacity_embedding`, `degree_embedding`, and `unused_embedding` (or can be concatenated alongside them during an A/B comparison).

## What each state encodes

The vocabulary gives every constraint situation its own learned vector. Key state classes:

| state pattern | meaning |
|---|---|
| `(*, 0)` | **satisfied** — node complete, don't touch |
| `(*, n<0)` | **over-saturated** — bridges need removing |
| `(1, n)` for `n>0` | **forced** — only 1 direction, must place exactly `n` there (and `n ≤ 2`) |
| `(d, n)` where `n > (d-1)×2` | **pigeonhole** — at least `n − (d−1)×2` bridges forced in every direction |
| `(d, d×2)` | **fully forced** — 2 bridges in every direction |
| `(d, 1)` | **almost done** — exactly 1 bridge left to place |

These are the exact deductive rules a human Hashi solver applies. With separate embeddings, the model must discover these interaction patterns through MLP weight composition. With the vocabulary, each pattern is a distinct token that receives its own gradient signal from the first training step.

## Why not extend to edge-level pair vocabulary?

A natural extension: for each edge, encode `(source_state, target_state)` as a pair embedding (52² ≈ 2,700 entries). However, the `EdgeHead` already concatenates `[src_h, dst_h]` and feeds them through an MLP — it is already a learned pairwise interaction function over post-GNN node representations (which contain the vocabulary embedding). An additional edge-level vocabulary would duplicate this mechanism. The node-level vocabulary is high-value because it encodes *local, self-contained* constraint state that exists before any message passing. Pairwise interactions are what the GNN + EdgeHead are designed for.

## Why not encode full neighborhoods?

A degree-4 node has 4 neighbors, each in one of ~68 states. The number of possible neighborhood multisets (~870K) is too large for a flat embedding and would require aggregation — which is literally message passing. The vocabulary works precisely because it captures pre-GNN, per-node knowledge. Once you need neighbor information, the GNN is the right tool.

## Capacity as a separate axis

The original capacity is *not* needed as a third axis. During iterative solving, the constraint logic depends only on how many bridges are still needed (net_capacity) and how many directions are available (degree). Two islands both at `net_capacity=3` with `degree=2` face identical constraints regardless of whether they started at `capacity=3` or `capacity=7`. The original capacity is an artifact of the initial state, fully subsumed by the (degree, net_capacity) pair.
