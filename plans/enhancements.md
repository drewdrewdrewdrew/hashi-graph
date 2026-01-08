# Hashi Graph Model Enhancements

This document outlines strategies to improve the performance of the Graph Neural Network (GNN) for solving Hashi puzzles. The suggestions range from simple configuration changes to architectural overhauls.

## 1. Data & Feature Engineering

The current model treats the puzzle as a generic graph, ignoring key geometric and directional properties of Hashi.

### A. Edge Features (Distance)
*   **Current:** No edge attributes are used. All connections are treated as equal "1-hop" neighbors, regardless of whether they span 2 units or 8 units.
*   **Recommendation:** Calculate Euclidean or Manhattan distance between connected islands during graph creation.
*   **Transformation Evaluation:** Compare two strategies for handling OOD generalization (Train on 8x8, Test on 50x50):
    1.  **Inverse Distance:** ($1/d$) Maps all distances to $[0, 1]$, treating large unseen distances as "weak connections" close to 0. Safe and simple.
    2.  **Sinusoidal Encoding:** Maps distance to a vector of sine/cosine waves (like Transformers). Converts unbounded scalar extrapolation into bounded pattern matching $[-1, 1]$.
*   **Risks/Drawbacks:** 
    *   **Overfitting to Grid Size:** If raw distance is used, the model might fail on larger puzzles where valid connections are longer than anything seen in training. The proposed transformations (Inverse/Sinusoidal) mitigate this but add complexity.
    *   **Computational Cost:** Sinusoidal encoding increases the dimensionality of edge features.

### B. Directional Features (Egocentric)
*   **Current:** A generic GCN/GAT aggregates all neighbors into a single soup. The model cannot distinguish a "North" neighbor from an "East" neighbor.
*   **Recommendation:** Encode the **Egocentric Direction** of each edge.
*   **Implementation:** 
    *   For edge $u \to v$, calculate direction relative to $u$ (e.g., if $v$ is above $u$, direction is North).
    *   For the reverse edge $v \to u$, the direction is inverted (South).
    *   Encode as discrete categories (N, S, E, W) on the edge.
*   **Why:** Helps the model distinguish that satisfying a "3" node might require specific directional combinations. It grounds the graph in the physical grid layout.
*   **Risks/Drawbacks:** 
    *   **Symmetry Breaking:** The model might learn biases (e.g., "always prefer North bridges") if the training data isn't perfectly rotationally invariant.
    *   **Requirement:** Requires switching to an edge-aware GNN (e.g., `GINEConv`) to utilize these features. Standard `GCNConv` will ignore them.

### C. Conflict Modeling
*   **Current:** The graph contains all potential edges. If a vertical edge and horizontal edge cross geometrically, the GNN has no explicit knowledge that they are mutually exclusive.
*   **Recommendation:** Implement **Conflict Loss** (Soft Physics Approach).
*   **Implementation:** 
    *   Pre-compute pairs of edges that geometrically intersect.
    *   During training, add an auxiliary loss term: `Loss_conflict = Sum(Prob(e1) * Prob(e2))` for all crossing pairs.
*   **Why:** If the model predicts high probability for two crossing edges, the loss explodes. This forces the model to "pick a winner" and learn the exclusion principle directly from optimization signals.
*   **Alternative (Ghost Nodes):** Creating nodes at intersection points to manage "traffic flow." Dropped for now due to graph complexity, but conceptually powerful.
*   **Risks/Drawbacks:**
    *   **Computation:** Requires $O(E^2)$ check to find crossings (done once per puzzle) and calculating the sum during training. Manageable for Hashi sizes.

### D. Dynamic Unused Capacity (Node Features)
*   **Concept:** For any island node, the most critical piece of information is how many bridges are *already* placed vs. how many *remain* to be placed.
*   **Implementation (One-Shot with Masking):**
    *   In the progressive masking setup, some edges are "visible" (ground truth labels provided as features).
    *   Calculate `known_degree` for each node using only the visible edges.
    *   Update node features to include `remaining_capacity = initial_capacity - known_degree`.
*   **Why:** This explicitly encodes the arithmetic of Hashi into the node embeddings. It allows the model to reason: "This island needs 3 bridges, I see 1 bridge on a visible edge, so I must find 2 more among my masked edges." 
*   **Inference Note:** At 100% masking (validation/testing), `known_degree` is 0 and `remaining_capacity` equals `initial_capacity`, which is consistent with the starting state of a puzzle. This avoids distributional shift while providing a powerful hint during the easier phases of curriculum learning.

## 2. Configuration & Hyperparameters

The current configuration (`base_config.yaml`) is likely too shallow for the reasoning required.

### A. Embedding Dimension
*   **Current:** `8` or `16`. This is extremely small, forcing the model to compress island capacity, degree, and propagated constraints into a tiny vector.
*   **Recommendation:** Increase to **64**. This allows the model to encode island capacity, current degree, and complex propagated constraints.
*   **Risks/Drawbacks:**
    *   **Overfitting:** Larger embeddings mean more parameters, which requires more data or stronger regularization (dropout) to prevent overfitting on small datasets.

### B. Network Depth (Receptive Field)
*   **Current:** `2` layers. The model only "sees" neighbors 2 hops away.
*   **Recommendation:** Increase to **6-10 layers**. Hashi puzzles require global constraint propagation (e.g., a corner decision affecting the opposite side).
*   **Requirement:** To train deep networks, **Residual Connections (Skip Connections)** must be added to the model architecture to prevent oversmoothing and vanishing gradients.
*   **Risks/Drawbacks:**
    *   **Oversmoothing:** Without proper residuals or architecture (like GatedGCN), deep GNNs tend to make all node representations identical, degrading performance.
    *   **Vanishing Gradients:** Harder to train without careful initialization and normalization.

### C. Class Imbalance Handling
*   **Current:** Standard Cross Entropy Loss. Since most potential connections are `0` (no bridge), the model is biased towards predicting "no bridge".
*   **Recommendation:** Use weighted `CrossEntropyLoss` in `train.py`.
*   **Implementation:** Calculate class frequencies and assign higher weights to classes `1` and `2`.
*   **Risks/Drawbacks:**
    *   **False Positives:** Aggressive weighting might cause the model to hallucinate bridges where there shouldn't be any, violating constraints.

## 3. Architecture

### A. Edge-Conditioned Convolutions
*   **Current:** `GCNConv` ignores edge attributes entirely. `GATConv` *can* use them but isn't currently configured to do so robustly for geometric data.
*   **Recommendation:** Switch to **`GINEConv`** (Graph Isomorphism Network with Edge features) or **`NNConv`**.
*   **Why:** Allows message passing to be modulated by bridge distance and direction.
*   **Risks/Drawbacks:**
    *   **Implementation Complexity:** These layers are more complex to implement and tune than standard GCN.

### B. Graph Transformer / Global Attention
*   **Current:** A single "Meta Node" connects to all others. This creates an information bottleneck where all global context must squeeze through one vector.
*   **Recommendation:** Use **`TransformerConv`** or a full Graph Transformer architecture.
*   **Why:** Allows every node to attend to every other node, capturing global counting constraints without a bottleneck.
*   **Risks/Drawbacks:**
    *   **Memory Usage:** Full attention is $O(N^2)$, which is fine for small Hashi puzzles (<100 nodes) but could scale poorly if you move to massive puzzles.

### C. Row/Column Meta Nodes
*   **Current:** Global Meta Node (connects to everyone) or None.
*   **Recommendation:** Create specific meta nodes for each row and each column.
*   **Enhancement:** Create a **"Meta-Grid"** by connecting:
    *   Row Meta $i$ $\leftrightarrow$ Row Meta $i+1$ (North/South)
    *   Col Meta $j$ $\leftrightarrow$ Col Meta $j+1$ (East/West)
    *   *Why:* Allows constraints to propagate sequentially across the grid backbone, reusing standard distance/direction embeddings.
*   **Why:** Explicitly models the "line of sight" constraint, allowing efficient communication along rows/cols (2 hops).
*   **Risks/Drawbacks:**
    *   **Graph Size:** Adds $2 \times \text{Size}$ extra nodes and many edges, increasing graph size.
    *   **Redundancy:** Might provide redundant information if the GNN is already deep enough to propagate along the grid.

### D. Conflict Modeling
A fundamental rule is that bridges cannot cross.
*   **Current:** The graph contains all potential edges. If a vertical edge and horizontal edge cross geometrically, the GNN has no explicit knowledge that they are mutually exclusive.
*   **Recommendation 1 (Structural):** Add **Conflict Edges** between nodes whose potential bridges would cross.
    *   If Edge A (1-2) crosses Edge B (3-4), add edges (1,3), (1,4), (2,3), (2,4) with a special `edge_type="collision"`.
    *   *Why:* Allows explicit message passing ("My path is blocked by yours").
*   **Recommendation 2 (Loss):** Implement **Conflict Loss** (Soft Physics Approach).
    *   During training, add an auxiliary loss term: `Loss_conflict = Sum(Prob(e1) * Prob(e2))` for all crossing pairs.
    *   *Why:* Forces the model to "pick a winner" via optimization pressure.
*   **Risks/Drawbacks:**
    *   **Graph Density:** Structural approach adds edges. If crossings are frequent, this creates cliques that slow down processing.
    *   **Computation:** Loss approach requires $O(E^2)$ pre-computation of crossings.

### E. Constraint-Aware Auxiliary Losses

Beyond standard cross-entropy on edge labels, we can add auxiliary losses that explicitly encode Hashi puzzle rules. These act as "soft constraints" that guide the model toward valid solutions.

#### 1. Degree Violation Loss (Island Counting)
*   **Current:** Edges are predicted independently. The model doesn't "know" that the sum of bridges must equal the island's number.
*   **Implementation:**
    ```python
    # For each node, sum predicted bridge values from incident edges
    predicted_degree = scatter_add(edge_predictions, edge_index[0])
    target_degree = node_features['n']  # Island capacity (1-8)
    loss_degree = F.mse_loss(predicted_degree, target_degree)
    
    # Or use L1 for sparsity:
    loss_degree = F.l1_loss(predicted_degree, target_degree)
    ```
*   **Why:** Forces the model to "learn to count" and respect the fundamental constraint: `sum(bridges) = island_capacity`.
*   **Weighting:** Start with `λ_degree = 0.1` and tune. Too high causes conflict with CE loss early in training.
*   **Risks/Drawbacks:**
    *   **Optimization Conflict:** The counting loss might conflict with the classification loss early in training, making optimization unstable. Needs careful weighting.
    *   **Masking Interaction:** During progressive masking, should we count predicted values for masked edges or ground truth for visible ones? (Recommendation: Use predictions for ALL edges to force true inference.)

#### 2. Bridge Crossing Loss (Mutual Exclusion)
*   **Current:** Model can predict high probabilities for two edges that geometrically intersect, violating the no-crossing rule.
*   **Implementation:**
    ```python
    # Pre-compute crossing edge pairs (e_i, e_j) during data creation
    for (edge_i, edge_j) in crossing_pairs:
        # Get predicted probabilities for non-zero bridges
        prob_i = softmax(logits[edge_i])[1:].sum()  # P(bridge exists)
        prob_j = softmax(logits[edge_j])[1:].sum()  # P(bridge exists)
        
        # Penalize both being active
        loss_crossing += prob_i * prob_j
    
    loss_crossing = loss_crossing / len(crossing_pairs)
    ```
*   **Why:** Soft physics constraint - if both edges have high probability, the loss explodes. Model learns to "pick a winner" automatically.
*   **Alternative:** Harder constraint using logits directly:
    ```python
    # Penalize max logit (strongest prediction) for both edges
    loss_crossing = F.relu(logits[edge_i].max() + logits[edge_j].max() - threshold)
    ```
*   **Weighting:** Start with `λ_crossing = 0.5` - this is a critical constraint.
*   **Efficiency Note:** Pre-compute crossing pairs once during dataset creation (stored in `edge_conflicts`). We already have this infrastructure!
*   **Risks/Drawbacks:**
    *   **Optimization Instability:** Multiplicative terms can have vanishing/exploding gradients. Consider using `log(prob_i * prob_j)` or clamping.
    *   **Computational Cost:** O(E²) to find crossings (one-time), O(C) per batch where C = number of crossing pairs.

#### 3. Connectivity Loss (Graph Must Be Connected)
*   **Current:** Model can predict disconnected solutions (island subgraphs that don't connect to the main component).
*   **Implementation (Hard - Requires Graph Ops):**
    ```python
    # Construct predicted graph from edge predictions
    predicted_edges = (edge_preds > 0).nonzero()  # Edges with bridges
    
    # Use torch_geometric utilities or networkx
    num_components = compute_connected_components(predicted_edges, num_nodes)
    
    # Penalize multiple components
    loss_connectivity = F.relu(num_components - 1)  # Should be 1 component
    ```
*   **Implementation (Soft - Spectral Approximation):**
    ```python
    # Use graph Laplacian eigenvalues
    # For connected graph, 2nd smallest eigenvalue (Fiedler value) > 0
    L = compute_laplacian(edge_index, edge_weights=predicted_probs)
    eigenvalues = torch.linalg.eigvalsh(L)
    fiedler_value = eigenvalues[1]
    
    loss_connectivity = F.relu(-fiedler_value + epsilon)  # Encourage > 0
    ```
*   **Why:** Ensures global connectivity, preventing "island within an island" solutions.
*   **Weighting:** Start with `λ_connectivity = 0.2` - important but harder to compute.
*   **Risks/Drawbacks:**
    *   **Computational Cost:** Graph algorithms or eigenvalue decomposition are expensive. Consider computing only every N steps or on a subset of batch.
    *   **Differentiability:** Hard version (counting components) is not differentiable. Soft version approximates with spectral methods but adds complexity.
    *   **May Be Unnecessary:** With 21k training samples, the model might learn connectivity implicitly. Consider this an "advanced" loss to add if you plateau.

#### 4. Edge Symmetry Loss (Bidirectional Consistency)
*   **Current:** We predict labels for both `(u,v)` and `(v,u)` independently. The model could predict different values for the same edge!
*   **Problem:** This is nonsensical - an edge has ONE state (0, 1, or 2 bridges), not two.
*   **Options:**

    **Option A: Enforce in Loss (Soft Constraint)**
    ```python
    # For each bidirectional edge pair
    edge_forward = edge_index[0]  # u -> v
    edge_backward = edge_index[1]  # v -> u (if stored separately)
    
    # Find reverse edge indices (requires index mapping)
    reverse_idx = find_reverse_edges(edge_index)
    
    # Penalize prediction mismatch
    logits_fwd = model_output[edge_forward]
    logits_bwd = model_output[reverse_idx]
    
    loss_symmetry = F.kl_div(
        F.log_softmax(logits_fwd), 
        F.softmax(logits_bwd),
        reduction='batchmean'
    )
    ```
    *Pros:* Flexible, model can learn to be consistent  
    *Cons:* Adds computational overhead, may not fully solve the issue

    **Option B: Average Predictions (Post-Processing)**
    ```python
    # At inference time, average the two predictions
    logits_fwd = model_output[edge_idx]
    logits_bwd = model_output[reverse_idx]
    logits_avg = (logits_fwd + logits_bwd) / 2
    final_pred = logits_avg.argmax()
    ```
    *Pros:* Simple, guaranteed consistency at inference  
    *Cons:* Training still wastes capacity learning redundant info

    **Option C: Structural Change (Predict Once Per Undirected Edge)**
    ```python
    # Only keep forward edges (u < v), remove backward edges
    # At data loading time, filter edge_index to canonical direction
    edge_mask = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, edge_mask]
    y = y[edge_mask]
    
    # Message passing still uses bidirectional graph
    # But predictions/loss only computed on forward edges
    ```
    *Pros:* Most efficient, no redundancy, exact consistency  
    *Cons:* Requires rethinking edge_mask logic, more invasive change

*   **Recommendation:** **Option C (Structural)** is cleanest for new training runs. **Option B (Averaging)** is quickest band-aid for existing models.
*   **Discussion:**
    *   Current approach predicts 2× the edges needed (redundant work)
    *   With 21k samples, the model might learn approximate symmetry on its own
    *   BUT: For 100% accuracy goal, explicit consistency is important
    *   Consider implementing Option C when you do next round of dataset regeneration (after clearing processed cache)
    * YOU CAN DO BOTH! CHOOSE ONE FOR TRAINING, TAKE AN AVERAGE FOR INFERENCE
* TEST WHETHER THIS IS NECESSARY BY LOOKING AT CORRELATION BETWEEN THE DIRECTIONS

#### Combined Loss Function
```python
total_loss = loss_ce + \
             λ_degree * loss_degree + \
             λ_crossing * loss_crossing + \
             λ_connectivity * loss_connectivity + \
             λ_symmetry * loss_symmetry

# Suggested starting weights:
# λ_degree = 0.1
# λ_crossing = 0.5
# λ_connectivity = 0.2 (optional)
# λ_symmetry = 0.1 (if using Option A)
```

**Phased Introduction:** Don't add all losses at once. Start with degree + crossing (easiest to implement), then add others if you plateau.

## 4. Adaptive Curriculum Learning (Progressive Masking)

The current implementation uses a **fixed schedule** (cosine/linear) for ramping masking difficulty. However, this can be too aggressive if the model hasn't fully learned at the current difficulty level. **Adaptive curriculum learning** adjusts difficulty based on model performance, ensuring the model "stays behind the ramp."

### Goal: Train Accuracy Should Stay High Throughout

**Ideal trajectory:**
```
Train Acc:  100% ────────────────── 95-100% (stays high!)
Val Acc:     40% ───────────→→→───→  95-100% (catches up!)
                                         ↑
                                    Convergence!
Masking:      0% ───→───→───→───→→  100%
            Warmup         Gradual Ramp
```

If train accuracy drops significantly as masking increases (e.g., 100% → 50%), the curriculum is too aggressive or the model isn't learning at each step.

### Technique 1: Performance-Gated Progression

Only increase masking when model hits a performance threshold.

```python
def get_adaptive_masking_rate(epoch, current_train_acc, masking_state):
    """
    Only increase masking if train accuracy is high enough.
    """
    current_rate = masking_state['current_rate']
    
    # Gate: Don't increase masking unless train acc > 90%
    if current_train_acc > 0.90:
        # Model is comfortable, increase difficulty
        current_rate += 0.02  # Small increment
        current_rate = min(current_rate, 1.0)
    elif current_train_acc < 0.80:
        # Model struggling, decrease difficulty!
        current_rate = max(current_rate - 0.01, 0.0)
    # else: stay at current level
    
    masking_state['current_rate'] = current_rate
    return current_rate
```

**Pros:** Simple, interpretable, guarantees model stays above threshold  
**Cons:** Can be slow if threshold is too strict

### Technique 2: Plateau Detection

Only increase difficulty when learning has plateaued (no improvement for N epochs).

```python
class PlateauBasedCurriculum:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_acc = 0.0
        self.plateau_counter = 0
        self.masking_rate = 0.0
    
    def update(self, train_acc):
        # Check if we've improved
        if train_acc > self.best_acc + self.min_delta:
            self.best_acc = train_acc
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        # If plateaued for N epochs, increase difficulty
        if self.plateau_counter >= self.patience and train_acc > 0.85:
            self.masking_rate = min(self.masking_rate + 0.05, 1.0)
            self.plateau_counter = 0  # Reset
            print(f"Plateau detected! Increasing masking to {self.masking_rate:.2%}")
        
        return self.masking_rate
```

**Pros:** Ensures model fully learns at each level  
**Cons:** Can be very slow, may need maximum time limit

### Technique 3: Loss-Based Adaptive Schedule

Use loss trend instead of accuracy (more sensitive signal).

```python
def get_adaptive_masking_rate(epoch, recent_losses, masking_state):
    """
    Adjust masking based on loss trend (3-epoch window).
    """
    current_rate = masking_state['current_rate']
    
    if len(recent_losses) >= 3:
        loss_trend = recent_losses[-1] - recent_losses[-3]
        
        if loss_trend < -0.05:  # Loss decreasing significantly
            current_rate += 0.03  # Increase difficulty faster
        elif loss_trend < 0:  # Loss decreasing slowly
            current_rate += 0.01  # Increase gradually
        elif loss_trend > 0.05:  # Loss increasing (struggling!)
            current_rate -= 0.02  # Decrease difficulty
    
    current_rate = np.clip(current_rate, 0.0, 1.0)
    return current_rate
```

**Pros:** More fine-grained response to learning dynamics  
**Cons:** Noisy signal, needs smoothing/EMA

### Technique 4: Moving Average + Target Accuracy

Use smoothed performance with proportional control (PID-like).

```python
class AdaptiveCurriculum:
    def __init__(self, target_acc=0.90, smoothing=0.9):
        self.target_acc = target_acc
        self.smoothing = smoothing
        self.smoothed_acc = 1.0
        self.masking_rate = 0.0
    
    def update(self, train_acc):
        # Exponential moving average (reduces noise)
        self.smoothed_acc = (self.smoothing * self.smoothed_acc + 
                             (1 - self.smoothing) * train_acc)
        
        # Proportional control: adjust based on error
        error = self.smoothed_acc - self.target_acc
        
        if error > 0.05:  # Well above target
            self.masking_rate += 0.02
        elif error < -0.05:  # Below target
            self.masking_rate -= 0.02
        
        self.masking_rate = np.clip(self.masking_rate, 0.0, 1.0)
        return self.masking_rate
```

**Pros:** Stable, self-regulating, handles noise well  
**Cons:** Hyperparameters (target_acc, smoothing) need tuning

### Technique 5: Loss-Based Adaptive Masking (Easy-First vs. Hard-First)

Instead of random masking, use **per-edge loss** to strategically select which edges to mask. Two opposing philosophies:

#### Defining "Hard" vs. "Easy"

**Use loss, not just confidence:**
- **High loss = Hard** 
  - Confident + Wrong = HIGH loss (confidently wrong! needs focus)
  - Uncertain + Wrong = HIGH loss (confused, needs focus)
- **Low loss = Easy**
  - Confident + Correct = LOW loss (nailed it)
  - Uncertain + Correct = LOW loss (got lucky, but ok)

**Why loss is better than confidence alone:**
- Catches confident mistakes (model thinks it's right but isn't)
- Directly measures what we care about (prediction error)
- Aligns with hard negative mining in other domains

#### Philosophy A: Easy-First (Traditional Curriculum)
Mask edges with **lowest loss** first.
- **Rationale**: Remove "training wheels" gradually — let model rely on hints for hard cases longer
- **Progression**: High-loss edges stay visible → provide context for easier inference
- Model practices solving easy cases without hints

#### Philosophy B: Hard-First (Adaptive Hard Negative Mining)  
Mask edges with **highest loss** first.
- **Rationale**: Focus on biggest errors — force model to learn hard reasoning patterns early
- **Progression**: Low-loss edges stay visible → provide scaffolding while attacking hard cases
- **"Whack-a-Mole" Dynamic**: As model masters high-loss cases, NEW high-loss cases emerge and become the focus
- Model practices solving hard cases WITH full context

**Key Optimization:** Cache per-edge losses from the end of epoch N and use them for masking in epoch N+1. This is equivalent to a forward pass at the start of the next epoch (since model state hasn't changed yet) but requires no extra computation!

#### The Whack-a-Mole Effect (Hard-First Only)

```
Epoch 10 (1% masking):
  Highest loss edges: A, B, C (loss = 2.5, 2.3, 2.1)
  Mask A, B, C → Model must predict them (with 99% other edges visible as hints)

Epoch 11 (2% masking):
  Model improved! A, B, C now have loss = 0.8, 0.9, 0.7
  NEW highest loss edges: D, E, F (loss = 2.2, 2.0, 1.9)
  Mask D, E, F + next tier → Focus shifts dynamically

Epoch 12 (3% masking):
  Model improved on D, E, F! Their loss dropped.
  NEW highest loss edges: G, H, I
  ...continues...

Result: You're always attacking biggest errors (adaptive).
        As you improve, "highest loss" becomes objectively lower.
        Natural progression from hard → medium → easy absolute difficulty.
        
Key: Loss automatically identifies BOTH:
  - Uncertain predictions (model doesn't know)
  - Confident mistakes (model is wrong but doesn't realize it)
```

#### Implementation

```python
def train_epoch(model, loader, optimizer, criterion, device, 
                masking_rate=0.0, cached_losses=None, mask_strategy='easy_first'):
    """
    Train with adaptive masking using cached per-edge losses.
    
    Args:
        cached_losses: Dict mapping batch_idx -> per-edge losses from previous epoch
        mask_strategy: 'easy_first' or 'hard_first'
    
    Returns:
        avg_loss, accuracy, new_losses (for next epoch)
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_edges = 0
    new_losses = {}  # Cache for next epoch
    
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        
        # Apply masking using CACHED losses from previous epoch
        if masking_rate > 0.0 and cached_losses is not None:
            edge_losses = cached_losses.get(batch_idx)
            if edge_losses is not None:
                # Only consider original edges (not meta edges)
                original_edge_losses = edge_losses[data.edge_mask]
                num_to_mask = int(len(original_edge_losses) * masking_rate)
                
                if mask_strategy == 'easy_first':
                    # Mask LOWEST loss edges (easiest, model gets them right)
                    _, indices_to_mask = original_edge_losses.topk(num_to_mask, largest=False)
                elif mask_strategy == 'hard_first':
                    # Mask HIGHEST loss edges (hardest, biggest errors)
                    _, indices_to_mask = original_edge_losses.topk(num_to_mask, largest=True)
                
                # Map back to full edge_attr indices
                original_indices = torch.where(data.edge_mask)[0]
                full_indices_to_mask = original_indices[indices_to_mask]
                
                data.edge_attr[full_indices_to_mask, 3] = 0  # Zero out label
                data.edge_attr[full_indices_to_mask, 4] = 0  # Zero out is_labeled
        
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        
        # Compute loss (average for backprop)
        logits_original = logits[data.edge_mask]
        loss = criterion(logits_original, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = logits_original.argmax(dim=-1)
        correct_predictions += (pred == data.y).sum().item()
        total_edges += data.edge_mask.sum().item()
        
        # CACHE per-edge losses for NEXT epoch (no extra cost!)
        with torch.no_grad():
            # Compute loss per edge (reduction='none')
            per_edge_loss = F.cross_entropy(
                logits[data.edge_mask], 
                data.y,
                reduction='none'
            )
            new_losses[batch_idx] = per_edge_loss.cpu()
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct_predictions / total_edges
    return avg_loss, accuracy, new_losses

# In main training loop:
loss_cache = None

for epoch in range(1, epochs + 1):
    # Choose strategy
    strategy = 'hard_first'  # or 'easy_first' or 'random'
    
    train_loss, train_acc, loss_cache = train_epoch(
        model, train_loader, optimizer, criterion, device,
        masking_rate=current_masking_rate,
        cached_losses=loss_cache,
        mask_strategy=strategy
    )
```

#### Advanced: Stochastic Sampling (Exploration vs. Exploitation)

Instead of **deterministically** masking the top-K highest/lowest loss edges, **sample** edges with probability proportional to their loss. This adds exploration and robustness.

**Deterministic Selection (Current):**
```python
losses = [2.5, 2.3, 0.8, 0.5, 0.3]
# If masking top-2: Always mask indices 0,1 (losses 2.5, 2.3)
# Same edges every time at this masking rate
```

**Stochastic Sampling:**
```python
losses = [2.5, 2.3, 0.8, 0.5, 0.3]
probs = losses / losses.sum()  # [0.44, 0.40, 0.14, 0.01, 0.01]

# Epoch 10: Sample → might get [0, 1] (high prob, focus on worst)
# Epoch 11: Sample → might get [0, 2] (explore medium-loss case!)
# Epoch 12: Sample → might get [1, 3] (rare, but possible)
```

**Why Stochastic Helps:**
- ✅ **Avoids overfitting to outliers**: If some edges are just noisy/ambiguous, deterministic wastes epochs on them
- ✅ **Natural exploration**: Sometimes practice medium-difficulty cases, not just extremes
- ✅ **Diversity**: Different edges masked each epoch → varied training signal (like data augmentation)
- ✅ **Robustness**: Less sensitive to "unlearnable" hard cases

**Implementation with Temperature Control:**

```python
def sample_edges_to_mask(edge_losses, num_to_mask, temperature=1.0, strategy='hard_first'):
    """
    Sample edges to mask with probability proportional to loss.
    
    Args:
        edge_losses: Per-edge losses from previous epoch
        num_to_mask: How many edges to mask
        temperature: Controls sharpness (lower = more deterministic)
                     - 0.5: Conservative (mostly deterministic, slight randomness)
                     - 1.0: Balanced exploration
                     - 2.0: Exploratory (flatter distribution)
        strategy: 'hard_first' (high loss = high prob) or 'easy_first' (low loss = high prob)
    
    Returns:
        Indices of edges to mask
    """
    if strategy == 'hard_first':
        # Higher loss → higher probability of being masked
        weights = edge_losses
    elif strategy == 'easy_first':
        # Lower loss → higher probability of being masked
        # Invert: max_loss - loss
        weights = edge_losses.max() - edge_losses + 1e-8
    
    # Convert to probabilities with temperature
    probs = F.softmax(weights / temperature, dim=0)
    
    # Sample without replacement
    indices_to_mask = torch.multinomial(
        probs,
        num_samples=num_to_mask,
        replacement=False
    )
    
    return indices_to_mask

# In train_epoch, replace deterministic topk with:
if mask_strategy == 'hard_first':
    indices_to_mask = sample_edges_to_mask(
        original_edge_losses, 
        num_to_mask, 
        temperature=1.0,
        strategy='hard_first'
    )
```

**Variant: Misplaced Confidence Sampling**

Explicitly target **calibration errors** (confident mistakes):

```python
def compute_misplaced_confidence(logits, targets):
    """
    Score how wrong the model's confidence is.
    
    High score = confidently wrong (needs focus!)
    Low score = correct or appropriately uncertain
    """
    probs = F.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values  # Model's confidence [0, 1]
    
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()   # 1.0 if correct, 0.0 if wrong
    
    # Misplaced confidence = |confidence - correctness|
    # Examples:
    #   - confidence=0.95, correct=0 → score=0.95 (bad! confidently wrong)
    #   - confidence=0.40, correct=0 → score=0.40 (uncertain, ok)
    #   - confidence=0.95, correct=1 → score=0.05 (good! confident and right)
    misplaced = torch.abs(confidence - correct)
    
    return misplaced

# Use misplaced confidence as sampling weights
misplaced_scores = compute_misplaced_confidence(logits, targets)
probs = misplaced_scores / misplaced_scores.sum()
indices_to_mask = torch.multinomial(probs, num_to_mask, replacement=False)
```

**When to Use Stochastic vs. Deterministic:**

| Approach | Best For | Risk |
|----------|----------|------|
| **Deterministic** | Clean datasets, clear hard cases | May overfit to outliers |
| **Stochastic (temp=0.5)** | Mostly deterministic, slight robustness | Small exploration benefit |
| **Stochastic (temp=1.0)** | Balanced, good default for experiments | May occasionally miss worst cases |
| **Stochastic (temp=2.0)** | Noisy data, many outliers | May be too random |
| **Misplaced Confidence** | Calibration issues, confident mistakes | More complex to implement |

**Recommendation:** Start with **deterministic** (simpler, more interpretable). If you notice the model getting stuck on the same hard outliers epoch after epoch, switch to **stochastic with temperature=1.0**.

#### Comparison: Easy-First vs. Hard-First

| Aspect | **Easy-First** | **Hard-First** |
|--------|---------------|---------------|
| **Philosophy** | Traditional curriculum (walk before run) | Adaptive hard negative mining (attack weaknesses) |
| **Progression** | Easy → Hard (remove scaffolding) | Hard → Easier (whack-a-mole dynamics) |
| **Literature Support** | ✅ Strong (Bengio 2009, widely used) | ⚠️ Weaker (some evidence, less common) |
| **Optimization Stability** | ✅ High (avoids bad local minima) | ⚠️ Medium (requires warmup to be safe) |
| **For Reasoning Tasks** | ⚠️ May delay learning key patterns | ✅ Forces deep reasoning early |
| **Hard Case Coverage** | ⚠️ Hard cases trained shortest | ✅ Hard cases trained longest |
| **Risk** | Model may never master hardest cases | Model may struggle early, fail to converge |

**Easy-First Pros:**
- ✅ Well-validated in literature across many domains
- ✅ Stable optimization (avoids early chaos)
- ✅ Natural "remove training wheels" metaphor
- ✅ Works well for compositional tasks (learn A, then A+B, then A+B+C)

**Easy-First Cons:**
- ❌ Hard cases get least training time (masked latest)
- ❌ Model might learn shortcuts on easy cases, struggle when scaffolding removed
- ❌ For reasoning tasks, easy cases may not teach the key patterns

**Hard-First Pros:**
- ✅ Focuses on weaknesses (like hard negative mining)
- ✅ Hard cases get maximum training time (masked earliest)
- ✅ Dynamic "whack-a-mole" adapts to model's changing weaknesses
- ✅ May be better for constraint reasoning (Hashi's core challenge)
- ✅ Natural progression: As model improves, "hardest" becomes easier (automatic difficulty adjustment)

**Hard-First Cons:**
- ❌ Less empirical validation than easy-first
- ❌ Risk of optimization instability (mitigated by warmup)
- ❌ May focus on noise/outliers rather than learnable patterns
- ❌ Could miss compositional structure if hard cases require understanding easy patterns first

#### Recommended Approach for Hashi

**For 100% accuracy goal, try Hard-First:**

Reasons:
1. **You have warmup** (epochs 0-10 at 0% masking) → Learns basics, avoids instability
2. **Hashi is a reasoning task** → Hard edges teach constraint propagation patterns
3. **Goal is mastery** → Need to excel at hard cases, not just average cases
4. **Strong architecture** → Transformer with residuals handles hard examples well
5. **Large dataset** (21k) → Reduces risk of overfitting to outliers

**But run ablation study:**
```python
# Train 3 models in parallel:
Model A: Random masking (baseline)
Model B: Easy-first (well-validated)
Model C: Hard-first (experimental, but promising for reasoning)

# If hard-first wins by 2-3%+, you've validated it for constraint solving!
```

**Alternative: Clean Eval Pass (Optional)**

If you want cleaner per-edge losses (without dropout/BatchNorm noise from training mode):

```python
@torch.no_grad()
def cache_losses_for_next_epoch(model, loader, criterion, device):
    """
    Single eval pass at end of epoch for clean per-edge losses.
    Optional - only adds ~1 epoch worth of time over entire training.
    """
    model.eval()
    losses = {}
    
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.edge_attr)
        
        # Per-edge loss
        per_edge_loss = F.cross_entropy(
            logits[data.edge_mask],
            data.y,
            reduction='none'
        )
        losses[batch_idx] = per_edge_loss.cpu()
    
    return losses

# Use after training epoch:
loss_cache = cache_losses_for_next_epoch(model, train_loader, F.cross_entropy, device)
```

**General Pros (Both Strategies):** 
- Curriculum is personalized to model's current errors
- Zero extra computation cost (when caching from training)
- Adaptive to model's learning progress
- Loss-based selection is principled (directly measures prediction error)
- Catches both uncertain predictions AND confident mistakes

**General Cons (Both Strategies):** 
- Requires careful indexing to map losses to correct edges across batches
- Slightly more complex implementation than random masking
- Need to choose between easy-first vs. hard-first (or run experiments)
- Per-edge loss computation adds small overhead (but cached, so minimal)

### Recommended Implementation: Hybrid Approach

Combine fixed warmup with adaptive gating:

```python
def hybrid_curriculum(epoch, train_acc, config):
    warmup_epochs = config['warmup_epochs']
    
    if epoch < warmup_epochs:
        # Phase 1: Fixed warmup (0% masking)
        return 0.0
    else:
        # Phase 2: Adaptive progression with gating
        # Use plateau detection or target accuracy approach
        return adaptive_masking(train_acc, config)
```

### Configuration Extension

Add to `masking_experiment.yaml`:

```yaml
masking:
  enabled: true
  schedule: "adaptive"  # "cosine", "linear", or "adaptive"
  start_rate: 0.0
  end_rate: 1.0
  warmup_epochs: 10
  
  # Adaptive settings
  adaptive_mode: "plateau"  # "plateau", "gated", "target_acc", "loss_trend"
  target_train_acc: 0.90     # For gated/target_acc modes
  plateau_patience: 3        # For plateau mode
  max_increase_per_epoch: 0.03  # Safety cap
```

### Key Insight

The goal is **train accuracy staying high (95-100%)** while masking increases, with **val accuracy rising to meet it**. If train drops significantly, the curriculum is outpacing the model's learning.

## 5. Self-Critique Verification (Hybrid Graph Approach)

Instead of training a separate verifier, integrate verification directly into the training loop. The model learns to both predict edges AND recognize when its predictions form valid solutions, creating a self-supervised constraint learning mechanism.

### A. Core Concept: Hybrid Graph Verification

**Training flow per masked puzzle:**
1. **Input:** Graph with some edges masked (unknown), others visible (ground truth)
2. **Predict:** Model predicts labels for masked edges
3. **Construct hybrid graph:**
   - Visible edges: Use ground truth labels as features
   - Masked edges: Use model predictions as features
   - Update node degrees based on combined edge predictions
4. **Re-run model:** Forward pass on hybrid graph to get updated global meta embedding
5. **Verify:** Binary classification on global meta embedding
6. **Target:** 1.0 if predictions are perfect (hybrid = ground truth), 0.0 otherwise

### B. Why This is Powerful

| Traditional Self-Critique | Hybrid Graph Approach |
|---------------------------|-----------------------|
| Tests: "Are my predictions globally valid?" | Tests: "Do my predictions + known constraints form a valid solution?" |
| Learns from complete predictions | Learns consistency with partial supervision |
| Unrealistic (assumes no known edges) | Realistic (partial information setting) |
| May overfit to model's own patterns | Learns general constraint satisfaction |

### C. Key Insights

*   **Constraint consistency:** Model learns that predictions must be consistent with known correct edges
*   **Realistic supervision:** Mimics real-world puzzle solving with partial information
*   **Self-supervised:** Verification provides additional gradient signal without extra labeled data
*   **Computational cost:** Requires second forward pass but provides rich learning signal

### D. Architecture: Dual-Head with Global Meta

**Prediction Head:** Per-edge 3-class classification (0, 1, 2 bridges)  
**Verification Head:** Binary classification on global meta node embedding

**Implementation:**
- Global meta node aggregates information from all nodes after message passing
- Verification head: `MLP(global_meta_embedding) → Sigmoid → P(valid)`
- Combined loss: `λ_edge * edge_loss + λ_verify * verify_loss`

### E. Training Dynamics & Curriculum

**Phase 1:** Edge prediction only (λ_verify = 0) until baseline performance  
**Phase 2:** Introduce verification gradually as model learns basic patterns  
**Phase 3:** Full dual-head training with balanced loss weights

**Expected behavior:**
- Early training: Verification loss noisy, provides weak signal
- Mid training: Verification becomes meaningful constraint teacher
- Late training: Verification reinforces perfect solutions, penalizes constraint violations

### F. Risks & Gotchas

| Risk | Mitigation |
|------|------------|
| **Task interference** | Start with low λ_verify (0.1), gradually increase |
| **Verification noise** | Only meaningful when edge predictions are reasonably good |
| **Computational overhead** | Second forward pass (~2x cost) but rich learning signal |
| **Training instability** | Monitor both losses; reduce λ_verify if edge loss degrades |
| **False negatives** | Hybrid graph might be "invalid" due to prediction errors, not fundamental issues |
| **Degree update complexity** | Node degree features must reflect hybrid predictions, not pre-computed values |

### G. Expected Benefits

*   **Improved constraint learning:** Model learns global consistency, not just local patterns
*   **Better generalization:** Handles partial information scenarios better
*   **Self-regularization:** Verification provides additional gradient signal
*   **Iterative refinement ready:** Architecture naturally supports inference-time verification

Instead of two separate models, use **one backbone with two heads**:
*   **Prediction Head:** Per-edge 3-class classification (0, 1, 2 bridges)
*   **Verification Head:** **Graph-level** binary classification (valid solution vs. invalid solution)

#### Why Graph-Level Classification?

The verification head judges the **entire puzzle solution**, not individual edges:
*   Input: Puzzle graph with proposed edge labels as features
*   Output: Single binary score—"Is this a valid Hashi solution?"

This is closer to a **true GAN discriminator**—it evaluates the full "sample" holistically.

| Per-Edge Verification | Graph-Level Verification |
|-----------------------|--------------------------|
| Flags specific errors | Judges overall validity |
| N outputs per puzzle | 1 output per puzzle |
| Can overfit to local patterns | Must learn global constraints |
| Useful for iterative fixing | Useful for rejection sampling / confidence |

#### Training Flow (Per Batch)

1.  Forward pass through shared backbone
2.  **Prediction head** outputs edge labels → CE loss vs. ground truth
3.  Create two versions of the puzzle:
    *   **Valid:** Use ground truth edge labels as features
    *   **Invalid:** Use corrupted labels (or prediction head's own outputs)
4.  **Verification head:** Graph pooling → binary classifier
    *   Should output 1 for valid, 0 for invalid
5.  Combined loss backprops through shared backbone

#### Benefits of Shared Backbone + Graph-Level Verification

| Benefit | Why It Helps |
|---------|--------------|
| **Holistic judgment** | Must learn ALL constraints (degree, crossing, connectivity) to judge validity |
| **Stronger learning signal** | A puzzle is solved or it isn't—no partial credit, forces precision |
| **GAN-like dynamics** | Discriminator sees "real" (valid) and "fake" (corrupted) solutions |
| **Shared representations** | Features for generating solutions also useful for recognizing them |
| **Implicit regularization** | Backbone learns what valid solutions "look like" globally |
| **Simpler verification** | One output per puzzle, not per edge |

#### Key Differences from True GANs

*   **Cooperative, not adversarial:** Both heads trained together, not competing
*   **No fooling dynamic:** Generator isn't actively trying to trick discriminator
*   **Supervised discriminator:** Trained on known valid/invalid labels, not learning to distinguish
*   **Closer to auxiliary classifier GAN (AC-GAN)** or **energy-based model**

#### Training Variants

1.  **Joint Training:** Both losses every batch (simple, shared gradients)
2.  **Self-Critique:** Feed prediction head's outputs to verification head—model judges its own work
3.  **Contrastive Batches:** Each puzzle appears twice (valid + corrupted), verification head must distinguish

#### Verification Head Architecture

**Option A: Standard Pooling**
```
Node embeddings (from backbone)
    ↓
Global pooling (mean, max, or attention)
    ↓
MLP → Sigmoid → P(valid)
```

**Option B: Use Existing Global Meta Node (Preferred)**

We already have infrastructure for a global meta node that connects to all puzzle nodes. After message passing, the meta node's embedding IS a graph-level representation—no separate pooling needed!

```
Puzzle nodes ←→ Global Meta Node (existing toggle: use_meta_node)
                      ↓
              Meta node embedding (after GNN layers)
                      ↓
              MLP → Sigmoid → P(valid)
```

**Option C: Hierarchical Meta Aggregation**

Connect global meta node to row/col meta nodes for structured aggregation:

```
Puzzle nodes ←→ Row Meta Nodes ←→ Global Meta Node
             ←→ Col Meta Nodes ←↗
```

*   Row/col metas summarize their respective lines
*   Global meta aggregates row/col summaries
*   Hierarchical structure mirrors puzzle geometry
*   Final global meta embedding captures both local (row/col) and global structure

**Why Meta Node for Classification?**

| Benefit | Explanation |
|---------|-------------|
| **Already implemented** | `use_meta_node` toggle exists, just unused |
| **Natural aggregation** | Message passing automatically creates graph summary |
| **No information loss** | Attention-based aggregation (TransformerConv) preserves important nodes |
| **Geometric hierarchy** | Row/col → global mirrors Hashi's line-of-sight structure |
| **Single embedding** | Meta node embedding = graph representation, feed directly to classifier |

#### Risks/Considerations

*   **Graph pooling bottleneck:** Must compress full puzzle into single vector—may lose edge-level detail
*   **Easy negative examples:** Random corruption may be too easy to detect; consider adversarial corruptions
*   **Task interference:** Two heads may want different features—need careful loss weighting
*   **Binary signal is sparse:** Only 1 bit of feedback per puzzle (vs. N bits for per-edge)

#### Verdict

Worth experimenting with. Graph-level verification provides a **holistic validity prior** that per-edge approaches miss. The shared backbone learns "what valid Hashi solutions look like" as a global property. Start with joint training and λ_verify = 0.3 to provide meaningful gradient signal without dominating.

### G. Recommended Approach

1.  **Start simple:** Corruption pretraining on separate verifier (independent baseline)
2.  **Experiment:** Single-model dual-head with joint training
3.  **Fine-tune:** On proposer's actual errors after initial training
4.  **Deploy:** Iterative refinement loop for hard puzzles

This approach is particularly promising for Hashi because constraint violations (degree, crossing, connectivity) create learnable error signatures that a verifier can exploit.

## 6. Graph Reinforcement Learning (The "Human-Like" Solver)

While one-shot classification is powerful, Hashi is fundamentally a sequential constraint satisfaction problem. A human doesn't solve the whole board instantly; they fill in "obvious" bridges first, update the island counts, and then solve the next easiest constraints. This suggests moving from a classification paradigm to a **Reinforcement Learning (RL)** paradigm.

### A. Core Concept: Sequential Construction

Instead of predicting all edge states at once (0, 1, or 2), the agent acts iteratively:
1.  **Observe** the current partial graph state (island capacities, current bridges).
2.  **Select** a valid edge to modify.
3.  **Action:** Add 1 bridge (increment) or add 2 bridges (double-increment).
4.  **Update** the graph state (reduce island remaining capacity, update edge features).
5.  **Repeat** until solved or stuck.

### B. Action Space: The "Add-K" Paradigm

We define a discrete action space that maps efficiently to graph edges.
*   **Actions per Edge:**
    *   **Increment (+1):** Change bridge count $0 \to 1$ or $1 \to 2$.
    *   **Double Increment (+2):** Change bridge count $0 \to 2$.
    *   *(No-Op is implicit by not selecting the edge)*
    * experiement with concatenating global node feat to edge node feats 

*   **Validity Masking (Critical):**
    The agent should only be allowed to take valid actions. We must compute a **legal action mask** at each step:
    *   Mask out edges that already have 2 bridges.
    *   Mask out **+2** action if an edge already has 1 bridge.
    *   Mask out actions that would exceed an island's remaining capacity (e.g., trying to add a bridge to an island with 0 remaining capacity).
    *   Mask out actions that would cross an existing bridge.

### C. State Representation (The Environment)

The `HashiEnv` (Gym/PettingZoo environment) maintains the state:
*   **Static Features:** Island initial capacities, positions, edge distances.
*   **Dynamic Features (Updated per step):**
    *   `edge_attr`: Current bridge count (0, 1, 2) on each edge.
    *   `node_attr`: Current **remaining capacity** (initial capacity - current degree).
        - migght want to generate a joint embedding for capacity/remaining capacity
    *   `mask`: Boolean flag for "solved" islands (visual aid for agent).

### D. Policy Network (Pointer Network)

We can reuse the existing `TransformerEdgeClassifier` backbone but change the readout head.
*   **Input:** Dynamic graph state.
*   **Backbone:** Transformer layers propagate constraints (e.g., "This island needs 1 more bridge").
*   **Output Head:** Instead of 3-class classification, output **2 policy logits** per edge:
    *   Logit A: Score for taking **+1** action.
    *   Logit B: Score for taking **+2** action.
    * experiemnt with c
*   **Selection:** Flatten to `[num_edges * 2]`, apply legal action mask (set invalid logits to $-\infty$), and sample via Softmax (Categorical distribution).

### E. Reward Function (Shaping Behavior)

RL is sensitive to rewards. We need a dense signal to guide the agent through long episodes.

*   **Sparse Terminal Reward:**
    *   **+10** for solving the entire puzzle perfectly.
    *   **-10** for reaching a "dead end" (no legal moves left but puzzle unsolved).

*   **Dense Step Rewards:**
    *   **+0.5** for satisfying an island (reducing remaining capacity to exactly 0).
    *   **-0.1** step penalty (encourages efficiency/speed).

*   **Immediate Punishments (Safety Rails):**
    While masking prevents *illegal* moves, we can also soft-punish *bad* moves if we relax the masking (e.g., allowing over-filling to let the agent learn counting):
    *   **-5.0** if an action exceeds island capacity (if not masked).
    *   **-5.0** if an action creates a crossing (if not masked).
    *   **-5.0** if an edge count goes > 2.
    *   *Note: Hard masking is generally better than learning-via-punishment for hard constraints.*

### F. Implementation Steps for RL

1.  **Environment Class (`HashiEnv`):** Create a Gym-compatible environment that wraps `HashiDataset`.
    *   `reset()`: Loads a new puzzle, resets dynamic features.
    *   `step(action)`: Applies logic, updates graph, computes reward/done.
    *   `get_mask()`: Computes the legal action mask.
2.  **Trajectory Buffer:** A data structure to store `(state, action, reward, next_state, mask)` transitions.
3.  **RL Loop:** Implement a simple policy gradient loop (like REINFORCE or PPO) using the GNN as the policy network.
4.  **Inference Strategy:**
    *   **Greedy:** Pick max-probability action.
    *   **Sampling:** Sample from distribution (good for exploration/training).
    *   **Beam Search:** Keep top-K partial solutions (advanced).

### G. Curriculum Learning for RL ("Finish the Game")

The "Progressive Masking" strategy that succeeded in one-shot classification has a direct and powerful analogue in RL: **Reverse Curriculum Generation** (or "Start from the End").

Instead of forcing a baby agent to solve an empty board from scratch (where the horizon is long and rewards are sparse), we train it to **"finish the game."**

#### 1. The Concept
We initialize the environment not with an empty board, but with a **partially solved board**.
*   **Easy Mode (90% Solved):** The board has almost all bridges. The agent only needs to place the final 1-2 bridges to get the "Solved!" reward. This teaches the "endgame" mechanics and credit assignment instantly.
*   **Medium Mode (50% Solved):** The agent must solve the second half of the puzzle.
*   **Hard Mode (0% Solved):** The standard empty board.

#### 2. Implementation: "Reverse Masking"
We reuse the `masking_rate` logic but invert its purpose for the RL environment initialization:

```python
class HashiEnv(gym.Env):
    def reset(self, initial_solution_rate=0.9):
        """
        initial_solution_rate: 0.0 = Empty board (Hard)
                               1.0 = Fully solved board (Trivial)
        """
        # Load puzzle
        data = self.dataset.get_random()
        
        # Determine which edges are "pre-solved"
        # We REUSE the masking logic: 
        # - Masked edges = What the agent needs to find (Hidden)
        # - Visible edges = Pre-filled bridges (Given)
        
        # If rate is 0.9, we reveal 90% of the ground truth bridges
        # The agent starts with these bridges already added to the graph state
        self.current_bridges = reveal_bridges(data.y, rate=initial_solution_rate)
        
        # Update node capacities based on these pre-filled bridges
        self.update_capacities()
        
        return self.get_observation()
```

#### 3. Curriculum Schedule
We gradually reduce the `initial_solution_rate` as the agent improves, lengthening the episode horizon.

*   **Phase 1 (Endgame):** Start with `rate=0.8`. Agent learns "If I connect these two 1-islands, I win." High reward frequency.
*   **Phase 2 (Midgame):** Linearly ramp `rate` down to `0.0`. Agent learns to chain decisions.
*   **Phase 3 (Full Game):** Train on empty boards (`rate=0.0`).

#### 4. Why This Wins
*   **Dense Rewards:** The agent hits the `+10` terminal reward frequently in early training, stabilizing the policy gradient.
*   **Value Function Anchoring:** The agent learns the value of "good states" (near-solved boards) first, then learns how to reach those states from further away.
*   **Shared Infrastructure:** It uses the exact same `RandomHashiAugment` / Masking logic we built for the classifier, just applied to the environment state initialization.

## 7. Immediate Action Plan

- [ ] **Config:** Update `node_embedding_dim` to `64` and `num_layers` to `8`.
- [ ] **Code (Model):** Implement Residual Connections in `gcn.py` and `gat.py` to support depth.
- [ ] **Code (Train):** Implement weighted Loss function to handle class imbalance (0 vs 1/2).
- [ ] **Code (Data):** Add `distance` (Euclidean/Manhattan) as an edge attribute.
- [ ] **Code (Data):** Add `direction` (Egocentric N/S/E/W) as an edge attribute.
- [ ] **Code (Data/Model):** Implement **Inverse Distance** and **Sinusoidal Encoding** for edge features (Sweep to compare).
- [ ] **Code (Data):** Implement **Conflict Edges** (Structural) or **Conflict Loss** (Training).
- [ ] **Code (Data):** Implement **Meta-Grid** (Row/Col Meta Nodes with backbone connectivity).
- [ ] **metrics suite**
    - measure 100% puzzle accuracy rate during trainig
    - bridge count confusion matrix
    - some kind of anchor-target island "degree" accuracy heatmap?
    - some kind of edge location accuracy heatmap?
- [ ] **hyperparameter sweep**
- [ ] ** augmentations?** on the fly flip / rotate (or row/col stretching!)
    - will need to investigate how this interacts with presaved data
- [ ] **verifier classification**
