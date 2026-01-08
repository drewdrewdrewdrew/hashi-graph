# Hashi Graph Model Enhancements

This document outlines strategies to improve the performance of the Hashi GNN solver, moving from the current One-Shot Classification baseline towards a human-like Sequential Solver.

## 0. Current Status & Methodology

We are currently treating Hashi as a **One-Shot Edge Classification** problem with **Curriculum Learning**.

*   **Task:** Given a puzzle state, predict the final label (0, 1, or 2 bridges) for every edge simultaneously.
*   **Architecture:** Graph Transformer with Global Meta Nodes and Row/Col Meta Nodes for long-range constraint propagation.
*   **Curriculum (Progressive Masking):**
    *   **Start:** Most edges are visible (Ground Truth provided). The model learns to fill in small gaps ("easy mode").
    *   **Ramp:** Over training epochs, we mask more edges (Hidden).
    *   **End:** 100% of edges are masked. The model solves an empty board from scratch.
*   **Feature Engineering:** We have heavily engineered the input space to make logic explicit:
    *   **Geometric:** Distance ($1/d$), Direction (N/S/E/W).
    *   **Topological:** Spectral Fingerprinting (Eigenvectors), Articulation Points, Bridges (Cut Edges).
    *   **Logic:** Dynamic "Unused Capacity" (remaining bridges needed), Conflict Edges (crossing constraints).

*Note: Core feature engineering (distance/direction/conflict), architecture updates (Transformer/Meta Nodes), and auxiliary losses (Degree/Crossing/Verification) have been implemented.*

## 1. Adaptive Curriculum Learning (Refinements)

While basic progressive masking (cosine schedule) is implemented, the "Adaptive" and "Hard-First" strategies remain as powerful optimizations to ensure the model masters difficult constraints.

### A. Adaptive Schedules
Instead of a fixed cosine ramp, adjust masking based on model performance:
*   **Plateau Detection:** Only increase masking when validation loss plateaus.
*   **Performance Gating:** Only increase masking if training accuracy > 95%.

### B. Loss-Based Masking (Hard-First)
Instead of random masking, use per-edge loss to select which edges to mask.
*   **Strategy:** Mask edges with the **highest loss** from the previous epoch.
*   **Why:** Forces the model to focus on its "confidently wrong" or "uncertain" predictions (Hard Negative Mining).
*   **Dynamic:** As the model improves on hard edges, new edges become the "hardest," creating a natural "Whack-a-Mole" curriculum that targets weaknesses.

## 2. Graph Theoretical Enhancements (Bridges & Cut Nodes)

To bridge the gap between pattern matching and global topology, we can introduce explicit graph theory metrics.

### A. The "Potential Graph"
These metrics must be calculated on the **Potential Graph** (all valid physical connections, `n=0`), NOT the solution graph. This represents the "space of possibilities."

### B. Cut Edges (Bridges)
*   **Concept:** An edge is a bridge if removing it increases the number of connected components in the potential graph.
*   **Implication:** If an edge is a bridge, it represents the *only* physical path between two sets of islands. Therefore, **it must be part of the solution** (assuming the puzzle is valid and requires global connectivity).
*   **Feature:** `is_cut_edge` (Binary Edge Feature).

### C. Articulation Points (Cut Vertices)
*   **Concept:** A node is an articulation point if removing it disconnects the graph.
*   **Implication:** These nodes act as critical gateways or bottlenecks. All flow between separated components must pass through them.
*   **Feature:** `is_articulation_point` (Binary Node Feature).

### D. Spectral Fingerprinting (Wacky/Esoteric)
*   **Concept:** Encode the "resonant frequencies" of the graph topology.
*   **Implementation:** Compute the top $k$ eigenvectors of the Normalized Graph Laplacian.
*   **Normalization (Crucial):** Use **L-Infinity normalization** (scaling each vector to max absolute value of 1.0) to ensure the features are invariant to puzzle size ($N$ vs $N_{train}$).
*   **Why:** Captures global structural symmetry and partitioning that is invisible to local message passing. Helps the model distinguish between "The Northern Cluster" and "The Southern Cluster" regardless of geometric distance.

### E. Advanced Graph Topology (Deep Cuts)
These features target intermediate structure (loops and flow) to complement the global (spectral) and local (capacity) features.

#### 1. Cycle Participation (Edge Feature)
*   **Concept:** Generalizes the "Bridge" concept.
*   **Logic:** Compute the **Cycle Basis** of the potential graph. Count how many fundamental cycles each edge belongs to.
*   **Implication:**
    *   Count = 0: **Bridge** (Critical).
    *   Count = 1: Part of a simple loop (Structural).
    *   Count > 5: Part of a dense mesh (Optional/Redundant).
*   **Why:** Tells the model "how redundant" an edge is. High redundancy means the edge is likely optional; zero redundancy means it's mandatory.

#### 2. Flow Centrality (Node Feature)
*   **Concept:** Measures a node's role as a "capacity highway."
*   **Logic:** Treat islands as Sources/Sinks with demand = capacity. Run Max Flow between random pairs.
*   **Implication:** Continuous version of Articulation Points.
*   **Why:** Highlights nodes that sit on critical capacity transfer paths, even if they aren't strictly articulation points.

### F. Aesthetic Note: The "Cheat" vs. The Curriculum
*   **The Cheat:** Providing these features allows the model to "cheat" on connectivity, instantly solving global topology without learning the underlying logic. This might hurt OOD generalization if the input graph is noisy or constructed from vision data.
*   **The Recommendation:** **Do not implement these immediately.** Let the Transformer architecture (with its global attention) attempt to learn these topological properties first.
*   **The Fallback:** If the model solves local constraints perfectly but fails to create a single connected component, introduce these features as "training wheels" (hints) that are eventually dropped out, or as a diagnostic tool.

## 3. Graph Reinforcement Learning (The "Human-Like" Solver)

Moving from one-shot classification to sequential construction mimics human solving behavior. This breaks the "Ambiguity Ceiling" by allowing the agent to collapse uncertainty one step at a time.

### A. The "Incremental Autoregressive" Bridge (Behavior Cloning)
Before full RL, we can train the existing architecture to act sequentially using supervised data.
*   **Concept:** Instead of predicting `Final_State` (0/1/2), predict `Next_Action`.
*   **Actions:** `Add +1 Bridge`, `Add +2 Bridges`, `No-Op`.
*   **Training:** Deconstruct completed puzzles into trajectories.
    *   State $S_t$: Partial graph.
    *   Target: The action that moves edge $E_i$ closer to its ground truth.
*   **Result:** A pre-trained policy that understands "increments" rather than just "states."

### B. Core RL Concept
*   **Action Space:** Discrete Edge Selection + Increment.
    *   Option 1: `(Edge_Index, Action_Type)` tuple.
    *   Option 2: Flattened logic `[Edge_0_Add1, Edge_0_Add2, ..., Edge_N_Add2]`.
*   **State:** Dynamic graph with `remaining_capacity` updated per step.
*   **Masking (Safety Rails):** Invalid actions (creating crossings, over-filling islands, exceeding 2 bridges) are strictly masked out. The agent literally *cannot* make an illegal move.

### C. Reverse Curriculum ("Finish the Game")
Train the agent on partially solved puzzles first. This creates a dense reward signal early on.
*   **Phase 1 (Endgame):** 90% bridges placed. Agent learns "Connect these two 1s to win." Immediate +10 reward.
*   **Phase 2 (Midgame):** 50% bridges placed. Agent learns deeper lookahead and chain reactions.
*   **Phase 3 (Empty Board):** Full solving. Agent must plan from the start.

### D. Advanced "Wacky" Features for RL
*   **The "Satisfiability Probe" (Monte Carlo):** Run 10 random rollouts from the current state. Feed the statistics ("Edge X was chosen 90% of the time") as an input feature.
*   **Time-Travel Gradients (Denoising):** Train the model to "denoise" a corrupted solution, treating the solving process as a diffusion process from noise to order.


## 4. Immediate Action Plan

- [ ] **Code (Train):** Implement **Hard-First / Loss-Based Masking** in `train.py`.
- [ ] **Code (Train):** Implement **Plateau-Based Adaptive Schedule** in `train.py`.
- [ ] **Metrics:** Add connectivity failure tracking (how many disjoint components in predicted graphs?).
- [ ] **RL Env:** Create `HashiEnv` gym wrapper around `HashiDataset`.
- [ ] **RL Agent:** Implement simple PPO/REINFORCE loop using `TransformerEdgeClassifier` as policy backbone.
