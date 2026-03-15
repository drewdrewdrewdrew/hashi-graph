# TO_CONSIDER

Things that came up during BPTT implementation worth thinking through.

---

## 1. Fair comparison: BPTT vs baseline

n_blocks was doing something subtle — it runs the full batch for step 0, then subsamples to ~14/32 graphs for steps 1-9. So in the baseline only 14 samples ever complete a full multi-step trajectory. BPTT (with n_blocks disabled) runs all 32 samples through all 10 steps.

To get a true apples-to-apples baseline:
- Set `n_blocks: 9` (= num_inference_steps_training - 1) — subsample_size equals batch_size, subsampling silently no-ops
- All 32 samples complete the full trajectory, no cross-step gradient
- Compare val_perfect_acc curves at same epoch count

The question you're actually answering: does gradient signal flowing across diffusion steps improve puzzle-solving accuracy, holding everything else equal?

---

## 2. Gradient checkpointing in non-BPTT mode

Currently the non-BPTT path stores all 10 step computation graphs in memory simultaneously until `torch.stack(step_losses).mean().backward()`. This is what OOM'd with n_blocks=9 on the shared GPU — BPTT ran fine because `checkpoint()` keeps only one step's activations alive at a time.

You could add checkpointing to the non-BPTT path too, which would:
- Bring peak memory in line with BPTT (making the comparison fair on memory too)
- Make the n_blocks=9 baseline viable on contested GPUs

The obstacle: the non-BPTT forward is written inline in the loop, not factored into a callable. You'd need to extract it before wrapping in `checkpoint()`.

---

## 3. Refactor opportunity: shared step forward

Right now `_step_forward` is a closure defined inside `_run_bptt_window`. A cleaner architecture would lift it to a proper method on `DiffusionTrainer` that both paths call:

- BPTT path: calls it via `checkpoint()` inside the window loop
- Non-BPTT path: calls it via `checkpoint()` inside the training loop

Benefits:
- Both paths get memory-efficient training
- The two modes become symmetric and easier to reason about
- Easier to add future features (e.g. per-step logging, different loss weighting) in one place

This is a meaningful refactor — not urgent, but worth doing before adding more complexity on top.

---

## 4. n_blocks + BPTT are fundamentally incompatible

Current fix: `and not bptt_enabled` in the n_blocks condition, so subsampling is skipped when BPTT is on. This is correct but worth documenting why:

n_blocks subsamples `batch` and `scales` after step 0. BPTT re-runs from `start_data=data` (the full pre-subsampling batch). After subsampling, `scales` has fewer entries than `data`'s graph count, so `scales[edge_batch_i]` goes out of bounds inside `_run_bptt_window`.

The two mechanisms serve overlapping purposes — n_blocks provides more compute per batch by running fewer samples through more steps, BPTT provides it via window re-runs. They shouldn't coexist. The current fix is the right call; just make sure it's obvious in code comments if someone tries to re-enable n_blocks with BPTT later.

---

## 5. Latent bug: scales undefined for non-diff-cont modes with BPTT

Flagged in the milestone audit. If `bptt.enabled: true` and `mode != "diff-cont"` (e.g. flow-blind or diff-discrete), the `scales` variable is undefined when `_run_bptt_window` is called at line 545 → `UnboundLocalError`.

Not a problem now (YAML has `mode: diff-cont`), but will surface immediately if BPTT is ever tested with another mode. Fix: initialize `scales = None` before the mode branch, or add an explicit guard.
