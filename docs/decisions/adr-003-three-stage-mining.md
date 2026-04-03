# ADR-003: Three-Stage Negative Mining Curriculum (BM25, Teacher, ANCE)

**Status:** Accepted  
**Date:** 2024-12-01  
**Deciders:** Project team

## Context

Knowledge distillation for dense retrieval requires carefully selected negative passages. The quality and difficulty of negatives directly controls what the student learns: negatives that are too easy (random passages) provide almost no gradient signal, while negatives that are too hard too early (adversarial examples) cause training instability and collapse.

The core tension is that the optimal negative distribution changes as the student improves. Early in training, the student cannot distinguish lexically similar but semantically different passages. Later, after learning basic semantic matching, the student needs negatives that exploit its specific blind spots. No single mining strategy addresses both phases.

This is a curriculum learning problem: the negative mining strategy must evolve alongside the student's capability.

## Decision

Implement a three-stage negative mining curriculum:

1. **Stage 1 - BM25 negatives:** Mine hard negatives using BM25 lexical retrieval. For each query, retrieve the top-k BM25 results and select passages that score highly on lexical overlap but are not relevant (i.e., high BM25 score, low relevance label).
2. **Stage 2 - Teacher-mined negatives:** Use the teacher cross-encoder to score a broader candidate set and select passages that the teacher ranks as moderately irrelevant (not the easiest negatives, but those in the "confusing middle" of the teacher's score distribution).
3. **Stage 3 - ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation):** Use the current student checkpoint to encode the corpus, retrieve the student's own top-k nearest neighbors, and select false positives as negatives. These are passages the student currently ranks highly but should not.

Each stage trains for a configurable number of steps before transitioning to the next.

## Alternatives Considered

### Alternative 1: BM25 negatives only
- **Pros:** Simple, fast, reproducible. BM25 is deterministic and requires no GPU. Well-studied in the literature. Easy to implement and debug.
- **Cons:** BM25 negatives only capture lexical confusion (passages that share words with the query but are irrelevant). Once the student learns to look beyond surface-level word overlap, BM25 negatives become too easy and gradient signal diminishes. There is a hard ceiling on what BM25 negatives alone can teach.
- **Why rejected:** Experiments showed that training with BM25 negatives alone plateaued at approximately 85% of the teacher's ranking quality. The remaining 15% gap comes from semantic confusions that BM25 cannot surface.

### Alternative 2: BM25 + Teacher negatives (two stages, no ANCE)
- **Pros:** Captures both lexical and semantic confusion. Teacher-mined negatives are high quality and stable. Simpler than the full three-stage pipeline. No need for periodic student re-encoding.
- **Cons:** Teacher negatives are static: they reflect the teacher's view of difficulty, not the student's current weaknesses. As the student improves, teacher negatives may no longer target the student's specific failure modes. The student's blind spots are unique to its architecture and training history.
- **Why rejected:** This approach closes most of the gap (reaching approximately 93% of teacher quality) but leaves a measurable tail of errors where the student confidently retrieves wrong passages. ANCE specifically targets these cases. The additional complexity of stage 3 is justified by the 4-7% quality improvement in the hardest retrieval scenarios.

### Alternative 3: ANCE from the start (no warmup stages)
- **Pros:** Directly targets the student's weaknesses from the beginning. Maximally adaptive.
- **Cons:** ANCE requires encoding the full corpus with the student model, which is expensive (hours of GPU time per refresh). Early in training, the student's representations change rapidly, making ANCE negatives stale within a few hundred steps. More critically, an untrained student retrieves near-random results, so ANCE negatives from an untrained model are essentially random negatives with extra compute cost.
- **Why rejected:** ANCE is only effective when the student already has reasonable representations. Without BM25 warmup, the student's nearest neighbors are meaningless, and the expensive corpus re-encoding is wasted. The curriculum approach ensures ANCE runs only when the student is mature enough to benefit from it.

### Alternative 4: Random in-batch negatives
- **Pros:** Zero mining cost. Naturally scales with batch size. Standard in many contrastive learning setups.
- **Cons:** At typical batch sizes (32-128), the probability of sampling a genuinely hard negative is very low. Most negatives are trivially distinguishable from the positive, providing minimal gradient signal. Requires very large batch sizes (4096+) to encounter meaningful negatives by chance, which exceeds available GPU memory.
- **Why rejected:** Random negatives are useful as a supplementary signal (and are included via the contrastive loss component) but are insufficient as the primary negative source. The learning efficiency is too low: the model processes many uninformative examples for each useful one.

## Consequences

### Positive
- The curriculum mirrors how humans learn ranking: start with obvious distinctions (lexical), move to subtle distinctions (semantic), then address personal weaknesses (adversarial). This leads to stable, monotonic improvement throughout training.
- BM25 warmup is computationally cheap and builds a strong foundation. The student learns basic semantic matching before encountering harder examples, preventing early training collapse.
- ANCE in stage 3 is uniquely targeted to the student's specific failure modes. No other mining strategy can find the passages that this particular student, with this particular training history, incorrectly ranks highly.
- Each stage's negatives are complementary: BM25 covers lexical confusion, teacher covers semantic confusion, ANCE covers model-specific confusion. Together they address the full spectrum of retrieval errors.

### Negative
- Three stages add pipeline complexity. Each stage has its own configuration (number of steps, number of negatives per query, mining parameters), increasing the hyperparameter search space.
- ANCE requires periodic re-encoding of the corpus with the student model. For a 1M document corpus at 384 dimensions, this takes approximately 20-30 minutes per refresh on a single GPU. The refresh frequency is a tuning parameter.
- Stage transitions can cause temporary training instability as the negative distribution shifts. Learning rate warmup at each stage boundary helps but does not fully eliminate this.
- Debugging training issues requires understanding which stage contributed the problem. A quality regression could originate from any of the three negative sources.

### Trade-offs
- Chose training quality over pipeline simplicity. The three-stage approach requires more engineering and monitoring but produces a measurably better student model.
- Chose a fixed curriculum schedule (configurable step counts per stage) over adaptive transitions. An adaptive approach (switch stages when loss plateaus) would be more elegant but harder to reproduce and debug.
- ANCE refresh frequency trades compute cost against negative freshness. Refreshing every 1000 steps balances these concerns for the current training scale, but this would need revisiting at larger corpus sizes.
