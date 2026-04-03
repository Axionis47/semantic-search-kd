# ADR-004: Multi-Loss Combination (60% MarginMSE + 20% Listwise-KD + 20% Contrastive)

**Status:** Accepted  
**Date:** 2024-12-01  
**Deciders:** Project team

## Context

Knowledge distillation from a cross-encoder teacher to a bi-encoder student requires a loss function that transfers ranking knowledge effectively. The fundamental challenge is a representation mismatch: the teacher cross-encoder outputs raw relevance scores (approximately [-10, +10]) from joint query-passage attention, while the student bi-encoder outputs cosine similarity scores ([-1, +1]) from independently encoded query and passage vectors.

This mismatch means that naive approaches like direct MSE on raw scores fail: the score scales are incompatible, and the student's cosine similarity space cannot replicate the teacher's unbounded score distribution. The loss function must bridge this gap while preserving the teacher's ranking signal.

Additionally, a single loss function captures only one aspect of ranking quality. Pairwise ordering, listwise distribution shape, and in-batch discrimination each provide different gradient signals. The question is whether combining multiple losses produces a better student than any single loss.

## Decision

Use a weighted combination of three losses:

1. **MarginMSE (60% weight):** Compute the margin (score difference) between positive and negative passages according to both the teacher and the student. Minimize the MSE between these margins. This operates on relative score differences, eliminating the scale mismatch.
2. **Listwise KD (20% weight):** Apply softmax over the teacher's scores for all candidates of a query to produce a distribution. Apply softmax over the student's scores for the same candidates. Minimize the KL divergence between these distributions. This captures the full ranking shape.
3. **Contrastive loss (20% weight):** Standard InfoNCE over in-batch negatives using the student's own similarity scores. This provides direct gradient signal from the student's embedding space without teacher involvement.

Final loss: `L = 0.6 * L_margin_mse + 0.2 * L_listwise_kd + 0.2 * L_contrastive`

## Alternatives Considered

### Alternative 1: KL divergence only (standard distillation)
- **Pros:** The textbook approach to knowledge distillation. Well-understood theoretically. Single loss, simple to implement and tune. Captures the full distribution shape of teacher scores via softmax.
- **Cons:** KL divergence on softmax distributions is sensitive to the temperature parameter and can over-emphasize the tail of the distribution. It does not explicitly optimize pairwise ordering margins. When the teacher's score distribution is peaked (one clearly relevant result), KL divergence provides weak gradient for distinguishing among the negatives.
- **Why rejected:** KL divergence alone ignores the pairwise margin structure that is critical for retrieval ranking. In retrieval, the margin between the top result and the second result matters as much as the overall distribution shape. MarginMSE directly optimizes these margins. Empirically, KL-only training converged to approximately 90% of the target quality, while the combined loss reached 96%.

### Alternative 2: MSE on raw teacher scores
- **Pros:** Simplest possible distillation loss. Directly regresses student output toward teacher output. Easy to implement.
- **Cons:** Cross-encoder scores span [-10, +10] while bi-encoder cosine similarity spans [-1, +1]. The student physically cannot produce values in the teacher's range. Even with normalization, the distributions have different shapes. MSE on raw scores optimizes for score magnitude reproduction rather than ranking order.
- **Why rejected:** The scale mismatch is fundamental. Normalizing scores introduces assumptions about the score distribution that may not hold across different queries. MarginMSE avoids this entirely by operating on score differences, which are scale-invariant.

### Alternative 3: RankNet pairwise loss
- **Pros:** Directly optimizes pairwise ordering probability. Well-established in learning-to-rank literature. Theoretically principled.
- **Cons:** Quadratic cost in the number of candidates per query (every pair must be compared). For 20 candidates per query, this is 190 pairs. Does not leverage the teacher's score magnitudes, only orderings. Loses information about how confident the teacher is in each ordering.
- **Why rejected:** The quadratic cost is prohibitive for the candidate list sizes used in training (20-50 per query). More importantly, RankNet discards the teacher's score magnitudes. If the teacher gives passage A a score of 9.5 and passage B a score of 9.3, RankNet treats this the same as scores of 9.5 and 2.0. MarginMSE preserves these magnitude differences as learning signal.

### Alternative 4: Single loss (any one of the three)
- **Pros:** Simpler training loop. No weight hyperparameters to tune. Clearer attribution of training dynamics to a single loss.
- **Cons:** Each loss captures a different aspect of ranking quality. MarginMSE focuses on pairwise margins but ignores distribution shape. Listwise-KD captures distribution shape but can be dominated by easy negatives. Contrastive learns the embedding space geometry but has no teacher signal. Any single loss leaves gaps.
- **Why rejected:** Ablation experiments showed consistent improvement from combining losses. MarginMSE alone reached 93%. Adding Listwise-KD brought it to 95%. Adding Contrastive reached 96%. The gains are diminishing but meaningful, and the implementation cost of combining three losses is low.

## Consequences

### Positive
- MarginMSE at 60% weight provides the dominant gradient signal through scale-invariant pairwise margin matching. This is the single most impactful loss for transferring ranking knowledge across incompatible score scales.
- Listwise-KD captures distributional information that pairwise losses miss: the overall shape of the ranking, including the relative spacing between all candidates, not just adjacent pairs.
- Contrastive loss ensures the student's embedding space has good geometric properties (well-separated clusters) independent of teacher signal. This acts as a regularizer that prevents the student from overfitting to teacher-specific scoring patterns.
- The 60/20/20 weighting was determined empirically by grid search over {40,50,60,70} for MarginMSE and splitting the remainder equally. 60% MarginMSE consistently produced the best NDCG@10 across evaluation queries.

### Negative
- Three losses introduce three sets of hyperparameters (temperature for Listwise-KD, margin scale for MarginMSE, in-batch negative count for Contrastive). The interaction effects make systematic tuning harder.
- Training dynamics are harder to interpret. A spike in total loss could originate from any of the three components. Monitoring requires tracking each loss independently.
- The 60/20/20 weights were found empirically on the current dataset and model combination. These may not transfer to different domains or model sizes without re-tuning.

### Trade-offs
- Chose multi-loss richness over single-loss simplicity. The marginal quality gain of 3-6% justifies the moderate increase in training complexity.
- Chose fixed weights over dynamic/adaptive weighting. Adaptive schemes (e.g., GradNorm, uncertainty weighting) could theoretically adjust weights during training, but add significant complexity and were not explored.
- The contrastive loss at 20% weight acts partly as a regularizer. Increasing its weight would improve embedding space geometry but dilute the teacher's ranking signal. The current balance prioritizes teacher fidelity.
