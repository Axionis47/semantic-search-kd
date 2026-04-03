# ADR-005: Linear Temperature Annealing from T=4.0 to T=2.0

**Status:** Accepted  
**Date:** 2024-12-01  
**Deciders:** Project team

## Context

Temperature scaling is a core mechanism in knowledge distillation. When computing softmax over the teacher's relevance scores, the temperature parameter T controls how "soft" the resulting probability distribution is:

- **High temperature (T >> 1):** Flattens the distribution, making all candidates appear closer in probability. This reveals the teacher's "dark knowledge" about relative ordering among non-top results. A passage ranked 5th vs. 10th may have nearly identical probabilities at T=1 but distinguishable probabilities at T=4.
- **Low temperature (T -> 1):** Sharpens the distribution toward the top-ranked result, emphasizing the teacher's most confident judgments. The gradient signal becomes concentrated on distinguishing the best result from the rest.

The question is whether temperature should be fixed throughout training or varied, and if varied, what schedule to use.

The intuition behind annealing: early in training, the student benefits from broad, coarse-grained signal about how all candidates relate to each other (high T). As the student improves and has already learned the coarse ordering, it benefits more from precise, fine-grained signal about the top of the ranking (low T).

## Decision

Apply **linear temperature annealing** from T=4.0 at the start of training to T=2.0 at the end:

```
T(step) = 4.0 - (4.0 - 2.0) * (step / total_steps)
T(step) = 4.0 - 2.0 * (step / total_steps)
```

This schedule applies to the Listwise-KD loss component. MarginMSE and Contrastive losses are not temperature-dependent.

## Alternatives Considered

### Alternative 1: Fixed T=2.0 (Hinton standard)
- **Pros:** The value recommended in the original Hinton et al. (2015) knowledge distillation paper. Well-studied default. No schedule to tune. Produces a moderately soft distribution that balances dark knowledge revelation with signal clarity.
- **Cons:** A fixed temperature cannot adapt to the student's learning progress. Early in training, T=2.0 may be too sharp, causing the student to over-focus on the top result before it has learned the basic ordering. Late in training, T=2.0 may be too soft, wasting gradient signal on fine distinctions among irrelevant passages.
- **Why rejected:** While T=2.0 is a reasonable fixed choice, it represents a static compromise. The student's needs change during training, and a fixed temperature cannot reflect this. Experiments showed a 1-2% NDCG@10 improvement from annealing vs. fixed T=2.0, with the largest gains in early training stability.

### Alternative 2: Fixed T=4.0 (high temperature throughout)
- **Pros:** Maximizes dark knowledge transfer throughout training. Every gradient step receives information about the full ranking distribution. Useful when the candidate list contains many near-relevant passages that the student should learn to distinguish.
- **Cons:** High temperature persistently dilutes the signal about the most important distinctions (top-1 vs. top-5). Late in training, when the student needs to refine its top-of-ranking precision, T=4.0 spreads gradient signal too thin across the full candidate list. The student may learn good overall ordering but poor precision at the very top.
- **Why rejected:** Retrieval evaluation metrics (NDCG@10, MRR) are top-heavy. The top few results matter far more than results at position 20. Persistent high temperature under-weights this critical region. Annealing to T=2.0 restores focus on top-of-ranking precision when the student is ready for it.

### Alternative 3: Cosine annealing schedule
- **Pros:** Smoother transition with slower changes at the beginning and end of the schedule. Theoretically more gradual adaptation. Well-established in learning rate scheduling literature.
- **Cons:** Adds complexity for marginal benefit. In experiments, cosine annealing from T=4.0 to T=2.0 performed within 0.3% NDCG@10 of linear annealing. The shape of the annealing curve matters less than the start and end temperatures. Harder to reason about and explain.
- **Why rejected:** Performance was statistically indistinguishable from linear annealing. Linear is simpler to implement, easier to debug (the temperature at any step is immediately calculable), and more interpretable. Following the principle of choosing the simplest approach that achieves comparable results.

### Alternative 4: No temperature (T=1.0 fixed)
- **Pros:** Uses the teacher's raw softmax distribution without modification. No hyperparameter to tune. The student sees the teacher's "true" confidence levels.
- **Cons:** At T=1.0, the teacher's softmax distribution is typically very peaked. For a query with one clearly relevant passage, the softmax probability on that passage may be 0.98+, leaving less than 0.02 of probability mass spread across all other candidates. The gradient signal about relative ordering among non-top results is almost zero.
- **Why rejected:** T=1.0 effectively discards dark knowledge. The student learns to identify the top result but receives almost no signal about how to order the remaining candidates. This matters in practice because retrieval queries often return a ranked list, and users examine multiple results.

## Consequences

### Positive
- High initial temperature (T=4.0) provides a broad, stabilizing gradient signal early in training. The student receives information about the relative ordering of all candidates, not just the top result. This prevents early over-specialization and builds a robust foundation.
- The gradual reduction to T=2.0 sharpens the signal as training progresses, focusing the student's later learning on the most important distinctions at the top of the ranking.
- Linear schedule is trivially implementable (one line of code) and fully deterministic given the current step and total steps. No state tracking or complex schedule objects needed.
- The annealing approach decouples the "early training" and "late training" temperature requirements, avoiding the forced compromise of any fixed temperature.

### Negative
- Introduces two additional hyperparameters (start temperature, end temperature) that require tuning for new domains or model combinations.
- The linear schedule assumes that the student's learning progress is roughly uniform across training steps. In practice, learning is often faster early and slower late, which might call for a non-linear schedule. The linear approximation is good enough but not optimal.
- Temperature annealing only affects the Listwise-KD loss (20% of total loss weight). The impact on overall training is modulated by this weight. If Listwise-KD weight is reduced in future experiments, the benefit of annealing diminishes proportionally.

### Trade-offs
- Chose linear over cosine schedule for simplicity, accepting that cosine might be marginally better in theory.
- Chose T=4.0 as the starting temperature based on the observation that the teacher's score distribution spans roughly 20 points. T=4.0 flattens this into a distribution where the 5th-ranked passage retains at least 5% of probability mass, which provides meaningful gradient signal.
- The end temperature of T=2.0 (rather than T=1.0) deliberately retains some dark knowledge even at the end of training. Fully sharpening to T=1.0 was found to cause slight overfitting to the teacher's top-1 choice in the final training steps.
