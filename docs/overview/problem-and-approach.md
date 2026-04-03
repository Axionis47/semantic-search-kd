# Problem and Approach

## The Speed-Accuracy Tradeoff in Neural Search

Neural information retrieval has two dominant architectures, and each makes a painful compromise.

**Cross-encoders** process a query and document jointly through a single transformer. They see both texts at once, so they can attend across them and capture fine-grained relevance signals. The result: high accuracy. The cost: you must run the full model for every (query, document) pair. At 100ms per pair, scoring 10,000 candidates takes 1,000 seconds. Cross-encoders cannot serve real-time search.

**Bi-encoders** encode queries and documents independently into dense vectors. You pre-compute all document embeddings offline, then at query time you encode the query once and run a nearest-neighbor lookup. The result: sub-millisecond search over millions of documents. The cost: because the query and document never "see" each other inside the model, the relevance signal is weaker. Bi-encoders consistently underperform cross-encoders by 5-15% on ranking benchmarks.

> **Why does this matter?**
> For any search system serving real users, you need both speed and accuracy. Users expect results in under 200ms. But if those results are noticeably worse than what a slower model could produce, the speed is wasted.

## Why Existing Approaches Fall Short

Several strategies try to bridge this gap. None fully succeed.

**Caching cross-encoder results** works for known query-document pairs, but fails for open-ended search. You cannot pre-compute scores for queries you have never seen. Every new query requires fresh cross-encoder inference.

**Using cross-encoders as rerankers** is a common compromise: retrieve candidates with a fast bi-encoder, then rerank the top-K with a cross-encoder. This helps, but it adds latency (100ms+ per reranking pass), increases infrastructure complexity, and still depends on the bi-encoder's recall. If the bi-encoder misses a relevant document in the initial retrieval, the reranker never sees it.

**Training better bi-encoders from scratch** has diminishing returns. Models like E5, BGE, and GTE have pushed bi-encoder quality steadily upward, but the gap to cross-encoders persists. The architectural limitation is fundamental: without joint attention, there is a ceiling on how well you can score relevance from independent embeddings alone.

> **Why can't we just use a bigger bi-encoder?**
> Scaling up bi-encoder parameters improves accuracy, but eventually the latency cost offsets the benefit. A 1B-parameter bi-encoder may close the gap to cross-encoders, but it encodes queries in 10-50ms rather than 1ms, and its embeddings consume more memory. The tradeoff shifts rather than disappears.

## Why Knowledge Distillation Is the Answer

Knowledge distillation sidesteps the architectural limitation. Instead of training the bi-encoder to predict relevance labels directly, we train it to mimic a cross-encoder's scoring behavior.

The key insight: a cross-encoder's output contains richer information than binary relevance labels. When a cross-encoder scores document A at 0.92 and document B at 0.87, that margin encodes nuanced relevance distinctions that binary labels ("relevant" / "not relevant") throw away. By training the bi-encoder to reproduce these soft scores, we transfer the cross-encoder's understanding of *relative* relevance into the bi-encoder's embedding space.

This works because the bottleneck in bi-encoder quality is training signal, not model capacity. A 33M-parameter bi-encoder has enough representational power to encode fine relevance distinctions; it just needs the right supervision to learn them.

> **Why distillation over other transfer methods?**
> Feature-level distillation (matching hidden states) requires architectural compatibility between teacher and student. Attention transfer has similar constraints. Score-level distillation is architecture-agnostic: any model that outputs a relevance score can teach any model that produces embeddings. This gives us freedom to pick the best teacher and best student independently.

## Our Approach

### 3-Stage Hard Negative Mining

What the student learns depends entirely on what training examples it sees. Easy negatives (completely irrelevant documents) teach nothing useful. The student needs *hard* negatives: documents that look relevant but are not, or are less relevant than the true positive.

We mine negatives in three stages, forming a curriculum from easy to hard:

1. **BM25 negatives** - Lexically similar but semantically off. These are documents that share keywords with the query but miss the intent. They teach the student to look beyond surface overlap. (See [ADR-004](../decisions/adr-004.md))

2. **Teacher-scored negatives** - The cross-encoder teacher scores a pool of candidates and identifies documents it rates as "almost relevant." These are harder than BM25 negatives because they fool lexical matching *and* have partial semantic overlap. (See [ADR-004](../decisions/adr-004.md))

3. **ANCE negatives** - Mined using the student's own embeddings (Approximate Nearest Neighbor Negative Contrastive Estimation). These are the documents the student currently thinks are relevant but the teacher disagrees. They target the student's specific blind spots. (See [ADR-004](../decisions/adr-004.md))

> **Why a curriculum instead of just using the hardest negatives?**
> Starting with the hardest negatives causes training instability. The student has no baseline to distinguish subtle differences. The curriculum lets it build competence incrementally: first learn obvious distinctions, then refine.

### 3-Component Loss Function

No single loss captures everything the student needs to learn. We combine three losses, each targeting a different aspect of ranking quality:

| Loss | Weight | What It Teaches |
|------|--------|-----------------|
| **Margin-MSE** | 60% | Score *differences* between document pairs. If the teacher says doc A is 0.05 better than doc B, the student learns to preserve that margin. (See [ADR-003](../decisions/adr-003.md)) |
| **Listwise KD** | 20% | Ranking *order* across a full candidate list. Uses KL divergence between teacher and student score distributions. (See [ADR-003](../decisions/adr-003.md)) |
| **Contrastive** | 20% | Absolute query-document similarity. Pushes relevant documents closer to the query and irrelevant documents farther away in embedding space. (See [ADR-003](../decisions/adr-003.md)) |

> **Why not just Margin-MSE at 100%?**
> Margin-MSE alone makes the student good at pairwise comparisons but can produce inconsistent global rankings. Listwise loss ensures the full ordering is coherent. Contrastive loss ensures the embedding space is well-structured for nearest-neighbor retrieval (which operates on absolute distances, not pairwise margins).

### Temperature Annealing

Both the listwise and contrastive losses use a temperature parameter that controls how "peaked" the score distribution is. High temperature produces a soft, uniform distribution; low temperature produces a sharp distribution concentrated on the top-ranked documents.

We anneal temperature from 4.0 to 2.0 over the course of training:

- **Early training (T=4.0):** Soft distributions. The student sees broad ranking relationships and learns general relevance patterns. This prevents premature convergence on noisy early-stage predictions. (See [ADR-005](../decisions/adr-005.md))

- **Late training (T=2.0):** Sharper distributions. The student focuses on getting the top ranks exactly right, which is what matters most for search quality (nDCG@10 weights top results heavily). (See [ADR-005](../decisions/adr-005.md))

> **Why not a fixed temperature?**
> Fixed high temperature under-trains on top-rank precision. Fixed low temperature causes training instability early on when the student's scores are noisy. Annealing gets the benefits of both regimes.

## Model Choices

### Student: intfloat/e5-small-v2 (33M parameters)

E5-small-v2 is a strong baseline bi-encoder that punches above its weight. At 33M parameters, it is small enough to encode queries in ~1ms on CPU and to deploy on constrained infrastructure (Cloud Run with 1 vCPU). Despite its size, it has a well-structured embedding space from its pre-training on large text pair datasets, which gives knowledge distillation a solid foundation to build on. (See [ADR-001](../decisions/adr-001.md))

### Teacher: BAAI/bge-reranker-large (560M parameters)

BGE-reranker-large is one of the strongest open-source cross-encoders available. It was trained on diverse relevance data and generalizes well across domains. Its 560M parameters are large enough to produce high-quality soft labels for distillation, but small enough to run scoring on CPU within our training budget (~$4 total). (See [ADR-002](../decisions/adr-002.md))

> **Why not a larger teacher?**
> Larger cross-encoders (1B+ parameters) would produce marginally better soft labels, but the scoring cost during training would multiply. BGE-reranker-large hits the sweet spot: strong enough to teach effectively, affordable enough to score hundreds of thousands of query-document pairs.

## Results Summary

After knowledge distillation, the student reaches 97% of the teacher's accuracy while running 100x faster:

| Metric | Pre-KD Student | Post-KD Student | Teacher | Recovery |
|--------|---------------|-----------------|---------|----------|
| nDCG@10 | 0.719 | 0.882 | 0.91 | 97% |
| MRR@10 | 0.632 | 0.775 | 0.80 | 97% |
| Query latency | 1ms | 1ms | 100ms | 100x faster |
| Parameters | 33M | 33M | 560M | 17x smaller |

The student improved by 23% on nDCG@10 through distillation alone, with no architectural changes and no increase in inference cost. The remaining 3% gap to the teacher is the price of independent encoding; it is a fundamental architectural tradeoff, not a training deficiency.

For detailed architecture diagrams, see [C4 Context](../architecture/c4-context.md) and [C4 Container](../architecture/c4-container.md).
