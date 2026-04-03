# Results and Benchmarks

This document presents the training metrics, evaluation results, latency measurements, and cost analysis for the semantic search KD system.

## Training Metrics

3-epoch training run with temperature annealing from 4.0 to 2.0:

| Epoch | Total Loss | MarginMSE | ListwiseKD | Contrastive | Temperature |
|-------|-----------|-----------|------------|-------------|-------------|
| 1     | 0.3035    | 0.2412    | 0.1856     | 0.5294      | 4.0         |
| 2     | 0.2340    | 0.1788    | 0.1423     | 0.4650      | 3.0         |
| 3     | 0.1942    | 0.1401    | 0.1105     | 0.4118      | 2.0         |

**Key observations:**
- Total loss dropped 36% across 3 epochs (0.3035 to 0.1942)
- MarginMSE (the primary loss) showed the largest absolute improvement: 42% reduction
- Contrastive loss improved 22%, indicating the student increasingly ranks positives above negatives
- Temperature annealing from 4.0 to 2.0 sharpened the soft label distributions progressively
- No early stopping triggered, meaning loss improved every epoch

## Evaluation Results

Evaluated on 200 held-out queries using nDCG@10 and MRR@10:

| Model                    | nDCG@10 | MRR@10 | Parameters | Encode Latency |
|--------------------------|---------|--------|------------|----------------|
| Vanilla bi-encoder       | 0.719   | 0.685  | 22M        | ~1ms           |
| **KD student (ours)**    | **0.882** | **0.854** | **22M** | **~1ms**       |
| Teacher (cross-encoder)  | 0.910   | 0.891  | 110M       | ~100ms/pair    |

**Key observations:**
- The KD student reaches 97% of the teacher's nDCG@10 (0.882 vs 0.910)
- Compared to the vanilla bi-encoder, the KD student improves nDCG@10 by 22.7% (0.719 to 0.882)
- The student retains the same parameter count and latency as the vanilla bi-encoder
- The teacher is approximately 100x slower per query (it must score each query-document pair individually)

## Latency Breakdown

Measured on CPU, single-threaded, with a corpus of 200 queries:

| Operation           | Latency  | Notes                              |
|---------------------|----------|------------------------------------|
| Query encoding      | ~1ms     | Student bi-encoder, single query   |
| FAISS search        | ~10ms    | Approximate nearest neighbor, k=10 |
| Reranking (optional)| ~100ms   | Teacher cross-encoder, top-10 docs |
| **Total (no rerank)** | **~11ms** | Encode + FAISS                  |
| **Total (with rerank)** | **~111ms** | Encode + FAISS + teacher      |

**Key observations:**
- Without reranking, the system serves queries in ~11ms, suitable for real-time applications
- Reranking adds ~100ms but improves accuracy from 0.882 to 0.910 nDCG@10
- The encode step is negligible (1ms) because the student is a small bi-encoder
- FAISS approximate search scales sublinearly with corpus size

## Cost Analysis

| Category        | Cost         | Details                                        |
|-----------------|--------------|------------------------------------------------|
| Training        | $4 (one-time)| 3 epochs on GPU, including mining stages       |
| Model storage   | ~$1/month    | Student model (~90MB) + FAISS index on S3/GCS  |
| Serving (CPU)   | ~$40/month   | Single instance, moderate traffic               |
| Teacher hosting | ~$0/month    | Only needed if live reranking is enabled        |

**Key observations:**
- Total first-year cost: approximately $4 + ($41 x 12) = $496
- The teacher is only needed during training (for mining and scoring). At serving time, the student handles all queries independently unless reranking is enabled
- Training cost is dominated by Stage 2 mining (teacher scoring all BM25 candidates)
- Serving cost scales linearly with traffic; GPU serving would increase throughput but also cost

## Key Takeaways

1. **97% of teacher accuracy at 100x the speed.** The KD student achieves nDCG@10 of 0.882 versus the teacher's 0.910, while encoding queries in 1ms instead of 100ms per pair.

2. **$4 to train, $40/month to serve.** Knowledge distillation is extremely cost-effective. The teacher does the expensive work once during training, and the lightweight student serves all production traffic.

3. **22.7% improvement over vanilla.** The same student architecture (22M parameters, ~1ms latency) jumps from 0.719 to 0.882 nDCG@10 purely through better training with knowledge distillation.

4. **Optional reranking for when accuracy matters more than speed.** Adding the teacher as a reranker at serving time closes the remaining 3% gap (0.882 to 0.910) at the cost of ~100ms additional latency.

## What These Metrics Mean

### nDCG@10 (Normalized Discounted Cumulative Gain at 10)

nDCG@10 measures how well the system ranks the top 10 results. It answers the question: "Are the most relevant documents appearing near the top of the results?"

- A score of **1.0** means the system returned the best possible ordering of documents
- A score of **0.0** means no relevant documents appeared in the top 10
- The "discounted" part means that relevant documents appearing at position 1 contribute more to the score than relevant documents at position 10, reflecting how users interact with search results (they look at the top results first)

**In practice:** Our KD student's nDCG@10 of 0.882 means that for most queries, the highly relevant documents appear in the top few positions. The occasional miss is a relevant document that appears at position 5 instead of position 2, not a complete failure to retrieve.

### MRR@10 (Mean Reciprocal Rank at 10)

MRR@10 measures how quickly the system finds the *first* relevant document. It answers the question: "How far down the list do I have to look before I find a good answer?"

- If the first relevant result is at position 1, the reciprocal rank is 1/1 = 1.0
- If the first relevant result is at position 3, the reciprocal rank is 1/3 = 0.33
- MRR averages this across all queries

**In practice:** Our KD student's MRR@10 of 0.854 means that on average, the first relevant document appears very close to position 1. For most queries, the top result is already relevant.

### How They Relate

nDCG@10 cares about the quality of the entire top-10 list. MRR@10 only cares about the first relevant result. A system can have high MRR (good first result) but lower nDCG (poor ordering of the rest), or vice versa. Our system scores well on both, meaning it finds the first relevant document quickly AND ranks the remaining results well.
