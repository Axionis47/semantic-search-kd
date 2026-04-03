# ADR-002: BAAI/bge-reranker-large as Teacher Cross-Encoder

**Status:** Accepted  
**Date:** 2024-12-01  
**Deciders:** Project team

## Context

Knowledge distillation requires a high-quality teacher model that produces reliable soft labels for training the student bi-encoder. The teacher's role is critical: noisy or poorly calibrated teacher scores propagate directly into the student's learned ranking behavior. A teacher that confidently assigns incorrect orderings will teach the student to reproduce those errors.

The teacher operates offline during training (scoring query-passage pairs to generate distillation labels) and optionally at inference time as a reranker over the student's top-k candidates. Because the teacher runs offline for training, inference speed is secondary to output quality and score calibration.

Requirements for the teacher:
1. **Ranking accuracy:** Must produce state-of-the-art relevance judgments, since the student's ceiling is bounded by teacher quality.
2. **Score calibration:** Scores must be well-ordered and reasonably scaled. Poorly calibrated scores (e.g., all scores clustered in a narrow range) destroy the margin signal that MarginMSE loss depends on.
3. **Cross-encoder architecture:** Must attend jointly to query and passage tokens for deep interaction modeling.
4. **Practical scoring throughput:** Must be able to score millions of query-passage pairs in reasonable time (hours, not weeks) on available GPU hardware.

## Decision

Use **BAAI/bge-reranker-large** as the teacher cross-encoder for knowledge distillation and optional inference-time reranking.

Key properties:
- 560M parameters (XLM-RoBERTa-large backbone)
- Cross-encoder architecture with joint query-passage attention
- Trained on large-scale relevance data including MS MARCO
- Outputs well-calibrated relevance scores in approximately the [-10, +10] range
- Dual-purpose: serves as both KD teacher and optional production reranker

## Alternatives Considered

### Alternative 1: cross-encoder/ms-marco-MiniLM-L-12-v2
- **Pros:** Much smaller (33M parameters), faster inference. Well-established in the sentence-transformers ecosystem. Good enough for many reranking tasks. Lower GPU cost for scoring large training sets.
- **Cons:** Measurably lower ranking quality on NDCG@10 and MRR benchmarks compared to bge-reranker-large. Score distribution is narrower, with less separation between relevant and irrelevant passages. English-only architecture.
- **Why rejected:** The quality gap matters more for KD than for direct reranking. When a teacher produces soft labels, every ordering mistake and every poorly separated score pair becomes a learning signal that the student internalizes. The MiniLM cross-encoder's narrower score distribution compresses the margin signal, making it harder for MarginMSE loss to learn clean pairwise orderings. The GPU cost savings do not justify the quality ceiling reduction for the student.

### Alternative 2: castorini/monot5-base-msmarco
- **Pros:** T5-based generative reranker with a different inductive bias. Produces relevance probability via "true"/"false" token logits, which is naturally calibrated as a probability. Competitive ranking quality.
- **Cons:** Fundamentally different scoring paradigm (generative vs. discriminative). Relevance scores are log-probabilities of a "true" token, not direct relevance scores. Slower inference due to autoregressive decoding. Score distribution behaves differently from discriminative cross-encoders, requiring different loss function tuning.
- **Why rejected:** The generative scoring paradigm introduces a domain mismatch with the distillation losses (MarginMSE, Listwise-KD) that expect discriminative relevance scores. Converting T5 log-probabilities to margin-compatible scores requires additional calibration steps that add complexity without clear benefit. The autoregressive decoding also makes large-scale offline scoring significantly slower.

## Consequences

### Positive
- State-of-the-art ranking quality ensures the student has the highest possible learning ceiling. The teacher's accuracy directly bounds what the student can achieve through distillation.
- Well-calibrated score distribution (spanning roughly [-10, +10] with clear separation between relevant and irrelevant) provides clean gradient signal for MarginMSE loss. Score margins between positive and negative passages are large and consistent.
- The XLM-RoBERTa backbone means the teacher has been pre-trained on multilingual data. While the current pipeline is English-focused, this provides a foundation for multilingual expansion without changing the teacher.
- Dual-purpose architecture means the same model can serve as an optional second-stage reranker at inference time, reranking the student's top-k results for maximum precision when latency budget allows.

### Negative
- 560M parameters requires a GPU for practical scoring throughput. Scoring 1M query-passage pairs takes approximately 4-6 hours on a single A100, which is manageable but non-trivial for iterating on training data.
- The model cannot be deployed as the primary retrieval mechanism (cross-encoders require joint encoding of each query-passage pair, making them O(n) per query over the corpus).
- Dependency on BAAI's model release. If the model is updated or deprecated, the distillation pipeline's reproducibility depends on pinning the exact model version.

### Trade-offs
- Chose maximum teacher quality over scoring speed. The offline nature of KD label generation means throughput is a one-time cost per training run, while quality permanently affects the student.
- Chose a discriminative cross-encoder over a generative alternative to maintain compatibility with the multi-loss distillation framework without additional calibration layers.
- The large model size means the teacher cannot be co-located with the student in the same 2GB container for inference-time reranking without upgrading infrastructure. If reranking is desired in production, it requires a separate serving instance or a larger container.
