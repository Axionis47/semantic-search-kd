# ADR-001: intfloat/e5-small-v2 as Student Bi-Encoder

**Status:** Accepted  
**Date:** 2024-12-01  
**Deciders:** Project team

## Context

The retrieval pipeline needs a lightweight bi-encoder that can run in production on resource-constrained infrastructure (2GB Cloud Run instances). The student model must satisfy several constraints simultaneously:

1. **Latency budget:** Query encoding must complete in under 5ms on CPU to keep total search latency under 50ms (encoding + FAISS lookup + post-processing).
2. **Memory footprint:** The full model, tokenizer, and runtime overhead must fit within a 2GB container alongside the FAISS index and application code.
3. **Embedding dimensionality:** Dimensions must be small enough for FAISS HNSW to maintain high throughput at the target index scale (up to 1M documents), while retaining enough capacity to capture semantic nuance.
4. **Knowledge distillation compatibility:** The model must be a strong enough learner to absorb soft-label signal from the teacher cross-encoder without saturating too early.
5. **Asymmetric retrieval support:** Queries and passages have fundamentally different characteristics. The model should handle this natively rather than requiring external wrappers.

## Decision

Use **intfloat/e5-small-v2** as the student bi-encoder for knowledge distillation and production serving.

Key properties:
- 33M parameters (MiniLM architecture)
- 384-dimensional output embeddings
- Built-in `query:` and `passage:` prefixes for asymmetric retrieval
- Pre-trained on large-scale text pairs with contrastive learning
- Apache 2.0 license

## Alternatives Considered

### Alternative 1: sentence-transformers/all-MiniLM-L6-v2
- **Pros:** Most widely adopted sentence embedding model. Extensive community benchmarks and examples. Well-tested in production across many organizations. Same 384-dim output and similar parameter count.
- **Cons:** Slightly lower baseline on MTEB retrieval tasks (NDCG@10 ~0.5 lower on average). No native query/passage prefix convention, meaning symmetric treatment of queries and documents. Community adoption skews toward semantic similarity rather than retrieval specifically.
- **Why rejected:** The lack of asymmetric prefix support means the model treats "What is Python?" and "Python is a programming language" identically during encoding. E5's prefix mechanism lets the model learn distinct representations for query intent vs. passage content, which is a meaningful advantage for retrieval workloads. The baseline quality gap, while small, compounds across the distillation stages.

### Alternative 2: BAAI/bge-small-en-v1.5
- **Pros:** Strong MTEB scores, comparable or slightly better than e5-small-v2 on some benchmarks. Supports instruction-based prefixing. Active maintenance from BAAI.
- **Cons:** Newer model with less production deployment history at the time of evaluation. Instruction prefix format (`Represent this sentence:`) is more verbose, adding tokens and latency. Fewer published knowledge distillation experiments using BGE as a student.
- **Why rejected:** While benchmark numbers are competitive, the limited track record in KD pipelines introduced risk. The verbose prefix format wastes tokens in the 512-token budget. E5's simpler prefix convention (`query:`, `passage:`) is more token-efficient and has more published KD results to reference.

### Alternative 3: thenlper/gte-small
- **Pros:** Competitive retrieval quality. Lightweight architecture similar to e5-small.
- **Cons:** No prefix convention at all for asymmetric retrieval. Less documentation on fine-tuning behavior. Smaller ecosystem of tooling and examples.
- **Why rejected:** Without a prefix mechanism, the model cannot distinguish between query-side and passage-side encoding. Adding a custom prefix post-hoc does not replicate the benefit of pre-training with prefix-aware objectives. This would require additional engineering to approximate what e5 provides natively.

## Consequences

### Positive
- 33M parameters loads in under 200MB of RAM, leaving ample headroom in the 2GB container for the FAISS index and application logic.
- 384-dimensional embeddings keep the FAISS HNSW index compact: 1M documents at 384 dims in float32 is roughly 1.5GB, fitting within deployment constraints.
- The `query:`/`passage:` prefix convention aligns naturally with the retrieval pipeline's asymmetric design, requiring zero custom preprocessing.
- Strong pre-trained baseline means the model already captures meaningful semantic structure before distillation begins, reducing the number of KD training steps needed.

### Negative
- 384 dimensions is a hard ceiling on representational capacity. For domains requiring very fine-grained semantic distinctions (e.g., legal clause differentiation), this may be insufficient.
- The e5 prefix convention is non-standard outside the e5 family. Switching student models later would require updating all encoding paths to handle different prefix formats.
- 33M parameters limits the model's ability to absorb arbitrarily complex teacher signal. There is a distillation ceiling beyond which a larger student would be needed.

### Trade-offs
- Chose deployment simplicity and latency over maximum representational power. A 768-dim model would capture more nuance but would double index size and increase encoding time.
- Chose a well-tested model over potentially higher-scoring newer alternatives. Production reliability outweighed marginal benchmark gains.
- The prefix convention creates a mild vendor lock-in to the e5 encoding style, but this is acceptable given the clear retrieval quality benefits.
