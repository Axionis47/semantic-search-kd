# ADR-006: FAISS HNSW Index (M=32, efConstruction=200, efSearch=64)

**Status:** Accepted  
**Date:** 2024-12-01  
**Deciders:** Project team

## Context

The retrieval pipeline needs an approximate nearest neighbor (ANN) index to search over document embeddings at query time. The index must satisfy latency, recall, and deployment constraints:

1. **Latency:** ANN lookup must complete in under 20ms for a single query to stay within the total 50ms search budget (encoding + ANN + post-processing).
2. **Recall:** At least 95% recall@100 (the true top-100 nearest neighbors must appear in the ANN's returned top-100 at least 95% of the time). Retrieval errors from the index directly reduce end-to-end search quality.
3. **Scale:** Target corpus size is 10K-1M documents. The solution must handle the upper end comfortably with room for moderate growth.
4. **Deployment:** The index must be servable from a single Cloud Run container with limited memory (2GB total). The index is built offline and loaded at startup.
5. **Operational simplicity:** Fewer moving parts are preferred. The index should be a single file loadable by FAISS with no external dependencies.

## Decision

Use **FAISS HNSW** (Hierarchical Navigable Small World) with the following configuration:

- **M=32:** Each node connects to 32 neighbors per layer. Higher M improves recall at the cost of memory and build time.
- **efConstruction=200:** Search depth during index construction. Higher values produce a better-connected graph at the cost of longer build times.
- **efSearch=64:** Search depth at query time. This is the primary recall-vs-latency knob and can be adjusted without rebuilding the index.

At 384 dimensions and 1M documents in float32, the index consumes approximately 1.5GB of memory (embeddings) plus roughly 250MB of graph structure, fitting within the 2GB deployment target with margin for the model and application.

## Alternatives Considered

### Alternative 1: FAISS IVF-PQ (Inverted File with Product Quantization)
- **Pros:** Superior memory efficiency through product quantization (compresses 384-dim float32 to ~48 bytes per vector using PQ with 48 sub-quantizers). Handles 50M+ document collections where HNSW's memory overhead becomes prohibitive. Well-suited for very large-scale retrieval. Supports GPU-accelerated index building.
- **Cons:** Requires a training step to learn the quantization codebook and cluster centroids. Quantization introduces irreversible information loss, reducing recall at small scales. At 1M documents, IVF-PQ's recall@100 is typically 90-93% with standard settings, below the 95% target without aggressive nprobe tuning. More parameters to tune (nlist, nprobe, PQ sub-quantizers, training set size).
- **Why rejected:** At the current scale (10K-1M documents), IVF-PQ's quantization overhead reduces recall below HNSW without providing a meaningful memory advantage. HNSW achieves 97%+ recall@100 at this scale. IVF-PQ's strengths emerge at 50M+ documents where HNSW's memory becomes impractical. The configuration is designed to support an IVF-PQ migration path if corpus size grows beyond 10M.

### Alternative 2: Flat (brute-force) index
- **Pros:** Exact nearest neighbor search with perfect recall. No index construction step. No approximation errors. Simplest possible implementation. Results are deterministic and reproducible.
- **Cons:** O(n) search time per query. At 1M documents with 384 dimensions, brute-force search takes 50-100ms on CPU, exceeding the 20ms latency budget. Latency scales linearly with corpus size, providing no path to growth.
- **Why rejected:** Brute-force is acceptable at 10K documents (under 5ms) but violates latency constraints at the target upper bound of 1M. Since the pipeline must handle growth toward 1M, an approximate method is necessary. Flat index remains useful as a recall benchmark during evaluation.

### Alternative 3: Google ScaNN
- **Pros:** Competitive or superior performance to HNSW on many benchmarks. Supports asymmetric hashing for memory-efficient search. Developed and maintained by Google with active support.
- **Cons:** Smaller open-source ecosystem than FAISS. Python bindings are less mature, with occasional installation issues on non-standard platforms. Fewer examples of production deployment patterns outside Google Cloud. Integration with existing FAISS-based tooling would require rewriting index management code.
- **Why rejected:** Performance is comparable to HNSW at the target scale. FAISS has a larger ecosystem, more community resources, and more proven deployment patterns on Cloud Run. The operational risk of a less-tested library outweighs the marginal performance differences. If ScaNN's ecosystem matures, it could be reconsidered.

## Consequences

### Positive
- 97%+ recall@100 at 1M documents with the chosen parameters. This exceeds the 95% target and provides margin for edge cases.
- Sub-15ms query latency at 1M documents on CPU. Well within the 20ms budget, leaving room for post-processing and network overhead.
- Single-file index that loads with a single FAISS call. No cluster centroids, codebooks, or auxiliary files to manage. Deployment is straightforward: copy the index file to the container.
- efSearch is adjustable at query time without rebuilding the index. This provides a runtime knob to trade latency for recall as requirements evolve.
- HNSW supports incremental addition of new vectors without full rebuild (though graph quality degrades gradually). This enables partial index updates for small corpus changes.

### Negative
- Memory consumption is higher than quantized alternatives. At 1M documents, the full float32 embeddings plus HNSW graph structure consume approximately 1.7GB. This limits the maximum corpus size within the 2GB container.
- Index build time grows super-linearly with M and efConstruction. Building the index for 1M documents with M=32 and efConstruction=200 takes approximately 15-20 minutes on a single CPU core. Rebuilds are not instant.
- HNSW does not support deletion of individual vectors. Removing a document requires rebuilding the index from scratch (or maintaining a deletion blacklist at query time, which is an additional layer of complexity).
- No native GPU acceleration for HNSW in FAISS. Index building and search are CPU-only. This is not a bottleneck at current scale but could become one at larger sizes.

### Trade-offs
- Chose recall quality over memory efficiency. HNSW's unquantized float32 storage uses more memory than IVF-PQ but preserves the full precision of the distilled student embeddings.
- Chose operational simplicity over maximum scalability. HNSW is simpler to deploy and operate than IVF-PQ, at the cost of a lower scaling ceiling. The configuration includes a documented migration path to IVF-PQ for future growth.
- M=32 is on the higher end of typical HNSW configurations (16-48). This biases toward recall over memory. For use cases where memory is tighter, M=16 with efSearch=128 achieves similar recall with lower memory but higher query latency.
