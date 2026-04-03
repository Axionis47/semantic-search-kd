# ADR-007: Native PyTorch Serving Over ONNX Runtime

**Status:** Accepted  
**Date:** 2024-12-01  
**Deciders:** Project team

## Context

The student bi-encoder must be served in production to encode incoming queries into 384-dimensional vectors for FAISS lookup. The serving infrastructure must balance inference latency, deployment complexity, and maintainability.

Profiling the retrieval pipeline reveals the following latency breakdown for a single query:

| Component | Latency (p50) | Latency (p99) |
|-----------|--------------|--------------|
| Query tokenization | <0.5ms | <1ms |
| Model forward pass | ~1ms | ~2ms |
| FAISS HNSW search | ~10ms | ~15ms |
| Post-processing | <1ms | <1ms |
| **Total** | **~12.5ms** | **~19ms** |

The model forward pass (query encoding) accounts for approximately 8% of total latency. The FAISS search dominates at approximately 80%.

## Decision

Serve the student model using **native PyTorch** (torch.no_grad() inference) rather than converting to ONNX Runtime, TorchScript, or TensorRT.

An ONNX export script is maintained in the repository as an escape hatch for future optimization if the latency profile changes.

## Alternatives Considered

### Alternative 1: ONNX Runtime
- **Pros:** Typically provides 2-3x inference speedup through graph optimizations (operator fusion, constant folding, memory planning). Cross-platform runtime with minimal dependencies. CPU and GPU execution providers. Industry standard for production ML serving. Would reduce the forward pass from 1ms to approximately 0.3-0.5ms.
- **Cons:** ONNX export introduces version compatibility constraints between PyTorch, ONNX opset versions, and ONNX Runtime versions. Dynamic shapes (variable sequence length) require careful export configuration. Debugging ONNX models is significantly harder than PyTorch: no breakpoints, no intermediate tensor inspection, limited error messages. Custom tokenization logic must be handled separately outside the ONNX graph. Export can silently produce numerically different results for certain operations (especially around attention masking edge cases).
- **Why rejected:** The forward pass is 1ms. Optimizing it to 0.3ms saves 0.7ms per query, a 5.6% reduction in total pipeline latency. This saving does not materially change the user experience or enable a meaningful throughput increase. Meanwhile, the ONNX export and version management complexity is ongoing: every model retrain requires re-exporting and re-validating the ONNX model. The maintenance cost exceeds the latency benefit.

### Alternative 2: TorchScript (torch.jit.trace / torch.jit.script)
- **Pros:** Stays within the PyTorch ecosystem. Enables graph-mode optimizations without leaving the framework. Can be loaded without Python (via LibTorch) for C++ serving. Moderate speedup (1.3-1.5x) from graph optimizations.
- **Cons:** TorchScript has a restricted Python subset. Models using dynamic control flow, certain Python builtins, or complex tokenization logic may fail to script. Traced models are fragile to input shape changes. Error messages from TorchScript compilation are often cryptic. The PyTorch team has signaled reduced investment in TorchScript in favor of torch.compile (PyTorch 2.x).
- **Why rejected:** The speedup is smaller than ONNX (1.3x vs. 2-3x), so the latency argument is even weaker. TorchScript's restrictions add friction to model iteration: changes to the model architecture or preprocessing may break the script. With PyTorch moving toward torch.compile, investing in TorchScript is misaligned with the framework's direction.

### Alternative 3: TensorRT
- **Pros:** Maximum possible inference speed on NVIDIA GPUs. Kernel auto-tuning for specific GPU architectures. INT8 quantization support with calibration. Can achieve 5-10x speedup over native PyTorch on GPU.
- **Cons:** NVIDIA GPU-only. The production environment is CPU-based Cloud Run. TensorRT compilation is hardware-specific: a model compiled for T4 may not work on A100. The compilation process is lengthy and must be repeated for each target GPU. Adds a hard dependency on NVIDIA's proprietary toolchain.
- **Why rejected:** The production deployment target is CPU-based Cloud Run instances. TensorRT provides no benefit on CPU. Even if the deployment moved to GPU, the query encoding latency (1ms on CPU) is not the bottleneck. TensorRT would optimize the wrong component.

## Consequences

### Positive
- Zero additional build steps or export processes. The model checkpoint saved during training is the same artifact served in production. This eliminates an entire class of bugs related to export/conversion mismatch.
- Full PyTorch debugging capabilities in production: print statements, breakpoints, intermediate tensor inspection, profiling with torch.profiler. When issues arise, the debugging experience is identical to development.
- Model updates are deploy-and-restart. No re-export, no re-validation, no re-compilation. This reduces the time from training completion to production deployment.
- The team can use any PyTorch feature without worrying about ONNX/TorchScript compatibility. This removes a constraint on model architecture evolution.

### Negative
- The 1ms forward pass is roughly 2-3x slower than it could be with ONNX Runtime. While this does not matter at current throughput levels, it means each Cloud Run instance handles fewer concurrent queries per second.
- If query encoding latency ever becomes the bottleneck (e.g., through batched encoding of multiple queries, longer input sequences, or a larger student model), the decision to stay on native PyTorch would need revisiting.
- PyTorch's eager-mode execution includes Python interpreter overhead that compiled runtimes avoid. At very high concurrency, this overhead could become visible.

### Trade-offs
- Chose development velocity and debugging simplicity over maximum inference performance. The 0.7ms savings from ONNX does not justify the ongoing maintenance burden of an export pipeline.
- Chose to maintain an ONNX export script as an uncommitted escape hatch rather than removing the option entirely. This provides a migration path if the latency profile changes without committing to the complexity now.
- Chose to optimize the bottleneck (FAISS search at 10ms) rather than the non-bottleneck (model forward pass at 1ms). Engineering effort is better spent on FAISS parameter tuning, caching, or index sharding than on shaving sub-millisecond time from encoding.
