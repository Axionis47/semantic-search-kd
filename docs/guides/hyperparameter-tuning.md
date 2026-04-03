# Hyperparameter Tuning Guide

## When you need this

You have a working training pipeline and want to improve model quality. This guide tells you what to tune, in what order, and what effect each parameter has. Start from the top of the list and work down; the parameters are ordered by impact.

## Tuning order

Tune in this sequence for the best return on effort:

1. Loss weights (highest impact on ranking quality)
2. Temperature schedule (controls knowledge transfer fidelity)
3. Mining stages (controls negative difficulty)
4. Learning rate and batch size (standard training knobs)
5. FAISS index parameters (affects serving latency and recall)

## 1. Loss weights

The combined KD loss has three components, configured in `configs/kd.yaml`:

```yaml
kd:
  loss_weights:
    margin_mse: 0.6
    listwise_kd: 0.2
    contrastive: 0.2
```

### What each component does

- **Margin MSE (0.6)**: trains the student to preserve the teacher's pairwise score differences. This is the primary distillation signal. Higher weight means the student focuses more on matching the teacher's relative ordering.
- **Listwise KD (0.2)**: aligns the student's full ranking distribution with the teacher's. Complements margin MSE by capturing list-level structure rather than just pairs.
- **Contrastive (0.2)**: standard contrastive loss that pushes positives closer and negatives farther in embedding space. Provides a direct retrieval signal independent of the teacher.

### What happens when you change them

| Change | Effect |
|--------|--------|
| Increase margin_mse to 0.8, reduce others | Stronger teacher mimicry; student rankings track the teacher more closely, but may overfit to teacher errors |
| Increase contrastive to 0.4, reduce margin_mse | More emphasis on embedding geometry; better recall but may lose fine-grained ranking quality |
| Equal weights (0.33 each) | Balanced approach; reasonable starting point if the default split underperforms |
| Set contrastive to 0 | Pure distillation mode; only useful when the teacher is very accurate |

**Recommendation**: start with the default 0.6/0.2/0.2 split. If nDCG improves but MRR does not, try increasing listwise_kd to 0.3 and reducing margin_mse to 0.5.

## 2. Temperature schedule

```yaml
kd:
  temperature_start: 4.0
  temperature_end: 2.0
  temperature_schedule: "linear"
```

Temperature controls the softness of the teacher's score distribution during distillation.

| Temperature | Effect on teacher targets |
|-------------|-------------------------|
| High (4.0-8.0) | Very soft distributions. The student sees smooth probability differences between positives and negatives. Good for early training when the student is far from the teacher. |
| Low (1.0-2.0) | Sharp distributions. The student sees nearly one-hot targets. Good for late training when fine-tuning precise rankings. |

### Annealing strategies

| Schedule | Behavior |
|----------|----------|
| `linear` | Temperature decreases linearly from start to end over training. Default and generally robust. |
| `cosine` | Temperature decreases following a cosine curve (slower at start and end, faster in middle). |
| `constant` | No annealing. Use `temperature_start` throughout. Set this if you want to isolate the effect of a single temperature. |

**Recommendation**: start with `linear` from 4.0 to 2.0. If the student converges too quickly and plateaus, try starting at 6.0. If training is unstable, try narrowing the range to 3.0 to 2.0.

## 3. Mining stages

The curriculum has three stages of increasing difficulty:

### Stage 1: BM25 negatives

```yaml
mining:
  stage_a:
    strategy: "in_batch"
    negatives_per_query: 7
```

BM25 finds lexically similar but semantically irrelevant passages. These are the easiest negatives. Use this stage for:

- Warmup before harder mining
- Quick iteration when prototyping
- CPU-only training (no teacher inference needed)

### Stage 2: Teacher-mined negatives

```yaml
mining:
  stage_b:
    strategy: "teacher"
    teacher_top_k: 100
    teacher_select_k: 20
```

The teacher cross-encoder re-scores the top-100 BM25 candidates and selects the 20 hardest negatives (highest teacher score among non-relevant passages). Use this stage when:

- You have GPU access for teacher inference
- BM25-only training has plateaued
- You want the student to learn nuanced relevance distinctions

**Denoising**: passages with teacher score >= 0.7 or text overlap > 0.8 are dropped to avoid training on false negatives.

### Stage 3: ANCE (iterative mining)

```yaml
mining:
  stage_c:
    strategy: "ance"
    ance_top_k: 50
    ance_refresh_every_n_steps: 500
```

The student uses its own embeddings to find its current hardest negatives, refreshing the negative set every 500 training steps. Use this stage when:

- You are doing a full production training run
- Earlier stages have converged
- You want maximum ranking quality

**When to skip stages**: for small datasets (under 10K samples), Stage 1 alone is often sufficient. Stages 2 and 3 provide diminishing returns on small data but significant gains at scale.

## 4. Learning rate, batch size, epochs

```yaml
training:
  learning_rate: 2.0e-5
  batch_size: 32
  num_epochs: 3
  warmup_steps: 1000
  gradient_accumulation_steps: 2
```

### Learning rate

The default 2e-5 works well for fine-tuning pre-trained transformers. If loss oscillates, reduce to 1e-5. If loss decreases very slowly, try 5e-5.

### Batch size

Larger batch sizes provide more in-batch negatives and more stable gradients. If GPU memory is limited, use `gradient_accumulation_steps` to simulate larger batches:

| GPU memory | Batch size | Accumulation steps | Effective batch |
|------------|------------|-------------------|----------------|
| 8 GB | 8 | 4 | 32 |
| 16 GB | 16 | 2 | 32 |
| 24 GB+ | 32 | 1 | 32 |
| 48 GB+ | 64 | 1 | 64 |

### Epochs

For the demo, 2 epochs is enough. For production with 50K+ samples, 3 epochs is the default. Early stopping with `patience: 2` on nDCG@10 prevents overfitting, so setting a higher epoch count (5-10) is safe as long as early stopping is active.

## 5. FAISS index parameters

These parameters affect serving latency and recall, not training quality. Tune them after you have a trained model.

Configured in `configs/index.yaml`:

```yaml
hnsw:
  M: 32
  efConstruction: 200
  efSearch: 64
```

### HNSW parameters

| Parameter | Default | Range | Effect | Tune when |
|-----------|---------|-------|--------|-----------|
| M | 32 | 16-64 | Number of bi-directional links per node. Higher M improves recall but increases memory and build time. | Recall@10 is below target |
| efConstruction | 200 | 100-400 | Search depth during index build. Higher values produce a better-connected graph. | Index quality is poor (high recall gap vs. brute force) |
| efSearch | 64 | 32-128 | Search depth at query time. Higher values improve recall at the cost of latency. | Latency budget allows for more thoroughness, or recall is below target |

### Recall vs. latency tradeoff

The index validation in `index.yaml` requires recall@10 >= 0.97 against brute-force search:

```yaml
validation:
  recall_threshold: 0.97
  brute_force_top_k: 10
```

If validation fails, increase `efConstruction` and `M`. If latency is too high, reduce `efSearch` (at the cost of recall).

## Reference table

| Parameter | Default | Range | Effect | Tune when |
|-----------|---------|-------|--------|-----------|
| `kd.loss_weights.margin_mse` | 0.6 | 0.3-0.8 | Teacher pairwise score matching | nDCG is low |
| `kd.loss_weights.listwise_kd` | 0.2 | 0.1-0.4 | Full ranking distribution alignment | MRR is low relative to nDCG |
| `kd.loss_weights.contrastive` | 0.2 | 0.0-0.4 | Embedding space geometry | Recall@100 is low |
| `kd.temperature_start` | 4.0 | 2.0-8.0 | Softness of early teacher targets | Student converges too fast or too slow |
| `kd.temperature_end` | 2.0 | 1.0-4.0 | Sharpness of late teacher targets | Final ranking precision is poor |
| `kd.confidence_threshold` | 0.6 | 0.4-0.8 | Minimum teacher confidence for distillation | Noisy teacher predictions hurt training |
| `training.learning_rate` | 2e-5 | 1e-5 to 5e-5 | Training speed vs. stability | Loss oscillates or does not decrease |
| `training.batch_size` | 32 | 4-64 | Gradient stability, in-batch negatives | GPU memory constrained |
| `training.num_epochs` | 3 | 2-10 | Training duration (with early stopping) | Model has not converged |
| `mining.stage_b.teacher_top_k` | 100 | 50-200 | Candidate pool for teacher scoring | Teacher mining is too slow |
| `mining.stage_c.ance_refresh_every_n_steps` | 500 | 200-1000 | How often ANCE re-mines negatives | Training loss plateaus mid-epoch |
| `hnsw.M` | 32 | 16-64 | Graph connectivity, memory usage | Recall below target |
| `hnsw.efConstruction` | 200 | 100-400 | Index build quality | Recall gap vs. brute force |
| `hnsw.efSearch` | 64 | 32-128 | Query-time thoroughness vs. latency | Need faster queries or higher recall |
