# Custom Dataset Guide

## When you need this

You want to train a knowledge-distilled semantic search model on your own documents instead of (or in addition to) MS MARCO. This guide covers the required data format, how to adapt the data pipeline, chunking strategies, and tips for domain-specific datasets.

## Required data format

The training pipeline expects data in JSONL format with the following structure:

```json
{
  "query_id": "unique-query-id",
  "query": "the user's search query",
  "passages": {
    "passage_text": ["passage 1 text", "passage 2 text", "passage 3 text"],
    "is_selected": [0, 1, 0]
  }
}
```

Each line represents one query with its candidate passages:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_id` | string or int | Yes | Unique identifier for the query |
| `query` | string | Yes | The query text |
| `passages.passage_text` | list of strings | Yes | Candidate passages for this query |
| `passages.is_selected` | list of ints (0/1) | Yes | Relevance labels: 1 = relevant, 0 = not relevant |

**Important**: queries with no relevant passages (`is_selected` all zeros) are skipped during training. Each query needs at least one passage marked as relevant.

## Adapting the data pipeline

### Option A: Convert your data to JSONL format

The simplest approach is to convert your data into the expected JSONL format and place it in `data/raw/your_dataset/train.jsonl`. Then run the pipeline pointing at your file:

```bash
poetry run python scripts/train_kd_pipeline.py \
    --data-dir ./data \
    --max-samples 10000 \
    --epochs 3 \
    --batch-size 32 \
    --device cuda \
    --output-dir artifacts/models/kd_student_custom
```

The pipeline will look for `data/raw/msmarco/train.jsonl` by default. To use a different path, you need to modify the data loading in `scripts/train_kd_pipeline.py` or create a symlink:

```bash
mkdir -p data/raw/msmarco
ln -s /path/to/your/train.jsonl data/raw/msmarco/train.jsonl
```

### Option B: Modify fetch.py for a new data source

If your data lives in a database, API, or non-JSONL file format, create a new fetch function in `src/data/fetch.py`:

```python
def fetch_custom_dataset(output_dir: Path, max_samples: Optional[int] = None) -> Dict:
    """Fetch your custom dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load your data from wherever it lives
    # (database, CSV, API, etc.)
    records = load_your_data()

    # Write to JSONL in the expected format
    output_file = output_dir / "train.jsonl"
    with open(output_file, "w") as f:
        for i, record in enumerate(records):
            if max_samples and i >= max_samples:
                break
            row = {
                "query_id": record["id"],
                "query": record["question"],
                "passages": {
                    "passage_text": record["documents"],
                    "is_selected": record["relevance_labels"],
                },
            }
            f.write(json.dumps(row) + "\n")

    return {
        "dataset": "custom",
        "splits": {"train": {"file": str(output_file), "num_samples": i + 1}},
    }
```

### Option C: Modify prepare.py for a different passage structure

If your documents are not pre-segmented into passages (for example, full articles or PDFs), adapt `src/data/prepare.py`. The key function is `prepare_msmarco_split`, which:

1. Reads each JSONL line
2. Extracts passages and relevance labels
3. Chunks each passage using `TextChunker`
4. Writes the results as Parquet

The Parquet schema used downstream:

| Column | Type | Description |
|--------|------|-------------|
| `chunk_id` | string | Unique chunk identifier |
| `doc_id` | string | Parent document identifier |
| `query_id` | string | Associated query ID |
| `query_text` | string | Query text |
| `text` | string | Chunk text content |
| `tokens` | int | Token count |
| `is_relevant` | int | 0 or 1 |
| `split` | string | train/validation/test |

## Chunking strategy for your documents

The `TextChunker` in `src/utils/chunk.py` is configured with two parameters:

```python
chunker = TextChunker(max_tokens=512, stride=80)
```

- **max_tokens**: maximum number of tokens per chunk. Set this to match the student model's max input length (512 for E5-small).
- **stride**: overlap between consecutive chunks. An 80-token stride ensures context is not lost at chunk boundaries.

### Choosing chunk parameters for your domain

| Document type | Recommended max_tokens | Recommended stride | Notes |
|--------------|----------------------|-------------------|-------|
| Short passages (1-2 paragraphs) | 512 | 80 | Default. Works for FAQ, support articles. |
| Long articles (1000+ words) | 512 | 128 | Higher stride preserves more cross-chunk context. |
| Technical documentation | 256 | 64 | Shorter chunks if queries target specific details. |
| Legal or medical text | 512 | 160 | Higher stride because key clauses often span paragraph boundaries. |

If your documents are already short enough to fit in 512 tokens, the chunker will produce one chunk per document with no splitting.

## Training with custom data

### Step 1: Prepare your JSONL file

Place your data at `data/raw/msmarco/train.jsonl` (or symlink it as shown above).

### Step 2: Run the training pipeline

```bash
poetry run python scripts/train_kd_pipeline.py \
    --max-samples 50000 \
    --epochs 3 \
    --batch-size 32 \
    --stage 1 \
    --device cuda \
    --output-dir artifacts/models/kd_student_custom \
    --log-level INFO
```

Start with `--stage 1` (BM25 only) to verify the pipeline works with your data before running the full curriculum.

### Step 3: Evaluate

```bash
poetry run python scripts/simple_eval.py \
    --model-path artifacts/models/kd_student_custom/best_model \
    --data-path data/raw/msmarco/train.jsonl \
    --output-path artifacts/evaluation/custom_results.json \
    --max-samples 1000 \
    --device cuda
```

## Evaluating on your domain

The built-in evaluation computes nDCG@1, nDCG@5, nDCG@10, and MRR@10. These metrics require relevance judgments in the `is_selected` field.

### If you have graded relevance (not just binary)

The pipeline currently treats relevance as binary (0/1). If your dataset has graded relevance (0-3), you can:

1. Threshold at a level (e.g., grade >= 2 counts as relevant)
2. Map grades to the `is_selected` field accordingly

### If you have no relevance judgments

You can still train using the teacher model as a source of pseudo-labels. The teacher cross-encoder will score query-passage pairs during mining, and those scores serve as soft relevance labels for distillation. However, evaluation without ground-truth labels is unreliable. Consider:

- Creating a small manually annotated evaluation set (100-500 queries)
- Using the teacher's rankings as a proxy evaluation signal
- Running A/B tests in production

## Tips for domain-specific datasets

### 1. Start with a pre-trained student, not from scratch

The default student (`intfloat/e5-small-v2`) is pre-trained on general web text. Fine-tuning it on your domain via KD is much more effective than training from scratch.

### 2. Balance query and passage lengths

If your queries are very short (2-3 words) but passages are long, the teacher may struggle to score accurately. Consider:

- Expanding short queries with context before feeding to the teacher
- Using the passage title as additional query context

### 3. Handle domain vocabulary

If your domain has specialized vocabulary (medical terms, legal jargon, code identifiers), verify that the student tokenizer handles it reasonably. E5-small uses a WordPiece tokenizer that will split unknown words into subwords. This is usually fine, but for highly specialized domains, consider:

- Adding domain text to BM25 index building for better lexical negatives
- Increasing `teacher_top_k` in Stage 2 to get more diverse candidates

### 4. Watch for class imbalance

If most passages are irrelevant (common in retrieval), the contrastive loss may dominate. Consider increasing `margin_mse_weight` to 0.7 or 0.8 to emphasize the teacher signal.

### 5. Data size guidelines

| Dataset size | Expected quality | Training time (L4 GPU) |
|-------------|-----------------|----------------------|
| 1K-5K queries | Marginal improvement over vanilla | 10-30 minutes |
| 5K-50K queries | Solid improvement, especially with Stage 2 | 1-4 hours |
| 50K-500K queries | Production quality | 4-24 hours |
| 500K+ queries | Diminishing returns; focus on mining quality | 1-3 days |

### 6. Iterate quickly, then scale

1. Start with 1K samples and Stage 1 to validate the data format works
2. Scale to 10K samples and add Stage 2 to check for meaningful improvement
3. Run the full dataset with all three stages for the final model
