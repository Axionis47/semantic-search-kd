# Quickstart: 5-Minute Local Demo

## When you need this

You want to see the full knowledge distillation pipeline running on your machine before committing time to understanding the architecture. This guide gets you from zero to a trained model and evaluation report in about five minutes using a small data sample on CPU.

## Prerequisites

- **Python 3.10+** installed
- **Poetry** installed (`curl -sSL https://install.python-poetry.org | python3 -`)
- About 2 GB of free disk space (for models and data)
- No GPU required; the demo runs entirely on CPU

## 1. Install dependencies

```bash
cd semantic-search-kd
make install
```

This runs `poetry install` and sets up pre-commit hooks. All dependencies are pinned in `pyproject.toml`.

## 2. Run the demo pipeline

```bash
./scripts/run_demo_pipeline.sh
```

The script runs four steps end to end:

1. **Baseline evaluation**: evaluates the vanilla `intfloat/e5-small-v2` student model on 200 MS MARCO samples
2. **KD training**: trains the student for 2 epochs with batch size 4, using BM25-mined hard negatives and teacher scores from `BAAI/bge-reranker-large`
3. **Post-KD evaluation**: evaluates the distilled student on the same samples
4. **Report generation**: writes a comparison report to `artifacts/evaluation_demo/KD_REPORT.md`

Default demo configuration:

| Setting | Value |
|---------|-------|
| Max samples | 200 |
| Epochs | 2 |
| Batch size | 4 |
| Mining stage | 1 (BM25 only) |
| Device | cpu |

## 3. Check results

Open the generated report:

```bash
cat artifacts/evaluation_demo/KD_REPORT.md
```

The report shows nDCG@10 and MRR@10 for both the vanilla and KD student, plus the percentage improvement.

Your trained model is saved at:

```
artifacts/models/kd_student_demo/best_model/
```

## 4. Start the API and test with curl

Launch the FastAPI service locally:

```bash
make serve-local
```

This starts uvicorn on `http://0.0.0.0:8080` with hot-reload enabled. In another terminal, verify the service is running:

```bash
curl http://localhost:8080/health
```

Send a search query:

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is knowledge distillation", "top_k": 5}'
```

## 5. What just happened

The demo pipeline executed a compressed version of the full training workflow:

1. **Data fetch**: downloaded 200 samples from MS MARCO v2.1 via HuggingFace Datasets
2. **Data preparation**: chunked passages into Parquet format using a `TextChunker` with 512 max tokens and 80-token stride
3. **BM25 index build**: built a BM25 index over the corpus for hard negative mining
4. **Model loading**: loaded the teacher (BGE reranker, 560M params cross-encoder) and student (E5-small, 33M params bi-encoder)
5. **Hard negative mining**: used BM25 to find challenging negatives for each query
6. **KD training**: trained the student using a combined loss with three components: margin MSE (weight 0.6), listwise KD (weight 0.2), and contrastive loss (weight 0.2)
7. **Evaluation**: compared the distilled student against the vanilla baseline

The student learns to mimic the teacher's ranking behavior while remaining small enough to serve at low latency. For production-quality results, see the [Training Guide](training-guide.md) to run the full pipeline on GPU with more data.

## Next steps

- [Training Guide](training-guide.md): full training on GCP with all three mining stages
- [Deployment Guide](deployment-guide.md): Docker and Cloud Run deployment
- [Hyperparameter Tuning](hyperparameter-tuning.md): what to tune and in what order
- [Custom Dataset Guide](custom-dataset-guide.md): bring your own data
