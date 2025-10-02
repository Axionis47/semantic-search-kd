#!/bin/bash
set -e

echo "=========================================="
echo "Complete Semantic-KD Pipeline with L4 GPU"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID="plotpointe"
REGION="us-central1"
DEVICE="cuda"
MAX_SAMPLES=50000
EPOCHS=3
BATCH_SIZE=32
STAGE=3

# Paths
DATA_DIR="data/chunks/msmarco"
OUTPUT_DIR="artifacts/models/kd_student"
GCS_OUTPUT_DIR="gs://plotpointe-semantic-kd-models/kd_student"
EVAL_DIR="artifacts/evaluation"

echo "Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Device: ${DEVICE}"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Mining Stage: ${STAGE}"
echo ""

# Check if running on GCP (has GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    USE_GPU=true
else
    echo "⚠️  No GPU detected - will use CPU (slower)"
    echo ""
    DEVICE="cpu"
    BATCH_SIZE=4
    MAX_SAMPLES=1000
    USE_GPU=false
fi

# Step 1: Prepare data
echo "=========================================="
echo "Step 1: Data Preparation"
echo "=========================================="
echo ""

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Fetching and preparing data..."
    poetry run python -c "
from src.data.fetch import fetch_msmarco
from src.data.prepare import prepare_msmarco
from pathlib import Path

# Fetch
fetch_msmarco(
    output_dir=Path('data/raw/msmarco'),
    max_samples_per_split=${MAX_SAMPLES}
)

# Prepare
prepare_msmarco(
    input_dir=Path('data/raw/msmarco'),
    output_dir=Path('${DATA_DIR}'),
    chunk_size=512,
    stride=80
)
"
    echo "✓ Data prepared"
else
    echo "✓ Data already prepared"
fi
echo ""

# Step 2: Evaluate vanilla student (baseline)
echo "=========================================="
echo "Step 2: Baseline Evaluation (Vanilla Student)"
echo "=========================================="
echo ""

VANILLA_MODEL="intfloat/e5-small-v2"
VANILLA_RESULTS="${EVAL_DIR}/vanilla_results.json"

if [ ! -f "${VANILLA_RESULTS}" ]; then
    echo "Evaluating vanilla student model..."
    poetry run python -c "
from src.models.student import StudentModel
from src.kd.eval import KDEvaluator
from pathlib import Path
import pandas as pd
import json

# Load test data
df = pd.read_parquet('${DATA_DIR}/test.parquet').head(1000)
queries = []
documents_list = []
relevance_list = []

for query_id, group in df.groupby('query_id'):
    queries.append(group.iloc[0]['query_text'])
    documents_list.append(group['text'].tolist())
    relevance_list.append(group['is_relevant'].tolist())

# Load model
student = StudentModel('${VANILLA_MODEL}', device='${DEVICE}')

# Evaluate
evaluator = KDEvaluator(student=student)
results = evaluator.evaluate_retrieval(
    queries=queries,
    documents_list=documents_list,
    relevance_list=relevance_list,
    k_values=[1, 5, 10, 20],
    use_teacher=False
)

# Save results
Path('${EVAL_DIR}').mkdir(parents=True, exist_ok=True)
with open('${VANILLA_RESULTS}', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Vanilla Student nDCG@10: {results[\"ndcg@10\"]:.4f}')
"
    echo "✓ Vanilla student evaluated"
else
    echo "✓ Vanilla student already evaluated"
fi

# Print baseline results
echo ""
echo "Baseline Results:"
poetry run python -c "
import json
with open('${VANILLA_RESULTS}') as f:
    results = json.load(f)
print(f\"  nDCG@10: {results['ndcg@10']:.4f}\")
print(f\"  MRR@10: {results['mrr@10']:.4f}\")
"
echo ""

# Step 3: Train KD student
echo "=========================================="
echo "Step 3: Knowledge Distillation Training"
echo "=========================================="
echo ""

if [ ! -d "${OUTPUT_DIR}/best_model" ]; then
    echo "Training KD student model..."
    echo "  Device: ${DEVICE}"
    echo "  Samples: ${MAX_SAMPLES}"
    echo "  Epochs: ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo ""
    
    poetry run python scripts/train_kd_pipeline.py \
        --max-samples ${MAX_SAMPLES} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --stage ${STAGE} \
        --device ${DEVICE} \
        --output-dir ${OUTPUT_DIR} \
        --gcs-output-dir ${GCS_OUTPUT_DIR} \
        --log-level INFO
    
    echo "✓ KD training complete"
else
    echo "✓ KD model already trained"
fi
echo ""

# Step 4: Evaluate KD student
echo "=========================================="
echo "Step 4: Post-KD Evaluation"
echo "=========================================="
echo ""

KD_MODEL="${OUTPUT_DIR}/best_model"
KD_RESULTS="${EVAL_DIR}/kd_results.json"

if [ ! -f "${KD_RESULTS}" ]; then
    echo "Evaluating KD student model..."
    poetry run python -c "
from src.models.student import StudentModel
from src.kd.eval import KDEvaluator
from pathlib import Path
import pandas as pd
import json

# Load test data
df = pd.read_parquet('${DATA_DIR}/test.parquet').head(1000)
queries = []
documents_list = []
relevance_list = []

for query_id, group in df.groupby('query_id'):
    queries.append(group.iloc[0]['query_text'])
    documents_list.append(group['text'].tolist())
    relevance_list.append(group['is_relevant'].tolist())

# Load model
student = StudentModel('${KD_MODEL}', device='${DEVICE}')

# Evaluate
evaluator = KDEvaluator(student=student)
results = evaluator.evaluate_retrieval(
    queries=queries,
    documents_list=documents_list,
    relevance_list=relevance_list,
    k_values=[1, 5, 10, 20],
    use_teacher=False
)

# Save results
with open('${KD_RESULTS}', 'w') as f:
    json.dump(results, f, indent=2)

print(f'KD Student nDCG@10: {results[\"ndcg@10\"]:.4f}')
"
    echo "✓ KD student evaluated"
else
    echo "✓ KD student already evaluated"
fi

# Print KD results
echo ""
echo "KD Results:"
poetry run python -c "
import json
with open('${KD_RESULTS}') as f:
    results = json.load(f)
print(f\"  nDCG@10: {results['ndcg@10']:.4f}\")
print(f\"  MRR@10: {results['mrr@10']:.4f}\")
"
echo ""

# Step 5: Generate comparison report
echo "=========================================="
echo "Step 5: Generate Comparison Report"
echo "=========================================="
echo ""

REPORT_PATH="${EVAL_DIR}/KD_REPORT.md"

echo "Generating comparison report..."
poetry run python -c "
import json
from pathlib import Path
import pandas as pd

# Load results
with open('${VANILLA_RESULTS}') as f:
    vanilla = json.load(f)
with open('${KD_RESULTS}') as f:
    kd = json.load(f)

# Generate report
report = []
report.append('# Knowledge Distillation Results')
report.append('')
report.append(f'**Generated:** {pd.Timestamp.now()}')
report.append('')
report.append('---')
report.append('')

# Summary
vanilla_ndcg10 = vanilla.get('ndcg@10', 0)
kd_ndcg10 = kd.get('ndcg@10', 0)
improvement = ((kd_ndcg10 - vanilla_ndcg10) / vanilla_ndcg10) * 100

report.append('## Summary')
report.append('')
report.append(f'- **Vanilla Student nDCG@10:** {vanilla_ndcg10:.4f}')
report.append(f'- **KD Student nDCG@10:** {kd_ndcg10:.4f}')
report.append(f'- **Improvement:** {improvement:+.2f}%')
report.append('')

# Detailed metrics
report.append('## Detailed Metrics')
report.append('')
report.append('| Metric | Vanilla | KD | Improvement |')
report.append('|--------|---------|----|-----------| ')

for metric in ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'mrr@10']:
    v = vanilla.get(metric, 0)
    k = kd.get(metric, 0)
    imp = ((k - v) / v * 100) if v > 0 else 0
    report.append(f'| {metric} | {v:.4f} | {k:.4f} | {imp:+.2f}% |')

report.append('')

# Conclusion
report.append('## Conclusion')
report.append('')
if improvement > 0:
    report.append('✅ **Knowledge distillation improved model performance!**')
else:
    report.append('⚠️ **Knowledge distillation did not improve performance.**')
report.append('')

# Write report
Path('${EVAL_DIR}').mkdir(parents=True, exist_ok=True)
Path('${REPORT_PATH}').write_text('\\n'.join(report))
print('✓ Report generated')
"

echo "✓ Report saved to: ${REPORT_PATH}"
echo ""

# Step 6: Display final results
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
echo ""

cat ${REPORT_PATH}

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Artifacts:"
echo "  - Trained Model: ${OUTPUT_DIR}/best_model"
echo "  - GCS Model: ${GCS_OUTPUT_DIR}"
echo "  - Evaluation Report: ${REPORT_PATH}"
echo "  - Vanilla Results: ${VANILLA_RESULTS}"
echo "  - KD Results: ${KD_RESULTS}"
echo ""

