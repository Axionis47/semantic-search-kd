#!/bin/bash
set -e

echo "=========================================="
echo "Quick Semantic-KD Demo"
echo "=========================================="
echo ""

# Small-scale demo configuration
MAX_SAMPLES=200
EPOCHS=2
BATCH_SIZE=4
STAGE=1
DEVICE="cpu"
DATA_FILE="data/chunks/msmarco/train.parquet"
OUTPUT_DIR="artifacts/models/kd_student_demo"
EVAL_DIR="artifacts/evaluation_demo"

echo "Configuration:"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Mining Stage: ${STAGE} (BM25 only)"
echo "  Device: ${DEVICE}"
echo ""

# Step 1: Evaluate vanilla student (baseline)
echo "=========================================="
echo "Step 1: Baseline Evaluation (Vanilla Student)"
echo "=========================================="
echo ""

VANILLA_MODEL="intfloat/e5-small-v2"
VANILLA_RESULTS="${EVAL_DIR}/vanilla_results.json"

mkdir -p ${EVAL_DIR}

echo "Evaluating vanilla student model on ${MAX_SAMPLES} samples..."
poetry run python -c "
from src.models.student import StudentModel
from src.kd.eval import KDEvaluator
from pathlib import Path
import pandas as pd
import json
import numpy as np

# Load data
df = pd.read_parquet('${DATA_FILE}').head(${MAX_SAMPLES})
queries = []
documents_list = []
relevance_list = []

for query_id, group in df.groupby('query_id'):
    queries.append(group.iloc[0]['query_text'])
    documents_list.append(group['text'].tolist())
    relevance_list.append(group['is_relevant'].tolist())

print(f'Loaded {len(queries)} queries')

# Load model
student = StudentModel('${VANILLA_MODEL}', device='${DEVICE}')

# Evaluate
evaluator = KDEvaluator(student=student)
results = evaluator.evaluate_retrieval(
    queries=queries,
    documents_list=documents_list,
    relevance_list=relevance_list,
    k_values=[1, 5, 10],
    use_teacher=False
)

# Save results
with open('${VANILLA_RESULTS}', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\\nVanilla Student Results:')
print(f'  nDCG@10: {results[\"ndcg@10\"]:.4f}')
print(f'  MRR@10: {results[\"mrr@10\"]:.4f}')
"

echo ""

# Step 2: Train KD student
echo "=========================================="
echo "Step 2: Knowledge Distillation Training"
echo "=========================================="
echo ""

echo "Training KD student model..."
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
    --log-level INFO

echo ""
echo "✓ KD training complete"
echo ""

# Step 3: Evaluate KD student
echo "=========================================="
echo "Step 3: Post-KD Evaluation"
echo "=========================================="
echo ""

KD_MODEL="${OUTPUT_DIR}/best_model"
KD_RESULTS="${EVAL_DIR}/kd_results.json"

echo "Evaluating KD student model..."
poetry run python -c "
from src.models.student import StudentModel
from src.kd.eval import KDEvaluator
from pathlib import Path
import pandas as pd
import json

# Load data
df = pd.read_parquet('${DATA_FILE}').head(${MAX_SAMPLES})
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
    k_values=[1, 5, 10],
    use_teacher=False
)

# Save results
with open('${KD_RESULTS}', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\\nKD Student Results:')
print(f'  nDCG@10: {results[\"ndcg@10\"]:.4f}')
print(f'  MRR@10: {results[\"mrr@10\"]:.4f}')
"

echo ""

# Step 4: Generate comparison report
echo "=========================================="
echo "Step 4: Generate Comparison Report"
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
report.append('# Knowledge Distillation Results (Demo)')
report.append('')
report.append(f'**Generated:** {pd.Timestamp.now()}')
report.append(f'**Dataset:** ${MAX_SAMPLES} samples from MS MARCO')
report.append(f'**Training:** ${EPOCHS} epochs, batch size ${BATCH_SIZE}')
report.append(f'**Mining:** Stage ${STAGE} (BM25 only)')
report.append('')
report.append('---')
report.append('')

# Summary
vanilla_ndcg10 = vanilla.get('ndcg@10', 0)
kd_ndcg10 = kd.get('ndcg@10', 0)
improvement = ((kd_ndcg10 - vanilla_ndcg10) / vanilla_ndcg10) * 100 if vanilla_ndcg10 > 0 else 0

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

for metric in ['ndcg@1', 'ndcg@5', 'ndcg@10', 'mrr@10']:
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
    report.append('')
    report.append(f'The KD student achieved a {improvement:.2f}% improvement over the vanilla student.')
else:
    report.append('⚠️ **Knowledge distillation did not improve performance in this demo.**')
    report.append('')
    report.append('Note: This is a small-scale demo with limited data and training.')

report.append('')
report.append('## Next Steps for Production')
report.append('')
report.append('To achieve better results:')
report.append('- Train on larger dataset (50K+ samples)')
report.append('- Use GPU for faster training')
report.append('- Run all 3 mining stages (BM25 → Teacher → ANCE)')
report.append('- Train for more epochs (3-5)')
report.append('- Use larger batch size (32-64)')
report.append('')

# Write report
Path('${REPORT_PATH}').write_text('\\n'.join(report))
print('✓ Report generated')
"

echo "✓ Report saved to: ${REPORT_PATH}"
echo ""

# Display final results
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
echo ""

cat ${REPORT_PATH}

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Artifacts:"
echo "  - Trained Model: ${OUTPUT_DIR}/best_model"
echo "  - Evaluation Report: ${REPORT_PATH}"
echo "  - Vanilla Results: ${VANILLA_RESULTS}"
echo "  - KD Results: ${KD_RESULTS}"
echo ""
echo "For production training with L4 GPU on GCP, use:"
echo "  ./scripts/run_full_training.sh"
echo ""

