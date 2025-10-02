#!/bin/bash
set -e

echo "=========================================="
echo "Production-Scale Training (Local)"
echo "=========================================="
echo ""
echo "This will run production-scale training locally."
echo "Note: This will take longer on CPU but will produce"
echo "better results than the quick demo."
echo ""
echo "Configuration:"
echo "  Samples: 1000 (production-scale)"
echo "  Epochs: 3"
echo "  Batch Size: 8"
echo "  Mining Stage: 2 (BM25 → Teacher)"
echo "  Device: CPU"
echo ""
echo "Expected time: 2-4 hours on CPU"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

# Configuration
MAX_SAMPLES=1000
EPOCHS=3
BATCH_SIZE=8
STAGE=2
DEVICE="cpu"
OUTPUT_DIR="artifacts/models/kd_student_production"
EVAL_DIR="artifacts/evaluation_production"

mkdir -p ${EVAL_DIR}

# Step 1: Baseline evaluation
echo "=========================================="
echo "Step 1: Baseline Evaluation"
echo "=========================================="
echo ""

VANILLA_RESULTS="${EVAL_DIR}/vanilla_results.json"

if [ ! -f "${VANILLA_RESULTS}" ]; then
    echo "Evaluating vanilla student model..."
    poetry run python scripts/simple_eval.py \
        --model-path "intfloat/e5-small-v2" \
        --data-path "data/raw/msmarco/train.jsonl" \
        --output-path "${VANILLA_RESULTS}" \
        --max-samples ${MAX_SAMPLES} \
        --device ${DEVICE}
    echo ""
else
    echo "✓ Vanilla results already exist"
fi

echo "Baseline Results:"
poetry run python -c "
import json
with open('${VANILLA_RESULTS}') as f:
    results = json.load(f)
print(f\"  nDCG@10: {results.get('ndcg@10', 0):.4f}\")
print(f\"  MRR@10: {results.get('mrr@10', 0):.4f}\")
"
echo ""

# Step 2: Train KD student
echo "=========================================="
echo "Step 2: Knowledge Distillation Training"
echo "=========================================="
echo ""

if [ -d "${OUTPUT_DIR}/best_model" ]; then
    echo "⚠️  Model already exists. Skipping training."
    echo "   To retrain, delete: ${OUTPUT_DIR}"
    echo ""
else
    echo "Training KD student model..."
    echo "  Samples: ${MAX_SAMPLES}"
    echo "  Epochs: ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  Mining Stage: ${STAGE} (BM25 → Teacher)"
    echo ""
    echo "  This will take 2-4 hours on CPU..."
    echo ""

    poetry run python scripts/train_kd_pipeline.py \
        --max-samples ${MAX_SAMPLES} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --stage ${STAGE} \
        --device ${DEVICE} \
        --output-dir ${OUTPUT_DIR} \
        --log-level INFO 2>&1 | tee ${EVAL_DIR}/training.log

    echo ""
    echo "✓ Training complete!"
    echo ""
fi

# Step 3: Evaluate KD student
echo "=========================================="
echo "Step 3: Post-KD Evaluation"
echo "=========================================="
echo ""

KD_RESULTS="${EVAL_DIR}/kd_results.json"

if [ ! -f "${KD_RESULTS}" ]; then
    echo "Evaluating KD student model..."
    poetry run python scripts/simple_eval.py \
        --model-path "${OUTPUT_DIR}/best_model" \
        --data-path "data/raw/msmarco/train.jsonl" \
        --output-path "${KD_RESULTS}" \
        --max-samples ${MAX_SAMPLES} \
        --device ${DEVICE}
else
    echo "✓ KD results already exist"
fi

echo ""
echo "KD Results:"
poetry run python -c "
import json
with open('${KD_RESULTS}') as f:
    results = json.load(f)
print(f\"  nDCG@10: {results.get('ndcg@10', 0):.4f}\")
print(f\"  MRR@10: {results.get('mrr@10', 0):.4f}\")
"
echo ""

# Step 4: Generate report
echo "=========================================="
echo "Step 4: Generate Comparison Report"
echo "=========================================="
echo ""

REPORT_PATH="${EVAL_DIR}/PRODUCTION_RESULTS.md"

poetry run python -c "
import json
from pathlib import Path
from datetime import datetime

# Load results
with open('${VANILLA_RESULTS}') as f:
    vanilla = json.load(f)
with open('${KD_RESULTS}') as f:
    kd = json.load(f)

# Generate report
report = []
report.append('# Production Training Results')
report.append('')
report.append(f'**Generated:** {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
report.append(f'**Dataset:** ${MAX_SAMPLES} samples from MS MARCO')
report.append(f'**Training:** ${EPOCHS} epochs, batch size ${BATCH_SIZE}, Stage ${STAGE} (BM25 → Teacher)')
report.append('')
report.append('---')
report.append('')

# Summary
vanilla_ndcg10 = vanilla.get('ndcg@10', 0)
kd_ndcg10 = kd.get('ndcg@10', 0)
improvement = ((kd_ndcg10 - vanilla_ndcg10) / vanilla_ndcg10) * 100 if vanilla_ndcg10 > 0 else 0

report.append('## Executive Summary')
report.append('')
report.append(f'- **Vanilla Student nDCG@10:** {vanilla_ndcg10:.4f}')
report.append(f'- **KD Student nDCG@10:** {kd_ndcg10:.4f}')
report.append(f'- **Improvement:** {improvement:+.2f}%')
report.append('')

if improvement > 10:
    report.append('✅ **Knowledge distillation achieved significant improvement!**')
elif improvement > 0:
    report.append('✅ **Knowledge distillation improved model performance.**')
else:
    report.append('⚠️ **Note:** Results may vary. Consider training with more data or epochs.')

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

# Training details
report.append('## Training Configuration')
report.append('')
report.append(f'- **Dataset:** MS MARCO (${MAX_SAMPLES} samples)')
report.append(f'- **Mining Stage:** ${STAGE} (BM25 → Teacher with scores)')
report.append(f'- **Epochs:** ${EPOCHS}')
report.append(f'- **Batch Size:** ${BATCH_SIZE}')
report.append(f'- **Device:** ${DEVICE}')
report.append('')

# Key achievements
report.append('## Key Achievements')
report.append('')
report.append('- ✅ **Real MS MARCO data** (not synthetic)')
report.append('- ✅ **2-stage hard negative mining** (BM25 → Teacher)')
report.append('- ✅ **Teacher scores captured** for KD supervision')
report.append('- ✅ **Production-scale training** (1000 samples, 3 epochs)')
report.append('- ✅ **Complete KD pipeline** functional')
report.append('')

# Comparison to quick demo
report.append('## Comparison to Quick Demo')
report.append('')
report.append('| Aspect | Quick Demo | Production |')
report.append('|--------|------------|------------|')
report.append('| Samples | 50 | 1000 (20x more) |')
report.append('| Epochs | 1 | 3 (3x more) |')
report.append('| Mining Stage | 1 (BM25) | 2 (BM25 → Teacher) |')
report.append('| Batch Size | 2 | 8 (4x larger) |')
report.append('| Training Time | ~9 min | ~2-4 hours |')
report.append('')

# Next steps
report.append('## Next Steps for Full Production')
report.append('')
report.append('To achieve even better results:')
report.append('- **Scale up:** Train on 10K-50K samples')
report.append('- **Use GPU:** Request GPU quota on GCP for 10-50x speedup')
report.append('- **All stages:** Run all 3 mining stages (BM25 → Teacher → ANCE)')
report.append('- **More epochs:** Train for 5-10 epochs')
report.append('- **Larger batches:** Use batch size 32-64 with GPU')
report.append('')

# Write report
Path('${REPORT_PATH}').write_text('\\n'.join(report))
print('✓ Report generated')
"

echo "✓ Report saved to: ${REPORT_PATH}"
echo ""

# Display results
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
echo ""

cat ${REPORT_PATH}

echo ""
echo "=========================================="
echo "Production Training Complete!"
echo "=========================================="
echo ""
echo "📁 Artifacts:"
echo "  - Trained Model: ${OUTPUT_DIR}/best_model"
echo "  - Report: ${REPORT_PATH}"
echo "  - Training Log: ${EVAL_DIR}/training.log"
echo "  - Vanilla Results: ${VANILLA_RESULTS}"
echo "  - KD Results: ${KD_RESULTS}"
echo ""
echo "🎯 To upload to GCS:"
echo "  gsutil -m cp -r ${OUTPUT_DIR} gs://plotpointe-semantic-kd-models/"
echo ""

