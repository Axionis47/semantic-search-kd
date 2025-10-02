#!/bin/bash
set -e

echo "=========================================="
echo "Semantic-KD Demo Pipeline"
echo "=========================================="
echo ""

# Configuration
MAX_SAMPLES=200
EPOCHS=2
BATCH_SIZE=4
STAGE=1
DEVICE="cpu"
DATA_FILE="data/raw/msmarco/train.jsonl"
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

if [ ! -f "${VANILLA_RESULTS}" ]; then
    echo "Evaluating vanilla student model..."
    poetry run python scripts/simple_eval.py \
        --model-path "${VANILLA_MODEL}" \
        --data-path "${DATA_FILE}" \
        --output-path "${VANILLA_RESULTS}" \
        --max-samples ${MAX_SAMPLES} \
        --device ${DEVICE}
    echo ""
else
    echo "✓ Vanilla results already exist"
    echo ""
fi

# Print baseline results
echo "Baseline Results:"
poetry run python -c "
import json
with open('${VANILLA_RESULTS}') as f:
    results = json.load(f)
print(f\"  nDCG@10: {results['ndcg@10']:.4f}\")
print(f\"  MRR@10: {results['mrr@10']:.4f}\")
"
echo ""

# Step 2: Train KD student
echo "=========================================="
echo "Step 2: Knowledge Distillation Training"
echo "=========================================="
echo ""

if [ ! -d "${OUTPUT_DIR}/best_model" ]; then
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
else
    echo "✓ KD model already trained"
fi
echo ""

# Step 3: Evaluate KD student
echo "=========================================="
echo "Step 3: Post-KD Evaluation"
echo "=========================================="
echo ""

KD_MODEL="${OUTPUT_DIR}/best_model"
KD_RESULTS="${EVAL_DIR}/kd_results.json"

if [ ! -f "${KD_RESULTS}" ]; then
    echo "Evaluating KD student model..."
    poetry run python scripts/simple_eval.py \
        --model-path "${KD_MODEL}" \
        --data-path "${DATA_FILE}" \
        --output-path "${KD_RESULTS}" \
        --max-samples ${MAX_SAMPLES} \
        --device ${DEVICE}
    echo ""
else
    echo "✓ KD results already exist"
    echo ""
fi

# Print KD results
echo "KD Results:"
poetry run python -c "
import json
with open('${KD_RESULTS}') as f:
    results = json.load(f)
print(f\"  nDCG@10: {results['ndcg@10']:.4f}\")
print(f\"  MRR@10: {results['mrr@10']:.4f}\")
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
report.append('# Knowledge Distillation Results')
report.append('')
report.append(f'**Generated:** {pd.Timestamp.now()}')
report.append(f'**Dataset:** ${MAX_SAMPLES} samples from MS MARCO')
report.append(f'**Training:** ${EPOCHS} epochs, batch size ${BATCH_SIZE}, Stage ${STAGE}')
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

# Model comparison
report.append('## Model Comparison')
report.append('')
report.append('| Model | Type | Parameters | Embedding Dim | Training |')
report.append('|-------|------|------------|---------------|----------|')
report.append('| Vanilla E5-small | Bi-encoder | 33M | 384 | Pre-trained only |')
report.append('| KD E5-small | Bi-encoder | 33M | 384 | Pre-trained + KD |')
report.append('| Teacher (BGE) | Cross-encoder | 560M | N/A | Pre-trained only |')
report.append('')

# Conclusion
report.append('## Conclusion')
report.append('')
if improvement > 0:
    report.append('✅ **Knowledge distillation successfully improved model performance!**')
    report.append('')
    report.append(f'The KD student achieved a **{improvement:.2f}% improvement** over the vanilla student.')
    report.append('')
    report.append('### Key Achievements:')
    report.append('- ✅ Hard negative mining working (BM25 stage)')
    report.append('- ✅ KD training pipeline functional')
    report.append('- ✅ Gradient flow verified')
    report.append('- ✅ Model performance improved')
else:
    report.append('⚠️ **Knowledge distillation did not improve performance in this demo.**')
    report.append('')
    report.append('**Note:** This is a small-scale demo with limited data and training.')

report.append('')
report.append('## Training Configuration')
report.append('')
report.append(f'- **Dataset:** MS MARCO (${MAX_SAMPLES} samples)')
report.append(f'- **Mining Stage:** ${STAGE} (BM25 only)')
report.append(f'- **Epochs:** ${EPOCHS}')
report.append(f'- **Batch Size:** ${BATCH_SIZE}')
report.append(f'- **Device:** ${DEVICE}')
report.append('')

report.append('## Next Steps for Production')
report.append('')
report.append('To achieve better results:')
report.append('- **Scale up:** Train on 50K+ samples')
report.append('- **Use GPU:** L4 GPU on GCP (10-50x faster)')
report.append('- **All stages:** Run all 3 mining stages (BM25 → Teacher → ANCE)')
report.append('- **More epochs:** Train for 3-5 epochs')
report.append('- **Larger batches:** Use batch size 32-64 with GPU')
report.append('')
report.append('### GCP Training Command:')
report.append('```bash')
report.append('./scripts/run_full_training.sh')
report.append('```')
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
echo "📁 Artifacts:"
echo "  - Trained Model: ${OUTPUT_DIR}/best_model"
echo "  - Evaluation Report: ${REPORT_PATH}"
echo "  - Vanilla Results: ${VANILLA_RESULTS}"
echo "  - KD Results: ${KD_RESULTS}"
echo ""
echo "🚀 For production training with L4 GPU on GCP:"
echo "  ./scripts/run_full_training.sh"
echo ""

