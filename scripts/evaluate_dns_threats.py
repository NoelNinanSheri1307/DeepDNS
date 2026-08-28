"""
Zero-Shot Evaluation of Pre-Trained Character-CNN on External dns_threats Dataset.

Evaluates the pre-trained Lexical Character-CNN model (multiview_lexical_only.pt)
on 620,703 real-world external test domains from dns_threats without retraining.

Generates:
  - reports/diagnostics/dns_threats_zero_shot_evaluation.json
  - reports/diagnostics/dns_threats_zero_shot_evaluation.txt
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.char_cnn import CharacterTokenizer
from src.models.multiview_fusion import DeepDNSMultiViewClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EvalDNSThreats")


def evaluate_dns_threats_zero_shot(
    max_samples: int = 100000,
    batch_size: int = 512,
):
    logger.info("=========================================================")
    logger.info("STARTING ZERO-SHOT EVALUATION ON EXTERNAL DNS_THREATS DATASET")
    logger.info("=========================================================")

    data_path = PROJECT_ROOT / "data" / "raw" / "dns_threats" / "original" / "test_combined_multiclass.csv.gz"
    model_path = PROJECT_ROOT / "data" / "processed" / "models" / "multiview_lexical_only.pt"

    if not data_path.exists():
        raise FileNotFoundError(f"dns_threats dataset not found at: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Pre-trained lexical model not found at: {model_path}")

    logger.info(f"Loading up to {max_samples:,} domains from {data_path}...")
    df = pd.read_csv(data_path, compression="gzip", nrows=max_samples)
    domains = df["domain"].fillna("").astype(str).tolist()
    # Binary mapping: class 0 -> Benign (0), class >= 1 -> Malicious (1)
    raw_classes = df["class"].values
    y_true = np.where(raw_classes == 0, 0, 1)

    logger.info(f"Loaded {len(domains):,} domains. Positive prevalence: {np.mean(y_true)*100:.2f}%")

    tokenizer = CharacterTokenizer()
    logger.info("Tokenizing domain strings...")
    token_matrix = tokenizer.batch_encode(domains)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading pre-trained lexical model on {device}...")
    classifier = DeepDNSMultiViewClassifier.load(model_path, device=device)
    classifier.model.eval()

    # Create TensorDataset with dummy sequence dimension (B, 1, 128)
    tokens_tensor = torch.tensor(token_matrix, dtype=torch.long).unsqueeze(1)
    loader = DataLoader(TensorDataset(tokens_tensor), batch_size=batch_size, shuffle=False)

    logger.info("Running zero-shot inference...")
    all_probs = []
    with torch.no_grad():
        for (batch_tokens,) in loader:
            batch_tokens = batch_tokens.to(device)
            # Forward pass in lexical_only mode
            step_logits, _, _, _, _ = classifier.model(None, batch_tokens, mode="lexical_only")
            probs = torch.softmax(step_logits[:, 0, :], dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    results = {
        "dataset_name": "dns_threats_test_combined_multiclass",
        "evaluation_type": "Zero-Shot External Generalization (No Fine-Tuning)",
        "model_evaluated": "multiview_lexical_only.pt (Trained strictly on CIC-Bell)",
        "total_samples_evaluated": len(y_true),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1, 4),
        "fpr": round(fpr * 100, 4),
        "fnr": round(fnr * 100, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }

    out_dir = PROJECT_ROOT / "reports" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dns_threats_zero_shot_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    txt_path = out_dir / "dns_threats_zero_shot_evaluation.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("ZERO-SHOT EXTERNAL GENERALIZATION REPORT: DNS_THREATS DATASET\n")
        f.write("================================================================================\n\n")
        f.write(f"Model:                   {results['model_evaluated']}\n")
        f.write(f"Evaluation Mode:         {results['evaluation_type']}\n")
        f.write(f"Domains Evaluated:       {results['total_samples_evaluated']:,}\n\n")
        f.write(f"Accuracy:                {results['accuracy']}%\n")
        f.write(f"Precision:               {results['precision']}%\n")
        f.write(f"Recall (Detection Rate): {results['recall']}%\n")
        f.write(f"F1-Score:                {results['f1_score']}\n")
        f.write(f"False Positive Rate:     {results['fpr']}%\n")
        f.write(f"ROC-AUC:                 {results['roc_auc']}\n")
        f.write(f"PR-AUC:                  {results['pr_auc']}\n")
        f.write(f"Confusion Matrix:        {results['confusion_matrix']}\n")
        f.write("================================================================================\n")

    logger.info(f"Saved zero-shot reports to {json_path} and {txt_path}")
    logger.info("=========================================================")
    logger.info("DNS_THREATS ZERO-SHOT EVALUATION COMPLETE")
    logger.info("=========================================================")


if __name__ == "__main__":
    evaluate_dns_threats_zero_shot(max_samples=100000)
