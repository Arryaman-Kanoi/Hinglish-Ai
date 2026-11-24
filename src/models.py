"""
src/models.py
-------------
Architectures for 3-class Hinglish Sentiment Analysis:
1. Baselines:
   - Combined Word & Character n-gram TF-IDF + Logistic Regression / LinearSVC.
2. Pretrained Transformers:
   - mBERT (bert-base-multilingual-cased)
   - XLM-RoBERTa (xlm-roberta-base)
   - MuRIL (google/muril-base-cased)
   - IndicBERTv2 (ai4bharat/IndicBERTv2-MLM-only)
   - HingBERT (l3cube-pune/hing-bert)
3. Custom WeightedTrainer for Class-Weighted Loss.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer
)

try:
    from src.data import LABEL2ID, ID2LABEL
except ImportError:
    from data import LABEL2ID, ID2LABEL

# Hugging Face Model Hub identifiers
MODEL_REGISTRY = {
    "mbert": "bert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "hingbert": "l3cube-pune/hing-bert"
}


def build_tfidf_baseline(
    classifier_type: str = "logreg",
    use_class_weights: bool = True
) -> Pipeline:
    """
    Builds a strong lexical baseline combining word and character n-grams.
    Character n-grams are critical for Romanized Hinglish due to frequent spelling variations.
    """
    class_weight = "balanced" if use_class_weights else None

    # Feature Union: Word n-grams (1-2) + Char n-grams (2-5)
    union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            min_df=2,
            sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            ngram_range=(2, 5),
            analyzer="char",
            min_df=3,
            sublinear_tf=True
        ))
    ])

    if classifier_type.lower() == "logreg":
        clf = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight=class_weight,
            random_state=42
        )
    elif classifier_type.lower() == "svm":
        clf = LinearSVC(
            C=1.0,
            class_weight=class_weight,
            random_state=42,
            dual="auto",
            max_iter=5000,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Choose 'logreg' or 'svm'.")

    pipeline = Pipeline([
        ("features", union),
        ("classifier", clf)
    ])
    return pipeline


def resolve_model_name(model_key_or_name: str) -> str:
    """Resolves short alias (e.g. 'hingbert') to full Hugging Face model identifier."""
    return MODEL_REGISTRY.get(model_key_or_name.lower(), model_key_or_name)


def load_transformer_and_tokenizer(
    model_name_or_key: str,
    num_labels: int = 3,
    freeze_encoder: bool = False
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Loads Hugging Face tokenizer and classification model configured with canonical label mappings.
    Optionally freezes the encoder backbone (for ablation analysis).
    """
    hf_model_name = resolve_model_name(model_name_or_key)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

    config = AutoConfig.from_pretrained(
        hf_model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    if freeze_encoder:
        # Freeze all layers of the base model (e.g., bert, roberta)
        base_model = getattr(model, model.base_model_prefix, None)
        if base_model is not None:
            for param in base_model.parameters():
                param.requires_grad = False
            print(f"[Model] Freezing backbone: {model.base_model_prefix}. Only classification head is trainable.")

    return tokenizer, model


class WeightedTrainer(Trainer):
    """
    Custom Hugging Face Trainer that supports class-weighted Cross-Entropy loss
    to tackle class imbalance (e.g., small neutral class or skewed Twitter sentiment).
    """
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # Mean cross-entropy is per micro-batch. Trainer must divide by grad accumulation.
        self.loss_is_scaled_for_ga = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device, dtype=torch.float32)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        # Cross-entropy in fp32. fp16 logits overflow the log-softmax on a 4GB card.
        loss = loss_fct(
            logits.float().view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


class BiLSTMClassifier(nn.Module):
    """Word-level bidirectional LSTM. Optional neural baseline, not a transformer."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128,
                 num_labels: int = 3, pad_idx: int = 0, dropout: float = 0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        outputs, _ = self.lstm(embedded)
        mask = (input_ids != self.pad_idx).unsqueeze(-1).to(outputs.dtype)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(self.dropout(pooled))
