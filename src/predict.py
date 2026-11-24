"""
src/predict.py
--------------
Inference module for 3-class Hinglish Sentiment Analysis.
Provides:
- `predict_sentiment(text, model, tokenizer)`
- `HinglishPredictor` class supporting Hugging Face models, Scikit-learn baselines,
  and offline heuristic fallback.
"""

import os
import argparse
import numpy as np
import torch

try:
    from src.data import ID2LABEL, LABEL2ID
    from src.preprocess import clean_tweet
    from src.models import resolve_model_name
except ImportError:
    from data import ID2LABEL, LABEL2ID
    from preprocess import clean_tweet
    from models import resolve_model_name


class HinglishSentimentPredictor:
    """
    Unified inference predictor for Roman Hinglish Sentiment Analysis.
    Handles preprocessing, tokenization, model inference, and probability formatting.
    """
    def __init__(
        self,
        model_path_or_name: str = None,
        model_type: str = "transformer",
        device: str = None
    ):
        self.model_type = model_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.baseline_pipeline = None
        self.system = None

        if model_path_or_name is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for name in ("best", "hingbert"):
                candidate = os.path.join(root, "results", "checkpoints", name)
                if os.path.isdir(candidate):
                    model_path_or_name = candidate
                    break
        if model_path_or_name is not None and os.path.exists(model_path_or_name):
            self._load_from_path(model_path_or_name)
        elif model_path_or_name is not None:
            self._load_from_hf(model_path_or_name)
        else:
            print("[Predictor] No fine-tuned checkpoint found. Using the word-list fallback.")

    def _load_from_path(self, path: str):
        if self.model_type == "transformer":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
            self.model.eval()
            print(f"[Predictor] Loaded Transformer checkpoint from: {path}")
        else:
            import joblib
            self.baseline_pipeline = joblib.load(path)
            print(f"[Predictor] Loaded Scikit-Learn baseline from: {path}")

    def _load_from_hf(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        resolved = resolve_model_name(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(resolved).to(self.device)
        self.model.eval()
        print(f"[Predictor] Loaded model from Hugging Face: {resolved}")

    def predict(self, text: str) -> dict:
        """
        Runs prediction for a single Hinglish input text.

        Returns:
        --------
        dict:
            {
                "label": "positive" | "neutral" | "negative",
                "confidence": float,
                "probabilities": {"negative": float, "neutral": float, "positive": float},
                "cleaned_text": str
            }
        """
        if not text or not str(text).strip():
            return {
                "label": "neutral",
                "confidence": 0.3333,
                "probabilities": {"negative": 0.3333, "neutral": 0.3334, "positive": 0.3333},
                "cleaned_text": ""
            }

        cleaned = clean_tweet(str(text))

        if self.system is not None:
            try:
                from src.flip import annotate
            except ImportError:
                from flip import annotate
            return annotate(self.system.explain(text), text, self.system)

        # 1. Transformer inference
        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            pred_id = int(np.argmax(probs))
            pred_label = ID2LABEL.get(pred_id, str(pred_id))
            prob_dict = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))}

            return {
                "label": pred_label,
                "confidence": round(float(probs[pred_id]), 4),
                "probabilities": prob_dict,
                "cleaned_text": cleaned,
                "reading": "HingBERT, the saved checkpoint.",
            }

        # 2. Baseline pipeline inference
        elif self.baseline_pipeline is not None:
            pred_id = int(self.baseline_pipeline.predict([cleaned])[0])
            pred_label = ID2LABEL.get(pred_id, str(pred_id))

            if hasattr(self.baseline_pipeline, "predict_proba"):
                probs = self.baseline_pipeline.predict_proba([cleaned])[0]
                prob_dict = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))}
                confidence = round(float(probs[pred_id]), 4)
            else:
                prob_dict = {ID2LABEL[i]: 1.0 if i == pred_id else 0.0 for i in range(3)}
                confidence = 1.0

            return {
                "label": pred_label,
                "confidence": confidence,
                "probabilities": prob_dict,
                "cleaned_text": cleaned
            }

        # 3. Rule-based lexical fallback (for immediate UI demonstration before model training)
        else:
            return self._lexical_fallback(cleaned)

    def _lexical_fallback(self, cleaned: str) -> dict:
        """Lightweight lexical sentiment scoring when weights are not yet downloaded."""
        lower = cleaned.lower()
        pos_words = ["zabardast", "mast", "superb", "shandar", "badhiya", "best", "love", "amazing", "proud", "party", "dil", "khoobsurat"]
        neg_words = ["bekaar", "ghatiya", "wahiyat", "useless", "worst", "disappointed", "crushed", "scam", "barbad", "kharab", "late", "fail"]
        negation = ["nahi", "nahin", "nhi", "mat", "na"]

        pos_score = sum(1 for w in pos_words if w in lower)
        neg_score = sum(1 for w in neg_words if w in lower)
        has_negation = any(w in lower.split() for w in negation)

        if "🔥" in cleaned or "❤️" in cleaned or "🎉" in cleaned or "😊" in cleaned or "🥳" in cleaned:
            pos_score += 2
        if "😡" in cleaned or "🤮" in cleaned or "😭" in cleaned or "💔" in cleaned:
            neg_score += 2

        if has_negation and pos_score > 0:
            # Polarity flip e.g. "accha nahi hai"
            neg_score += pos_score
            pos_score = 0

        total = pos_score + neg_score + 1.0
        p_pos = (pos_score + 0.3) / (total + 0.9)
        p_neg = (neg_score + 0.3) / (total + 0.9)
        p_neu = 1.0 - (p_pos + p_neg)

        probs = [p_neg, p_neu, p_pos]
        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_id]

        return {
            "label": pred_label,
            "confidence": round(float(probs[pred_id]), 4),
            "probabilities": {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(3)},
            "cleaned_text": cleaned
        }


def predict(text: str, predictor: HinglishSentimentPredictor = None) -> dict:
    """Convenience top-level predict function."""
    if predictor is None:
        predictor = HinglishSentimentPredictor()
    return predictor.predict(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Hinglish tweet sentiment")
    parser.add_argument("--text", type=str, default="Bhai yeh movie sach me ekdum zabardast hai! 🔥❤️", help="Input Hinglish text")
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned checkpoint")
    args = parser.parse_args()

    pred = HinglishSentimentPredictor(model_path_or_name=args.model_path)
    res = pred.predict(args.text)
    print("\n[Prediction Output]")
    print(f"Input:         {args.text}")
    print(f"Cleaned:       {res['cleaned_text']}")
    print(f"Sentiment:     {res['label'].upper()} (Confidence: {res['confidence']:.2%})")
    print(f"Probabilities: {res['probabilities']}")
