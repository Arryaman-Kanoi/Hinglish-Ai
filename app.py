"""
app.py
------
Interactive Gradio Web Demo for Roman-script Hinglish Sentiment Analysis.
Features:
- Real-time 3-class sentiment prediction (Positive / Neutral / Negative)
- Detailed class probability distribution
- Preprocessing & normalization inspector
- Curated Hinglish test examples showcasing sarcasm, Hindi negation, code-mixing, and emojis
"""

import os
import sys
import gradio as gr

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import HinglishSentimentPredictor, ID2LABEL

# Initialize predictor
predictor = HinglishSentimentPredictor()

# Curated test examples covering edge cases
EXAMPLE_INPUTS = [
    ["Bhai yeh movie sach me ekdum zabardast hai, mind-blowing acting! 🔥❤️"],
    ["Delivery itni late thi aur customer support ne call disconnect kar diya. Bilkul bekaar service 😡"],
    ["Match kal shaam 7:30 baje start hoga live Hotstar par."],
    ["Ye phone lene ki galti mat karna, battery 2 ghante bhi nahi chalti."],
    ["Kya baat hai, itna kharab update dene ke liye dhanyawad! 👏"],
    ["Me acha nahi hu na"],
    ["Finally result aa gaya, passed with distinction! Party toh banti hai 🎉"],
    ["Package deliver ho gaya hai but maine abhi open nahi kiya."],
    ["Service was okay okay, na zyada acchi na buri."],
    ["Bohot hi mast track hai bro, loop pe sun raha hu tabse! 🎧🔥"],
    ["Flop picture hai bhai, time aur paise dono waste ho gaye 🤮"]
]


def analyze_hinglish_sentiment(text: str):
    """
    Inference callback for Gradio UI.
    """
    if not text or not text.strip():
        return "Please enter a Hinglish sentence.", {}, "", ""

    result = predictor.predict(text)

    label = result["label"].upper()
    conf = result["confidence"]
    probs = result["probabilities"]
    cleaned = result["cleaned_text"]
    reading = result.get("reading") or (
        f"Gap {result.get('gap', 0):.2f}. Specialist: {result.get('specialist', 'model')}. "
        f"Trigger: {result.get('trigger') or 'none'}."
    )

    emoji_map = {
        "POSITIVE": "POSITIVE",
        "NEUTRAL": "NEUTRAL",
        "NEGATIVE": "NEGATIVE",
    }
    badge = f"{emoji_map.get(label, label)}  —  Confidence: {conf * 100:.1f}%"

    return badge, probs, cleaned, reading


def build_app():
    custom_css = """
    .gradio-container { max-width: 850px !important; margin: auto; }
    .badge-box { font-size: 1.25rem; font-weight: bold; text-align: center; }
    """

    with gr.Blocks(title="Hinglish Sentiment Analysis", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🇮🇳 Hinglish Sentiment Analyzer (Roman-Script SentiMix)
            ### **CS3255: Natural Language Processing — SEE Project**
            **Student:** Krishnendu Nayak | **USN:** 1RUA24SCS0052
            
            The label comes from the saved HingBERT checkpoint, the best model on the official SentiMix test split. SentiMix has no sarcasm label, so a praise cue is not marked sarcastic.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                input_box = gr.Textbox(
                    label="Enter Hinglish Tweet / Sentence:",
                    placeholder="e.g. Bhai yeh movie bohottt mast hai! 🔥",
                    lines=3
                )
                analyze_btn = gr.Button("Analyze Sentiment 🚀", variant="primary")

            with gr.Column(scale=2):
                output_badge = gr.Textbox(label="Detected Sentiment & Confidence", elem_classes=["badge-box"])
                output_probs = gr.Label(label="Class Probability Distribution", num_top_classes=3)
                output_clean = gr.Textbox(label="Normalized / Preprocessed Text", interactive=False)
                output_reading = gr.Textbox(label="Model", interactive=False)

        gr.Markdown("### 🧪 Try Test Examples (Click to Load):")
        gr.Examples(
            examples=EXAMPLE_INPUTS,
            inputs=[input_box],
            outputs=[output_badge, output_probs, output_clean, output_reading],
            fn=analyze_hinglish_sentiment,
            cache_examples=False
        )

        analyze_btn.click(
            fn=analyze_hinglish_sentiment,
            inputs=[input_box],
            outputs=[output_badge, output_probs, output_clean, output_reading]
        )

        gr.Markdown(
            """
            ---
            **Key Features & Ablations Covered:**
            - **Models Supported:** Lexical Baseline (TF-IDF + LogReg/SVM), mBERT, XLM-RoBERTa, MuRIL, IndicBERTv2, HingBERT.
            - **Metrics:** Macro-F1 (primary), Accuracy, Per-class Precision/Recall/F1 over 3 seeds (mean ± std).
            - **Linguistic Robustness:** Preserves emojis, captures Hindi negation (`nahi`, `mat`), and normalizes elongated slang (`bohottt` → `bohatt`).
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=False)
