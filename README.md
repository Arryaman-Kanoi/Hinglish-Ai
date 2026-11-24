# Hinglish sentiment

CS3255 Natural Language Processing, SEE project. Krishnendu Nayak, USN 1RUA24SCS0052.

Three-class sentiment on Roman-script Hindi-English: negative (0), neutral (1), positive (2). The label map does not change between scripts.

The training data is SemEval-2020 Task 9, SentiMix Hinglish (Patwa et al., 2020), Zenodo record 3974927. Train, validation, and test stay the official files. After dropping duplicate text inside train and validation, and dropping train or validation tweets whose text also appears in a later split, the sizes are 13,849 / 2,993 / 3,000. The released test file is not edited. Three duplicate test texts are kept.

A second Roman-script set (Mendeley DOI 10.17632/wtp7htm3ss.1) is scored only as a transfer test. It is not Twitter text and it is not used for training.

## The system

The submitted model is not a single checkpoint. It has three parts:

- A character TF-IDF model, fit on the training split, for Roman spelling.
- A polarity lexicon mined from the training split. A negation word (`nahi`, `nhi`, `mat`, `na`, `not`) flips the next three word scores.
- A logistic stacker and a neutral margin, both chosen on the validation split. If the gap between the top two probabilities is below that margin, the label is neutral.

```
.\.venv\Scripts\python.exe -m src.system
```

That writes `results/system_metrics.csv`, `results/slices.csv` (macro-F1 for mostly-one-language tweets, mixed tweets, and tweets with a negation), and `results/disagreements.csv` (test tweets where the character model and the saved HingBERT predictions disagree). The demo loads `results/system/` and does not need the GPU.

## Scope, and what is not sarcasm

SentiMix has no sarcasm label. A line is not given a sarcasm class.

`python -m src.flip` measures two shapes. A negation flips a content word two tokens back and three tokens forward, so Hindi order (`achha nahi`) reverses the compliment. A sentence-final `na` is a tag, "right?", not a second negation. Together, `me acha nahi hu na` is an echo: the negated word is thrown back at the other person. The earlier turn is not in the tweet. On this data that echo is rare: 7 training tweets, 1 validation tweet, 4 test tweets. The sentiment of that shape is still negative, because the flipped word is negative. Sincere "I am not good" and the sarcastic throwback are the same class.

The scope layer was fit on 70% of validation and judged on the other 30%. It was not kept. Holdout macro-F1 was 0.612 against 0.623 for the current system, and the negation slice got worse (−0.013). Test was scored once after that decision: the layer's macro-F1 is 0.685, the current system is 0.696, and the saved HingBERT predictions are 0.724. On tweets where the surface word and the scoped word disagree (248 test tweets), positive F1 stays weak: HingBERT 0.348, the system 0.059. That slice is the real gap. It is not fixed here.

The demo still prints the scope sentence.

The character model was then refit with 1,426 extra training rows: real flip tweets repeated once, 58 clear positive tweets with `nahi` inserted and the label set to negative, and 22 clear negated tweets with the negation removed and the label set to positive. Validation kept that refit. On the test set, scored once, overall macro-F1 moved from 0.696 to 0.698, and the negation slice from 0.570 to 0.585. Positive F1 on that slice moved from 0.382 to 0.415. HingBERT's weights were not changed. MuRIL was using the GPU.

## Install

The environment is `.venv`. Torch there is the CUDA 12.8 wheel. Installing torch from the default PyPI index replaces that wheel. Recreate it in this order:

```
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --upgrade-strategy only-if-needed
```

## Data

The CoNLL files are in `data/raw/sentimix`. Download them again with:

```
.\.venv\Scripts\python.exe data\download_data.py
```

## Train

From the project root:

```
.\.venv\Scripts\python.exe -m src.experiments
```

The script writes EDA plots, then TF-IDF logistic regression, TF-IDF linear SVM, and a word BiLSTM, each with seeds 42, 100, and 2024. It picks a learning rate on HingBERT using validation macro-F1 only, from 2e-5, 3e-5, and 5e-5. That rate is then used for HingBERT, mBERT, MuRIL, IndicBERTv2, and XLM-RoBERTa. Seeds change initialization, not the split. The test set is scored after the learning rate is chosen.

A row already present in `results/metrics_seeds.csv` is skipped, so a stopped run can be started again.

After those official-split models, the same command continues HingBERT for one epoch on the training tweets plus the mechanical flip rows. Validation decides whether that checkpoint is kept. The test split is scored only if it is kept. The result is `results/flip_model_metrics.csv`.

On this RTX 2050 (4 GB) the transformer runs use batch size 2, 8 accumulation steps, fp16, and 8-bit AdamW. Sequence length comes from the mBERT subword length distribution and is stored in `results/training_config.json`.

Outputs:

- `results/metrics.csv` — mean and sample standard deviation over seeds
- `results/lr_grid.csv`, `results/ablations.csv`, `results/transfer.csv`
- `results/plots/`
- `results/errors_best.csv` — up to 50 errors from the best transformer
- `results/checkpoints/best` — checkpoint used by the demo

Macro-F1 is the main metric. Accuracy and weighted F1 are reported beside it.

## Notebook

`notebooks/colab_sentimix_hinglish.ipynb` calls `src.experiments.main()`. On Colab, use a GPU runtime. The setup cell installs packages only when it is running inside Colab.

## Demo

```
.\.venv\Scripts\python.exe app.py
```

`predict` loads `results/checkpoints/best` when that directory exists. Until training has copied a checkpoint there, the demo uses a word list.
