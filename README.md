# Romanized-to-Devanagari Nepali Translator (Seq2Seq Transformer)
[![Hugging Face Model](https://shields.io)](https://huggingface.co)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/rabindra-dahal/romanized-to-devanagari-using-sentencepiece-tokenizer-nlp)


An offline, lightweight **Sequence-to-Sequence (Seq2Seq) Neural Machine Translation (NMT)** system built from scratch in PyTorch. It translates Romanized Nepali text (including common chat slang) into native Devanagari script using a custom 2-layer Encoder-Decoder Transformer and SentencePiece subword tokenization.

---

## Key Features

- **Custom Pure-PyTorch Transformer:** Built from scratch without high-level `nn.Transformer` wrappers—featuring custom Sinusoidal Positional Encoding, Scaled Dot-Product Multi-Head Attention, Causal Look-Ahead Masking, and 4D Padding Masks.
- **Lightweight & CPU-Friendly:** ~760K trainable parameters (~2.9 MB checkpoint) designed to train and run fast on standard CPUs.
- **100% Offline Inference:** Caches both the trained SentencePiece model (`spm_nepali.model`) and best model weights (`model.pt`) locally inside `app/` for instant offline translation.
- **Nepali Chat Slang Normalization:** Automatically expands common Romanized chat shorthand (e.g., `kto` → `keta`, `kti` → `keti`, `xa` → `chha`, `manxe` → `manchhe`) before inference.
- **Template-Breaking Data Augmentation:** Combines synthetic subject/noun swaps and 1–2 word prefix fragments during training so the model handles both full sentences and short phrases without hallucinating extra words.

---

## Project Structure

```
romanized-to-devanagari-using-sentencepiece-tokenizer-nlp
├── app/
│   ├── config.py          # Model hyperparameters, special token IDs, and device config
│   ├── vocabulary.py      # SentencePiece Unigram tokenizer wrapper (train & offline load)
│   ├── model.py           # Custom Encoder-Decoder Transformer architecture
│   ├── pipeline.py        # Teacher-forcing training loop & greedy autoregressive inference
│   ├── main.py            # Entry point: data augmentation, training, and interactive CLI
│   ├── spm_nepali.model   # Saved SentencePiece vocabulary (auto-generated on first run)
│   └── model.pt           # Saved best PyTorch model weights (auto-generated after training)
├── requirements.txt       # Python package dependencies
└── README.md
```

---

## Model Architecture Summary

| Component | Specification |
| :--- | :--- |
| **Architecture** | 2-Layer Encoder / 2-Layer Decoder Transformer |
| **Embedding Dimension (`D_MODEL`)** | 128 |
| **Attention Heads (`NHEAD`)** | 4 (`head_dim = 32`) |
| **Feed-Forward Dimension** | 256 (`ReLU` activation) |
| **Tokenizer** | Shared SentencePiece Unigram (`vocab_size = 256`) |
| **Total Parameters** | ~760,576 trainable parameters |
| **Loss & Optimizer** | `CrossEntropyLoss` (`label_smoothing = 0.05`) + `Adam` (`lr = 0.0015`) |

### Data Flow
1. **Preprocessing:** Raw user input passes through `normalize_chat_input()` to expand chat slang, capitalize the first letter, and ensure proper punctuation.
2. **Encoding:** SentencePiece converts the Romanized string into subword IDs (`[<sos>, ..., <eos>]`), which are embedded, scaled by sqrt(d_model), summed with sinusoidal positional signals, and passed through the 2-layer Encoder.
3. **Autoregressive Decoding:** Starting from `<sos>`, the 2-layer Decoder attends to the encoder memory via Cross-Attention and predicts Devanagari subword tokens step-by-step until `<eos>` or a dynamic input-proportional length cap is reached.

---

## Getting Started

### 1. Prerequisites
Install the required Python dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run Training & Interactive Translator
Navigate to the project root and run `app/main.py`:
```bash
python app/main.py
```
- **First Run:** Downloads the `rabindra-dahal/english-nepali-romanized-csv-text-for-kid-learning` dataset from Hugging Face, builds `app/spm_nepali.model`, trains for 20 epochs, saves the lowest-loss checkpoint to `app/model.pt`, and launches the interactive CLI.
- **Subsequent Runs:** Automatically detects `app/spm_nepali.model` and `app/model.pt` and starts the offline translator immediately.

### 3. Download Pre-trained Weights from Hugging Face
 - Pre-trained weights (`model.pt`) and the SentencePiece tokenizer (`spm_nepali.model`) are hosted at hugging face:
 - 👉 **[rabindra-dahal/romanized-to-devanagari-using-sentencepiece-tokenizer-nlp](https://huggingface.co/rabindra-dahal/romanized-to-devanagari-using-sentencepiece-tokenizer-nlp)**
---

## Example Outputs

```text
Romanized > kto naya gainda herchha.
Devanagari > केटा नयाँ गैंडा हेर्छ।

Romanized > meri aama chamcha man parauchhin.
Devanagari > मेरी आमा चम्चा मन पराउँछिन्।

Romanized > kti.
Devanagari > केटी।
```

---

## Retraining from Scratch
If you modify the augmentation rules or hyperparameters in `app/main.py`:
1. Delete `app/model.pt` (keep `app/spm_nepali.model` to preserve vocabulary ID alignment, or delete both to rebuild the vocabulary from scratch).
2. Re-run `python app/main.py`.

---

## Limitations & Future Work

- **Domain & Vocabulary Scope:** Trained on a compact kid-learning corpus with a 256-token shared vocabulary, which can struggle with out-of-vocabulary proper nouns, complex idioms, or long multi-clause sentences. Scaling `VOCAB_SIZE` to `1,000–2,000` and training on broader conversational datasets will improve generalization.
- **Greedy Decoding:** Inference currently selects the highest-probability token at each step (`argmax`). Implementing **Beam Search** with length normalization will yield smoother, more grammatically accurate translations.
- **Phonetic Transliteration Fallback:** Unseen names or English loanwords written in Romanized Nepali can be misspelled; adding a rule-based or character-level transliteration fallback for unknown words would make out-of-domain inputs more reliable.
- **Evaluation Metrics:** Adding automated **BLEU** and **Character Error Rate (CER)** validation splits to benchmark model improvements across training runs.
