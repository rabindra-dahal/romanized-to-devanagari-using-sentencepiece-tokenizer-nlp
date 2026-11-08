import os
import warnings
import time
import random

# 1. Prevent Hugging Face Hub from triggering administrative symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# 2. Suppress internal PyTorch prototype and canopy formatting messages
warnings.filterwarnings("ignore", category=UserWarning, message=".*nested tensors is in prototype stage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Support for mismatched key_padding_mask.*")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from config import Config
from vocabulary import SentencePieceVocabulary
from model import NepaliTransformer
from pipeline import train_one_epoch, evaluate_translation


class KidLearningDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def build_augmented_pairs(hf_dataset):
    """Combines original pairs with synthetic noun/subject swaps and short prefix fragments."""
    base_pairs = [
        (str(row.get('romanized', row.get('english', ''))), str(row['nepali']))
        for row in hf_dataset
    ]

    swaps = [
        (("Meri bahini", "मेरी बहिनी"), ("Meri aama", "मेरी आमा")),
        (("Meri aama", "मेरी आमा"), ("Meri didi", "मेरी दिदी")),
        (("naya chamcha", "नयाँ चम्चा"), ("chamcha", "चम्चा")),
        (("naya gainda", "नयाँ गैंडा"), ("gainda", "गैंडा")),
        (("Keti", "केटी"), ("Keta", "केटा")),
    ]

    augmented = list(base_pairs)
    random.seed(42)

    for rom, nep in base_pairs:
        # 1. Noun/subject swaps to break template memorization
        for (r_old, n_old), (r_new, n_new) in swaps:
            if r_old in rom and n_old in nep and random.random() < 0.35:
                augmented.append((rom.replace(r_old, r_new, 1), nep.replace(n_old, n_new, 1)))

        # 2. 1-word and 2-word prefix fragments so short inputs emit <eos> early
        r_words = rom.rstrip(".?!").split()
        n_words = nep.rstrip("।?!").split()
        if len(r_words) == len(n_words) and len(r_words) >= 3 and random.random() < 0.25:
            cut = random.choice([1, 2])
            r_frag = " ".join(r_words[:cut]) + "."
            n_frag = " ".join(n_words[:cut]) + "।"
            augmented.append((r_frag, n_frag))

    return augmented


def collate_fn(batch, vocab):
    from torch.nn.utils.rnn import pad_sequence
    src_list, tgt_list = zip(*batch)

    src_tensors = [torch.tensor(vocab.encode(s)) for s in src_list]
    tgt_tensors = [torch.tensor(vocab.encode(t)) for t in tgt_list]

    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)

    return src_batch, tgt_batch


def normalize_chat_input(text):
    slang_map = {
        # People & Family
        "kto": "keta",
        "kta": "keta",
        "keto": "keta",
        "kti": "keti",
        "kt": "keti",
        "manxe": "manchhe",
        "xora": "chhora",
        "xori": "chhori",
        "tmi": "timi",
        "hjr": "hajur",
        "sathi": "saathi",
        # Verbs ('chha' forms & common verbs)
        "x": "chha",
        "xa": "chha",
        "xaina": "chhaina",
        "xain": "chhaina",
        "xu": "chhu",
        "xau": "chhau",
        "xan": "chhan",
        "garxa": "garchha",
        "garxau": "garchhau",
        "garxan": "garchhan",
        "janxa": "janchha",
        "khanxa": "khanchha",
        "herxa": "herchha",
        "khna": "khana",
        # Question words & Adjectives
        "k": "ke",
        "kina": "kina",
        "kna": "kina",
        "kasto": "kasto",
        "ksto": "kasto",
        "rmro": "ramro",
        "narmro": "naramro",
        "thulo": "thulo",
        "sano": "sano",
        # Time & Everyday Nouns
        "aja": "aaja",
        "voli": "bholi",
        "hjo": "hijo",
        "ghr": "ghar",
        "pani": "paani",
        "bhat": "bhaat",
        "dudh": "doodh"
    }

    words = text.strip().split()
    normalized_words = []
    for w in words:
        core = w.rstrip(".?!")
        punct = w[len(core):]
        mapped = slang_map.get(core.lower(), core)
        normalized_words.append(mapped + punct)

    clean_text = " ".join(normalized_words)
    if clean_text:
        clean_text = clean_text[0].upper() + clean_text[1:]
    if clean_text and clean_text[-1] not in {".", "?", "!"}:
        clean_text += "."
    return clean_text


def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(app_dir, "model.pt")
    spm_model_path = os.path.join(app_dir, "spm_nepali.model")

    # Architecture & training settings
    Config.D_MODEL = 128
    Config.NHEAD = 4
    Config.NUM_ENCODER_LAYERS = 2
    Config.NUM_DECODER_LAYERS = 2
    Config.DIM_FEEDFORWARD = 256
    Config.DROPOUT = 0.05
    Config.MAX_LEN = 50
    epochs = 20

    raw_dataset = None
    if os.path.exists(spm_model_path):
        print(f"[!] Loading saved SentencePiece model from '{spm_model_path}' (Offline Mode)...")
        vocab = SentencePieceVocabulary(corpus=None, vocab_size=256)
    else:
        print("Loading Kid-Learning parallel corpus from HF to build SentencePiece vocabulary...")
        raw_dataset = load_dataset("rabindra-dahal/english-nepali-romanized-csv-text-for-kid-learning", split="train")
        all_text_corpus = []
        for row in raw_dataset:
            rom = row.get('romanized', row.get('english', ''))
            nep = row.get('nepali', '')
            if rom:
                all_text_corpus.append(str(rom))
            if nep:
                all_text_corpus.append(str(nep))

        vocab = SentencePieceVocabulary(all_text_corpus, vocab_size=256)
        print(f"[+] SentencePiece initialized! Vocabulary size: {len(vocab)}")

    model = NepaliTransformer(len(vocab), len(vocab)).to(Config.DEVICE)

    if os.path.exists(checkpoint_path):
        print(f"[!] Loading saved model weights from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        if raw_dataset is None:
            raw_dataset = load_dataset("rabindra-dahal/english-nepali-romanized-csv-text-for-kid-learning", split="train")

        pairs = build_augmented_pairs(raw_dataset)
        print(f"[+] Training set prepared with swaps & short fragments: {len(pairs)} pairs.")

        dataset = KidLearningDataset(pairs)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, vocab),
            drop_last=True
        )

        criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX, label_smoothing=0.05)
        optimizer = optim.Adam(model.parameters(), lr=0.0015)

        print(f"\nBeginning 2-layer padded training on device: {Config.DEVICE}")
        total_start_time = time.time()
        interval_start_time = time.time()
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epochs=epochs)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

            if epoch % 5 == 0:
                interval_elapsed = time.time() - interval_start_time
                time_str = f"{interval_elapsed / 60:.1f} min" if interval_elapsed >= 60 else f"{interval_elapsed:.1f} sec"
                print(f"Epoch {epoch:02d}/{epochs} | CrossEntropyLoss: {avg_loss:.4f} (Best: {best_loss:.4f}) | Time Taken: {time_str}")
                interval_start_time = time.time()

        total_elapsed = time.time() - total_start_time
        print(f"\nTraining Complete! Total Session Duration: {total_elapsed / 60:.1f} min")
        print(f"[+] Best checkpoint saved to '{checkpoint_path}' (Loss: {best_loss:.4f})!")

    print("\n--- Quick Sanity Check ---")
    sample_phrases = [
        "Uni ahile chamcha man parauchhin.",
        "Meri aama chamcha man parauchhin.",
        "Kto naya gainda herchha.",
        "Meri aama.",
        "kto.",
        "kti."
    ]
    for text in sample_phrases:
        formatted = normalize_chat_input(text)
        prediction = evaluate_translation(model, formatted, vocab, vocab)
        print(f"Romanized : {text}")
        print(f"Devanagari: {prediction}\n")

    print("=" * 55)
    print("LIVE ROMANIZED-TO-DEVANAGARI TRANSLATOR (OFFLINE)")
    print("Type a Romanized Nepali sentence and press Enter.")
    print("Type 'exit' or 'quit' to close.")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nRomanized > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Closing translator. Goodbye!")
                break
            formatted_input = normalize_chat_input(user_input)
            translation = evaluate_translation(model, formatted_input, vocab, vocab)
            print(f"Devanagari > {translation}")
        except (KeyboardInterrupt, EOFError):
            print("\nClosing translator. Goodbye!")
            break


if __name__ == '__main__':
    main()
