import os
import sentencepiece as spm


class SentencePieceVocabulary:
    def __init__(self, corpus=None, vocab_size=256):
        app_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_prefix = os.path.join(app_dir, "spm_nepali")
        self.vocab_size = vocab_size

        if not os.path.exists(f"{self.model_prefix}.model"):
            if corpus is None:
                raise ValueError("Corpus must be provided when 'spm_nepali.model' does not exist.")
            self._train_spm(corpus, vocab_size)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{self.model_prefix}.model")

    def _train_spm(self, corpus, vocab_size):
        # Save corpus to a temporary file for SentencePiece trainer
        temp_corpus_path = f"{self.model_prefix}_temp_corpus.txt"
        with open(temp_corpus_path, "w", encoding="utf-8") as f:
            for text in corpus:
                f.write(text + "\n")

        # Train Unigram model with strict parameters
        spm.SentencePieceTrainer.train(
            input=temp_corpus_path,
            model_prefix=self.model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='UNIGRAM',
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3
        )

        if os.path.exists(temp_corpus_path):
            os.remove(temp_corpus_path)

    def encode(self, text):
        # Returns list of token IDs wrapped with SOS (1) and EOS (2)
        return [1] + self.sp.encode(text) + [2]

    def decode(self, ids):
        # Filter out padding (0), SOS (1), and EOS (2) tokens before decoding
        filtered_ids = [i for i in ids if i not in (0, 1, 2)]
        return self.sp.decode(filtered_ids)

    def __len__(self):
        return self.sp.get_piece_size()
