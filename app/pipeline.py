import torch
import re
from config import Config


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epochs=20):
    """
    Runs one full pass (epoch) over the training DataLoader using Teacher Forcing,
    computes cross-entropy loss, clips gradients, and updates model weights.
    """
    model.train()
    total_loss = 0
    total_batches = len(dataloader)

    print(f"  Epoch {epoch:02d}/{total_epochs} Progress: [", end="", flush=True)
    dot_interval = max(1, total_batches // 12)

    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
        src_batch = src_batch.to(Config.DEVICE)
        tgt_batch = tgt_batch.to(Config.DEVICE)

        optimizer.zero_grad()

        tgt_input = tgt_batch[:, :-1]
        tgt_expected = tgt_batch[:, 1:]

        output = model(src_batch, tgt_input)
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % dot_interval == 0:
            print("■", end="", flush=True)

    print("] Done")
    return total_loss / total_batches


def evaluate_translation(model, sentence, src_vocab, tgt_vocab):
    """
    Translates a single Romanized Nepali sentence into Devanagari using
    autoregressive greedy decoding (selecting the highest-probability token at each step).
    """
    model.eval()

    encode_fn = src_vocab.encode if hasattr(src_vocab, 'encode') else tgt_vocab.encode
    src_tokens = torch.tensor([encode_fn(sentence)], device=Config.DEVICE)

    tgt_tokens = [Config.SOS_IDX]
    max_target_tokens = min(Config.MAX_LEN, max(6, int(src_tokens.size(1) * 2.2) + 4))

    with torch.no_grad():
        memory, src_mask = model.encode(src_tokens)

        for _ in range(max_target_tokens):
            tgt_tensor = torch.tensor([tgt_tokens], device=Config.DEVICE)
            out = model.decode(tgt_tensor, memory, src_mask)
            next_token = model.fc_out(out)[0, -1].argmax().item()

            if next_token == Config.EOS_IDX or len(tgt_tokens) > max_target_tokens:
                break

            tgt_tokens.append(next_token)

    raw_output = tgt_vocab.decode(tgt_tokens)
    text_clean = re.sub(r'\s+।', '।', raw_output)
    return re.sub(r'\s+', ' ', text_clean).strip()
