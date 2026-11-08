import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config


# ----------------------------------------------------
# 1. SINUSOIDAL POSITIONAL ENCODING BLOCK
# ----------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Injects deterministic sine and cosine wave signals into token embeddings
    so the Transformer knows the relative and absolute order of tokens in a sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ----------------------------------------------------
# 2. BROADCAST-COMPATIBLE MULTI-HEAD ATTENTION BLOCK
# ----------------------------------------------------
class CustomMultiHeadAttention(nn.Module):
    """
    Implements Scaled Dot-Product Multi-Head Attention in pure PyTorch.
    Bypasses native C++ nn.Transformer thread deadlocks on Windows CPUs and supports
    both 2D causal masks and 4D source padding masks.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.q_linear(q).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(context)


# ----------------------------------------------------
# 3. TRANSFORMER ENCODER & DECODER LAYERS
# ----------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    """
    Single Encoder block consisting of:
    1. Multi-Head Self-Attention (with source padding mask) + Residual + LayerNorm
    2. Position-wise Feed-Forward Network + Residual + LayerNorm
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_out = self.self_attn(src, src, src, mask=src_mask)
        x = self.norm1(src + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_out))


class TransformerDecoderLayer(nn.Module):
    """
    Single Decoder block consisting of:
    1. Masked Multi-Head Self-Attention (prevents peeking at future target tokens)
    2. Multi-Head Cross-Attention (attends to encoded Romanized memory while ignoring <pad>)
    3. Position-wise Feed-Forward Network + Residual + LayerNorm
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        self_attn_out = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        x = self.norm1(tgt + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_out))


# ----------------------------------------------------
# 4. 2-LAYER SEQ2SEQ TRANSFORMER WRAPPER
# ----------------------------------------------------
class NepaliTransformer(nn.Module):
    """
    Complete Sequence-to-Sequence Transformer for Romanized-to-Devanagari translation.
    Exposes standalone encode() and decode() methods so inference only encodes the
    source sentence once before running autoregressive decoding steps.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, Config.D_MODEL)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, Config.D_MODEL)
        self.positional_encoding = PositionalEncoding(Config.D_MODEL)
        self.emb_dropout = nn.Dropout(Config.DROPOUT)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(Config.D_MODEL, Config.NHEAD, Config.DIM_FEEDFORWARD, Config.DROPOUT)
            for _ in range(Config.NUM_ENCODER_LAYERS)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(Config.D_MODEL, Config.NHEAD, Config.DIM_FEEDFORWARD, Config.DROPOUT)
            for _ in range(Config.NUM_DECODER_LAYERS)
        ])

        self.fc_out = nn.Linear(Config.D_MODEL, tgt_vocab_size)

    def _generate_causal_mask(self, sz, device):
        return torch.tril(torch.ones(sz, sz, device=device))

    def _generate_src_mask(self, src):
        return (src != Config.PAD_IDX).unsqueeze(1).unsqueeze(2)

    def encode(self, src):
        src_mask = self._generate_src_mask(src)
        memory = self.emb_dropout(
            self.positional_encoding(self.src_embedding(src) * math.sqrt(Config.D_MODEL))
        )
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask=src_mask)
        return memory, src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)
        out = self.emb_dropout(
            self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(Config.D_MODEL))
        )
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask=tgt_mask, src_mask=src_mask)
        return out

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        out = self.decode(tgt, memory, src_mask)
        return self.fc_out(out)
