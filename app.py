# app.py — Clean Streamlit Urdu Chatbot (Transformer)
# Author: Muhammad Usman Awan | FAST NUCES
# ---------------------------------------------------
# Run: streamlit run app.py
# ---------------------------------------------------

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm

# =========================================================
#                 MODEL COMPONENTS
# =========================================================

# Scaled Dot-Product Attention
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout_p=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn_output, _ = attention(q, k, v, mask, self.dropout)
        concat = attn_output.transpose(1, 2).contiguous().view(bs, -1, self.dim)
        return self.out(concat)


# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout_p=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, dropout_p=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, dropout_p)
        self.ff = FeedForward(dim, ff_dim, dropout_p)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, dropout_p=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads, dropout_p)
        self.cross_attn = MultiHeadAttention(dim, n_heads, dropout_p)
        self.ff = FeedForward(dim, ff_dim, dropout_p)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


# Transformer
class Transformer(nn.Module):
    def __init__(self, vocab_size, dim=128, n_heads=4, n_enc_layers=2,
                 n_dec_layers=2, ff_dim=512, max_len=512, dropout_p=0.2):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.src_embed = nn.Embedding(vocab_size, dim)
        self.tgt_embed = nn.Embedding(vocab_size, dim)
        self.pos_enc = PositionalEncoding(dim, max_len)
        self.enc_layers = nn.ModuleList([EncoderBlock(dim, n_heads, ff_dim, dropout_p) for _ in range(n_enc_layers)])
        self.dec_layers = nn.ModuleList([DecoderBlock(dim, n_heads, ff_dim, dropout_p) for _ in range(n_dec_layers)])
        self.dropout = nn.Dropout(dropout_p)
        self.final_layer = nn.Linear(dim, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return tgt_pad_mask & tgt_sub_mask

    def encode(self, src, src_mask):
        x = self.src_embed(src) * math.sqrt(self.dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.tgt_embed(tgt) * math.sqrt(self.dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.final_layer(dec_out)

# =========================================================
#                INFERENCE FUNCTIONS
# =========================================================

def beam_search(model, src, tokenizer, device, beam_size=3, max_len=80):
    model.eval()
    src_mask = model.make_src_mask(src)
    enc_out = model.encode(src, src_mask)
    bos_id, eos_id = tokenizer.bos_id(), tokenizer.eos_id()  # ✅ use tokenizer values
    beams = [([bos_id], 0.0)]
    with torch.no_grad():
        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                if seq[-1] == eos_id:
                    candidates.append((seq, score))
                    continue
                tgt = torch.tensor([seq], device=device)
                tgt_mask = model.make_tgt_mask(tgt)
                dec_out = model.decode(tgt, enc_out, src_mask, tgt_mask)
                logits = model.final_layer(dec_out)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                top_probs, top_ids = torch.topk(log_probs, beam_size)
                for prob, idx in zip(top_probs, top_ids):
                    candidates.append((seq + [idx.item()], score + prob.item()))
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(seq[-1] == eos_id for seq, _ in beams):
                break
    best_seq = beams[0][0]
    answer_ids = [i for i in best_seq if i not in [0, bos_id, eos_id]]
    return tokenizer.decode_ids(answer_ids)


def greedy_decode(model, src, tokenizer, device, max_len=80):
    model.eval()
    bos_id, eos_id = tokenizer.bos_id(), tokenizer.eos_id()  # ✅ use tokenizer values
    with torch.no_grad():
        src_mask = model.make_src_mask(src)
        enc_out = model.encode(src, src_mask)
        tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_mask = model.make_tgt_mask(tgt)
            dec_out = model.decode(tgt, enc_out, src_mask, tgt_mask)
            logits = model.final_layer(dec_out)
            next_token = logits[0, -1].argmax().item()
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == eos_id:
                break
        answer_ids = [i for i in tgt[0].tolist() if i not in [0, bos_id, eos_id]]
        return tokenizer.decode_ids(answer_ids)


# =========================================================
#                STREAMLIT USER INTERFACE
# =========================================================

st.set_page_config(page_title="Urdu Chatbot (Transformer)", page_icon="🗣️", layout="centered")

st.title("🗣️ Urdu Conversational Chatbot — Transformer (from scratch)")
st.caption("Loads your trained checkpoint (state_dict) and SentencePiece tokenizer for live inference.")

with st.sidebar:
    st.header("⚙️ Configuration")
    ckpt_path = st.text_input("Model checkpoint (.pt)", value="urdu_chatbot.pt")
    spm_path = st.text_input("SentencePiece model (.model)", value="urdu_tokenizer.model")
    decoding = st.radio("Decoding", ["Beam Search", "Greedy"], index=0)
    beam_size = st.slider("Beam size", 2, 8, 3) if decoding == "Beam Search" else None
    max_len = st.slider("Max reply tokens", 16, 256, 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Device: **{device}**")

st.divider()

@st.cache_resource(show_spinner=False)
def load_tokenizer(spm_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    return sp

@st.cache_resource(show_spinner=True)
def load_model(ckpt_path, vocab_size, device):
    model = Transformer(vocab_size=vocab_size, dim=128, n_heads=4,
                        n_enc_layers=2, n_dec_layers=2, ff_dim=512,
                        max_len=512, dropout_p=0.2).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

try:
    sp = load_tokenizer(spm_path)
    vocab_size = sp.GetPieceSize()
    model = load_model(ckpt_path, vocab_size, device)
except Exception as e:
    st.error(f"Error loading model/tokenizer: {e}")
    st.stop()

st.success("✅ Model and tokenizer loaded successfully.")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("اپنا سوال یہاں لکھیں… (Urdu)")
if user_input:
    st.session_state.chat.append(("user", user_input))
    with st.chat_message("assistant"):
        with st.spinner("Generating…"):
            bos = sp.bos_id() if sp.bos_id() != -1 else 1
            eos = sp.eos_id() if sp.eos_id() != -1 else 2
            ids = [bos] + sp.EncodeAsIds(user_input) + [eos]
            src = torch.tensor([ids], dtype=torch.long, device=device)
            if decoding == "Beam Search":
                reply = beam_search(model, src, sp, device, beam_size=beam_size, max_len=max_len)
            else:
                reply = greedy_decode(model, src, sp, device, max_len=max_len)
            st.markdown(reply if reply.strip() else "...")
            st.session_state.chat.append(("assistant", reply))
