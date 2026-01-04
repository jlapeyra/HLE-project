import re
import random
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------
# 1) Dades i preprocessat
# -------------------------
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w·'\s]", " ", s)  # ajusta segons llengua
    s = re.sub(r"\s+", " ", s).strip()
    return s

with open("data/europarl.ca.txt", encoding="utf-8") as f:
    DATA_CA = f.readlines()
with open("data/europarl.es.txt", encoding="utf-8") as f:
    DATA_ES = f.readlines()

#pairs: List[Tuple[str, str]] = []

DATA = list(zip(DATA_CA, DATA_ES))
N = len(DATA)
DATA = DATA[:int(0.7*N)]
pairs = [(normalize(a), normalize(b)) for a, b in tqdm(DATA)]

# -------------------------
# 2) Vocabulari
# -------------------------
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

@dataclass
class Vocab:
    stoi: dict
    itos: list

def build_vocab(sentences: List[str], min_freq: int = 1) -> Vocab:
    c = Counter()
    for s in sentences:
        c.update(s.split())
    itos = [PAD, SOS, EOS, UNK]
    for w, f in c.items():
        if f >= min_freq:
            itos.append(w)
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)

src_sentences = [a for a, _ in pairs]
tgt_sentences = [b for _, b in pairs]
src_vocab = build_vocab(src_sentences, min_freq=1)
tgt_vocab = build_vocab(tgt_sentences, min_freq=1)

def encode(sentence: str, vocab: Vocab, max_len: int) -> List[int]:
    ids = [vocab.stoi.get(w, vocab.stoi[UNK]) for w in sentence.split()]
    ids = [vocab.stoi[SOS]] + ids + [vocab.stoi[EOS]]
    if len(ids) < max_len:
        ids += [vocab.stoi[PAD]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = vocab.stoi[EOS]
    return ids

def decode(ids: List[int], vocab: Vocab) -> str:
    words = []
    for i in ids:
        w = vocab.itos[i]
        if w in (SOS, PAD):
            continue
        if w == EOS:
            break
        words.append(w)
    return " ".join(words)

# Longituds màximes (fes-les més grans amb dades reals)
MAX_SRC_LEN = 30
MAX_TGT_LEN = 30

data = []
for src, tgt in pairs:
    data.append((
        torch.tensor(encode(src, src_vocab, MAX_SRC_LEN), dtype=torch.long),
        torch.tensor(encode(tgt, tgt_vocab, MAX_TGT_LEN), dtype=torch.long),
    ))

# -------------------------
# 3) Model: Encoder, Attention, Decoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):  # src: [B, S]
        emb = self.embedding(src)              # [B, S, E]
        outputs, (h, c) = self.lstm(emb)       # outputs: [B, S, H]
        return outputs, (h, c)

class DotAttention(nn.Module):
    def forward(self, dec_hidden, enc_outputs, src_mask):
        # dec_hidden: [B, H]   enc_outputs: [B, S, H]
        scores = torch.bmm(enc_outputs, dec_hidden.unsqueeze(2)).squeeze(2)  # [B, S]
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = F.softmax(scores, dim=1)  # [B, S]
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)  # [B, H]
        return context, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attn = DotAttention()
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, vocab_size)

    def forward(self, input_tok, state, enc_outputs, src_mask):
        # input_tok: [B] (1 pas)
        (h, c) = state
        emb = self.embedding(input_tok).unsqueeze(1)  # [B, 1, E]
        dec_hidden = h[-1]                            # [B, H]
        context, _ = self.attn(dec_hidden, enc_outputs, src_mask)  # [B, H]
        lstm_in = torch.cat([emb, context.unsqueeze(1)], dim=2)    # [B, 1, E+H]
        out, (h2, c2) = self.lstm(lstm_in, (h, c))                 # out: [B,1,H]
        out = out.squeeze(1)                                       # [B,H]
        logits = self.fc(torch.cat([out, context], dim=1))         # [B,V]
        return logits, (h2, c2)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def forward(self, src, tgt, teacher_forcing=0.5):
        # src: [B,S], tgt:[B,T]
        B, T = tgt.size()
        enc_outputs, state = self.encoder(src)
        src_mask = (src != self.src_pad_idx)  # [B,S]

        logits_all = []
        input_tok = tgt[:, 0]  # SOS
        for t in range(1, T):
            logits, state = self.decoder(input_tok, state, enc_outputs, src_mask)
            logits_all.append(logits.unsqueeze(1))
            use_tf = random.random() < teacher_forcing
            top1 = logits.argmax(dim=1)
            input_tok = tgt[:, t] if use_tf else top1

        return torch.cat(logits_all, dim=1)  # [B, T-1, V]

# -------------------------
# 4) Entrenament
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SRC_PAD = src_vocab.stoi[PAD]
TGT_PAD = tgt_vocab.stoi[PAD]

emb_dim = 128
hid_dim = 256

enc = Encoder(len(src_vocab.itos), emb_dim, hid_dim, SRC_PAD)
dec = Decoder(len(tgt_vocab.itos), emb_dim, hid_dim, TGT_PAD)
model = Seq2Seq(enc, dec, SRC_PAD, device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD)

def batchify(data, batch_size):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def train_epoch(batch_size=4):
    model.train()
    total_loss = 0.0
    for batch in batchify(data, batch_size):
        src = torch.stack([x[0] for x in batch]).to(device)
        tgt = torch.stack([x[1] for x in batch]).to(device)

        optimizer.zero_grad()
        logits = model(src, tgt, teacher_forcing=0.7)  # [B,T-1,V]
        # Compareu amb tgt[:,1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, math.ceil(len(data)/batch_size))

for epoch in tqdm(range(1, 201)):
    loss = train_epoch(batch_size=4)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | loss={loss:.4f}")

checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "src_vocab": src_vocab,
    "tgt_vocab": tgt_vocab,
    "emb_dim": emb_dim,
    "hid_dim": hid_dim,
    "MAX_SRC_LEN": MAX_SRC_LEN,
    "MAX_TGT_LEN": MAX_TGT_LEN,
}

torch.save(checkpoint, "translator_seq2seq.pt")
print("Model guardat!")

