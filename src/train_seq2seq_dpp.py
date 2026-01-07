import os
import re
import math
import random
from tqdm import tqdm
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# =========================
# 0) Utils DDP
# =========================
def setup_ddp():
    """
    torchrun sets env vars: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    """
    # Windows: use gloo. Linux: prefer nccl when CUDA is available.
    if os.name == "nt":
        backend = "gloo"
    else:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return device, local_rank, backend


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


# =========================
# 1) Dades + vocab
# =========================
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

def normalize(s: str) -> str:
    s = s.lower().strip()
    # Ajusta regex segons llengües. Aquí deixem lletres llatines + accents comuns.
    s = re.sub(r"[^\w\-·'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


class PairDataset(Dataset):
    def __init__(self, pairs_tensors: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.data = pairs_tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # (src, tgt)


# =========================
# 2) Model: Encoder / Attention / Decoder / Seq2Seq
# =========================
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):  # [B,S]
        emb = self.embedding(src)              # [B,S,E]
        outputs, (h, c) = self.lstm(emb)       # outputs [B,S,H]
        return outputs, (h, c)


class DotAttention(nn.Module):
    def forward(self, dec_hidden, enc_outputs, src_mask):
        # dec_hidden: [B,H], enc_outputs: [B,S,H], src_mask: [B,S]
        scores = torch.bmm(enc_outputs, dec_hidden.unsqueeze(2)).squeeze(2)  # [B,S]
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = F.softmax(scores, dim=1)                                     # [B,S]
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)       # [B,H]
        return context, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attn = DotAttention()
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, vocab_size)

    def forward(self, input_tok, state, enc_outputs, src_mask):
        # input_tok: [B]
        h, c = state
        emb = self.embedding(input_tok).unsqueeze(1)  # [B,1,E]

        dec_hidden = h[-1]                            # [B,H]
        context, _ = self.attn(dec_hidden, enc_outputs, src_mask)  # [B,H]

        lstm_in = torch.cat([emb, context.unsqueeze(1)], dim=2)    # [B,1,E+H]
        out, (h2, c2) = self.lstm(lstm_in, (h, c))                 # out [B,1,H]
        out = out.squeeze(1)                                       # [B,H]

        logits = self.fc(torch.cat([out, context], dim=1))         # [B,V]
        return logits, (h2, c2)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def forward(self, src, tgt, teacher_forcing: float = 0.7):
        # src: [B,S], tgt: [B,T]
        B, T = tgt.size()
        enc_outputs, state = self.encoder(src)
        src_mask = (src != self.src_pad_idx)  # [B,S]

        logits_all = []
        input_tok = tgt[:, 0]  # SOS
        for t in range(1, T):
            logits, state = self.decoder(input_tok, state, enc_outputs, src_mask)
            logits_all.append(logits.unsqueeze(1))  # [B,1,V]

            use_tf = random.random() < teacher_forcing
            top1 = logits.argmax(dim=1)
            input_tok = tgt[:, t] if use_tf else top1

        return torch.cat(logits_all, dim=1)  # [B,T-1,V]


# =========================
# 3) Train + Translate
# =========================


def train_one_epoch(ddp_model, loader, sampler, optimizer, criterion, device, epoch):
    ddp_model.train()
    sampler.set_epoch(epoch)

    total_loss = 0.0

    # Només rank 0 mostra la barra
    is_main = (not dist.is_initialized()) or dist.get_rank() == 0

    iterable = loader
    if is_main:
        iterable = tqdm(
            loader,
            desc=f"Epoch {epoch}",
            leave=False,
            dynamic_ncols=True
        )

    for src, tgt in iterable:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = ddp_model(src, tgt, teacher_forcing=0.7)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if is_main:
            iterable.set_postfix(loss=f"{loss.item():.4f}")

    # Average loss across GPUs
    avg = torch.tensor(total_loss / max(1, len(loader)), device=device)
    dist.all_reduce(avg, op=dist.ReduceOp.SUM)
    avg = avg / dist.get_world_size()

    return avg.item()


@torch.no_grad()
def translate_greedy(model, sentence: str, src_vocab: Vocab, tgt_vocab: Vocab,
                     max_src_len: int, max_tgt_len: int, device):
    model.eval()
    s = normalize(sentence)
    src = torch.tensor(encode(s, src_vocab, max_src_len), dtype=torch.long).unsqueeze(0).to(device)

    enc_outputs, state = model.encoder(src)
    src_mask = (src != src_vocab.stoi[PAD])

    tok = torch.tensor([tgt_vocab.stoi[SOS]], dtype=torch.long).to(device)
    out_ids = [tgt_vocab.stoi[SOS]]

    for _ in range(max_tgt_len):
        logits, state = model.decoder(tok, state, enc_outputs, src_mask)
        tok = logits.argmax(dim=1)
        out_ids.append(tok.item())
        if tok.item() == tgt_vocab.stoi[EOS]:
            break

    return decode(out_ids, tgt_vocab)


# =========================
# 4) Main
# =========================
def main():
    device, local_rank, backend = setup_ddp()

    if is_main_process():
        print(f"DDP backend: {backend}")
        print(f"World size: {ddp_world_size()}")

    # Seeds (cada procés diferent però controlat)
    seed = 1234 + local_rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- DADES D'EXEMPLE (substitueix per les teves parelles reals) ----
    with open("data/europarl.en-ca.ca", encoding="utf-8") as f:
        DATA_CA = f.readlines()
    with open("data/europarl.en-ca.en", encoding="utf-8") as f:
        DATA_EN = f.readlines()
    DATA = list(zip(DATA_CA, DATA_EN))[:100_000]
    print("Preprocessing...")
    pairs = [(normalize(a), normalize(b)) for a, b in tqdm(DATA)]

    MAX_SRC_LEN = 30
    MAX_TGT_LEN = 30

    src_vocab = build_vocab([a for a, _ in pairs], min_freq=1)
    tgt_vocab = build_vocab([b for _, b in pairs], min_freq=1)

    pairs_tensors = []
    for src, tgt in pairs:
        pairs_tensors.append((
            torch.tensor(encode(src, src_vocab, MAX_SRC_LEN), dtype=torch.long),
            torch.tensor(encode(tgt, tgt_vocab, MAX_TGT_LEN), dtype=torch.long),
        ))

    dataset = PairDataset(pairs_tensors)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=64,              # batch PER GPU (global = 64 * world_size)
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(2 > 0),
    )

    # ---- MODEL ----
    SRC_PAD = src_vocab.stoi[PAD]
    TGT_PAD = tgt_vocab.stoi[PAD]

    emb_dim = 128
    hid_dim = 256

    enc = Encoder(len(src_vocab.itos), emb_dim, hid_dim, SRC_PAD)
    dec = Decoder(len(tgt_vocab.itos), emb_dim, hid_dim, TGT_PAD)
    base_model = Seq2Seq(enc, dec, SRC_PAD).to(device)

    ddp_model = DDP(
        base_model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
    )

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD)

    # ---- TRAIN ----
    EPOCHS = 8
    for epoch in range(1, EPOCHS + 1):
        print("Epoch", epoch, "/", EPOCHS)
        loss = train_one_epoch(ddp_model, loader, sampler, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch} | loss={loss:.4f}")

    # ---- SAVE CHECKPOINT (només rank 0) ----
    if is_main_process():
        ckpt = {
            "model_state": ddp_model.module.state_dict(),
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "emb_dim": emb_dim,
            "hid_dim": hid_dim,
            "MAX_SRC_LEN": MAX_SRC_LEN,
            "MAX_TGT_LEN": MAX_TGT_LEN,
        }
        torch.save(ckpt, "seq2seq_lstm_ddp.pt")
        print("Checkpoint guardat a seq2seq_lstm_ddp.pt")

        # prova traducció
        for s in ["bon dia", "on es el bany", "necessito ajuda", "m'agrada aprendre"]:
            print(f"{s} -> {translate_greedy(ddp_model.module, s, src_vocab, tgt_vocab, MAX_SRC_LEN, MAX_TGT_LEN, device)}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
