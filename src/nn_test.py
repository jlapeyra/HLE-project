import os
import torch
from typing import Any

# Reuse definitions from the training module to avoid duplication.
from nn_train import Encoder, Decoder, Seq2Seq, Vocab, normalize, encode  # type: ignore


def _ensure_vocab(obj: Any) -> Vocab:
    if isinstance(obj, Vocab):
        return obj
    if hasattr(obj, "stoi") and hasattr(obj, "itos"):
        return Vocab(stoi=obj.stoi, itos=list(obj.itos))
    if isinstance(obj, dict):
        return Vocab(stoi=obj.get("stoi", {}), itos=obj.get("itos", []))
    raise ValueError("Unsupported vocab object in checkpoint")


def load_checkpoint(path: str | None = None, device: torch.device | None = None):
    if path is None:
        path = "translator_seq2seq_v2.pt"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([Vocab])

    ckpt = torch.load(path, map_location=device, weights_only=False)

    src_vocab = _ensure_vocab(ckpt.get("src_vocab"))
    tgt_vocab = _ensure_vocab(ckpt.get("tgt_vocab"))

    emb_dim = ckpt.get("emb_dim")
    hid_dim = ckpt.get("hid_dim")
    MAX_SRC_LEN = ckpt.get("MAX_SRC_LEN")
    MAX_TGT_LEN = ckpt.get("MAX_TGT_LEN")

    SRC_PAD = src_vocab.stoi.get("<pad>")
    TGT_PAD = tgt_vocab.stoi.get("<pad>")

    enc = Encoder(len(src_vocab.itos), emb_dim, hid_dim, SRC_PAD)
    dec = Decoder(len(tgt_vocab.itos), emb_dim, hid_dim, TGT_PAD)
    model = Seq2Seq(enc, dec, SRC_PAD, device).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return {
        "model": model,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "device": device,
        "MAX_SRC_LEN": MAX_SRC_LEN,
        "MAX_TGT_LEN": MAX_TGT_LEN,
    }


@torch.no_grad()
def translate(sentence: str, model, src_vocab: Vocab, tgt_vocab: Vocab, MAX_SRC_LEN: int, MAX_TGT_LEN: int, device=None) -> str:
    device = device or next(model.parameters()).device
    sent = normalize(sentence)
    src = torch.tensor([encode(sent, src_vocab, MAX_SRC_LEN)], dtype=torch.long).to(device)

    enc_outputs, state = model.encoder(src)
    src_mask = (src != model.src_pad_idx)

    tok = torch.tensor([tgt_vocab.stoi.get("<sos>")], dtype=torch.long).to(device)
    out_ids = [tgt_vocab.stoi.get("<sos>")]

    for _ in range(MAX_TGT_LEN):
        logits, state = model.decoder(tok, state, enc_outputs, src_mask)
        tok = logits.argmax(dim=1)
        out_ids.append(tok.item())
        if tok.item() == tgt_vocab.stoi.get("<eos>"):
            break

    words = []
    for i in out_ids:
        w = tgt_vocab.itos[i]
        if w in ("<sos>", "<pad>"):
            continue
        if w == "<eos>":
            break
        words.append(w)
    return " ".join(words)


if __name__ == "__main__":
    info = load_checkpoint()
    model = info["model"]
    src_vocab = info["src_vocab"]
    tgt_vocab = info["tgt_vocab"]
    device = info["device"]
    MAX_SRC_LEN = info["MAX_SRC_LEN"]
    MAX_TGT_LEN = info["MAX_TGT_LEN"]

    examples = [
        "El Parlament Europeu ha aprovat la llei.",
        "Necessito ajuda amb el meu equipament informàtic.",
        "Quin temps farà demà a Barcelona?",
        "M'agrada escoltar música clàssica mentre treballo.",
        "On és la biblioteca més propera?",
        "Podries recomanar-me un bon restaurant per sopar?",
        "Estic aprenent a programar en Python.",
        "La intel·ligència artificial està transformant moltes indústries.",
        "Quines són les últimes notícies sobre tecnologia?",
        "M'agradaria reservar una habitació d'hotel per a dues persones.",
    ]
    for s in examples:
        print(f"{s} -> {translate(s, model, src_vocab, tgt_vocab, MAX_SRC_LEN, MAX_TGT_LEN, device)}")
